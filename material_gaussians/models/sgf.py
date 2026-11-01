import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from material_gaussians.orientation import (
    normalize_quaternions,
    canonicalize_quaternions,
    symmetry_equivalents,
    crystal_symmetry_quaternions,
    misorientation_deg,
    align_quaternions,
)

DEFAULTS = dict(
    iterations=240,
    fit_epochs=None,
    batch_size=768,
    candidates=24,
    learning_rate=0.025,
    seed_stride=6,
    max_gaussians=30000,
    orientation_cluster_deg=4.0,
    orientation_sample_limit=12000,
    orientation_sigma_deg=4.0,
    boundary_weight=0.15,
    boundary_batch_size=96,
    boundary_threshold_deg=10.0,
    geometry_regularization=0.001,
    anisotropic=True,
    decode="modal",
    device="cpu",
    threads=4,
    max_train_points=300000,
    log_every=40,
    candidate_refresh=40,
)


def _torch_mul(a, b):
    w = a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True)
    v = (
        a[..., :1] * b[..., 1:]
        + b[..., :1] * a[..., 1:]
        + torch.linalg.cross(a[..., 1:], b[..., 1:])
    )
    return torch.cat((w, v), dim=-1)


class GaussianOrientationField:
    def fit(
        self,
        q_observed,
        phase_observed,
        valid_observed,
        z_um,
        spacing_yx_um,
        config=None,
        seed=0,
    ):
        started = time.perf_counter()
        self.config = {**DEFAULTS, **(config or {})}
        c = self.config
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(int(c["threads"]))
        device = c["device"]
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        q = np.asarray(q_observed, dtype=np.float32)
        phase = np.asarray(phase_observed, dtype=np.int16)
        valid = (
            np.asarray(valid_observed, bool)
            & (phase > 0)
            & (np.linalg.norm(q, axis=-1) > 0.5)
        )
        z_um = np.asarray(z_um, dtype=float)
        self.spacing_yx_um = np.asarray(spacing_yx_um, dtype=float)
        if (
            q.ndim != 4
            or q.shape[-1] != 4
            or phase.shape != q.shape[:-1]
            or (len(z_um) != len(q))
        ):
            raise ValueError(
                "Expected observed q[Nz,Y,X,4], phase/valid[Nz,Y,X], z_um[Nz]."
            )
        if not np.all(np.diff(z_um) > 0) or np.any(self.spacing_yx_um <= 0):
            raise ValueError(
                "Observed z must increase strictly; physical spacing must be positive."
            )
        if not valid.any():
            raise ValueError("No valid observed orientations.")
        self.shape_yx = q.shape[1:3]
        self.observed_z_um = z_um
        self.phase_symmetry = {
            int(k): str(v) for k, v in c.get("phase_symmetry", {1: "cubic"}).items()
        }
        missing = set(np.unique(phase[valid])) - self.phase_symmetry.keys()
        if missing:
            raise ValueError(
                f"Missing explicit phase_symmetry entries for phases {missing}"
            )
        q = normalize_quaternions(q).astype(np.float32)
        iz, iy, ix = np.nonzero(valid)
        xyz = np.column_stack(
            (z_um[iz], iy * self.spacing_yx_um[0], ix * self.spacing_yx_um[1])
        )
        target_q, target_phase = (q[valid], phase[valid])
        modes, mode_phase = ([], [])
        for p in np.unique(target_phase):
            v = target_q[target_phase == p]
            if len(v) > c["orientation_sample_limit"]:
                v = v[rng.choice(len(v), c["orientation_sample_limit"], replace=False)]
            v = canonicalize_quaternions(v, self.phase_symmetry[int(p)])
            tree = cKDTree(v)
            assigned = np.zeros(len(v), bool)
            radius = 2 * np.sin(np.deg2rad(c["orientation_cluster_deg"]) / 4)
            for j in range(len(v)):
                if assigned[j]:
                    continue
                ids = np.asarray(tree.query_ball_point(v[j], radius))
                assigned[ids] = True
                modes.append(normalize_quaternions(v[ids].mean(0)))
                mode_phase.append(p)
        self.initial_mode_q = np.asarray(modes, np.float32)
        self.mode_phase = np.asarray(mode_phase, np.int16)
        labels = np.empty(len(target_q), np.int32)
        for p in np.unique(target_phase):
            mi = np.flatnonzero(self.mode_phase == p)
            eq = symmetry_equivalents(
                self.initial_mode_q[mi], self.phase_symmetry[int(p)]
            )
            ng = eq.shape[1]
            expanded = np.concatenate((eq.reshape(-1, 4), -eq.reshape(-1, 4)))
            _, hit = cKDTree(expanded).query(
                target_q[target_phase == p], workers=c["threads"]
            )
            labels[target_phase == p] = mi[hit % (len(mi) * ng) // ng]
        label_volume = np.full(valid.shape, -1, np.int32)
        label_volume[valid] = labels
        stride = int(c["seed_stride"])
        if stride < 1:
            raise ValueError("seed_stride must be a positive integer")
        key = np.column_stack((iz, iy // stride, ix // stride, labels))
        keys, inv, counts = np.unique(
            key, axis=0, return_inverse=True, return_counts=True
        )
        if len(keys) > int(c["max_gaussians"]):
            raise ValueError(
                f"Local initialization requires {len(keys)} Gaussians at fixed tile stride {stride}; max_gaussians={c['max_gaussians']}. Increase the explicit capacity or use a smaller spatial crop. Tile size is never increased automatically."
            )
        centers = np.column_stack(
            [np.bincount(inv, weights=xyz[:, d]) / counts for d in range(3)]
        )
        variances = (
            np.column_stack(
                [np.bincount(inv, weights=xyz[:, d] ** 2) / counts for d in range(3)]
            )
            - centers**2
        )
        gap = (
            float(np.median(np.diff(z_um)))
            if len(z_um) > 1
            else float(max(self.spacing_yx_um))
        )
        self.search_scale = np.array(
            [gap, stride * self.spacing_yx_um[0], stride * self.spacing_yx_um[1]]
        )
        covariance = np.zeros((len(keys), 3, 3), float)
        sig = np.sqrt(np.maximum(variances, 0))
        sig[:, 0] = 0.38 * gap
        sig[:, 1:] += 0.8 * self.spacing_yx_um
        for d in range(3):
            covariance[:, d, d] = sig[:, d] ** 2
        bridged = np.zeros(len(keys), bool)
        for k in range(len(z_um) - 1):
            for mode in np.intersect1d(
                keys[keys[:, 0] == k, 3], keys[keys[:, 0] == k + 1, 3]
            ):
                a = np.flatnonzero((keys[:, 0] == k) & (keys[:, 3] == mode))
                b = np.flatnonzero((keys[:, 0] == k + 1) & (keys[:, 3] == mode))
                distance, idx = cKDTree(centers[b, 1:] / self.search_scale[1:]).query(
                    centers[a, 1:] / self.search_scale[1:]
                )
                good = distance <= 1.5
                aa, bb = (a[good], b[idx[good]])
                delta = centers[bb] - centers[aa]
                centers[aa] += 0.5 * delta
                covariance[aa] += 0.25 * delta[:, :, None] * delta[:, None, :]
                bridged[aa] = True
        self.gaussian_mode = keys[:, 3].astype(np.int64)
        self.unit_um = float(np.mean(self.spacing_yx_um))
        base_chol = np.linalg.cholesky(covariance) / self.unit_um
        if not c["anisotropic"]:
            scalar = np.linalg.det(covariance) ** (1 / 6) / self.unit_um
            base_chol = scalar[:, None, None] * np.eye(3)[None]
        self.base_mu = torch.as_tensor(
            centers / self.unit_um, dtype=torch.float32, device=self.device
        )
        self.base_chol = torch.as_tensor(
            base_chol, dtype=torch.float32, device=self.device
        )
        self.mu_delta = nn.Parameter(torch.zeros_like(self.base_mu))
        self.chol_delta = nn.Parameter(
            torch.zeros((len(centers), 6), device=self.device)
        )
        self.log_amplitude = nn.Parameter(torch.zeros(len(centers), device=self.device))
        self.mode_q = nn.Parameter(
            torch.as_tensor(self.initial_mode_q, device=self.device).clone()
        )
        self._mode_index = torch.as_tensor(self.gaussian_mode, device=self.device)
        self._mode_phase = torch.as_tensor(
            self.mode_phase.astype(np.int64), device=self.device
        )
        self._symmetry = {}
        for p, symmetry in self.phase_symmetry.items():
            self._symmetry[p] = torch.as_tensor(
                crystal_symmetry_quaternions(symmetry),
                dtype=torch.float32,
                device=self.device,
            )
        self._refresh_tree()
        n_all = len(xyz)
        chosen = rng.choice(n_all, min(n_all, c["max_train_points"]), replace=False)
        train_x, train_q, train_p = (
            xyz[chosen],
            target_q[chosen],
            target_phase[chosen],
        )
        train_candidates = self._candidates(train_x)
        if c["fit_epochs"] is not None:
            c["iterations"] = int(
                np.ceil(
                    np.ceil(len(train_x) / c["batch_size"]) * float(c["fit_epochs"])
                )
            )
        pairs = []
        for axis in (1, 2):
            left = [slice(None)] * 3
            right = [slice(None)] * 3
            left[axis], right[axis] = (slice(None, -1), slice(1, None))
            vl = valid[tuple(left)] & valid[tuple(right)]
            coords = np.column_stack(np.nonzero(vl))
            if len(coords):
                take = rng.choice(len(coords), min(12000, len(coords)), replace=False)
                p0 = coords[take]
                p1 = p0.copy()
                p1[:, axis] += 1
                pairs.append((p0, p1))
        if pairs:
            p0 = np.concatenate([a for a, _ in pairs])
            p1 = np.concatenate([b for _, b in pairs])
            a, b = (label_volume[tuple(p0.T)], label_volume[tuple(p1.T)])
            boundary_target = (a != b).astype(np.float32)
            for p, sym in self.phase_symmetry.items():
                same_phase = (self.mode_phase[a] == p) & (self.mode_phase[b] == p)
                boundary_target[same_phase] = (
                    misorientation_deg(
                        q[tuple(p0[same_phase].T)], q[tuple(p1[same_phase].T)], sym
                    )
                    > c["boundary_threshold_deg"]
                )
            pair_x0 = np.column_stack((z_um[p0[:, 0]], p0[:, 1:] * self.spacing_yx_um))
            pair_x1 = np.column_stack((z_um[p1[:, 0]], p1[:, 1:] * self.spacing_yx_um))
            pair_c0, pair_c1 = (self._candidates(pair_x0), self._candidates(pair_x1))
        affinity = np.eye(len(modes), dtype=bool)
        for p, sym in self.phase_symmetry.items():
            ids = np.flatnonzero(self.mode_phase == p)
            for start in range(0, len(ids), 32):
                aa = ids[start : start + 32]
                affinity[np.ix_(aa, ids)] = (
                    misorientation_deg(
                        self.initial_mode_q[aa, None],
                        self.initial_mode_q[None, ids],
                        sym,
                    )
                    < c["boundary_threshold_deg"]
                )
        affinity = torch.as_tensor(affinity, device=self.device)
        optimizer = torch.optim.Adam(
            [self.mu_delta, self.chol_delta, self.log_amplitude], lr=c["learning_rate"]
        )
        q_optimizer = torch.optim.Adam([self.mode_q], lr=c["learning_rate"] * 0.08)
        history = []
        tau2 = max(np.sin(np.deg2rad(c["orientation_sigma_deg"]) / 2) ** 2, 1e-06)
        permutation = rng.permutation(len(train_x))
        cursor = 0
        for step in range(int(c["iterations"])):
            if (
                c["candidate_refresh"]
                and step
                and (step % int(c["candidate_refresh"]) == 0)
            ):
                self._refresh_tree()
                train_candidates = self._candidates(train_x)
                if pairs:
                    pair_c0, pair_c1 = (
                        self._candidates(pair_x0),
                        self._candidates(pair_x1),
                    )
            if cursor >= len(train_x):
                permutation, cursor = (rng.permutation(len(train_x)), 0)
            pick = permutation[cursor : cursor + c["batch_size"]]
            cursor += len(pick)
            x, qi, pi, cand = (
                train_x[pick],
                train_q[pick],
                train_p[pick],
                train_candidates[pick],
            )
            weights, gids = self._weights(x, cand)
            logw = torch.log(weights.clamp_min(1e-12))
            target = torch.as_tensor(qi, device=self.device)
            normalized_q = torch.nn.functional.normalize(self.mode_q, dim=-1)
            d2 = torch.full_like(logw, 1.0)
            for p, sym in self._symmetry.items():
                mask_np = pi == p
                if not mask_np.any():
                    continue
                mask = torch.as_tensor(mask_np, device=self.device)
                cq = normalized_q[gids[mask]]
                eq = _torch_mul(
                    cq[..., None, :].expand(-1, -1, len(sym), -1),
                    sym.expand(len(cq), cq.shape[1], -1, -1),
                )
                dot = (
                    (eq * target[mask, None, None, :])
                    .sum(-1)
                    .abs()
                    .amax(-1)
                    .clamp(max=1)
                )
                candidate_phase = self._mode_phase[gids[mask]]
                d2[mask] = torch.where(
                    candidate_phase == p, 1 - dot.square(), torch.ones_like(dot)
                )
            data_loss = -torch.logsumexp(logw - d2 / (2 * tau2), dim=-1).mean()
            boundary_loss = torch.zeros((), device=self.device)
            if c["boundary_weight"] > 0 and pairs:
                bi = rng.integers(0, len(pair_x0), size=c["boundary_batch_size"])
                w0, g0 = self._weights(pair_x0[bi], pair_c0[bi])
                w1, g1 = self._weights(pair_x1[bi], pair_c1[bi])
                same = affinity[g0[:, :, None], g1[:, None, :]].float()
                pb = (1 - (w0[:, :, None] * w1[:, None, :] * same).sum((1, 2))).clamp(
                    1e-05, 1 - 1e-05
                )
                bt = torch.as_tensor(boundary_target[bi], device=self.device)
                boundary_loss = torch.nn.functional.binary_cross_entropy(pb, bt)
            reg = self.mu_delta.square().mean() + self.chol_delta.square().mean()
            loss = (
                data_loss
                + c["boundary_weight"] * boundary_loss
                + c["geometry_regularization"] * reg
            )
            optimizer.zero_grad()
            q_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            q_optimizer.step()
            if step % c["log_every"] == 0 or step == c["iterations"] - 1:
                history.append(
                    dict(
                        step=step,
                        loss=float(loss.detach().cpu()),
                        orientation=float(data_loss.detach().cpu()),
                        boundary=float(boundary_loss.detach().cpu()),
                    )
                )
        self._refresh_tree()
        self.info_ = dict(
            seed=seed,
            n_observed_pixels=n_all,
            n_train_points=len(train_x),
            actual_iterations=int(c["iterations"]),
            requested_fit_epochs=c["fit_epochs"],
            n_observed_slices=len(z_um),
            n_gaussians=len(centers),
            n_orientation_modes=len(modes),
            n_bridged_gaussians=int(bridged.sum()),
            actual_seed_stride=stride,
            fit_seconds=time.perf_counter() - started,
            device=str(self.device),
            history=history,
            convention="wxyz crystal-to-sample; q*s right crystal symmetry",
            training_sources="visible slices only; no feature/grain IDs",
            config=self.config,
        )
        return self

    def _geometry(self, indices=None):
        base_chol = self.base_chol if indices is None else self.base_chol[indices]
        base_mu = self.base_mu if indices is None else self.base_mu[indices]
        mu_delta = self.mu_delta if indices is None else self.mu_delta[indices]
        chol_delta = self.chol_delta if indices is None else self.chol_delta[indices]
        diag0 = torch.diagonal(base_chol, dim1=-2, dim2=-1)
        mu = base_mu + torch.tanh(mu_delta) * diag0
        if self.config["anisotropic"]:
            diag = diag0 * torch.exp(0.9 * torch.tanh(chol_delta[..., :3]))
        else:
            diag = diag0 * torch.exp(0.9 * torch.tanh(chol_delta[..., :1]))
        chol = torch.diag_embed(diag)
        if self.config["anisotropic"]:
            chol[..., 1, 0] = base_chol[..., 1, 0] + diag0[..., 1] * torch.tanh(
                chol_delta[..., 3]
            )
            chol[..., 2, 0] = base_chol[..., 2, 0] + diag0[..., 2] * torch.tanh(
                chol_delta[..., 4]
            )
            chol[..., 2, 1] = base_chol[..., 2, 1] + diag0[..., 2] * torch.tanh(
                chol_delta[..., 5]
            )
        return (mu, chol)

    def _refresh_tree(self):
        with torch.no_grad():
            mu, _ = self._geometry()
            self._tree = cKDTree(
                mu.detach().cpu().numpy() * self.unit_um / self.search_scale
            )

    def _candidates(self, coords):
        k = min(int(self.config["candidates"]), len(self.gaussian_mode))
        _, ids = self._tree.query(
            np.asarray(coords) / self.search_scale, k=k, workers=self.config["threads"]
        )
        return np.asarray(ids).reshape(len(coords), k)

    def _weights(self, coords, candidates):
        ids = torch.as_tensor(candidates, dtype=torch.long, device=self.device)
        x = torch.as_tensor(
            np.asarray(coords) / self.unit_um, dtype=torch.float32, device=self.device
        )
        mu, l = self._geometry(ids)
        delta = x[:, None, :] - mu
        v0 = delta[..., 0] / l[..., 0, 0]
        v1 = (delta[..., 1] - l[..., 1, 0] * v0) / l[..., 1, 1]
        v2 = (delta[..., 2] - l[..., 2, 0] * v0 - l[..., 2, 1] * v1) / l[..., 2, 2]
        logits = self.log_amplitude[ids] - 0.5 * (
            v0.square() + v1.square() + v2.square()
        )
        return (torch.softmax(logits, dim=1), self._mode_index[ids])

    def query(self, coords_um, chunk_size=8192, decode=None):
        coords = np.asarray(coords_um, float).reshape(-1, 3)
        decode = decode or self.config["decode"]
        if decode not in {"modal", "arithmetic", "symmetrymean"}:
            raise ValueError("decode must be modal, arithmetic, or symmetrymean")
        out_q, out_p, out_conf = ([], [], [])
        with torch.no_grad():
            mq = torch.nn.functional.normalize(self.mode_q, dim=-1).cpu().numpy()
            for start in range(0, len(coords), chunk_size):
                xx = coords[start : start + chunk_size]
                ww, gg = self._weights(xx, self._candidates(xx))
                w, g = (ww.cpu().numpy(), gg.cpu().numpy())
                score = ((g[:, :, None] == g[:, None, :]) * w[:, None, :]).sum(-1)
                best = score.argmax(-1)
                mode = g[np.arange(len(g)), best]
                pq = mq[mode].copy()
                pp = self.mode_phase[mode]
                if decode != "modal":
                    candidates = mq[g].copy()
                    wp = w * (self.mode_phase[g] == pp[:, None])
                    if decode == "symmetrymean":
                        for p, sym in self.phase_symmetry.items():
                            m = pp == p
                            if m.any():
                                candidates[m] = align_quaternions(
                                    candidates[m], pq[m, None], sym
                                )
                    pq = normalize_quaternions(
                        (wp[..., None] * candidates).sum(1)
                    ).astype(np.float32)
                out_q.append(pq)
                out_p.append(pp)
                out_conf.append(score[np.arange(len(g)), best])
        return dict(
            q=np.concatenate(out_q).astype(np.float32),
            phase=np.concatenate(out_p),
            valid=np.ones(len(coords), bool),
            confidence=np.concatenate(out_conf).astype(np.float32),
        )

    def predict(self, z_um, shape_yx=None, chunk_size=8192, decode=None):
        z = np.atleast_1d(z_um).astype(float)
        sy, sx = shape_yx or self.shape_yx
        yy, xx = np.meshgrid(
            np.arange(sy) * self.spacing_yx_um[0],
            np.arange(sx) * self.spacing_yx_um[1],
            indexing="ij",
        )
        planes = []
        for zz in z:
            coords = np.column_stack((np.full(sy * sx, zz), yy.ravel(), xx.ravel()))
            flat = self.query(coords, chunk_size, decode)
            planes.append(
                {k: v.reshape((sy, sx) + v.shape[1:]) for k, v in flat.items()}
            )
        return {k: np.stack([plane[k] for plane in planes]) for k in planes[0]}

    def save(self, path):
        path = Path(path)
        with torch.no_grad():
            mu, chol = self._geometry()
            arrays = dict(
                mu_um=mu.cpu().numpy() * self.unit_um,
                cholesky_um=chol.cpu().numpy() * self.unit_um,
                amplitude=torch.exp(self.log_amplitude).cpu().numpy(),
                mode_q=torch.nn.functional.normalize(self.mode_q, dim=-1).cpu().numpy(),
                gaussian_mode=self.gaussian_mode,
                mode_phase=self.mode_phase,
                observed_z_um=self.observed_z_um,
                spacing_yx_um=self.spacing_yx_um,
            )
        np.savez_compressed(path, **arrays)
        path.with_suffix(".json").write_text(
            json.dumps(self.info_, indent=2), encoding="utf-8"
        )
