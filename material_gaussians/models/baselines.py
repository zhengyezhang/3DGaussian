import time
import numpy as np
from material_gaussians.orientation import crystal_symmetry_quaternions, quat_mul


def align_equivalent(q, reference, symmetry):
    best, score = (q.copy(), np.full(q.shape[:-1], -1.0, np.float32))
    for sym in crystal_symmetry_quaternions(symmetry):
        candidate = quat_mul(q, np.broadcast_to(sym, q.shape))
        dot = np.sum(candidate * reference, axis=-1)
        choose = np.abs(dot) > score
        best[choose] = candidate[choose] * np.where(dot[choose, None] >= 0, 1.0, -1.0)
        score = np.maximum(score, np.abs(dot))
    return best


class SliceInterpolation:
    def __init__(self, mode="nearest"):
        self.mode = mode

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
        from scipy.ndimage import distance_transform_edt

        self.q, self.phase, self.valid = (
            q_observed.copy(),
            phase_observed.copy(),
            valid_observed.copy(),
        )
        for k in range(len(self.q)):
            if not self.valid[k].all() and self.valid[k].any():
                ids = distance_transform_edt(
                    ~self.valid[k],
                    sampling=spacing_yx_um,
                    return_distances=False,
                    return_indices=True,
                )
                bad = ~self.valid[k]
                self.q[k][bad] = self.q[k][tuple(ids[:, bad])]
                self.phase[k][bad] = self.phase[k][tuple(ids[:, bad])]
                self.valid[k][bad] = True
        self.z = np.asarray(z_um)
        self.symmetries = (config or {}).get("phase_symmetry", {1: "cubic"})
        self.info_ = {"method": self.mode, "fit_seconds": 0.0, "seed": seed}
        return self

    def predict(self, z_um, **kwargs):
        z = np.asarray(z_um)
        nearest = np.argmin(abs(z[:, None] - self.z[None]), axis=1)
        q, phase, valid = (
            self.q[nearest].copy(),
            self.phase[nearest].copy(),
            self.valid[nearest].copy(),
        )
        if self.mode == "slerp":
            for k, zz in enumerate(z):
                hi = int(np.searchsorted(self.z, zz, side="right"))
                if hi == 0 or hi == len(self.z) or np.min(abs(zz - self.z)) < 1e-08:
                    continue
                lo = hi - 1
                t = (zz - self.z[lo]) / (self.z[hi] - self.z[lo])
                for ph, symmetry in self.symmetries.items():
                    mask = (
                        (self.phase[lo] == int(ph))
                        & (self.phase[hi] == int(ph))
                        & self.valid[lo]
                        & self.valid[hi]
                    )
                    a, b = (self.q[lo][mask], self.q[hi][mask])
                    if not len(a):
                        continue
                    b = align_equivalent(b, a, symmetry)
                    theta = np.arccos(np.clip(np.sum(a * b, axis=-1), 0.0, 1.0))
                    den = np.sin(theta)
                    close = theta < 0.0001
                    den[close] = 1.0
                    out = (
                        np.sin((1 - t) * theta)[:, None] * a
                        + np.sin(t * theta)[:, None] * b
                    ) / den[:, None]
                    out[close] = (1 - t) * a[close] + t * b[close]
                    out /= np.maximum(
                        np.linalg.norm(out, axis=-1, keepdims=True), 1e-12
                    )
                    q[k][mask] = out
        return dict(q=q, phase=phase, valid=valid)


class SirenOrientationField:
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
        import torch
        from torch import nn

        cfg = config or {}
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.device = torch.device(
            cfg.get(
                "siren_device",
                cfg.get(
                    "device", "mps" if torch.backends.mps.is_available() else "cpu"
                ),
            )
        )
        width, depth, omega = (
            int(cfg.get("siren_width", 256)),
            int(cfg.get("siren_depth", 3)),
            float(cfg.get("siren_omega", 30)),
        )
        phases = sorted(
            (int(x) for x in np.unique(phase_observed[valid_observed]) if x > 0)
        )
        self.phases = np.array(phases, np.int16)
        self.shape_yx = q_observed.shape[1:3]
        self.spacing = np.array(spacing_yx_um)
        self.minimum = np.array([min(z_um), 0.0, 0.0], np.float32)
        self.extent = np.array(
            [
                max(z_um) - min(z_um),
                (self.shape_yx[0] - 1) * self.spacing[0],
                (self.shape_yx[1] - 1) * self.spacing[1],
            ],
            np.float32,
        )
        self.extent = np.maximum(self.extent, 1e-06)

        class SineLayer(nn.Module):
            def __init__(self, din, dout, first=False):
                super().__init__()
                self.linear = nn.Linear(din, dout)
                with torch.no_grad():
                    bound = 1 / din if first else np.sqrt(6 / din) / omega
                    self.linear.weight.uniform_(-bound, bound)

            def forward(self, x):
                return torch.sin(omega * self.linear(x))

        layers = [SineLayer(3, width, True)] + [
            SineLayer(width, width) for _ in range(depth - 1)
        ]
        final = nn.Linear(width, 4 + len(phases))
        with torch.no_grad():
            final.weight.uniform_(
                -np.sqrt(6 / width) / omega, np.sqrt(6 / width) / omega
            )
        self.net = nn.Sequential(*layers, final).to(self.device)
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=float(cfg.get("siren_lr", 0.0001))
        )
        indices = np.argwhere(valid_observed)
        coords = indices.astype(np.float32)
        coords[:, 0] = np.asarray(z_um)[indices[:, 0]]
        coords[:, 1:] *= self.spacing
        coords = (coords - self.minimum) / self.extent * 2 - 1
        targets = q_observed[valid_observed]
        p = np.searchsorted(self.phases, phase_observed[valid_observed])
        symmetries = cfg.get("phase_symmetry", {1: "cubic"})
        sym = {
            ph: torch.as_tensor(
                crystal_symmetry_quaternions(
                    symmetries.get(ph, symmetries.get(str(ph), "cubic"))
                ),
                dtype=torch.float32,
                device=self.device,
            )
            for ph in phases
        }

        def multiply(a, b):
            aw, av, bw, bv = (a[..., :1], a[..., 1:], b[..., :1], b[..., 1:])
            return torch.cat(
                (
                    aw * bw - (av * bv).sum(-1, keepdim=True),
                    aw * bv + bw * av + torch.cross(av.expand_as(bv), bv, dim=-1),
                ),
                dim=-1,
            )

        iterations = int(cfg.get("siren_iterations", 1500))
        batch_size = int(cfg.get("siren_batch", 4096))
        history, started = ([], time.perf_counter())
        for step in range(iterations):
            ix = rng.integers(0, len(indices), min(batch_size, len(indices)))
            x = torch.as_tensor(coords[ix], device=self.device)
            target = torch.as_tensor(targets[ix], device=self.device)
            phase_target = torch.as_tensor(p[ix], dtype=torch.long, device=self.device)
            output = self.net(x)
            qpred = nn.functional.normalize(output[:, :4], dim=-1, eps=1e-08)
            loss = nn.functional.cross_entropy(output[:, 4:], phase_target)
            for cls, ph in enumerate(phases):
                subset = phase_target == cls
                if subset.any():
                    obs = target[subset, None, :]
                    ss = sym[ph][None].expand(obs.shape[0], -1, -1)
                    equiv = multiply(obs, ss)
                    similarity = torch.abs(
                        (equiv * qpred[subset, None, :]).sum(-1)
                    ).amax(-1)
                    loss = loss + (1 - similarity.square()).sum() / len(ix) * 10
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step == iterations - 1:
                history.append([step, float(loss.detach().cpu())])
        if self.device.type == "mps":
            torch.mps.synchronize()
        self.info_ = dict(
            method="siren",
            iterations=iterations,
            width=width,
            depth=depth,
            omega=omega,
            seed=seed,
            n_parameters=sum((p.numel() for p in self.net.parameters())),
            fit_seconds=time.perf_counter() - started,
            loss_history=history,
        )
        return self

    def predict(self, z_um, chunk_size=65536, **kwargs):
        import torch

        shape = (len(z_um), *self.shape_yx)
        coords = np.indices(shape, dtype=np.float32).reshape(3, -1).T
        coords[:, 0] = np.asarray(z_um)[coords[:, 0].astype(int)]
        coords[:, 1:] *= self.spacing
        coords = (coords - self.minimum) / self.extent * 2 - 1
        q, phase = (
            np.empty((len(coords), 4), np.float32),
            np.empty(len(coords), np.int16),
        )
        with torch.no_grad():
            for start in range(0, len(coords), chunk_size):
                out = (
                    self.net(
                        torch.as_tensor(
                            coords[start : start + chunk_size], device=self.device
                        )
                    )
                    .cpu()
                    .numpy()
                )
                qq = out[:, :4]
                qq /= np.maximum(np.linalg.norm(qq, axis=-1, keepdims=True), 1e-08)
                q[start : start + len(out)] = qq
                phase[start : start + len(out)] = self.phases[out[:, 4:].argmax(-1)]
        return dict(
            q=q.reshape(*shape, 4),
            phase=phase.reshape(shape),
            valid=np.ones(shape, bool),
        )
