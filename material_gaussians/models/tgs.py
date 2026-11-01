import math
import time
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from material_gaussians.models.baselines import align_equivalent
from material_gaussians.orientation import misorientation_deg


def _symmetric_moment(vectors, order):
    out = []
    for a in range(order + 1):
        for b in range(order + 1 - a):
            c = order - a - b
            factor = math.sqrt(
                math.factorial(order)
                / (math.factorial(a) * math.factorial(b) * math.factorial(c))
            )
            out.append(
                factor
                * np.mean(
                    vectors[..., 0] ** a * vectors[..., 1] ** b * vectors[..., 2] ** c,
                    axis=-1,
                )
            )
    return np.stack(out, axis=-1).astype(np.float32)


def invariant_features(q, phase, symmetries):
    phases = sorted((int(p) for p in symmetries))
    features = np.zeros((*phase.shape, 34 + len(phases)), np.float32)
    for j, ph in enumerate(phases):
        mask = phase == ph
        if not mask.any():
            continue
        rot = Rotation.from_quat(q[mask][:, [1, 2, 3, 0]]).as_matrix()
        name = str(symmetries[ph]).lower()
        if name in ("cubic", "fcc", "bcc", "o", "m-3m", "432"):
            values = _symmetric_moment(np.swapaxes(rot, -1, -2), 4)
            features[mask, :15] = values
        elif name in ("hexagonal", "hex", "hcp", "d6", "6/mmm", "622"):
            axial = rot[:, :, 2][:, None, :]
            theta = np.arange(6) * np.pi / 3
            basal = np.stack((np.cos(theta), np.sin(theta), np.zeros(6)), axis=-1)
            basal = np.einsum("nij,kj->nki", rot, basal)
            features[mask, :6] = _symmetric_moment(axial, 2)
            features[mask, 6:34] = _symmetric_moment(basal, 6)
        else:
            features[mask, :9] = rot.reshape(-1, 9)
        features[mask, 34 + j] = 1.0
    return features


def _resize_features(features, shape):
    if features.shape[:2] == tuple(shape):
        return features
    factors = (shape[0] / features.shape[0], shape[1] / features.shape[1], 1.0)
    return ndimage.zoom(features, factors, order=1, prefilter=False)


def _flow(reference, moving, cfg):
    active = np.any(reference != 0, axis=(0, 1)) | np.any(moving != 0, axis=(0, 1))
    reference, moving = (reference[..., active], moving[..., active])
    shape = reference.shape[:2]
    levels = int(cfg["transport_levels"])
    flow = None
    for level in range(levels - 1, -1, -1):
        size = tuple((max(8, int(round(n / 2**level))) for n in shape))
        a, b = (_resize_features(reference, size), _resize_features(moving, size))
        if flow is None:
            flow = np.zeros((2, *size), np.float32)
        else:
            factors = (1, size[0] / flow.shape[1], size[1] / flow.shape[2])
            flow = ndimage.zoom(flow, factors, order=1, prefilter=False)
            flow[0] *= factors[1]
            flow[1] *= factors[2]
        gy, gx = np.gradient(b, axis=(0, 1))
        grid = np.indices(size, dtype=np.float32)
        sigma = float(cfg["transport_window_sigma"])
        for _ in range(int(cfg["transport_iterations"])):
            coords = grid + flow
            bw = np.stack(
                [
                    ndimage.map_coordinates(b[..., c], coords, order=1, mode="nearest")
                    for c in range(b.shape[-1])
                ],
                axis=-1,
            )
            yy = np.stack(
                [
                    ndimage.map_coordinates(gy[..., c], coords, order=1, mode="nearest")
                    for c in range(b.shape[-1])
                ],
                axis=-1,
            )
            xx = np.stack(
                [
                    ndimage.map_coordinates(gx[..., c], coords, order=1, mode="nearest")
                    for c in range(b.shape[-1])
                ],
                axis=-1,
            )
            residual = bw - a
            weight = 1 / np.sqrt(
                np.sum(residual**2, axis=-1) + float(cfg["transport_robust_eps"]) ** 2
            )

            def smooth_sum(value):
                return ndimage.gaussian_filter(
                    np.sum(value, axis=-1) * weight, sigma, mode="nearest"
                )

            ayy, axx, ayx = (
                smooth_sum(yy * yy),
                smooth_sum(xx * xx),
                smooth_sum(yy * xx),
            )
            by, bx = (smooth_sum(yy * residual), smooth_sum(xx * residual))
            ridge = float(cfg["transport_ridge"])
            ayy += ridge
            axx += ridge
            determinant = np.maximum(ayy * axx - ayx * ayx, 1e-12)
            delta = np.stack(
                (
                    (-axx * by + ayx * bx) / determinant,
                    (ayx * by - ayy * bx) / determinant,
                )
            )
            length = np.sqrt(np.sum(delta**2, axis=0))
            delta /= np.maximum(1.0, length / float(cfg["transport_step_limit"]))
            flow += delta
            flow = ndimage.gaussian_filter(
                flow,
                (
                    0,
                    float(cfg["transport_flow_sigma"]),
                    float(cfg["transport_flow_sigma"]),
                ),
                mode="nearest",
            )
            limit = float(cfg["transport_max_displacement"]) / 2**level
            flow = np.clip(flow, -limit, limit)
    return flow


class GaussianTransportField:
    defaults = dict(
        transport_levels=3,
        transport_iterations=6,
        transport_window_sigma=1.5,
        transport_flow_sigma=0.65,
        transport_robust_eps=0.1,
        transport_ridge=0.001,
        transport_step_limit=1.0,
        transport_max_displacement=8.0,
        transport_splat_sigma=0.65,
        transport_candidates=6,
        transport_orientation_gate_deg=5.0,
        transport_merge_gate_deg=10.0,
        transport_motion_scale=1.0,
        transport_confidence_floor=0.35,
        transport_match_scale_deg=10.0,
    )

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
        cfg = dict(self.defaults)
        cfg.update(config or {})
        self.cfg = cfg
        self.symmetries = {
            int(k): v for k, v in cfg.get("phase_symmetry", {1: "cubic"}).items()
        }
        self.q = np.asarray(q_observed, np.float32).copy()
        self.phase = np.asarray(phase_observed, np.int16).copy()
        self.valid = np.asarray(valid_observed, bool).copy()
        self.z = np.asarray(z_um, float).copy()
        self.spacing = np.asarray(spacing_yx_um, float).copy()
        if len(self.z) == 0 or np.any(np.diff(self.z) <= 0):
            raise ValueError(
                "Measured section coordinates must be nonempty and increasing."
            )
        for k in range(len(self.q)):
            if not self.valid[k].any():
                raise ValueError(
                    "Each observed section must contain indexed orientations."
                )
            if not self.valid[k].all():
                ids = ndimage.distance_transform_edt(
                    ~self.valid[k],
                    sampling=self.spacing,
                    return_distances=False,
                    return_indices=True,
                )
                bad = ~self.valid[k]
                self.q[k][bad] = self.q[k][tuple(ids[:, bad])]
                self.phase[k][bad] = self.phase[k][tuple(ids[:, bad])]
                self.valid[k][bad] = True
        features = [
            invariant_features(q, p, self.symmetries)
            for q, p in zip(self.q, self.phase)
        ]
        self.forward, self.backward, self.conf_forward, self.conf_backward = (
            [],
            [],
            [],
            [],
        )
        for k in range(len(self.z) - 1):
            fw, bw = (
                _flow(features[k], features[k + 1], cfg),
                _flow(features[k + 1], features[k], cfg),
            )
            self.forward.append(fw)
            self.backward.append(bw)
            self.conf_forward.append(self._confidence(k, k + 1, fw))
            self.conf_backward.append(self._confidence(k + 1, k, bw))
        self.grid = np.indices(self.q.shape[1:3], dtype=np.float32).reshape(2, -1).T
        self.info_ = dict(
            method="gaussian_transport",
            seed=seed,
            deterministic=True,
            fit_seconds=time.perf_counter() - started,
            observed_sections=len(self.z),
            config=cfg,
            displacement_rms_pixels=float(
                np.sqrt(np.mean(np.asarray(self.forward) ** 2))
            )
            if self.forward
            else 0.0,
            input_policy="observed sections only; no dense-reference labels or orientations",
        )
        return self

    def _confidence(self, lo, hi, flow):
        coords = np.indices(self.q.shape[1:3], dtype=np.float32) + flow
        idx = np.rint(coords).astype(int)
        idx[0] = np.clip(idx[0], 0, self.q.shape[1] - 1)
        idx[1] = np.clip(idx[1], 0, self.q.shape[2] - 1)
        match_q, match_phase = (self.q[hi][tuple(idx)], self.phase[hi][tuple(idx)])
        angles = np.full(self.phase[lo].shape, 90.0, np.float32)
        for ph, symmetry in self.symmetries.items():
            mask = (self.phase[lo] == ph) & (match_phase == ph)
            angles[mask] = misorientation_deg(self.q[lo][mask], match_q[mask], symmetry)
        floor = float(self.cfg["transport_confidence_floor"])
        confidence = floor + (1 - floor) * np.exp(
            -0.5 * (angles / float(self.cfg["transport_match_scale_deg"])) ** 2
        )
        return confidence.reshape(-1)

    def _splat(self, k, flow, t, confidence):
        moved = (
            self.grid
            + t * float(self.cfg["transport_motion_scale"]) * flow.reshape(2, -1).T
        )
        tree = cKDTree(moved * self.spacing)
        count = min(int(self.cfg["transport_candidates"]), len(moved))
        distance, index = tree.query(self.grid * self.spacing, k=count, workers=1)
        if count == 1:
            distance, index = (distance[:, None], index[:, None])
        sigma = float(self.cfg["transport_splat_sigma"]) * float(np.mean(self.spacing))
        weight = np.exp(-0.5 * (distance / max(sigma, 1e-06)) ** 2) * confidence[index]
        best = weight.argmax(axis=1)
        ids = index[np.arange(len(index)), best]
        q_all, p_all = (self.q[k].reshape(-1, 4), self.phase[k].reshape(-1))
        anchor, phase = (q_all[ids], p_all[ids])
        qout = anchor.copy()
        support = np.zeros(len(index), np.float32)
        for ph, symmetry in self.symmetries.items():
            mask = phase == ph
            if not mask.any():
                continue
            candidate = q_all[index[mask]]
            reference = np.broadcast_to(anchor[mask, None], candidate.shape)
            aligned = align_equivalent(candidate, reference, symmetry)
            cosine = np.clip(np.sum(aligned * reference, axis=-1), 0.0, 1.0)
            compatible = cosine >= np.cos(
                np.deg2rad(float(self.cfg["transport_orientation_gate_deg"])) / 2
            )
            compatible &= p_all[index[mask]] == ph
            w = weight[mask] * compatible
            qq = np.sum(aligned * w[..., None], axis=1)
            norm = np.linalg.norm(qq, axis=-1, keepdims=True)
            qout[mask] = np.divide(
                qq, norm, out=anchor[mask].copy(), where=norm > 1e-10
            )
            support[mask] = w.sum(axis=1)
        return (qout, phase, support)

    def predict(self, z_um, **kwargs):
        z = np.asarray(z_um, float)
        shape = (len(z), *self.phase.shape[1:])
        qout, pout = (np.empty((*shape, 4), np.float32), np.empty(shape, np.int16))
        for i, zz in enumerate(z):
            nearest = int(np.argmin(abs(self.z - zz)))
            if zz <= self.z[0] or zz >= self.z[-1] or abs(self.z[nearest] - zz) < 1e-09:
                qout[i], pout[i] = (self.q[nearest], self.phase[nearest])
                continue
            hi = int(np.searchsorted(self.z, zz))
            lo, t = (hi - 1, (zz - self.z[hi - 1]) / (self.z[hi] - self.z[hi - 1]))
            qa, pa, wa = self._splat(lo, self.forward[lo], t, self.conf_forward[lo])
            qb, pb, wb = self._splat(
                hi, self.backward[lo], 1 - t, self.conf_backward[lo]
            )
            wa, wb = (wa * (1 - t), wb * t)
            choose_b = wb > wa
            qq, pp = (np.where(choose_b[:, None], qb, qa), np.where(choose_b, pb, pa))
            for ph, symmetry in self.symmetries.items():
                same = (pa == ph) & (pb == ph)
                if not same.any():
                    continue
                b = align_equivalent(qb[same], qa[same], symmetry)
                cosine = np.clip(np.sum(qa[same] * b, axis=-1), 0.0, 1.0)
                merge = cosine >= np.cos(
                    np.deg2rad(float(self.cfg["transport_merge_gate_deg"])) / 2
                )
                ids = np.flatnonzero(same)[merge]
                mixed = qa[ids] * wa[ids, None] + b[merge] * wb[ids, None]
                norm = np.linalg.norm(mixed, axis=-1, keepdims=True)
                qq[ids] = np.divide(mixed, norm, out=qa[ids].copy(), where=norm > 1e-10)
            qout[i], pout[i] = (qq.reshape(*shape[1:], 4), pp.reshape(shape[1:]))
        return dict(q=qout, phase=pout, valid=np.ones(shape, bool))

    def save(self, path):
        np.savez_compressed(
            path,
            q=self.q,
            phase=self.phase,
            valid=self.valid,
            z=self.z,
            spacing=self.spacing,
            forward=np.asarray(self.forward),
            backward=np.asarray(self.backward),
            conf_forward=np.asarray(self.conf_forward),
            conf_backward=np.asarray(self.conf_backward),
        )
