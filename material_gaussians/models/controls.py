import numpy as np
from scipy.spatial import cKDTree
from material_gaussians.models.sgf import GaussianOrientationField
from material_gaussians.models.cgf import GaussianMaterialField
from material_gaussians.models.tgs import GaussianTransportField, align_equivalent
from material_gaussians.models.gsr import GaussianGrainField
from material_gaussians.orientation import normalize_quaternions, align_quaternions


class LaplaceMaterialField(GaussianMaterialField):
    def query(self, coords_um, chunk_size=2048, decode=None):
        coords = np.asarray(coords_um, float).reshape(-1, 3)
        chunk_size = min(int(chunk_size), 2048)
        prediction = GaussianOrientationField.query(
            self, coords, chunk_size=chunk_size, decode=decode
        )
        k = min(int(self.config["material_candidates"]), len(self.material_q))
        sigma = np.deg2rad(float(self.config["material_orientation_sigma_deg"]))
        blend = float(self.config["material_blend"])
        for start in range(0, len(coords), chunk_size):
            stop = min(start + chunk_size, len(coords))
            anchor = prediction["q"][start:stop].astype(float)
            p = prediction["phase"][start:stop]
            distance, ids = self.material_tree.query(
                coords[start:stop] / self.material_scale,
                k=k,
                workers=int(self.config["threads"]),
            )
            distance, ids = (distance.reshape(-1, k), ids.reshape(-1, k))
            candidate = self.material_q[ids].astype(float)
            same = self.material_phase[ids] == p[:, None]
            for phase, symmetry in self.phase_symmetry.items():
                selected = p == phase
                if selected.any():
                    candidate[selected] = align_quaternions(
                        candidate[selected], anchor[selected, None], symmetry
                    )
            dot = np.clip(np.sum(candidate * anchor[:, None, :], axis=-1), 0.0, 1.0)
            angle = 2 * np.arccos(dot)
            logits = -2 * distance - 0.5 * (angle / sigma) ** 2
            logits[~same] = -np.inf
            good = same.any(axis=1)
            if good.any():
                weights = np.exp(
                    logits[good] - np.max(logits[good], axis=1, keepdims=True)
                )
                weights /= weights.sum(axis=1, keepdims=True)
                continuous = normalize_quaternions(
                    np.sum(weights[..., None] * candidate[good], axis=1)
                )
                output = normalize_quaternions(
                    (1 - blend) * anchor[good] + blend * continuous
                )
                ix = np.flatnonzero(good) + start
                prediction["q"][ix] = output.astype(np.float32)
        return prediction


class LaplaceTransportField(GaussianTransportField):
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
        weight = (
            np.exp(-np.sqrt(3) * (distance / max(sigma, 1e-06))) * confidence[index]
        )
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


class CalibratedDistanceField(GaussianGrainField):
    def _support(self, k, rid):
        score, q, phase = super()._support(k, rid)
        if rid:
            score = np.clip(
                score * float(self.config["distance_scale"]),
                -np.log(9999.0),
                np.log(9999.0),
            ).astype(np.float32)
        return (score, q, phase)
