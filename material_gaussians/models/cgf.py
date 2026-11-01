import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from material_gaussians.models.sgf import GaussianOrientationField
from material_gaussians.orientation import normalize_quaternions, align_quaternions

MATERIAL_DEFAULTS = dict(
    material_candidates=32,
    material_xy_sigma_pixels=1.25,
    material_z_sigma_gap=0.45,
    material_orientation_sigma_deg=3.0,
    material_blend=1.0,
)


class GaussianMaterialField(GaussianOrientationField):
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
        start = time.perf_counter()
        config = {**MATERIAL_DEFAULTS, **(config or {})}
        for key in (
            "material_xy_sigma_pixels",
            "material_z_sigma_gap",
            "material_orientation_sigma_deg",
        ):
            if config[key] <= 0:
                raise ValueError(f"{key} must be positive")
        if not 0 <= config["material_blend"] <= 1 or config["material_candidates"] < 1:
            raise ValueError(
                "material_blend must be in [0,1] and material_candidates must be positive"
            )
        super().fit(
            q_observed,
            phase_observed,
            valid_observed,
            z_um,
            spacing_yx_um,
            config=config,
            seed=seed,
        )
        valid = np.asarray(valid_observed, bool) & (np.asarray(phase_observed) > 0)
        valid &= np.linalg.norm(q_observed, axis=-1) > 0.5
        iz, iy, ix = np.nonzero(valid)
        self.material_xyz = np.column_stack(
            (
                np.asarray(z_um)[iz],
                iy * self.spacing_yx_um[0],
                ix * self.spacing_yx_um[1],
            )
        )
        self.material_q = normalize_quaternions(np.asarray(q_observed)[valid]).astype(
            np.float32
        )
        self.material_phase = np.asarray(phase_observed)[valid].astype(np.int16)
        gap = np.median(np.diff(z_um)) if len(z_um) > 1 else max(spacing_yx_um)
        self.material_scale = np.r_[
            gap * config["material_z_sigma_gap"],
            self.spacing_yx_um * config["material_xy_sigma_pixels"],
        ]
        self.material_tree = cKDTree(self.material_xyz / self.material_scale)
        self.info_.update(
            material_parameters={k: self.config[k] for k in MATERIAL_DEFAULTS},
            n_continuous_kernels=len(iz),
            material_mechanism="phase-restricted crystal-symmetry-gated Gaussian regression of observed orientations",
            fit_seconds=time.perf_counter() - start,
        )
        return self

    def query(self, coords_um, chunk_size=2048, decode=None):
        coords = np.asarray(coords_um, float).reshape(-1, 3)
        chunk_size = min(int(chunk_size), 2048)
        prediction = super().query(coords, chunk_size=chunk_size, decode=decode)
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
            logits = -0.5 * distance**2 - 0.5 * (angle / sigma) ** 2
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

    def save(self, path):
        super().save(path)
        path = Path(path)
        with np.load(path) as saved:
            arrays = {k: saved[k] for k in saved.files}
        arrays.update(
            material_xyz=self.material_xyz,
            material_q=self.material_q,
            material_phase=self.material_phase,
            material_scale=self.material_scale,
        )
        np.savez_compressed(path, **arrays)
        path.with_suffix(".json").write_text(json.dumps(self.info_, indent=2))
