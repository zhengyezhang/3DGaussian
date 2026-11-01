import time
import numpy as np
from scipy import ndimage, sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from material_gaussians.orientation import (
    normalize_quaternions,
    align_quaternions,
    misorientation_deg,
)


def _regions(q, phase, valid, symmetries, threshold):
    ids = np.arange(valid.size).reshape(valid.shape)
    rows, cols = ([], [])
    for axis in (0, 1):
        a, b = ([slice(None)] * 2, [slice(None)] * 2)
        a[axis], b[axis] = (slice(None, -1), slice(1, None))
        a, b = (tuple(a), tuple(b))
        good = np.zeros(valid[a].shape, bool)
        for p, symmetry in symmetries.items():
            selected = valid[a] & valid[b] & (phase[a] == p) & (phase[b] == p)
            good[selected] = (
                misorientation_deg(q[a][selected], q[b][selected], symmetry)
                <= threshold
            )
        rows.append(ids[a][good])
        cols.append(ids[b][good])
    rows, cols = (np.concatenate(rows), np.concatenate(cols))
    graph = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(valid.size, valid.size)
    )
    _, labels = connected_components(graph, directed=False)
    labels = labels.reshape(valid.shape)
    _, remapped = np.unique(labels[valid], return_inverse=True)
    output = np.zeros(valid.shape, np.int32)
    output[valid] = remapped + 1
    return output


class GaussianGrainField:
    defaults = dict(
        grain_split_deg=5.0,
        grain_match_deg=10.0,
        grain_min_match=0.05,
        grain_sigma_pixels=1.0,
        grain_shape="gaussian",
        grain_absent_logit=-6.0,
        grain_q_sigma_pixels=0.0,
        threads=2,
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
        self.config = {**self.defaults, **(config or {})}
        self.q = normalize_quaternions(np.asarray(q_observed)).astype(np.float32)
        self.phase = np.asarray(phase_observed, np.int16).copy()
        self.valid = np.asarray(valid_observed, bool).copy() & (self.phase > 0)
        self.valid &= np.linalg.norm(self.q, axis=-1) > 0.5
        self.z = np.asarray(z_um, float).copy()
        self.spacing = np.asarray(spacing_yx_um, float).copy()
        self.symmetries = {
            int(k): v
            for k, v in self.config.get("phase_symmetry", {1: "cubic"}).items()
        }
        if (
            self.q.shape[:-1] != self.phase.shape
            or self.phase.shape != self.valid.shape
        ):
            raise ValueError("q, phase and valid section shapes disagree")
        if (
            len(self.z) != len(self.q)
            or not len(self.z)
            or np.any(np.diff(self.z) <= 0)
        ):
            raise ValueError("Observed z must be nonempty and increasing")
        if np.any(self.spacing <= 0) or self.config["grain_sigma_pixels"] <= 0:
            raise ValueError("Physical spacing and Gaussian width must be positive")
        if self.config["grain_shape"] not in ("gaussian", "distance"):
            raise ValueError("grain_shape must be gaussian or distance")
        self.sections = []
        for q, phase, valid in zip(self.q, self.phase, self.valid):
            if not valid.any():
                raise ValueError("Each observed section needs indexed pixels")
            labels = _regions(
                q, phase, valid, self.symmetries, self.config["grain_split_deg"]
            )
            regions = {}
            for rid in range(1, labels.max() + 1):
                mask = labels == rid
                p = int(phase[mask][0])
                anchor = q[mask][0]
                aligned = align_quaternions(q[mask], anchor, self.symmetries[p])
                mean = normalize_quaternions(aligned.mean(0)).astype(np.float32)
                regions[rid] = dict(phase=p, mean=mean, area=int(mask.sum()))
            self.sections.append(dict(labels=labels, regions=regions))
        self.pairs = [self._match(k) for k in range(len(self.z) - 1)]
        self.info_ = dict(
            method="GaussianGrainField",
            fit_seconds=time.perf_counter() - started,
            config=self.config,
            seed=int(seed),
            observed_regions=[len(s["regions"]) for s in self.sections],
            matched_pairs=[sum((a > 0 and b > 0 for a, b in p)) for p in self.pairs],
            representation="Gaussian-kernel occupancy logits"
            if self.config["grain_shape"] == "gaussian"
            else "signed-distance shape interpolation, not Gaussian splatting",
            observed_sections_exactly_anchored=True,
        )
        return self

    def _match(self, k):
        sa, sb = self.sections[k : k + 2]
        a, b = (sa["labels"], sb["labels"])
        na, nb = (len(sa["regions"]), len(sb["regions"]))
        overlap = np.bincount(
            (a * (nb + 1) + b).ravel(), minlength=(na + 1) * (nb + 1)
        ).reshape(na + 1, nb + 1)[1:, 1:]
        score = np.zeros((na, nb))
        ia, ib = np.nonzero(overlap)
        for i, j in zip(ia, ib):
            ra, rb = (sa["regions"][i + 1], sb["regions"][j + 1])
            if ra["phase"] != rb["phase"]:
                continue
            angle = float(
                misorientation_deg(ra["mean"], rb["mean"], self.symmetries[ra["phase"]])
            )
            if angle <= self.config["grain_match_deg"]:
                dice = 2 * overlap[i, j] / (ra["area"] + rb["area"])
                score[i, j] = dice * np.exp(
                    -0.5 * (angle / self.config["grain_match_deg"]) ** 2
                )
        ia, ib = linear_sum_assignment(-score)
        keep = (score[ia, ib] > 0) & (score[ia, ib] >= self.config["grain_min_match"])
        matches = [(int(i + 1), int(j + 1)) for i, j in zip(ia[keep], ib[keep])]
        useda, usedb = ({a for a, b in matches}, {b for a, b in matches})
        return (
            matches
            + [(a, 0) for a in range(1, na + 1) if a not in useda]
            + [(0, b) for b in range(1, nb + 1) if b not in usedb]
        )

    def _support(self, k, rid):
        shape = self.phase.shape[1:]
        if rid == 0:
            return (
                np.full(shape, self.config["grain_absent_logit"], np.float32),
                None,
                None,
            )
        mask = self.sections[k]["labels"] == rid
        sigma = self.config["grain_sigma_pixels"] * min(self.spacing) / self.spacing
        if self.config["grain_shape"] == "gaussian":
            mass = ndimage.gaussian_filter(mask.astype(float), sigma, mode="constant")
            norm = ndimage.gaussian_filter(np.ones(shape), sigma, mode="constant")
            probability = np.clip(mass / norm, 0.0001, 1 - 0.0001)
            score = np.log(probability) - np.log1p(-probability)
        else:
            score = (
                ndimage.distance_transform_edt(mask, sampling=self.spacing)
                - ndimage.distance_transform_edt(~mask, sampling=self.spacing)
            ) / min(self.spacing)
        indices = ndimage.distance_transform_edt(
            ~mask, sampling=self.spacing, return_distances=False, return_indices=True
        )
        q = self.q[k].copy()
        qsigma = float(self.config["grain_q_sigma_pixels"])
        if qsigma > 0:
            region = self.sections[k]["regions"][rid]
            aligned = np.zeros_like(q)
            aligned[mask] = align_quaternions(
                q[mask], region["mean"], self.symmetries[region["phase"]]
            )
            denominator = ndimage.gaussian_filter(
                mask.astype(float), qsigma, mode="constant"
            )
            averaged = ndimage.gaussian_filter(
                aligned, (qsigma, qsigma, 0), mode="constant"
            )
            q[mask] = normalize_quaternions(averaged[mask] / denominator[mask, None])
        return (
            score.astype(np.float32),
            q[tuple(indices)],
            self.sections[k]["regions"][rid]["phase"],
        )

    def predict(self, z_um):
        target = np.asarray(z_um, float)
        shape = (len(target), *self.phase.shape[1:])
        output = dict(
            q=np.zeros((*shape, 4), np.float32),
            phase=np.zeros(shape, np.int16),
            valid=np.ones(shape, bool),
            confidence=np.zeros(shape, np.float32),
        )
        cache = {}
        for iz, z in enumerate(target):
            exact = np.flatnonzero(np.isclose(self.z, z, rtol=0, atol=1e-08))
            if len(exact) or len(self.z) == 1 or z < self.z[0] or (z > self.z[-1]):
                k = int(exact[0]) if len(exact) else int(np.argmin(abs(self.z - z)))
                output["q"][iz], output["phase"][iz], output["valid"][iz] = (
                    self.q[k],
                    self.phase[k],
                    self.valid[k],
                )
                output["confidence"][iz] = self.valid[k]
                continue
            k = int(np.searchsorted(self.z, z) - 1)
            t = (z - self.z[k]) / (self.z[k + 1] - self.z[k])
            if k not in cache:
                cache = {
                    k: [
                        (self._support(k, a), self._support(k + 1, b))
                        for a, b in self.pairs[k]
                    ]
                }
            supports = cache[k]
            scores = np.stack([(1 - t) * a[0] + t * b[0] for a, b in supports])
            winners = scores.argmax(0)
            best = scores.max(0)
            mass = np.exp(scores - best[None]).sum(0)
            output["confidence"][iz] = 1 / mass
            for g, (a, b) in enumerate(supports):
                mask = winners == g
                if not mask.any():
                    continue
                if a[1] is None:
                    q, p = (b[1][mask], b[2])
                elif b[1] is None:
                    q, p = (a[1][mask], a[2])
                else:
                    p = a[2]
                    qa = a[1][mask]
                    qb = align_quaternions(b[1][mask], qa, self.symmetries[p])
                    q = normalize_quaternions((1 - t) * qa + t * qb)
                output["q"][iz][mask], output["phase"][iz][mask] = (q, p)
        return output
