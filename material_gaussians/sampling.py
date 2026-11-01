import numpy as np
from scipy.ndimage import shift

PATTERNS = ("regular", "irregular", "block", "dispersed", "shift", "depth")


def observation_mask(nz, stride=4, pattern="regular", seed=0):
    if nz < 2 or stride < 2:
        raise ValueError("At least two sections and stride >= 2 are required.")
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown sampling pattern: {pattern}")
    obs = np.unique(np.r_[np.arange(0, nz, stride), nz - 1]).astype(int)
    if pattern == "irregular":
        rng = np.random.default_rng(seed + 1729)
        obs = np.sort(
            np.r_[
                0, rng.choice(np.arange(1, nz - 1), len(obs) - 2, replace=False), nz - 1
            ]
        )
    elif pattern == "block":
        obs = obs[
            (abs(obs - (nz - 1) / 2) > stride * 1.5) | (obs == 0) | (obs == nz - 1)
        ]
    elif pattern == "dispersed":
        if nz != 97 or stride != 4:
            raise ValueError(
                "The equal-budget dispersed control uses 97 sections and stride 4."
            )
        obs = obs[~np.isin(obs, [24, 48, 72])]
    return obs


def observed_input(reference, obs, z, pattern):
    q, phase, valid = (reference[key][obs].copy() for key in ("q", "phase", "valid"))
    depths = z[obs].copy()
    details = {}
    rng = np.random.default_rng(260527)
    if pattern == "shift":
        shifts = rng.integers(-1, 2, (len(obs), 2))
        for i, (dy, dx) in enumerate(shifts):
            q[i] = shift(q[i], (dy, dx, 0), order=0, mode="constant", cval=0)
            phase[i] = shift(phase[i], (dy, dx), order=0, mode="constant", cval=0)
            valid[i] = shift(valid[i], (dy, dx), order=0, mode="constant", cval=0)
        details["observation_shifts_yx"] = shifts.tolist()
    elif pattern == "depth":
        offsets = rng.uniform(-0.1, 0.1, len(obs)) * np.median(np.diff(depths))
        offsets[[0, -1]] = 0
        depths += offsets
        details["reported_depth_offsets_um"] = offsets.tolist()
    return q, phase, valid, depths, details
