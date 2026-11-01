import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, shift
from skimage.registration import phase_cross_correlation

from .io import digest, write_json


def register(dataset, data_root="data"):
    path = Path(data_root) / "processed" / f"{dataset}.npz"
    output = path.with_name(f"{dataset}_registered.npz")
    if output.exists():
        if not output.with_suffix(".json").exists():
            raise FileNotFoundError(output.with_suffix(".json"))
        stored = json.loads(output.with_suffix(".json").read_text())
        if (
            digest(output) != stored["npz_sha256"]
            or digest(path) != stored["source_npz_sha256"]
        ):
            raise ValueError(f"{output}: registered reference or its input has changed")
        return output
    with np.load(path) as raw:
        arrays = {key: raw[key] for key in raw.files}
    metadata = json.loads(path.with_suffix(".json").read_text())
    valid = arrays["valid"]
    key = next(
        (name for name in ("image_quality", "band_contrast") if name in arrays), None
    )
    if key:
        contrast = arrays[key].astype(np.float32)
        contrast[~valid] = 0
        low, high = np.quantile(contrast[valid], [0.02, 0.98])
        contrast = np.clip((contrast - low) / max(high - low, 1e-6), 0, 1)
    else:
        from material_gaussians.orientation import canonicalize_quaternions

        contrast = np.zeros(valid.shape, np.float32)
        for phase, symmetry in metadata["phase_symmetry"].items():
            mask = valid & (arrays["phase"] == int(phase))
            q = canonicalize_quaternions(arrays["q"][mask], symmetry)
            contrast[mask] = q[:, 1] + 2 * q[:, 2] + 3 * q[:, 3]
    contrast = np.stack([gaussian_filter(section, 0.8) for section in contrast])
    increment = np.zeros((len(valid), 2), np.float64)
    errors = []
    for z in range(1, len(valid)):
        delta, _, _ = phase_cross_correlation(
            contrast[z - 1], contrast[z], upsample_factor=1, normalization=None
        )
        if np.max(np.abs(delta)) > 0.15 * min(valid.shape[1:]):
            errors.append({"slice": z, "rejected_shift": delta.tolist()})
            delta = np.zeros(2)
        increment[z] = delta
    cumulative = np.cumsum(increment, axis=0)
    cumulative -= cumulative[len(valid) // 2]
    for value in arrays.values():
        if value.shape[:3] != valid.shape:
            continue
        for z, delta in enumerate(cumulative):
            value[z] = shift(
                value[z],
                tuple(delta) + (0,) * (value[z].ndim - 2),
                order=0,
                mode="constant",
                cval=0,
                prefilter=False,
            )
    arrays["valid"] = arrays["valid"].astype(bool)
    np.savez_compressed(output, **arrays)
    metadata.update(
        dataset_id=f"{dataset}_registered",
        registration="Full dense-stack adjacent quality-contrast integer translation registration; middle plane anchored. Subsequently sparse-masked aligned-reference benchmark.",
        registration_source=str(path.relative_to(data_root)),
        registration_contrast=key or "canonical orientation contrast",
        registration_shifts_yx_pixels=cumulative.astype(int).tolist(),
        registration_rejected_peaks=errors,
        source_npz_sha256=digest(path),
        npz_sha256=digest(output),
        valid_voxels=int(arrays["valid"].sum()),
        phase_counts={
            str(int(phase)): int(count)
            for phase, count in zip(*np.unique(arrays["phase"], return_counts=True))
        },
    )
    metadata.setdefault("caveats", []).append(
        "The full dense reference was used for registration before sparse masking."
    )
    write_json(output.with_suffix(".json"), metadata)
    print(f"{dataset}: registered reference saved to {output}", flush=True)
    return output
