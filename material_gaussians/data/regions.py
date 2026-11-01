import json
from pathlib import Path

import numpy as np

from .io import datasets, digest, write_json

SPLITS = ("test", "validation", "audit", "fresh", "compact")


def crop_ranges(shape, split):
    full = np.asarray(shape, dtype=int)
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if split == "validation":
        size = np.minimum(full, [65, 48, 48])
        start = np.zeros(3, dtype=int)
    elif split in ("audit", "fresh"):
        size = np.minimum(full, [65, 48, 48])
        start = [
            (full[0] - size[0]) // 2,
            48 if split == "fresh" else full[1] - size[1],
            0,
        ]
    elif split == "compact":
        main = np.minimum(full, [97, 128, 128])
        size = np.minimum(main, [64, 64, 64])
        origin = [(full[0] - main[0]) // 2, full[1] - main[1], full[2] - main[2]]
        start = [int(a + (n - m) // 2) for a, n, m in zip(origin, main, size)]
    else:
        size = np.minimum(full, [97, 128, 128])
        start = [(full[0] - size[0]) // 2, full[1] - size[1], full[2] - size[2]]
    return [[int(a), int(a + n)] for a, n in zip(start, size)]


def extract_region(volume, ranges):
    index = tuple(slice(start, stop) for start, stop in ranges)
    region = {
        key: np.asarray(volume[key][index]).copy() for key in ("q", "phase", "valid")
    }
    region["spacing_um"] = np.asarray(volume["spacing_um"]).copy()
    return region


def save_region(dataset, split, volume, source, source_metadata, data_root):
    target = Path(data_root) / "crops" / f"{dataset}_{split}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    ranges = crop_ranges(volume["q"].shape[:3], split)
    symmetry = source_metadata.get(
        "phase_symmetry", source_metadata.get("phase_symmetries")
    )
    if not symmetry:
        raise ValueError(f"{source}: crystal symmetry metadata is missing")
    np.savez_compressed(target, **extract_region(volume, ranges))
    metadata = {
        "source": str(Path(source).relative_to(data_root)),
        "source_metadata": source_metadata,
        "crop_ranges_zyx": ranges,
        "phase_symmetry": symmetry,
        "crop_sha256": digest(target),
    }
    write_json(target.with_suffix(".json"), metadata)
    print(f"{dataset}: prepared {split} {ranges}", flush=True)


def build_crops(dataset, data_root="data", splits=SPLITS):
    data_root = Path(data_root)
    source = data_root / "processed" / datasets()[dataset]["processed_filename"]
    source_metadata = json.loads(source.with_suffix(".json").read_text())
    missing = [
        split
        for split in splits
        if not (data_root / "crops" / f"{dataset}_{split}.npz").exists()
    ]
    if not missing:
        return
    with np.load(source) as archive:
        volume = {key: archive[key] for key in ("q", "phase", "valid", "spacing_um")}
    for split in missing:
        save_region(dataset, split, volume, source, source_metadata, data_root)


def load_region(dataset, split, data_root="data"):
    if dataset not in datasets():
        raise ValueError(f"Unknown dataset: {dataset}")
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    target = Path(data_root) / "crops" / f"{dataset}_{split}.npz"
    if not target.exists():
        build_crops(dataset, data_root, (split,))
    metadata = json.loads(target.with_suffix(".json").read_text())
    if digest(target) != metadata["crop_sha256"]:
        raise ValueError(f"{target}: the crop checksum differs from its metadata")
    with np.load(target) as archive:
        region = {key: archive[key] for key in archive.files}
    return region, metadata
