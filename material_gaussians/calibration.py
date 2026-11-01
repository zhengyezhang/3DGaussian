import sys
import subprocess
from pathlib import Path

import numpy as np

from .io import read_json, write_json
from .suites import DATASETS


def calibrate_distance(data_root="data", output="results"):
    root = Path(output) / "calibration"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for scale in (0.5, 1.0, 2.0):
        config = root / f"distance_{scale:g}.json"
        write_json(config, dict(distance_scale=scale))
        values = []
        for dataset in DATASETS:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "material_gaussians",
                    "run",
                    "--dataset",
                    dataset,
                    "--method",
                    "gsr",
                    "--variant",
                    "calibrated_distance",
                    "--split",
                    "validation",
                    "--tag",
                    f"scale{scale:g}".replace(".", "_"),
                    "--data-root",
                    str(data_root),
                    "--output",
                    str(output),
                    "--config",
                    str(config),
                ],
                check=True,
            )
            path = (
                Path(output)
                / dataset
                / "gsr"
                / (
                    "validation_regular_4x_m0_s0_calibrated_distance_scale"
                    + f"{scale:g}".replace(".", "_")
                )
            )
            values.append(read_json(path / "metrics.json")["angle_penalized_mean_deg"])
        rows.append(
            dict(
                scale=scale,
                mean_E90=float(np.mean(values)),
                dataset_E90=dict(zip(DATASETS, values)),
            )
        )
    chosen = min(rows, key=lambda row: row["mean_E90"])
    result = dict(
        selected_scale=chosen["scale"], selection_split="validation", candidates=rows
    )
    write_json(root / "distance_scale.json", result)
    print(f"Selected distance scale: {chosen['scale']:g}")
    return result
