import shlex
import subprocess
import sys
from pathlib import Path

from .io import read_json, write_csv

DATASETS = ("in100", "in718", "ti64", "zrnb")
CORE = ("nearest", "sgf", "cgf", "tgs", "gsr")
GROUPS = (
    "main",
    "sampling",
    "spatial",
    "repeats",
    "ablation",
    "kernels",
    "robustness",
    "baselines",
    "development",
)


def cases(group):
    if group == "all":
        return [case for name in GROUPS for case in cases(name)]
    result = []
    for dataset in DATASETS:

        def add(method, **kwargs):
            result.append(dict(dataset=dataset, method=method, **kwargs))

        if group == "main":
            for method in CORE:
                for stride in (2, 4, 8):
                    add(method, stride=stride)
        elif group == "sampling":
            for method in CORE:
                for pattern in ("irregular", "block", "dispersed"):
                    add(method, pattern=pattern)
        elif group == "spatial":
            for method in CORE:
                add(method, split="audit")
        elif group == "repeats":
            for method in ("sgf", "cgf"):
                for seed in (1, 2):
                    add(method, seed=seed)
        elif group == "ablation":
            add("cgf", variant="paired")
            for variant in ("no_motion", "no_endpoint_mixing"):
                add("tgs", variant=variant)
            for variant in ("no_matching", "no_smoothing", "distance"):
                add("gsr", variant=variant)
            for variant in (
                "initial",
                "isotropic",
                "mean",
                "symmetry_mean",
                "no_boundary",
            ):
                add("sgf", variant=variant)
        elif group == "kernels":
            for split in ("test", "fresh"):
                for method in ("cgf", "tgs", "gsr"):
                    add(method, split=split)
                    add(
                        method,
                        split=split,
                        variant="calibrated_distance" if method == "gsr" else "laplace",
                    )
        elif group == "robustness":
            for method in CORE:
                for pattern, mask in [
                    ("regular", 0),
                    ("irregular", 0),
                    ("irregular", 1),
                    ("irregular", 2),
                    ("shift", 0),
                    ("depth", 0),
                ]:
                    add(
                        method,
                        split="fresh",
                        pattern=pattern,
                        mask_seed=mask,
                        fit_seed=0,
                    )
        elif group == "baselines":
            for stride in (2, 4, 8):
                add("slerp", stride=stride)
            add("siren")
        elif group == "development":
            folder = Path(__file__).resolve().parents[1] / "configs"
            for candidate in read_json(folder / "development.json"):
                add(
                    candidate["method"],
                    split="validation",
                    tag=candidate["name"],
                    config="configs/development/" + candidate["name"] + ".json",
                )
        else:
            raise ValueError(f"Unknown experiment group: {group}")
    return result


def launch(
    group,
    datasets=DATASETS,
    methods=None,
    dry_run=False,
    data_root="data",
    output="results",
    config=None,
    tag="",
    device="cpu",
    threads=2,
):
    plan = [
        case
        for case in cases(group)
        if case["dataset"] in datasets and (not methods or case["method"] in methods)
    ]
    for case in plan:
        command = [sys.executable, "-m", "material_gaussians", "run"]
        for key, value in case.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        command.extend(
            [
                "--data-root",
                str(data_root),
                "--output",
                str(output),
                "--device",
                device,
                "--threads",
                str(threads),
            ]
        )
        if config and "config" not in case:
            command.extend(["--config", str(config)])
        if tag and "tag" not in case:
            command.extend(["--tag", tag])
        if dry_run:
            print(shlex.join(["python", *command[1:]]))
        else:
            subprocess.run(command, check=True)
    return len(plan)


def summarize(root="results"):
    root = Path(root)
    rows = []
    for file in sorted(root.rglob("metrics.json")):
        manifest_path = file.with_name("manifest.json")
        if not manifest_path.exists():
            continue
        manifest, metrics = read_json(manifest_path), read_json(file)
        row = {
            key: manifest.get(key)
            for key in (
                "dataset",
                "method",
                "split",
                "stride",
                "pattern",
                "variant",
                "seed",
                "mask_seed",
                "fit_seed",
                "paired_output",
            )
        }
        row["run"] = str(file.parent.relative_to(root))
        row.update(
            {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float)) or value is None
            }
        )
        for threshold, values in metrics.get("thresholds", {}).items():
            row.update(
                {
                    f"{key}_{threshold}deg": value
                    for key, value in values.items()
                    if isinstance(value, (int, float)) or value is None
                }
            )
        rows.append(row)
    write_csv(root / "summary.csv", rows)
    print(f"{len(rows)} runs: {root / 'summary.csv'}")
    return rows
