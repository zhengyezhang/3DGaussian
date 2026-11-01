import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")


def main():
    from .models import METHODS
    from .sampling import PATTERNS
    from .suites import DATASETS, GROUPS

    parser = argparse.ArgumentParser(prog="python -m material_gaussians")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--dataset", choices=DATASETS, required=True)
    run.add_argument("--method", choices=METHODS, required=True)
    run.add_argument("--variant", default="full")
    run.add_argument(
        "--split",
        choices=("test", "validation", "audit", "fresh", "compact"),
        default="test",
    )
    run.add_argument("--stride", type=int, default=4)
    run.add_argument("--pattern", choices=PATTERNS, default="regular")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--mask-seed", type=int)
    run.add_argument("--fit-seed", type=int)
    run.add_argument("--thresholds", type=float, nargs="+", default=[2, 5, 10])
    suite = commands.add_parser("suite")
    suite.add_argument("group", choices=(*GROUPS, "all"))
    suite.add_argument("--datasets", choices=DATASETS, nargs="+", default=DATASETS)
    suite.add_argument("--methods", choices=METHODS, nargs="+")
    suite.add_argument("--dry-run", action="store_true")
    for child in (run, suite):
        child.add_argument("--data-root", default="data")
        child.add_argument("--output", default="results")
        child.add_argument("--config", type=Path)
        child.add_argument("--tag", default="")
        child.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
        child.add_argument("--threads", type=int, default=2)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("paths", type=Path, nargs="+")
    analyze.add_argument(
        "--anchor", choices=("native", "anchored", "both"), default="both"
    )
    analyze.add_argument("--thresholds", type=float, nargs="+", default=[2, 5, 10])
    analyze.add_argument("--data-root")
    summary = commands.add_parser("summary")
    summary.add_argument("--output", default="results")
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--data-root", default="data")
    calibration.add_argument("--output", default="results")
    compare = commands.add_parser("compare")
    compare.add_argument("paths", type=Path, nargs="+")
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument(
        "--mode", choices=("connections", "budget"), default="connections"
    )
    args = parser.parse_args()
    if args.command == "run":
        from .experiment import run_experiment
        from .io import read_json

        overrides = read_json(args.config) if args.config else {}
        overrides.update(
            device=args.device, siren_device=args.device, threads=args.threads
        )
        if args.variant == "calibrated_distance" and "distance_scale" not in overrides:
            selection = Path(args.output) / "calibration" / "distance_scale.json"
            if selection.exists():
                overrides["distance_scale"] = read_json(selection)["selected_scale"]
        options = vars(args).copy()
        for key in ("command", "config", "device", "threads"):
            options.pop(key)
        run_experiment(**options, overrides=overrides)
    elif args.command == "suite":
        from .suites import launch

        options = vars(args).copy()
        options.pop("command")
        launch(**options)
    elif args.command == "analyze":
        from .analysis import evaluate_saved
        from .io import read_json

        runs = sorted(
            set(
                path if (path / "prediction.npz").exists() else file.parent
                for path in args.paths
                for file in (
                    [path / "prediction.npz"]
                    if (path / "prediction.npz").exists()
                    else path.rglob("prediction.npz")
                )
            )
        )
        runs = [
            path
            for path in runs
            if (path / "manifest.json").exists()
            and "dataset" in read_json(path / "manifest.json")
        ]
        if not runs:
            parser.error("No saved predictions found.")
        for path in runs:
            for anchored in (
                [False, True] if args.anchor == "both" else [args.anchor == "anchored"]
            ):
                print(f"Analyze: {path} | anchored={anchored}", flush=True)
                evaluate_saved(
                    path, args.thresholds, anchored, data_root=args.data_root
                )
    elif args.command == "summary":
        from .suites import summarize

        summarize(args.output)
    elif args.command == "calibrate":
        from .calibration import calibrate_distance

        calibrate_distance(args.data_root, args.output)
    elif args.command == "compare":
        from .compare import compare_saved, compare_equal_budget

        function = compare_saved if args.mode == "connections" else compare_equal_budget
        function(args.paths, args.output)


if __name__ == "__main__":
    main()
