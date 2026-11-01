import argparse


def parser():
    cli = argparse.ArgumentParser(description="Q-RBSA and PolyMicros comparisons")
    commands = cli.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser(
        "fetch", help="Download pinned source revisions and the PolyMicros checkpoint"
    )
    fetch.add_argument("--code-only", action="store_true")
    fetch.add_argument("--third-party", default="third_party")
    run = commands.add_parser("run", help="Run one fixed-observation comparison")
    run.add_argument("--dataset", choices=("in100", "in718"), required=True)
    run.add_argument(
        "--method",
        choices=("qrbsa", "polymicros", "nearest", "sgf", "cgf", "tgs", "gsr"),
        required=True,
    )
    run.add_argument(
        "--split", choices=("compact", "validation", "native128"), default="compact"
    )
    run.add_argument(
        "--regime", choices=("equal", "endpoint", "disjoint"), default="equal"
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--steps", type=int)
    develop = commands.add_parser(
        "develop", help="Select the Q-RBSA budget on both development regions"
    )
    summary = commands.add_parser(
        "summary", help="Score completed predictions on shared valid support"
    )
    summary.add_argument("--dataset", choices=("in100", "in718"), required=True)
    summary.add_argument("--split", choices=("compact", "native128"), default="compact")
    summary.add_argument(
        "--regime", choices=("equal", "endpoint", "disjoint"), default="equal"
    )
    for sub in (run, develop):
        sub.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    for sub in (run, develop, summary):
        sub.add_argument("--data-root", default="data")
        sub.add_argument("--output-root", default="results/external")
        sub.add_argument("--third-party", default="third_party")
    return cli


def main():
    args = parser().parse_args()
    if args.command == "fetch":
        from .fetch import fetch

        fetch(args.third_party, args.code_only)
    else:
        from .runner import develop, run, summary

        {"run": run, "develop": develop, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
