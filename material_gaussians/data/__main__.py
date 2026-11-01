import argparse

from .io import datasets


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare the four EBSD volumes."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("download", "prepare"):
        subparser = commands.add_parser(command)
        subparser.add_argument("datasets", nargs="+", choices=["all", *datasets()])
        subparser.add_argument("--data-root", default="data")
        if command == "download":
            subparser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    names = (
        list(datasets())
        if "all" in args.datasets
        else list(dict.fromkeys(args.datasets))
    )
    for name in names:
        if args.command == "download":
            from .download import download

            download(name, args.data_root, args.workers)
        else:
            from .prepare import prepare

            prepare(name, args.data_root)


if __name__ == "__main__":
    main()
