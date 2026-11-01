import hashlib
from pathlib import Path
import subprocess


SOURCES = {
    "Q-RBSA": (
        "https://github.com/UCSB-VRL/Q-RBSA.git",
        "2203058b2c6aa7d246448d7749f2fb2047c7cb12",
    ),
    "PolyMicros": (
        "https://github.com/Zraxix/PolyMicros.git",
        "9fb97e16d01565f99c08491fa9c2efdd40872993",
    ),
}
CHECKPOINT_SHA256 = "55f786d5a4980ba392e34f82d4914119b4b535bb3f1999460fc9bbb61fae6a6b"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(base="third_party", code_only=False):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    for name, (url, commit) in SOURCES.items():
        path = base / name
        if not path.exists():
            subprocess.run(["git", "clone", url, str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", commit], check=True)
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != commit:
            raise RuntimeError(f"{name}: expected {commit}, found {actual}")
    patch = base / "qrbsa_device_compatibility.patch"
    if not patch.is_file():
        patch = (
            Path(__file__).resolve().parents[2]
            / "third_party/qrbsa_device_compatibility.patch"
        )
    reverse = subprocess.run(
        [
            "git",
            "-C",
            str(base / "Q-RBSA"),
            "apply",
            "--reverse",
            "--check",
            str(patch.resolve()),
        ],
        capture_output=True,
    )
    if reverse.returncode:
        subprocess.run(
            [
                "git",
                "-C",
                str(base / "Q-RBSA"),
                "apply",
                "--check",
                str(patch.resolve()),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(base / "Q-RBSA"), "apply", str(patch.resolve())],
            check=True,
        )
    if not code_only:
        import gdown

        folder = base / "checkpoints/PolyMicros"
        folder.mkdir(parents=True, exist_ok=True)
        for filename, identifier in (
            ("config.json", "1yQ9K0oBMB1hB2wXHFds9dimMO9VddMhl"),
            ("PolyMicros.ckpt", "1WbfRH4SJ_SW2Fkibaf0Qc9BCKkQtsntR"),
        ):
            target = folder / filename
            if not target.exists():
                gdown.download(
                    id=identifier, output=str(target), quiet=False, resume=True
                )
        if digest(folder / "PolyMicros.ckpt") != CHECKPOINT_SHA256:
            raise RuntimeError("PolyMicros checkpoint checksum mismatch")
    print("External sources and requested checkpoint verified.", flush=True)
