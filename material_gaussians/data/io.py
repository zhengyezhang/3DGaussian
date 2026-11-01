import hashlib
import json
from pathlib import Path


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def datasets():
    path = Path(__file__).resolve().parents[2] / "configs" / "datasets.json"
    return json.loads(path.read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def verify(path, size, expected, algorithm="sha256"):
    path = Path(path)
    if path.stat().st_size != size:
        raise ValueError(f"{path}: expected {size} bytes, found {path.stat().st_size}")
    actual = digest(path, algorithm)
    if actual != expected:
        raise ValueError(f"{path}: {algorithm} mismatch ({actual})")
    return actual
