import concurrent.futures
import time
import urllib.request
from pathlib import Path

from .io import datasets, verify, write_json


def fetch_ranges(url, target, length, total=None, offset=0, etag=None, workers=8):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    total = length if total is None else total
    chunks = target.parent / "ranges" / target.name
    chunks.mkdir(parents=True, exist_ok=True)

    def fetch(bounds):
        start, stop = bounds
        path = chunks / f"{start}-{stop}"
        if path.exists() and path.stat().st_size == stop - start:
            return path
        headers = {"Range": f"bytes={offset + start}-{offset + stop - 1}"}
        if etag:
            headers["If-Match"] = etag
        for attempt in range(6):
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=90) as response:
                    expected_range = (
                        f"bytes {offset + start}-{offset + stop - 1}/{total}"
                    )
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != expected_range
                    ):
                        raise ValueError(
                            f"Unexpected byte range for {target.name}: {response.headers.get('Content-Range')}"
                        )
                    if etag and response.headers.get("ETag") != etag:
                        raise ValueError(f"The source version changed: {target.name}")
                    content = response.read(stop - start + 1)
                if len(content) != stop - start:
                    raise ValueError(
                        f"Incomplete byte range for {target.name}: {start}:{stop}"
                    )
                part = path.with_suffix(".part")
                part.write_bytes(content)
                part.replace(path)
                return path
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 * (attempt + 1))

    ranges = [
        (start, min(start + (4 << 20), length)) for start in range(0, length, 4 << 20)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(fetch, ranges))
    partial = target.with_suffix(target.suffix + ".part")
    with partial.open("wb") as stream:
        for part in parts:
            stream.write(part.read_bytes())
    partial.replace(target)


def download(name, data_root="data", workers=8):
    source = datasets()[name]
    folder = Path(data_root) / "raw" / source["raw_directory"]
    folder.mkdir(parents=True, exist_ok=True)
    record = dict(source)
    if name == "in718":
        for field, array in source["arrays"].items():
            target = folder / f"{field}.bin"
            fetch_ranges(
                source["url"],
                target,
                array["length_bytes"],
                source["source_size_bytes"],
                array["offset_bytes"],
                source["etag"],
                workers,
            )
            verify(target, array["length_bytes"], array["sha256"])
            print(f"{name}: {field} verified", flush=True)
    else:
        target = folder / source["filename"]
        fetch_ranges(source["url"], target, source["size_bytes"], workers=workers)
        verify(target, source["size_bytes"], source["sha256"])
        for algorithm, expected in source.get("publisher_checksum", {}).items():
            verify(target, source["size_bytes"], expected, algorithm)
        if source.get("publisher_checksum"):
            record["repository_checksum_verified"] = source["publisher_checksum"]
        print(f"{name}: {target.name} verified", flush=True)
    record["verified_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["verified_sha256"] = True
    write_json(folder / "source.json", record)
    return record
