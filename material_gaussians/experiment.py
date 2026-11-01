import hashlib
import json
import platform
import resource
import time
import types
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .data import load_region
from .io import file_hash, read_json, volume_hash, write_json
from .metrics import boundaries, boundary_metrics, evaluate, grain_labels, grain_metrics
from .models import build_model, model_config
from .sampling import observation_mask, observed_input


def reference_cache(reference, symmetries, threshold, cache_dir, fingerprint):
    path = Path(cache_dir) / f"{fingerprint[:24]}_{threshold:g}deg.npz"
    if path.exists():
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "boundary": boundaries(
            reference["q"],
            reference["phase"],
            reference["valid"],
            symmetries,
            threshold,
        ),
        "labels": grain_labels(
            reference["q"],
            reference["phase"],
            reference["valid"],
            symmetries,
            threshold,
        ),
    }
    np.savez_compressed(path, **result)
    return result


def score(prediction, reference, obs, symmetries, thresholds, cache_dir, fingerprint):
    cache = reference_cache(reference, symmetries, 5, cache_dir, fingerprint)
    result, maps = evaluate(prediction, reference, obs, symmetries, cache)
    hidden_z = np.ones(len(reference["q"]), bool)
    hidden_z[obs] = False
    hidden = reference["valid"] & hidden_z[:, None, None]
    values = {
        "5": {key: result[key] for key in ("boundary_f1", "grain_ari", "grain_vi_bits")}
    }
    for threshold in thresholds:
        if threshold == 5:
            continue
        cache = reference_cache(
            reference, symmetries, threshold, cache_dir, fingerprint
        )
        phase = np.where(prediction["valid"], prediction["phase"], 0)
        pb = boundaries(
            prediction["q"], phase, reference["valid"], symmetries, threshold
        )
        pl = grain_labels(
            prediction["q"], phase, reference["valid"], symmetries, threshold
        )
        row = boundary_metrics(
            pb,
            cache["boundary"],
            reference["valid"],
            hidden_z,
            reference["spacing_um"][1:],
            float(max(reference["spacing_um"][1:])),
        )
        row.update(grain_metrics(pl, cache["labels"], hidden))
        values[f"{threshold:g}"] = row
    result["thresholds"] = values
    return result, maps


def run_experiment(
    dataset,
    method,
    split="test",
    stride=4,
    pattern="regular",
    seed=0,
    mask_seed=None,
    fit_seed=None,
    variant="full",
    overrides=None,
    thresholds=(2, 5, 10),
    data_root="data",
    output="results",
    tag="",
):
    config = model_config(method, variant, overrides)
    mask_seed = seed if mask_seed is None else mask_seed
    fit_seed = seed if fit_seed is None else fit_seed
    name = f"{split}_{pattern}_{stride}x_m{mask_seed}_s{fit_seed}_{variant}"
    if tag:
        if any(
            char
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for char in tag
        ):
            raise ValueError(
                "Tags may contain letters, digits, underscores and hyphens."
            )
        name += "_" + tag
    out = Path(output) / dataset / method / name
    reference, metadata = load_region(dataset, split, data_root=data_root)
    symmetries = {int(key): value for key, value in metadata["phase_symmetry"].items()}
    config["phase_symmetry"] = symmetries
    z = np.arange(len(reference["q"])) * reference["spacing_um"][0]
    obs = observation_mask(len(z), stride, pattern, mask_seed)
    if not (reference["valid"][np.setdiff1d(np.arange(len(z)), obs)]).any():
        raise ValueError("No valid hidden voxels remain under this sampling mask.")
    fingerprint = volume_hash(reference)
    sources = {
        str(path.relative_to(Path(__file__).parent)): file_hash(path)
        for path in sorted(Path(__file__).parent.rglob("*.py"))
    }
    spec = dict(
        dataset=dataset,
        method=method,
        split=split,
        stride=stride,
        pattern=pattern,
        seed=seed,
        mask_seed=mask_seed,
        fit_seed=fit_seed,
        variant=variant,
        config=config,
        thresholds=list(thresholds),
        observed_z_indices=obs.tolist(),
        reference_sha256=fingerprint,
        source_hashes=sources,
        tag=tag,
    )
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    manifest_path = out / "manifest.json"
    complete_path = out / ("complete.json" if variant == "paired" else "metrics.json")
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous.get("signature") != signature:
            raise RuntimeError(
                f"Configuration, input or source changed: {out}. Use a new --tag or --output."
            )
        if complete_path.exists():
            print(f"Complete: {out}", flush=True)
            return out
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        spec,
        signature=signature,
        region=metadata,
        data_root=str(data_root),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        python=platform.python_version(),
        platform=platform.platform(),
    )
    write_json(manifest_path, manifest)
    q, phase, valid, depths, details = observed_input(reference, obs, z, pattern)
    manifest.update(details)
    write_json(manifest_path, manifest)
    print(f"Run: {out} | {len(obs)}/{len(z)} sections", flush=True)
    with threadpool_limits(limits=int(config["threads"])):
        model = build_model(method, variant)
        start = time.perf_counter()
        model.fit(
            q,
            phase,
            valid,
            depths,
            reference["spacing_um"][1:].copy(),
            config=config,
            seed=fit_seed,
        )
        fit_seconds = time.perf_counter() - start
        outputs = (
            [("full", out / "full"), ("coarse", out / "coarse")]
            if variant == "paired"
            else [("full", out)]
        )
        for label, folder in outputs:
            folder.mkdir(parents=True, exist_ok=True)
            if label == "coarse":
                from .models.sgf import GaussianOrientationField

                model.query = types.MethodType(GaussianOrientationField.query, model)
            started = time.perf_counter()
            prediction = model.predict(
                z_um=z, **({"chunk_size": 2048} if variant == "paired" else {})
            )
            predict_seconds = time.perf_counter() - started
            if variant == "paired":
                write_json(
                    folder / "manifest.json", dict(manifest, paired_output=label)
                )
            np.savez_compressed(folder / "prediction.npz", **prediction)
            write_json(folder / "model_info.json", getattr(model, "info_", {}))
            metrics, maps = score(
                prediction,
                reference,
                obs,
                symmetries,
                thresholds,
                Path(data_root) / "cache",
                fingerprint,
            )
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            metrics.update(
                fit_seconds=fit_seconds,
                predict_seconds=predict_seconds,
                peak_rss_bytes=int(
                    rss * (1 if platform.system() == "Darwin" else 1024)
                ),
                prediction_sha256=file_hash(folder / "prediction.npz"),
            )
            np.savez_compressed(folder / "evaluation_maps.npz", **maps)
            write_json(folder / "metrics.json", metrics)
            print(
                f"{label}: E90={metrics['angle_penalized_mean_deg']:.4f}, "
                f"F1={metrics['boundary_f1']:.4f}, ARI={metrics['grain_ari']:.4f}",
                flush=True,
            )
    if variant == "paired":
        write_json(complete_path, dict(signature=signature, shared_fit=True))
    return out
