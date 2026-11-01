import csv
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ..data import load_region
from ..orientation import misorientation_deg
from . import adapters
from .fetch import CHECKPOINT_SHA256, digest


CORE_METHODS = ("nearest", "sgf", "cgf", "tgs", "gsr")
BUDGETS = (1000, 5000, 10000)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def volume_digest(ref):
    hasher = hashlib.sha256()
    for key in ("q", "phase", "valid", "spacing_um"):
        value = np.ascontiguousarray(ref[key])
        hasher.update(f"{key}:{value.dtype}:{value.shape}".encode())
        hasher.update(memoryview(value).cast("B"))
    return hasher.hexdigest()


def region(dataset, split, data_root):
    ref, meta = load_region(
        dataset, "compact" if split == "native128" else split, data_root
    )
    if split == "validation":
        ref = {
            key: value[:64].copy() if key != "spacing_um" else value.copy()
            for key, value in ref.items()
        }
    elif split == "native128":
        if dataset != "in718":
            raise ValueError("The native-size comparison is defined for IN718 only")
        ranges = [[int(a - 32), int(b + 32)] for a, b in meta["crop_ranges_zyx"]]
        with np.load(Path(data_root) / "processed" / "in718.npz") as data:
            slices = tuple(slice(a, b) for a, b in ranges)
            ref = {key: data[key][slices].copy() for key in ("q", "phase", "valid")}
            ref["spacing_um"] = data["spacing_um"].copy()
        meta = {**meta, "crop_ranges_zyx": ranges}
    expected = (
        (64, 48, 48)
        if split == "validation"
        else ((128,) * 3 if split == "native128" else (64,) * 3)
    )
    if ref["q"].shape[:3] != expected:
        raise ValueError(f"Expected {expected}, received {ref['q'].shape[:3]}")
    return ref, meta


def masks(ref, observed):
    valid = ref["valid"] & (ref["phase"] == 1)
    acquired = np.zeros_like(valid)
    acquired[observed] = valid[observed]
    hidden = valid & ~acquired
    hidden[int(observed[-1]) + 1 :] = False
    return acquired, hidden


def output_valid(pred):
    if "rogsh" in pred:
        return np.isfinite(pred["rogsh"]).all(-1)
    return (
        np.isfinite(pred["q"]).all(-1)
        & (np.linalg.norm(pred["q"], axis=-1) > 0.5)
        & pred["valid"]
        & (pred["phase"] == 1)
    )


def descriptor(pred, mask):
    return (
        pred["rogsh"][mask]
        if "rogsh" in pred
        else adapters.quaternion_to_rogsh(pred["q"][mask])
    )


def score(pred, ref, observed, common=None):
    acquired, hidden = masks(ref, observed)
    valid = output_valid(pred)
    supported = hidden & valid if common is None else hidden & valid & common
    if not supported.any():
        raise ValueError("No valid hidden descriptor support")
    target = adapters.quaternion_to_rogsh(ref["q"][supported])
    error = float(np.abs(descriptor(pred, supported) - target).mean())
    observed_good = acquired & valid
    observed_error = None
    if observed_good.any():
        observed_error = float(
            np.abs(
                descriptor(pred, observed_good)
                - adapters.quaternion_to_rogsh(ref["q"][observed_good])
            ).mean()
        )
    angles = None
    if "q" in pred:
        values = np.full(int(hidden.sum()), 90.0)
        good = hidden & valid
        values[good[hidden]] = misorientation_deg(
            pred["q"][good], ref["q"][good], "cubic"
        )
        angles = dict(
            full_support_E90=float(values.mean()),
            angle_median_deg=float(np.median(values)),
            angle_p95_deg=float(np.quantile(values, 0.95)),
        )
    result = dict(
        hidden_rogsh_mae=error,
        hidden_voxels=int(hidden.sum()),
        descriptor_voxels=int(supported.sum()),
        prediction_coverage=float(valid[hidden].mean()),
        descriptor_coverage=float(supported.sum() / hidden.sum()),
        observed_descriptor_mae=observed_error,
        observed_coverage=float(valid[acquired].mean()),
        full_support_E90=None,
    )
    if angles:
        result.update(angles)
    if hidden.shape == (128, 128, 128):
        central = np.zeros_like(hidden)
        central[32:93, 32:96, 32:96] = True
        central &= supported
        result["central64_descriptor_mae"] = (
            float(
                np.abs(
                    descriptor(pred, central)
                    - adapters.quaternion_to_rogsh(ref["q"][central])
                ).mean()
            )
            if central.any()
            else None
        )
        result["central64_hidden_voxels"] = int(central.sum())
    return result


def seed_all(seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)
    elif device == "cuda":
        torch.cuda.manual_seed_all(seed)


def choose_steps(args):
    if args.steps is not None:
        return args.steps
    if args.method == "polymicros":
        return 100
    if args.method != "qrbsa":
        return 0
    if args.regime == "endpoint":
        return 1000
    if args.regime == "disjoint":
        return 5000
    selection = Path(args.output_root) / "qrbsa_budget.json"
    return (
        int(json.loads(selection.read_text())["selected_steps"])
        if selection.exists() and args.split != "validation"
        else 10000
    )


def train_qrbsa(args, ref, observed, out, steps):
    valid = ref["valid"] & (ref["phase"] == 1)
    train_obs = (
        np.unique(np.r_[observed, len(ref["q"]) - 1])
        if args.regime == "endpoint"
        else observed
    )
    training = {"kind": "observed_xy", "observed_indices": train_obs.tolist()}
    if args.regime == "disjoint":
        dense, meta = load_region(args.dataset, "validation", args.data_root)
        _, eval_meta = region(args.dataset, args.split, args.data_root)
        if not any(
            a >= d or b <= c
            for (a, b), (c, d) in zip(
                eval_meta["crop_ranges_zyx"], meta["crop_ranges_zyx"]
            )
        ):
            raise ValueError("Dense training and evaluation regions overlap")
        q = np.concatenate(
            (dense["q"].transpose(1, 0, 2, 3), dense["q"].transpose(2, 0, 1, 3))
        )
        mask = dense["valid"] & (dense["phase"] == 1)
        mask = np.concatenate((mask.transpose(1, 0, 2), mask.transpose(2, 0, 1)))
        training = {
            "kind": "disjoint_dense_xz_yz",
            "crop_ranges_zyx": meta["crop_ranges_zyx"],
        }
    else:
        q, mask = ref["q"][train_obs].copy(), valid[train_obs].copy()
    model = adapters.qrbsa_model(4, args.device, "cubic")
    torch.save(model.state_dict(), out / "initial_model.pt")
    checkpoints = {}

    def progress(step, loss):
        if args.split == "validation" and step in BUDGETS:
            model.eval()
            prediction = adapters.predict_qrbsa(
                model, ref["q"][observed].copy(), len(ref["q"]), symmetry="cubic"
            )
            fields = quaternion_fields(prediction)
            checkpoints[str(step)] = {
                **score(fields, ref, observed),
                "training_loss": loss,
            }
            np.savez_compressed(out / f"prediction_step_{step}.npz", **fields)
            torch.save(model.state_dict(), out / f"model_step_{step}.pt")
            write_json(out / "checkpoints.json", checkpoints)
            model.train()
        if step == 1 or step % 1000 == 0:
            print(f"Q-RBSA step {step}/{steps}: loss={loss:.6f}", flush=True)

    model, history = adapters.fit_qrbsa_observed(
        q,
        mask,
        "cubic",
        device=args.device,
        steps=steps,
        patch_size=32,
        batch_size=4,
        seed=args.seed,
        model=model,
        progress=progress,
        transpose_patches=args.regime != "disjoint",
    )
    torch.save(model.state_dict(), out / "model.pt")
    write_json(out / "training_history.json", history)
    if checkpoints:
        for step, item in checkpoints.items():
            item["training_loss_last100_mean"] = float(
                np.mean(history[max(0, int(step) - 100) : int(step)])
            )
        write_json(out / "checkpoints.json", checkpoints)
    prediction = adapters.predict_qrbsa(
        model, ref["q"][observed].copy(), len(ref["q"]), symmetry="cubic"
    )
    return quaternion_fields(prediction), {
        "training": training,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_steps": steps,
    }


def quaternion_fields(q):
    return dict(
        q=q,
        phase=np.ones(q.shape[:-1], np.int16),
        valid=np.isfinite(q).all(-1) & (np.linalg.norm(q, axis=-1) > 0.5),
    )


def run(args):
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; select --device cpu or cuda")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu or mps")
    if args.split == "validation" and (
        args.method != "qrbsa" or args.regime != "equal"
    ):
        raise ValueError("The development budget uses equal-input Q-RBSA only")
    if args.split == "native128" and (args.regime != "equal" or args.method == "qrbsa"):
        raise ValueError(
            "The native-size comparison uses equal-input core methods and PolyMicros"
        )
    adapters.BASE = Path(args.third_party).resolve()
    steps = choose_steps(args)
    if (args.method == "qrbsa" and steps <= 0) or (
        args.method == "polymicros" and steps < 2
    ):
        raise ValueError(
            "Training steps must be positive and sampling steps at least two"
        )
    out = (
        Path(args.output_root)
        / args.dataset
        / args.split
        / args.regime
        / f"{args.method}_s{args.seed}_steps{steps}"
    )
    out.mkdir(parents=True, exist_ok=True)
    ref, meta = region(args.dataset, args.split, args.data_root)
    observed = np.arange(0, len(ref["q"]), 4)
    conditioned = (
        np.unique(np.r_[observed, len(ref["q"]) - 1])
        if args.regime in ("endpoint", "disjoint") and args.method != "qrbsa"
        else observed
    )
    signature = dict(
        dataset=args.dataset,
        split=args.split,
        regime=args.regime,
        method=args.method,
        device=args.device,
        steps=steps,
        seed=args.seed,
        observed_z_indices=observed.tolist(),
        conditioned_z_indices=conditioned.tolist(),
        region=meta,
        reference_sha256=volume_digest(ref),
        source_sha256={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
    )
    if (out / "metrics.json").exists():
        previous = json.loads((out / "manifest.json").read_text())
        if previous["configuration"] != signature:
            raise RuntimeError(f"Configuration differs from completed run: {out}")
        print(f"Already complete: {out}", flush=True)
        return out
    write_json(out / "manifest.json", dict(configuration=signature))
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        seed_all(args.seed, args.device)
        if args.method == "qrbsa":
            pred, details = train_qrbsa(args, ref, observed, out, steps)
        elif args.method == "polymicros":
            checkpoint = adapters.BASE / "checkpoints/PolyMicros/PolyMicros.ckpt"
            if digest(checkpoint) != CHECKPOINT_SHA256:
                raise RuntimeError("PolyMicros checkpoint checksum mismatch")
            acquired, _ = masks(ref, conditioned)
            values = np.zeros(ref["valid"].shape + (3,), np.float32)
            values[acquired] = adapters.quaternion_to_rogsh(ref["q"][acquired])
            model = adapters.load_polymicros(len(ref["q"]), args.device)
            if args.split == "native128":
                from .memory_attention import install as install_attention
                from .tiled_convolution import install as install_convolution

                install_attention(model)
                install_convolution(model)
                if args.device == "mps":
                    torch.mps.set_per_process_memory_fraction(1.0)
            pred = dict(
                rogsh=adapters.sample_polymicros(
                    model,
                    values,
                    acquired,
                    steps=steps,
                    seed=args.seed,
                    schedule="paper75",
                )
            )
            details = dict(
                schedule="paper75",
                checkpoint_sha256=CHECKPOINT_SHA256,
                training_source_overlap=args.dataset == "in718",
                parameters=sum(p.numel() for p in model.parameters()),
            )
        else:
            from ..models import build_model, model_config

            config = model_config(
                args.method,
                overrides={
                    "device": args.device,
                    "threads": 2,
                    "phase_symmetry": {1: "cubic"},
                },
            )
            model = build_model(args.method)
            depths = np.arange(len(ref["q"])) * ref["spacing_um"][0]
            model.fit(
                ref["q"][conditioned].copy(),
                ref["phase"][conditioned].copy(),
                ref["valid"][conditioned].copy(),
                depths[conditioned].copy(),
                ref["spacing_um"][1:].copy(),
                config=config,
                seed=args.seed,
            )
            pred = model.predict(z_um=depths)
            details = dict(config=config, model_info=getattr(model, "info_", {}))
        np.savez_compressed(out / "prediction.npz", **pred)
        metrics = score(pred, ref, observed)
    metrics.update(
        elapsed_seconds=time.perf_counter() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        prediction_sha256=digest(out / "prediction.npz"),
    )
    write_json(out / "manifest.json", dict(configuration=signature, details=details))
    write_json(out / "metrics.json", metrics)
    print(
        f"Complete: {out}; descriptor MAE={metrics['hidden_rogsh_mae']:.6f}", flush=True
    )
    return out


def develop(args):
    from copy import copy

    records = []
    for dataset in ("in100", "in718"):
        case = copy(args)
        case.dataset, case.method, case.split, case.regime, case.seed, case.steps = (
            dataset,
            "qrbsa",
            "validation",
            "equal",
            0,
            10000,
        )
        out = run(case)
        records.append(json.loads((out / "checkpoints.json").read_text()))
    scores = {
        step: float(np.mean([r[str(step)]["full_support_E90"] for r in records]))
        for step in BUDGETS
    }
    selected = min(scores, key=scores.get)
    write_json(
        Path(args.output_root) / "qrbsa_budget.json",
        dict(mean_development_E90=scores, selected_steps=selected),
    )
    print(f"Selected Q-RBSA budget: {selected}", flush=True)


def summary(args):
    adapters.BASE = Path(args.third_party).resolve()
    root = Path(args.output_root) / args.dataset / args.split / args.regime
    paths = sorted(root.glob("*/prediction.npz"))
    paths = [p for p in paths if (p.parent / "metrics.json").is_file()]
    if not paths:
        raise FileNotFoundError(f"No completed predictions under {root}")
    ref, meta = region(args.dataset, args.split, args.data_root)
    reference_sha256 = volume_digest(ref)
    observed = np.arange(0, len(ref["q"]), 4)
    _, common = masks(ref, observed)
    for path in paths:
        cfg = json.loads((path.parent / "manifest.json").read_text())["configuration"]
        if (
            cfg["observed_z_indices"] != observed.tolist()
            or cfg["region"] != meta
            or cfg["reference_sha256"] != reference_sha256
        ):
            raise ValueError(f"Inconsistent observations or reference: {path}")
        with np.load(path) as pred:
            common &= output_valid(pred)
    if not common.any():
        raise ValueError("No shared valid descriptor support")
    rows = []
    for path in paths:
        with np.load(path) as pred:
            row = score(pred, ref, observed, common)
        rows.append(dict(run=path.parent.name, **row))
    _, hidden = masks(ref, observed)
    result = dict(
        dataset=args.dataset,
        split=args.split,
        regime=args.regime,
        hidden_voxels=int(hidden.sum()),
        common_descriptor_voxels=int(common.sum()),
        common_coverage=float(common.sum() / hidden.sum()),
        common_mask_sha256=hashlib.sha256(common.tobytes()).hexdigest(),
        records=rows,
    )
    write_json(root / "summary.json", result)
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {root / 'summary.csv'} ({len(rows)} runs)", flush=True)
