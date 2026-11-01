import itertools
import json
import os
from pathlib import Path

import numpy as np
from scipy import ndimage, sparse

from .analysis import _json_value, _write_csv, edge_angles, from_edges
from .data import load_region
from .io import file_hash, volume_hash
from .metrics import angle_map, grain_metrics


def incidence(reference, prediction, mask):
    reference_ids, reference_inverse = np.unique(reference[mask], return_inverse=True)
    prediction_ids, prediction_inverse = np.unique(
        prediction[mask], return_inverse=True
    )
    if not len(reference_inverse):
        raise ValueError("The comparison domain has no valid voxels")
    matrix = sparse.coo_matrix(
        (
            np.ones(len(reference_inverse), np.int64),
            (reference_inverse, prediction_inverse),
        ),
        shape=(len(reference_ids), len(prediction_ids)),
    ).tocsr()
    matrix.sum_duplicates()
    overlap = matrix.tocoo()
    reference_size = np.asarray(matrix.sum(1)).ravel()
    prediction_size = np.asarray(matrix.sum(0)).ravel()
    reference_overlap = (overlap.data >= 8) & (
        overlap.data / reference_size[overlap.row] >= 0.05
    )
    prediction_overlap = (overlap.data >= 8) & (
        overlap.data / prediction_size[overlap.col] >= 0.05
    )
    splits = np.bincount(overlap.row[reference_overlap], minlength=len(reference_ids))
    merges = np.bincount(overlap.col[prediction_overlap], minlength=len(prediction_ids))
    reference_qualified = (reference_ids > 0) & (reference_size >= 8)
    prediction_qualified = (prediction_ids > 0) & (prediction_size >= 8)
    reference_pairs = float(
        np.sum(reference_size.astype(float) * (reference_size - 1) / 2)
    )
    prediction_pairs = float(
        np.sum(prediction_size.astype(float) * (prediction_size - 1) / 2)
    )
    correct_pairs = float(np.sum(overlap.data.astype(float) * (overlap.data - 1) / 2))
    probability = overlap.data / overlap.data.sum()
    return dict(
        voxels=int(mask.sum()),
        reference_qualified=int(reference_qualified.sum()),
        predicted_qualified=int(prediction_qualified.sum()),
        reference_split_count=int(np.sum(reference_qualified & (splits > 1))),
        predicted_merge_count=int(np.sum(prediction_qualified & (merges > 1))),
        reference_split_fraction=float(np.mean(splits[reference_qualified] > 1))
        if reference_qualified.any()
        else None,
        predicted_merge_fraction=float(np.mean(merges[prediction_qualified] > 1))
        if prediction_qualified.any()
        else None,
        false_join_pair_fraction=1 - correct_pairs / prediction_pairs
        if prediction_pairs
        else None,
        false_separation_pair_fraction=1 - correct_pairs / reference_pairs
        if reference_pairs
        else None,
        H_reference_given_prediction_bits=float(
            -np.sum(probability * np.log2(overlap.data / prediction_size[overlap.col]))
        ),
        H_prediction_given_reference_bits=float(
            -np.sum(probability * np.log2(overlap.data / reference_size[overlap.row]))
        ),
        same_reference_pairs=reference_pairs,
        same_prediction_pairs=prediction_pairs,
        true_positive_same_pairs=correct_pairs,
    )


def edge_map(reference, prediction, valid, hidden, kind):
    result = np.zeros(reference.shape, bool)
    for axis in (1, 2):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        eligible = valid[a] & valid[b] & hidden[a] & hidden[b]
        if kind == "join":
            bad = (
                eligible
                & (reference[a] != reference[b])
                & (prediction[a] == prediction[b])
            )
        elif kind == "split":
            bad = (
                eligible
                & (reference[a] == reference[b])
                & (prediction[a] != prediction[b])
            )
        else:
            raise ValueError(f"Unknown connection error: {kind}")
        result[a] |= bad
        result[b] |= bad
    return result


def select_window(bad, observed_z, width=36):
    counts = bad.sum((1, 2))
    counts[observed_z] = 0
    z = int(np.argmax(counts))
    if not counts[z]:
        return None
    width = min(width, *bad.shape[1:])
    width -= width % 2
    if width < 2:
        raise ValueError(
            "A diagnostic window needs at least two voxels on each in-plane axis"
        )
    density = ndimage.uniform_filter(bad[z].astype(float), size=width, mode="constant")
    margin = width // 2
    allowed = np.zeros(bad.shape[1:], bool)
    allowed[
        margin : max(margin + 1, bad.shape[1] - margin),
        margin : max(margin + 1, bad.shape[2] - margin),
    ] = True
    density[~allowed] = -1
    y, x = np.unravel_index(np.argmax(density), density.shape)
    y0, x0 = int(y - margin), int(x - margin)
    below, above = observed_z[observed_z < z], observed_z[observed_z > z]
    if not len(below) or not len(above):
        raise ValueError(
            "The selected hidden section must be bracketed by acquired sections"
        )
    return dict(
        z=z,
        yx_center=[int(y), int(x)],
        window_yx=[y0, y0 + width, x0, x0 + width],
        diagnostic_voxels_in_plane=int(counts[z]),
        diagnostic_voxels_in_window=int(bad[z, y0 : y0 + width, x0 : x0 + width].sum()),
        input_z=[int(below.max()), int(above.min())],
    )


def _load_runs(run_dirs, data_root=None, same_mask=True):
    run_dirs = [Path(path) for path in run_dirs]
    if len(run_dirs) < 2:
        raise ValueError("Provide at least two prediction directories")
    records = []
    reference_hash = None
    for path in run_dirs:
        manifest = json.loads((path / "manifest.json").read_text())
        root = data_root if data_root is not None else manifest.get("data_root", "data")
        current, metadata = load_region(
            manifest["dataset"], manifest["split"], data_root=root
        )
        fingerprint = volume_hash(current)
        if (
            manifest.get("reference_sha256")
            and manifest["reference_sha256"] != fingerprint
        ):
            raise ValueError(f"{path}: reference volume differs from the fitted run")
        if records:
            first = records[0]["manifest"]
            if (
                any(manifest[key] != first[key] for key in ("dataset", "split"))
                or fingerprint != reference_hash
            ):
                raise ValueError(
                    "All predictions must use the same dataset, split and reference arrays"
                )
        else:
            reference, reference_hash = current, fingerprint
            symmetries = manifest["config"].get(
                "phase_symmetry", metadata["phase_symmetry"]
            )
            symmetries = {int(key): value for key, value in symmetries.items()}
        current_symmetries = manifest["config"].get(
            "phase_symmetry", metadata["phase_symmetry"]
        )
        if {int(key): value for key, value in current_symmetries.items()} != symmetries:
            raise ValueError("Crystal symmetry differs between compared runs")
        observed = np.asarray(manifest["observed_z_indices"], dtype=int)
        if (
            records
            and same_mask
            and not np.array_equal(observed, records[0]["observed"])
        ):
            raise ValueError(
                "Connection comparisons require identical acquired sections"
            )
        with np.load(path / "prediction.npz") as archive:
            prediction = {key: archive[key].copy() for key in ("q", "phase", "valid")}
        if prediction["phase"].shape != reference["phase"].shape:
            raise ValueError(f"{path}: prediction and reference shapes differ")
        records.append(
            dict(
                path=path,
                manifest=manifest,
                prediction=prediction,
                observed=observed,
                prediction_sha256=file_hash(path / "prediction.npz"),
            )
        )
    return reference, symmetries, records, reference_hash


def _write_json(path, result):
    Path(path).write_text(
        json.dumps(_json_value(result), indent=2, allow_nan=False) + "\n"
    )


def _sources(records, output):
    return [
        dict(
            index=index,
            run=os.path.relpath(record["path"], output),
            method=record["manifest"].get("method"),
            variant=record["manifest"].get("variant", record["manifest"].get("tag")),
            pattern=record["manifest"].get("pattern"),
            prediction_sha256=record["prediction_sha256"],
            observed_z_indices=record["observed"].tolist(),
        )
        for index, record in enumerate(records)
    ]


def _save_case(
    output, name, selected, bad, reference, records, labels, reference_labels
):
    if selected is None:
        return None
    lo, hi = selected["input_z"]
    y0, y1, x0, x1 = selected["window_yx"]
    region = (slice(lo, hi + 1), slice(y0, y1), slice(x0, x1))
    arrays = {
        f"reference_{key}": reference[key][region] for key in ("q", "phase", "valid")
    }
    arrays.update(
        spacing_um=reference["spacing_um"],
        global_origin_zyx=np.array([lo, y0, x0]),
        target_z_global=np.array(selected["z"]),
        observed_z_global=np.array(selected["input_z"]),
        diagnostic_mask=bad[region],
        reference_labels=reference_labels[region],
    )
    for index, record in enumerate(records):
        arrays.update(
            {
                f"prediction_{index}_{key}": record["prediction"][key][region]
                for key in ("q", "phase", "valid")
            }
        )
        arrays[f"prediction_{index}_labels"] = labels[index][region]
    path = output / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    return path.name


def compare_saved(run_dirs, output, thresholds=(2, 5), data_root=None):
    reference, symmetries, records, reference_hash = _load_runs(
        run_dirs, data_root, same_mask=True
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    thresholds = sorted(set(float(value) for value in thresholds) | {5.0})
    if any(not 0 < value < 90 for value in thresholds):
        raise ValueError("Grain thresholds must be between 0 and 90 degrees")
    observed = records[0]["observed"]
    domain = reference["valid"]
    hidden = domain.copy()
    hidden[observed] = False
    if not hidden.any():
        raise ValueError("No valid hidden voxels to compare")
    anchor = domain & ~hidden
    reference_edges = edge_angles(
        reference["q"], reference["phase"], domain, symmetries
    )
    for record in records:
        pred = record["prediction"]
        for key in pred:
            pred[key][anchor] = reference[key][anchor]
        phase = np.where(pred["valid"], pred["phase"], 0)
        record["phase"] = phase
        record["edges"] = edge_angles(pred["q"], phase, domain, symmetries)
        angles = angle_map(
            pred["q"], reference["q"], phase, reference["phase"], domain, symmetries
        )
        record["E90_deg"] = float(angles[hidden].mean())
    rows, cases = [], []
    for threshold in thresholds:
        ref_labels = from_edges(reference_edges, domain, threshold)
        labels = [from_edges(record["edges"], domain, threshold) for record in records]
        for index, (record, prediction_labels) in enumerate(zip(records, labels)):
            hidden_ari = grain_metrics(prediction_labels, ref_labels, hidden)[
                "grain_ari"
            ]
            for scope, mask in (("full_roi", domain), ("hidden", hidden)):
                rows.append(
                    dict(
                        run_index=index,
                        method=record["manifest"].get("method"),
                        threshold_deg=threshold,
                        scope=scope,
                        hidden_E90_deg=record["E90_deg"],
                        hidden_ARI=hidden_ari,
                        hidden_phase_error_fraction=float(
                            np.mean(
                                record["phase"][hidden] != reference["phase"][hidden]
                            )
                        ),
                        **incidence(ref_labels, prediction_labels, mask),
                    )
                )
        base_maps = {
            kind: edge_map(ref_labels, labels[0], domain, hidden, kind)
            for kind in ("join", "split")
        }
        for index in range(1, len(records)):
            excess = {
                kind: edge_map(ref_labels, labels[index], domain, hidden, kind)
                & ~base_maps[kind]
                for kind in ("join", "split")
            }
            kind = "split" if excess["split"].sum() >= excess["join"].sum() else "join"
            selected = select_window(excess[kind], observed)
            name = f"connection_{index}_vs_0_{threshold:g}deg"
            filename = _save_case(
                output,
                name,
                selected,
                excess[kind],
                reference,
                records,
                labels,
                ref_labels,
            )
            cases.append(
                dict(
                    kind="connection",
                    run_index=index,
                    baseline_index=0,
                    threshold_deg=threshold,
                    mechanism=kind,
                    excess_split_edge_voxels=int(excess["split"].sum()),
                    excess_join_edge_voxels=int(excess["join"].sum()),
                    selected=selected,
                    arrays=filename,
                )
            )
        if threshold == 5:
            labels_5, reference_labels_5 = labels, ref_labels
    baseline_correct = records[0]["phase"] == reference["phase"]
    for index in range(1, len(records)):
        correct = records[index]["phase"] == reference["phase"]
        introduced = hidden & baseline_correct & ~correct
        repaired = hidden & correct & ~baseline_correct
        for phase_id in symmetries:
            lost = introduced & (reference["phase"] == phase_id)
            selected = select_window(lost, observed)
            name = f"phase_loss_{index}_vs_0_phase{phase_id}"
            filename = _save_case(
                output,
                name,
                selected,
                lost,
                reference,
                records,
                labels_5,
                reference_labels_5,
            )
            cases.append(
                dict(
                    kind="phase_loss",
                    run_index=index,
                    baseline_index=0,
                    phase=phase_id,
                    introduced_phase_error_voxels=int(introduced.sum()),
                    repaired_phase_error_voxels=int(repaired.sum()),
                    phase_loss_voxels=int(lost.sum()),
                    hidden_valid_voxels=int(hidden.sum()),
                    selected=selected,
                    arrays=filename,
                )
            )
    result = dict(
        dataset=records[0]["manifest"]["dataset"],
        split=records[0]["manifest"]["split"],
        reference_sha256=reference_hash,
        sources=_sources(records, output),
        protocol=dict(
            anchoring="replace only valid acquired voxels; hidden predictions unchanged",
            partition="same-phase six-neighbor full-volume partitions",
            incidence="components >=8 voxels; substantial overlap >=8 voxels and >=5% of its component",
            pair_errors="all valid voxel pairs, including small components",
            baseline="first supplied run; compare each later run against it",
            window="largest excess XY error footprint; most affected hidden slice; densest centered 36x36 window; smallest-index ties",
            case_scope="diagnostic extreme; frequency is reported separately over the full ROI and hidden domain",
            arrays="same physical crop for acquired reference sections, target reference, and every prediction",
        ),
        incidence=rows,
        cases=cases,
    )
    _write_csv(output / "connection_incidence.csv", rows)
    _write_json(output / "connection_comparison.json", result)
    return _json_value(result)


def compare_equal_budget(run_dirs, output, data_root=None):
    reference, symmetries, records, reference_hash = _load_runs(
        run_dirs, data_root, same_mask=False
    )
    patterns = [record["manifest"].get("pattern") for record in records]
    if set(patterns) != {"block", "dispersed"}:
        raise ValueError("Provide block and dispersed sampling predictions")
    if len({len(record["observed"]) for record in records}) != 1:
        raise ValueError(
            "Every compared run must have the same number of acquired sections"
        )
    first = records[0]["manifest"]
    for record in records[1:]:
        manifest = record["manifest"]
        if (
            manifest.get("method") != first.get("method")
            or manifest["config"] != first["config"]
        ):
            raise ValueError(
                "Equal-budget comparisons must keep the method and fitted configuration fixed"
            )
        if manifest.get("fit_seed", manifest.get("seed", 0)) != first.get(
            "fit_seed", first.get("seed", 0)
        ):
            raise ValueError(
                "Equal-budget comparisons must keep the fitting seed fixed"
            )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    acquired_union = np.unique(
        np.concatenate([record["observed"] for record in records])
    )
    hidden_z = np.ones(len(reference["q"]), bool)
    hidden_z[acquired_union] = False
    common = reference["valid"] & hidden_z[:, None, None]
    if not common.any():
        raise ValueError(
            "No reference-valid voxels remain hidden in every sampling pattern"
        )
    arrays = dict(common_hidden=common, spacing_um=reference["spacing_um"])
    rows = []
    for index, record in enumerate(records):
        pred = record["prediction"]
        phase = np.where(pred["valid"], pred["phase"], 0)
        angles = angle_map(
            pred["q"],
            reference["q"],
            phase,
            reference["phase"],
            reference["valid"],
            symmetries,
        )
        values = angles[common]
        arrays[f"angles_{index}_deg"] = values
        rows.append(
            dict(
                run_index=index,
                pattern=record["manifest"]["pattern"],
                observed_planes=len(record["observed"]),
                n_common_hidden=int(common.sum()),
                common_hidden_E90_deg=float(values.mean()),
                common_hidden_phase_accuracy=float(
                    np.mean(phase[common] == reference["phase"][common])
                ),
            )
        )
    paired = []
    for block, dispersed in itertools.product(
        [row for row in rows if row["pattern"] == "block"],
        [row for row in rows if row["pattern"] == "dispersed"],
    ):
        paired.append(
            dict(
                block_index=block["run_index"],
                dispersed_index=dispersed["run_index"],
                block_E90_deg=block["common_hidden_E90_deg"],
                dispersed_E90_deg=dispersed["common_hidden_E90_deg"],
                delta_block_minus_dispersed_deg=block["common_hidden_E90_deg"]
                - dispersed["common_hidden_E90_deg"],
            )
        )
    result = dict(
        dataset=first["dataset"],
        split=first["split"],
        method=first.get("method"),
        reference_sha256=reference_hash,
        sources=_sources(records, output),
        common_hidden_z_indices=np.flatnonzero(hidden_z).tolist(),
        protocol="Equal acquired-section counts; fixed model, configuration and fit seed; evaluate reference-valid voxels hidden under every supplied pattern.",
        rows=rows,
        paired=paired,
    )
    np.savez_compressed(output / "common_hidden_errors.npz", **arrays)
    _write_csv(output / "equal_budget.csv", rows)
    _write_json(output / "equal_budget.json", result)
    return _json_value(result)
