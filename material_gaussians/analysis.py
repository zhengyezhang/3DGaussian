import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import wasserstein_distance

from .data import load_region
from .io import volume_hash
from .metrics import angle_map, boundaries, boundary_metrics, evaluate, grain_metrics

CONNECTIVITY = ndimage.generate_binary_structure(3, 1)


def edge_angles(q, phase, domain, syms):
    ans = []
    for axis in range(3):
        a = [slice(None)] * 3
        b = a.copy()
        a[axis] = slice(None, -1)
        b[axis] = slice(1, None)
        a, b = (tuple(a), tuple(b))
        eligible = domain[a] & domain[b]
        same = eligible & (phase[a] == phase[b])
        ans.append(angle_map(q[a], q[b], phase[a], phase[b], same, syms))
    return ans


def from_edges(edges, domain, t):
    indices = np.arange(domain.size, dtype=np.int32).reshape(domain.shape)
    rows = []
    cols = []
    for axis, dist in enumerate(edges):
        a = [slice(None)] * 3
        b = a.copy()
        a[axis] = slice(None, -1)
        b[axis] = slice(1, None)
        a, b = (tuple(a), tuple(b))
        good = (dist <= t) & domain[a] & domain[b]
        rows.append(indices[a][good])
        cols.append(indices[b][good])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    graph = sparse.coo_matrix(
        (np.ones(len(rows), np.uint8), (rows, cols)), shape=(domain.size, domain.size)
    ).tocsr()
    _, lab = connected_components(graph, directed=False)
    lab = lab.reshape(domain.shape).astype(np.int32) + 1
    lab[~domain] = 0
    return lab


def catalog(labels, phase, domain, spacing):
    effective = domain & (phase > 0) & (labels > 0)
    flat = np.flatnonzero(effective)
    ids, first, inv = np.unique(
        labels.ravel()[flat], return_index=True, return_inverse=True
    )
    n = np.bincount(inv).astype(np.int64)
    ph = phase.ravel()[flat[first]].astype(np.int16)
    assert np.all(phase.ravel()[flat] == ph[inv])
    coords = np.array(np.unravel_index(flat, labels.shape)).T
    physical = (coords + 0.5) * spacing
    means = np.stack(
        [
            np.bincount(inv, weights=physical[:, i], minlength=len(ids)) / n
            for i in range(3)
        ],
        axis=1,
    )
    cov = np.empty((len(ids), 3, 3))
    for i in range(3):
        for j in range(i, 3):
            v = (
                np.bincount(
                    inv, weights=physical[:, i] * physical[:, j], minlength=len(ids)
                )
                / n
                - means[:, i] * means[:, j]
            )
            if i == j:
                v += spacing[i] ** 2 / 12
            cov[:, i, j] = v
            cov[:, j, i] = v
    eigen = np.linalg.eigvalsh(cov)[:, ::-1]
    assert np.all(eigen > 0)
    roi_touch = np.any((coords == 0) | (coords == np.array(labels.shape) - 1), axis=1)
    near_invalid = ndimage.binary_dilation(
        ~effective, structure=ndimage.generate_binary_structure(3, 1), border_value=0
    ).ravel()[flat]
    br = np.bincount(inv, weights=roi_touch, minlength=len(ids)) > 0
    bi = np.bincount(inv, weights=near_invalid, minlength=len(ids)) > 0
    v = n * np.prod(spacing)
    qual = n >= 8
    complete = qual & ~br & ~bi
    return dict(
        grain_id=ids,
        phase=ph,
        voxels=n,
        volume_um3=v,
        deq_um=(6 * v / np.pi) ** (1 / 3),
        centroid_z_um=means[:, 0],
        centroid_y_um=means[:, 1],
        centroid_x_um=means[:, 2],
        lambda_major_um2=eigen[:, 0],
        lambda_middle_um2=eigen[:, 1],
        lambda_minor_um2=eigen[:, 2],
        axis_major_minor=np.sqrt(eigen[:, 0] / eigen[:, 2]),
        axis_middle_major=np.sqrt(eigen[:, 1] / eigen[:, 0]),
        axis_minor_major=np.sqrt(eigen[:, 2] / eigen[:, 0]),
        touches_roi_boundary=br,
        touches_invalid_domain=bi,
        size_qualified=qual,
        complete_eligible=complete,
    )


def counts(cat, phase=None):
    mask = (
        np.ones(len(cat["grain_id"]), bool) if phase is None else cat["phase"] == phase
    )
    q = mask & cat["size_qualified"]
    complete = mask & cat["complete_eligible"]
    return dict(
        all_positive_grains=int(mask.sum()),
        excluded_below8=int((mask & ~cat["size_qualified"]).sum()),
        qualified_grains=int(q.sum()),
        qualified_roi_boundary=int((q & cat["touches_roi_boundary"]).sum()),
        qualified_invalid_boundary=int((q & cat["touches_invalid_domain"]).sum()),
        excluded_any_boundary=int(
            (q & (cat["touches_roi_boundary"] | cat["touches_invalid_domain"])).sum()
        ),
        complete_eligible=int(complete.sum()),
        positive_voxels=int(cat["voxels"][mask].sum()),
        qualified_voxels=int(cat["voxels"][q].sum()),
        complete_voxels=int(cat["voxels"][complete].sum()),
    )


def overlap_graph(pl, rl, pc, rc, domain):
    valid = domain & (pl > 0) & (rl > 0)
    pv = pl[valid]
    rv = rl[valid]
    pi = np.searchsorted(pc["grain_id"], pv)
    ri = np.searchsorted(rc["grain_id"], rv)
    good = (pi < len(pc["grain_id"])) & (ri < len(rc["grain_id"]))
    pi = pi[good]
    ri = ri[good]
    pv = pv[good]
    rv = rv[good]
    good = (pc["grain_id"][pi] == pv) & (rc["grain_id"][ri] == rv)
    co = sparse.coo_matrix(
        (np.ones(int(good.sum()), np.int32), (ri[good], pi[good])),
        shape=(len(rc["grain_id"]), len(pc["grain_id"])),
    ).tocsr()
    co.sum_duplicates()
    return co.tocoo()


def matched(pl, rl, pc, rc, domain):
    co = overlap_graph(pl, rl, pc, rc, domain)
    iou = co.data / (rc["voxels"][co.row] + pc["voxels"][co.col] - co.data)
    candidate = (
        rc["size_qualified"][co.row]
        & pc["size_qualified"][co.col]
        & (rc["phase"][co.row] == pc["phase"][co.col])
        & (iou >= 0.1)
    )
    ix = np.flatnonzero(candidate)
    ix = ix[
        np.lexsort((pc["grain_id"][co.col[ix]], rc["grain_id"][co.row[ix]], -iou[ix]))
    ]
    assign = np.full(len(rc["grain_id"]), -1, int)
    miou = np.zeros(len(rc["grain_id"]))
    used = np.zeros(len(pc["grain_id"]), bool)
    for k in ix:
        a, b = (co.row[k], co.col[k])
        if assign[a] < 0 and (not used[b]):
            assign[a] = b
            miou[a] = iou[k]
            used[b] = True
    sig = (
        rc["size_qualified"][co.row]
        & pc["size_qualified"][co.col]
        & (rc["phase"][co.row] == pc["phase"][co.col])
    )
    split = np.bincount(
        co.row[sig & (co.data / rc["voxels"][co.row] >= 0.05)],
        minlength=len(rc["grain_id"]),
    )
    merge = np.bincount(
        co.col[sig & (co.data / pc["voxels"][co.col] >= 0.05)],
        minlength=len(pc["grain_id"]),
    )
    return (assign, miou, used, split, merge)


def describe(x):
    if not len(x):
        return dict(n=0, median=None, q25=None, q75=None, mean=None)
    return dict(
        n=len(x),
        median=float(np.median(x)),
        q25=float(np.quantile(x, 0.25)),
        q75=float(np.quantile(x, 0.75)),
        mean=float(np.mean(x)),
    )


def comparison(pc, rc, assign, miou, used, split, merge, phase=None):
    rq = rc["size_qualified"].copy()
    pq = pc["size_qualified"].copy()
    re = rc["complete_eligible"].copy()
    pe = pc["complete_eligible"].copy()
    if phase is not None:
        rq &= rc["phase"] == phase
        re &= rc["phase"] == phase
        pq &= pc["phase"] == phase
        pe &= pc["phase"] == phase
    matched_ref = rq & (assign >= 0)
    complete_ref_matched = re & (assign >= 0)
    complete_pairs = complete_ref_matched.copy()
    complete_pairs[complete_ref_matched] &= pc["complete_eligible"][
        assign[complete_ref_matched]
    ]
    n = int(re.sum())
    ref_vol = rc["volume_um3"][re]
    pred_vol = np.zeros(n)
    paired = assign[re] >= 0
    pred_vol[paired] = pc["volume_um3"][assign[re][paired]]
    out = dict(
        phase="all" if phase is None else int(phase),
        reference_counts=counts(rc, phase),
        prediction_counts=counts(pc, phase),
        matched_qualified=int(matched_ref.sum()),
        unmatched_reference_qualified=int((rq & (assign < 0)).sum()),
        unmatched_prediction_qualified=int((pq & ~used).sum()),
        matched_complete_references=int(complete_ref_matched.sum()),
        unmatched_complete_references=int((re & (assign < 0)).sum()),
        unmatched_complete_predictions=int((pe & ~used).sum()),
        complete_pair_n=int(complete_pairs.sum()),
        qualified_recall_iou50=float(np.mean(miou[rq] >= 0.5)) if rq.any() else None,
        complete_recall_iou50=float(np.mean(miou[re] >= 0.5)) if n else None,
        qualified_split_fraction=float(np.mean(split[rq] > 1)) if rq.any() else None,
        qualified_merge_fraction=float(np.mean(merge[pq] > 1)) if pq.any() else None,
        complete_split_fraction=float(np.mean(split[re] > 1)) if n else None,
        complete_merge_fraction=float(np.mean(merge[pe] > 1)) if pe.any() else None,
        complete_ref_volume_mae_unmatched_zero=float(
            np.mean(abs(pred_vol - ref_vol) / ref_vol)
        )
        if n
        else None,
    )
    pop = {}
    for key, transform in [
        ("volume_um3", np.log),
        ("deq_um", lambda x: x),
        ("axis_major_minor", lambda x: x),
        ("axis_middle_major", lambda x: x),
        ("axis_minor_major", lambda x: x),
    ]:
        r = rc[key][re]
        p = pc[key][pe]
        pop[key] = dict(
            reference=describe(r),
            prediction=describe(p),
            wasserstein=float(wasserstein_distance(transform(r), transform(p)))
            if len(r) and len(p)
            else None,
            wasserstein_scale="natural_log_um3" if key == "volume_um3" else "original",
            median_relative_bias=float(np.median(p) / np.median(r) - 1)
            if len(r) and len(p)
            else None,
        )
    out["population_complete"] = pop
    errors = {}
    for key in [
        "volume_um3",
        "deq_um",
        "axis_major_minor",
        "axis_middle_major",
        "axis_minor_major",
    ]:
        mask = (
            complete_ref_matched if key in ("volume_um3", "deq_um") else complete_pairs
        )
        r = rc[key][mask]
        p = pc[key][assign[mask]]
        errors[key] = dict(
            n=len(r),
            mean_absolute_relative_error=float(np.mean(abs(p - r) / r))
            if len(r)
            else None,
            median_absolute_relative_error=float(np.median(abs(p - r) / r))
            if len(r)
            else None,
            mean_signed_relative_error=float(np.mean((p - r) / r)) if len(r) else None,
        )
    out["paired_errors"] = errors
    qualified_pop = {}
    qualified_errors = {}
    for key in ("volume_um3", "deq_um"):
        r = rc[key][rq]
        p = pc[key][pq]
        transform = np.log if key == "volume_um3" else lambda x: x
        qualified_pop[key] = dict(
            reference=describe(r),
            prediction=describe(p),
            wasserstein=float(wasserstein_distance(transform(r), transform(p)))
            if len(r) and len(p)
            else None,
            wasserstein_scale="natural_log_um3" if key == "volume_um3" else "original",
            median_relative_bias=float(np.median(p) / np.median(r) - 1)
            if len(r) and len(p)
            else None,
        )
        r = rc[key][matched_ref]
        p = pc[key][assign[matched_ref]]
        qualified_errors[key] = dict(
            n=len(r),
            mean_absolute_relative_error=float(np.mean(abs(p - r) / r))
            if len(r)
            else None,
            median_absolute_relative_error=float(np.median(abs(p - r) / r))
            if len(r)
            else None,
            mean_signed_relative_error=float(np.mean((p - r) / r)) if len(r) else None,
        )
    qrv = rc["volume_um3"][rq]
    qpv = np.zeros(len(qrv))
    qmatch = assign[rq] >= 0
    qpv[qmatch] = pc["volume_um3"][assign[rq][qmatch]]
    out["qualified_ref_volume_mae_unmatched_zero"] = (
        float(np.mean(abs(qpv - qrv) / qrv)) if len(qrv) else None
    )
    out["population_qualified"] = qualified_pop
    out["paired_qualified_errors"] = qualified_errors
    return out


def pair_rows(rc, pc, a, iou, used):
    for i in np.flatnonzero(rc["size_qualified"]):
        j = a[i]
        yield dict(
            status="matched" if j >= 0 else "unmatched_reference",
            reference_id=int(rc["grain_id"][i]),
            prediction_id=int(pc["grain_id"][j]) if j >= 0 else None,
            reference_phase=int(rc["phase"][i]),
            prediction_phase=int(pc["phase"][j]) if j >= 0 else None,
            iou=float(iou[i]),
            reference_complete=bool(rc["complete_eligible"][i]),
            prediction_complete=bool(pc["complete_eligible"][j]) if j >= 0 else None,
            reference_volume_um3=float(rc["volume_um3"][i]),
            prediction_volume_um3=float(pc["volume_um3"][j]) if j >= 0 else None,
            reference_deq_um=float(rc["deq_um"][i]),
            prediction_deq_um=float(pc["deq_um"][j]) if j >= 0 else None,
            reference_axis_major_minor=float(rc["axis_major_minor"][i]),
            prediction_axis_major_minor=float(pc["axis_major_minor"][j])
            if j >= 0
            else None,
        )
    for j in np.flatnonzero(pc["size_qualified"] & ~used):
        yield dict(
            status="unmatched_prediction",
            reference_id=None,
            prediction_id=int(pc["grain_id"][j]),
            reference_phase=None,
            prediction_phase=int(pc["phase"][j]),
            iou=0.0,
            reference_complete=None,
            prediction_complete=bool(pc["complete_eligible"][j]),
            reference_volume_um3=None,
            prediction_volume_um3=float(pc["volume_um3"][j]),
            reference_deq_um=None,
            prediction_deq_um=float(pc["deq_um"][j]),
            reference_axis_major_minor=None,
            prediction_axis_major_minor=float(pc["axis_major_minor"][j]),
        )


def phase_metrics(mask, domain, spacing):
    mask = np.asarray(mask, bool) & domain
    labels, n = ndimage.label(mask, structure=CONNECTIVITY)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    n_phase = int(mask.sum())
    den = int(domain.sum())
    largest = int(counts.max()) if n else 0
    voxel_vol = float(np.prod(spacing))
    spans = {}
    ids = {}
    for axis, name in enumerate(("z", "y", "x")):
        a = np.unique(np.take(labels, 0, axis=axis))
        b = np.unique(np.take(labels, -1, axis=axis))
        shared = np.intersect1d(a[a > 0], b[b > 0])
        spans[name] = bool(len(shared))
        ids[name] = shared.tolist()
    return dict(
        reference_domain_voxels=den,
        phase_voxels=n_phase,
        phase_volume_um3=n_phase * voxel_vol,
        phase_fraction=n_phase / den,
        n_components=int(n),
        largest_component_voxels=largest,
        largest_component_volume_um3=largest * voxel_vol,
        largest_fraction_of_phase=largest / n_phase if n_phase else None,
        largest_fraction_of_roi=largest / den,
        spans_x=spans["x"],
        spans_y=spans["y"],
        spans_z=spans["z"],
        n_spanning_components_x=len(ids["x"]),
        n_spanning_components_y=len(ids["y"]),
        n_spanning_components_z=len(ids["z"]),
    )


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_csv(path, rows):
    rows = list(rows)
    fields = (
        list(rows[0])
        if rows
        else ["threshold_deg", "status", "reference_id", "prediction_id"]
    )
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_json_value(row) for row in rows)


def _save_catalog(path, objects):
    np.savez_compressed(path.with_suffix(".npz"), **objects)
    _write_csv(
        path.with_suffix(".csv"),
        (
            {key: values[index] for key, values in objects.items()}
            for index in range(len(objects["grain_id"]))
        ),
    )


def _phase_accuracy(prediction, reference, hidden, angles, symmetries):
    phase = np.where(prediction["valid"], prediction["phase"], 0)
    rows = []
    for phase_id, symmetry in symmetries.items():
        actual = hidden & (reference["phase"] == phase_id)
        positive = hidden & (phase == phase_id)
        true_positive = int((actual & positive).sum())
        n_reference = int(actual.sum())
        n_prediction = int(positive.sum())
        rows.append(
            dict(
                phase=phase_id,
                symmetry=symmetry,
                reference_voxels=n_reference,
                predicted_voxels=n_prediction,
                true_positive=true_positive,
                precision=true_positive / n_prediction if n_prediction else None,
                recall=true_positive / n_reference if n_reference else None,
                dice=2 * true_positive / (n_reference + n_prediction)
                if n_reference + n_prediction
                else None,
                reference_fraction=n_reference / int(hidden.sum()),
                predicted_fraction=n_prediction / int(hidden.sum()),
                matched_angle_deg=float(angles[actual & positive].mean())
                if true_positive
                else None,
            )
        )
    return rows


def _strata(prediction, reference, hidden, angles, distance):
    width = float(max(reference["spacing_um"]))
    phase = np.where(prediction["valid"], prediction["phase"], 0)
    result = {"interface_band_um": width}
    for name, mask in (
        ("interface", hidden & (distance <= width)),
        ("interior", hidden & (distance > width)),
    ):
        matched_phase = mask & (phase == reference["phase"])
        result[name] = dict(
            n=int(mask.sum()),
            E90_deg=float(angles[mask].mean()) if mask.any() else None,
            same_phase_n=int(matched_phase.sum()),
            same_phase_error_deg=float(angles[matched_phase].mean())
            if matched_phase.any()
            else None,
            phase_accuracy=float(matched_phase.sum() / mask.sum())
            if mask.any()
            else None,
        )
    return result


def _surface_metrics(pred_boundary, ref_boundary, hidden, spacing):
    diagonal = float(np.linalg.norm((np.asarray(hidden.shape) - 1) * spacing))
    dist_ref = (
        ndimage.distance_transform_edt(~ref_boundary, sampling=spacing)
        if ref_boundary.any()
        else np.full(hidden.shape, diagonal)
    )
    dist_pred = (
        ndimage.distance_transform_edt(~pred_boundary, sampling=spacing)
        if pred_boundary.any()
        else np.full(hidden.shape, diagonal)
    )
    pred_distances = dist_ref[pred_boundary & hidden]
    ref_distances = dist_pred[ref_boundary & hidden]
    tolerance = float(max(spacing))
    precision = (
        float(np.mean(pred_distances <= tolerance))
        if len(pred_distances)
        else float(not len(ref_distances))
    )
    recall = (
        float(np.mean(ref_distances <= tolerance))
        if len(ref_distances)
        else float(not len(pred_distances))
    )
    joined = np.concatenate((pred_distances, ref_distances))
    result = dict(
        asd_um=float(joined.mean()) if len(joined) else 0.0,
        hd95_um=float(np.quantile(joined, 0.95)) if len(joined) else 0.0,
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / max(precision + recall, 1e-12),
        tolerance_um=tolerance,
    )
    return result, dist_ref


def _gradients(prediction, reference, hidden, distance, symmetries):
    phase = np.where(prediction["valid"], prediction["phase"], 0)
    spacing = reference["spacing_um"]
    interior = hidden & (distance > max(spacing))
    rows = []
    for axis, name in enumerate(("z", "y", "x")):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        eligible = (
            interior[a] & interior[b] & (reference["phase"][a] == reference["phase"][b])
        )
        valid = (
            eligible
            & (phase[a] == reference["phase"][a])
            & (phase[b] == reference["phase"][b])
        )
        truth = (
            angle_map(
                reference["q"][a],
                reference["q"][b],
                reference["phase"][a],
                reference["phase"][b],
                valid,
                symmetries,
            )[valid]
            / spacing[axis]
        )
        estimate = (
            angle_map(
                prediction["q"][a],
                prediction["q"][b],
                phase[a],
                phase[b],
                valid,
                symmetries,
            )[valid]
            / spacing[axis]
        )
        rows.append(
            dict(
                axis=name,
                reference_pairs=int(eligible.sum()),
                correct_phase_pairs=int(valid.sum()),
                mae_deg_per_um=float(np.mean(abs(truth - estimate)))
                if len(truth)
                else None,
                reference_mean=float(truth.mean()) if len(truth) else None,
                predicted_mean=float(estimate.mean()) if len(truth) else None,
            )
        )
    return rows


def _connectivity(prediction, reference, symmetries):
    domain, spacing = reference["valid"], reference["spacing_um"]
    phase = np.where(prediction["valid"], prediction["phase"], 0)
    rows = []
    for phase_id in symmetries:
        ref = phase_metrics(reference["phase"] == phase_id, domain, spacing)
        pred = phase_metrics(phase == phase_id, domain, spacing)
        fractions = (
            pred["largest_fraction_of_phase"],
            ref["largest_fraction_of_phase"],
        )
        rows.append(
            dict(
                phase=phase_id,
                reference=ref,
                prediction=pred,
                phase_fraction_error_pp=100
                * abs(pred["phase_fraction"] - ref["phase_fraction"]),
                largest_fraction_error_pp=100 * abs(fractions[0] - fractions[1])
                if all(v is not None for v in fractions)
                else None,
                largest_roi_fraction_error_pp=100
                * abs(pred["largest_fraction_of_roi"] - ref["largest_fraction_of_roi"]),
                spanning_mismatches=sum(
                    pred[f"spans_{axis}"] != ref[f"spans_{axis}"] for axis in "xyz"
                ),
            )
        )
    return rows


def evaluate_saved(run_dir, thresholds=(2, 5, 10), anchor=False, data_root=None):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    data_root = (
        data_root if data_root is not None else manifest.get("data_root", "data")
    )
    reference, metadata = load_region(
        manifest["dataset"], manifest["split"], data_root=data_root
    )
    reference_hash = volume_hash(reference)
    if (
        manifest.get("reference_sha256")
        and manifest["reference_sha256"] != reference_hash
    ):
        raise ValueError(f"{run_dir}: reference volume differs from the fitted run")
    symmetries = manifest["config"].get("phase_symmetry", metadata["phase_symmetry"])
    symmetries = {int(key): value for key, value in symmetries.items()}
    with np.load(run_dir / "prediction.npz") as volume:
        prediction = {key: volume[key].copy() for key in ("q", "phase", "valid")}
    if prediction["phase"].shape != reference["phase"].shape:
        raise ValueError("Prediction and reference shapes differ")
    thresholds = sorted(set(float(value) for value in thresholds) | {5.0})
    if any(not 0 < value < 90 for value in thresholds):
        raise ValueError("Grain thresholds must be between 0 and 90 degrees")
    observed_z = np.asarray(manifest["observed_z_indices"], dtype=int)
    domain = reference["valid"]
    observed = np.zeros_like(domain)
    observed[observed_z] = domain[observed_z]
    hidden = domain & ~observed
    hidden_z = np.ones(len(domain), bool)
    hidden_z[observed_z] = False
    if not hidden.any():
        raise ValueError("No valid hidden voxels to evaluate")
    exact = all(
        np.array_equal(prediction[key][observed], reference[key][observed])
        for key in prediction
    )
    changed = {}
    for key in prediction:
        different = prediction[key][observed] != reference[key][observed]
        changed[key] = int(
            np.count_nonzero(np.any(different, axis=-1) if key == "q" else different)
        )
        if anchor:
            prediction[key][observed] = reference[key][observed]
    output = run_dir / ("analysis_anchored" if anchor else "analysis_native")
    output.mkdir(parents=True, exist_ok=True)
    phase = np.where(prediction["valid"], prediction["phase"], 0)
    spacing = reference["spacing_um"]
    reference_edges = edge_angles(
        reference["q"], reference["phase"], domain, symmetries
    )
    prediction_edges = edge_angles(prediction["q"], phase, domain, symmetries)
    reference_labels = from_edges(reference_edges, domain, 5)
    reference_boundary = boundaries(
        reference["q"], reference["phase"], domain, symmetries, 5
    )
    ordinary, maps = evaluate(
        prediction,
        reference,
        observed_z,
        symmetries,
        dict(labels=reference_labels, boundary=reference_boundary),
    )
    np.savez_compressed(output / "evaluation_maps.npz", **maps)
    reference_boundary_3d = reference_boundary | boundaries(
        reference["q"], reference["phase"], domain, symmetries, 5, axes=(0,)
    )
    prediction_boundary_3d = maps["boundary"] | boundaries(
        prediction["q"], phase, domain, symmetries, 5, axes=(0,)
    )
    surface, distance = _surface_metrics(
        prediction_boundary_3d, reference_boundary_3d, hidden, spacing
    )
    grain_results, match_rows = {}, []
    for threshold in thresholds:
        key = f"{threshold:g}"
        reference_labels = from_edges(reference_edges, domain, threshold)
        prediction_labels = from_edges(prediction_edges, domain, threshold)
        reference_objects = catalog(
            reference_labels, reference["phase"], domain, spacing
        )
        prediction_objects = catalog(prediction_labels, phase, domain, spacing)
        assignment, iou, used, splits, merges = matched(
            prediction_labels,
            reference_labels,
            prediction_objects,
            reference_objects,
            domain,
        )
        folder = output / f"grains_{key}deg"
        folder.mkdir(exist_ok=True)
        _save_catalog(folder / "reference", reference_objects)
        _save_catalog(folder / "prediction", prediction_objects)
        np.savez_compressed(
            folder / "partitions.npz",
            reference=reference_labels,
            prediction=prediction_labels,
        )
        comparisons = [
            comparison(
                prediction_objects,
                reference_objects,
                assignment,
                iou,
                used,
                splits,
                merges,
                phase_id,
            )
            for phase_id in (None, *symmetries)
        ]
        reference_boundary = boundaries(
            reference["q"], reference["phase"], domain, symmetries, threshold
        )
        prediction_boundary = boundaries(
            prediction["q"], phase, domain, symmetries, threshold
        )
        scores = boundary_metrics(
            prediction_boundary,
            reference_boundary,
            domain,
            hidden_z,
            spacing[1:],
            float(max(spacing[1:])),
        )
        scores.update(grain_metrics(prediction_labels, reference_labels, hidden))
        grain_results[key] = dict(hidden_metrics=scores, comparison=comparisons)
        match_rows.extend(
            dict(threshold_deg=threshold, **row)
            for row in pair_rows(
                reference_objects, prediction_objects, assignment, iou, used
            )
        )
    _write_csv(output / "grain_matches.csv", match_rows)
    with (run_dir / "prediction.npz").open("rb") as stream:
        prediction_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    result = dict(
        dataset=manifest["dataset"],
        split=manifest["split"],
        method=manifest.get("method"),
        anchored=bool(anchor),
        original_observed_arrays_exact=exact,
        observed_differing_voxels=changed,
        hidden_prediction_unchanged=True,
        prediction_sha256=prediction_hash,
        reference_sha256=reference_hash,
        observed_z_indices=observed_z.tolist(),
        spacing_um_zyx=spacing.tolist(),
        protocol=dict(
            grain_adjacency=6,
            grain_thresholds_deg=thresholds,
            minimum_grain_voxels=8,
            coordinates="voxel centers in physical ZYX coordinates",
            covariance="population covariance plus diagonal voxel variance spacing^2/12",
            shape="square roots of descending covariance eigenvalue ratios",
            complete_grains="at least 8 voxels; touch neither ROI faces nor invalid domain",
            matching="same phase; descending IoU >= 0.1; deterministic greedy one-to-one",
            unmatched_volume="zero for all-reference relative volume error; listed separately",
            split_merge="at least 5% overlap; same phase; both components at least 8 voxels",
            connectivity="6 face-neighbors; no periodic boundaries; all components retained",
            spanning="same component touches both outermost opposite ROI faces",
            phase_fraction_domain="reference-valid voxels in the fixed ROI",
            anchoring="replace only reference-valid observed voxels before full-volume partition",
        ),
        metrics=ordinary,
        phase_accuracy=_phase_accuracy(
            prediction, reference, hidden, maps["angles"], symmetries
        ),
        surface3d=surface,
        strata=_strata(prediction, reference, hidden, maps["angles"], distance),
        gradients=_gradients(prediction, reference, hidden, distance, symmetries),
        phase_connectivity=_connectivity(prediction, reference, symmetries),
        grains=grain_results,
    )
    result = _json_value(result)
    (output / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result
