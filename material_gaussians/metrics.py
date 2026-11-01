import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score
from material_gaussians.orientation import misorientation_deg


def angle_map(q1, q2, phase1, phase2, valid, symmetries):
    out = np.full(valid.shape, 90.0, dtype=np.float32)
    for phase, symmetry in symmetries.items():
        mask = valid & (phase1 == int(phase)) & (phase2 == int(phase))
        indices = np.flatnonzero(mask)
        a, b, flat = (q1.reshape(-1, 4), q2.reshape(-1, 4), out.ravel())
        for start in range(0, len(indices), 65536):
            ix = indices[start : start + 65536]
            flat[ix] = misorientation_deg(a[ix], b[ix], symmetry)
    return out


def neighbour_edges(q, phase, valid, symmetries, threshold_deg=5.0, axes=(0, 1, 2)):
    for axis in axes:
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[axis], b[axis] = (slice(None, -1), slice(1, None))
        a, b = (tuple(a), tuple(b))
        eligible = valid[a] & valid[b]
        same = eligible & (phase[a] == phase[b])
        angles = angle_map(q[a], q[b], phase[a], phase[b], same, symmetries)
        yield (a, b, eligible, same & (angles <= threshold_deg))


def boundaries(q, phase, valid, symmetries, threshold_deg=5.0, axes=(1, 2)):
    result = np.zeros(valid.shape, dtype=bool)
    for a, b, eligible, same in neighbour_edges(
        q, phase, valid, symmetries, threshold_deg, axes
    ):
        edge = eligible & ~same
        result[a] |= edge
        result[b] |= edge
    return result


def grain_labels(q, phase, valid, symmetries, threshold_deg=5.0):
    indices = np.arange(valid.size, dtype=np.int32).reshape(valid.shape)
    rows, cols = ([], [])
    for a, b, _, same in neighbour_edges(q, phase, valid, symmetries, threshold_deg):
        rows.append(indices[a][same])
        cols.append(indices[b][same])
    rows, cols = (np.concatenate(rows), np.concatenate(cols))
    graph = sparse.coo_matrix(
        (np.ones(len(rows), np.uint8), (rows, cols)), shape=(valid.size, valid.size)
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    labels = labels.reshape(valid.shape).astype(np.int32) + 1
    labels[~valid] = 0
    return labels


def boundary_metrics(pred, ref, valid, hidden_z, spacing_yx, tolerance_um):
    matched_p = matched_r = count_p = count_r = 0
    distances, maxdist = (
        [],
        float(np.hypot(*(np.array(ref.shape[1:]) - 1) * spacing_yx)),
    )
    for z in np.flatnonzero(hidden_z):
        p, r = (pred[z] & valid[z], ref[z] & valid[z])
        np_, nr = (int(p.sum()), int(r.sum()))
        count_p += np_
        count_r += nr
        if np_ and nr:
            dr = ndimage.distance_transform_edt(~r, sampling=spacing_yx)[p]
            dp = ndimage.distance_transform_edt(~p, sampling=spacing_yx)[r]
            matched_p += int((dr <= tolerance_um + 1e-08).sum())
            matched_r += int((dp <= tolerance_um + 1e-08).sum())
            distances.extend((dr, dp))
        elif np_ or nr:
            distances.append(np.full(np_ + nr, maxdist))
    precision = matched_p / count_p if count_p else float(count_r == 0)
    recall = matched_r / count_r if count_r else float(count_p == 0)
    joined = np.concatenate(distances) if distances else np.zeros(1)
    return dict(
        boundary_precision=precision,
        boundary_recall=recall,
        boundary_f1=2 * precision * recall / max(precision + recall, 1e-12),
        boundary_asd_um=float(joined.mean()),
        boundary_hd95_um=float(np.quantile(joined, 0.95)),
        boundary_tolerance_um=float(tolerance_um),
    )


def grain_metrics(pred, ref, hidden_mask):
    r, p = (ref[hidden_mask], pred[hidden_mask])
    _, ri = np.unique(r, return_inverse=True)
    _, pi = np.unique(p, return_inverse=True)
    contingency = sparse.coo_matrix(
        (np.ones(len(ri), dtype=np.int32), (ri, pi))
    ).tocsr()
    contingency.sum_duplicates()
    nr, np_ = (
        np.asarray(contingency.sum(1)).ravel(),
        np.asarray(contingency.sum(0)).ravel(),
    )
    c = contingency.tocoo()
    iou = c.data / (nr[c.row] + np_[c.col] - c.data)
    best = np.zeros(len(nr))
    np.maximum.at(best, c.row, iou)
    eligible = nr >= 8
    small = (
        eligible & (nr <= np.quantile(nr[eligible], 0.25))
        if eligible.any()
        else eligible
    )
    significant_r = c.data / nr[c.row] >= 0.05
    significant_p = c.data / np_[c.col] >= 0.05
    splits = np.bincount(c.row[significant_r], minlength=len(nr))
    merges = np.bincount(c.col[significant_p], minlength=len(np_))
    prob = c.data / max(c.data.sum(), 1)
    vi = -np.sum(prob * (np.log2(c.data / nr[c.row]) + np.log2(c.data / np_[c.col])))
    return dict(
        grain_ari=float(adjusted_rand_score(ri, pi)),
        grain_vi_bits=float(vi),
        grain_recall_iou50=float(np.mean(best[eligible] >= 0.5))
        if eligible.any()
        else None,
        small_grain_recall_iou50=float(np.mean(best[small] >= 0.5))
        if small.any()
        else None,
        reference_grains_ge8=int(eligible.sum()),
        small_reference_grains=int(small.sum()),
        split_fraction=float(np.mean(splits[eligible] > 1)) if eligible.any() else None,
        merge_fraction=float(np.mean(merges[np_ >= 8] > 1))
        if (np_ >= 8).any()
        else None,
    )


def two_point_errors(pred_labels, ref_labels, mask, max_lag=16):
    curves, errors = ({}, [])
    for axis, name in enumerate(("z", "y", "x")):
        curves[name] = []
        for lag in range(1, min(max_lag + 1, mask.shape[axis])):
            a, b = ([slice(None)] * 3, [slice(None)] * 3)
            a[axis], b[axis] = (slice(None, -lag), slice(lag, None))
            a, b = (tuple(a), tuple(b))
            m = mask[a] & mask[b]
            if not m.any():
                continue
            ref = float(np.mean(ref_labels[a][m] == ref_labels[b][m]))
            pred = float(np.mean(pred_labels[a][m] == pred_labels[b][m]))
            curves[name].append([lag, ref, pred])
            errors.append(abs(ref - pred))
    return dict(
        two_point_same_grain_mae=float(np.mean(errors)), two_point_curves=curves
    )


def evaluate(prediction, reference, observed_z, symmetries, reference_cache=None):
    q, phase, valid = (reference["q"], reference["phase"], reference["valid"])
    spacing = reference["spacing_um"]
    hidden_z = np.ones(len(q), bool)
    hidden_z[np.asarray(observed_z, dtype=int)] = False
    hidden = valid & hidden_z[:, None, None]
    pred_valid = prediction.get("valid", np.ones_like(valid))
    pred_phase = np.where(pred_valid, prediction["phase"], 0)
    angles = angle_map(prediction["q"], q, pred_phase, phase, valid, symmetries)
    values = angles[hidden]
    correct_phase = hidden & (pred_phase == phase)
    observed_mask = valid & ~hidden_z[:, None, None]
    result = dict(
        n_hidden_voxels=int(hidden.sum()),
        n_observed_voxels=int(observed_mask.sum()),
        n_hidden_slices=int(hidden_z.sum()),
        angle_penalized_mean_deg=float(values.mean()),
        angle_penalized_median_deg=float(np.median(values)),
        angle_penalized_p95_deg=float(np.quantile(values, 0.95)),
        within_5deg_fraction=float(np.mean(values <= 5)),
        phase_accuracy=float(np.mean(pred_phase[hidden] == phase[hidden])),
        angle_phase_matched_mean_deg=float(angles[correct_phase].mean())
        if correct_phase.any()
        else None,
        observed_angle_mean_deg=float(angles[observed_mask].mean()),
    )
    cache = reference_cache or {}
    ref_boundary = cache.get("boundary")
    if ref_boundary is None:
        ref_boundary = boundaries(q, phase, valid, symmetries)
    pred_boundary = boundaries(prediction["q"], pred_phase, valid, symmetries)
    result.update(
        boundary_metrics(
            pred_boundary,
            ref_boundary,
            valid,
            hidden_z,
            spacing[1:],
            float(max(spacing[1:])),
        )
    )
    result["boundary_angle_mean_deg"] = (
        float(angles[hidden & ref_boundary].mean())
        if (hidden & ref_boundary).any()
        else None
    )
    result["interior_angle_mean_deg"] = (
        float(angles[hidden & ~ref_boundary].mean())
        if (hidden & ~ref_boundary).any()
        else None
    )
    ref_labels = cache.get("labels")
    if ref_labels is None:
        ref_labels = grain_labels(q, phase, valid, symmetries)
    pred_labels = grain_labels(prediction["q"], pred_phase, valid, symmetries)
    result.update(grain_metrics(pred_labels, ref_labels, hidden))
    result.update(two_point_errors(pred_labels, ref_labels, hidden))
    slice_errors = [
        float(angles[z][valid[z]].mean()) if valid[z].any() else None
        for z in np.flatnonzero(hidden_z)
    ]
    result["hidden_slice_mean_angles_deg"] = slice_errors
    result["hidden_slice_indices"] = np.flatnonzero(hidden_z).tolist()
    return (result, dict(angles=angles, boundary=pred_boundary, labels=pred_labels))
