from functools import lru_cache
import numpy as np
from scipy.spatial.transform import Rotation


def normalize_quaternions(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.divide(q, n, out=np.zeros_like(q), where=n > 1e-12)


def quat_mul(a, b):
    a, b = (np.asarray(a), np.asarray(b))
    w = a[..., :1] * b[..., :1] - np.sum(
        a[..., 1:] * b[..., 1:], axis=-1, keepdims=True
    )
    v = (
        a[..., :1] * b[..., 1:]
        + b[..., :1] * a[..., 1:]
        + np.cross(a[..., 1:], b[..., 1:])
    )
    return np.concatenate((w, v), axis=-1)


def _group_name(symmetry):
    s = str(symmetry).lower()
    if s in {"cubic", "o", "m-3m", "432", "fcc", "bcc"}:
        return "O"
    if s in {"hexagonal", "hex", "hcp", "d6", "6/mmm", "622"}:
        return "D6"
    if s in {"identity", "triclinic", "1", "none"}:
        return "C1"
    raise ValueError(f"Unsupported crystal symmetry: {symmetry}")


@lru_cache(maxsize=16)
def crystal_symmetry_quaternions(symmetry="cubic"):
    return Rotation.create_group(_group_name(symmetry), axis="Z").as_quat()[
        :, [3, 0, 1, 2]
    ]


def symmetry_equivalents(q, symmetry="cubic"):
    return quat_mul(
        normalize_quaternions(q)[..., None, :], crystal_symmetry_quaternions(symmetry)
    )


def canonicalize_quaternions(q, symmetry="cubic"):
    shape = np.shape(q)
    q = normalize_quaternions(q).reshape(-1, 4)
    r = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    r = r.reduce(right=Rotation.create_group(_group_name(symmetry), axis="Z"))
    out = r.as_quat()[:, [3, 0, 1, 2]]
    out *= np.where(out[:, :1] < 0, -1.0, 1.0)
    return out.reshape(shape)


def align_quaternions(q, reference, symmetry="cubic"):
    q, reference = np.broadcast_arrays(
        normalize_quaternions(q), normalize_quaternions(reference)
    )
    eq = symmetry_equivalents(q, symmetry)
    dots = np.sum(eq * reference[..., None, :], axis=-1)
    k = np.argmax(np.abs(dots), axis=-1)
    chosen = np.take_along_axis(eq, k[..., None, None], axis=-2)[..., 0, :]
    sign = np.take_along_axis(dots, k[..., None], axis=-1)
    return chosen * np.where(sign < 0, -1.0, 1.0)


def misorientation_deg(q1, q2, symmetry="cubic", chunk_size=65536):
    a, b = np.broadcast_arrays(np.asarray(q1), np.asarray(q2))
    shape = a.shape[:-1]
    a, b = (a.reshape(-1, 4), b.reshape(-1, 4))
    out = np.empty(len(a), dtype=np.float64)
    for start in range(0, len(a), chunk_size):
        aa, bb = (a[start : start + chunk_size], b[start : start + chunk_size])
        good = (np.linalg.norm(aa, axis=-1) > 1e-12) & (
            np.linalg.norm(bb, axis=-1) > 1e-12
        )
        eq = symmetry_equivalents(bb, symmetry)
        dots = np.abs(np.sum(normalize_quaternions(aa)[:, None, :] * eq, axis=-1)).max(
            axis=-1
        )
        angle = np.degrees(2 * np.arccos(np.clip(dots, 0, 1)))
        angle[~good] = np.nan
        out[start : start + len(aa)] = angle
    return out.reshape(shape)
