import io
import json
from pathlib import Path
import re
import tarfile
import zipfile

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .io import digest, datasets, verify

CONVENTION = {
    "quaternion_order": "wxyz",
    "rotation": "active crystal-to-sample; Hamilton quaternion multiplication",
    "euler_conversion": "SciPy intrinsic ZXZ, Bunge angles in radians; equivalent to conjugate of DREAM3D passive xyzw quaternion, up to antipodal sign",
    "crystal_symmetry_action": "right: q_equivalent = q * s; s rotates crystal coordinates",
    "sample_symmetry": "none",
    "canonical_sign": "w>=0 (q and -q represent the same rotation)",
}


def euler_to_q(eu, degrees=False):
    shape = eu.shape[:-1]
    q = (
        Rotation.from_euler("ZXZ", eu.reshape(-1, 3), degrees=degrees)
        .as_quat(scalar_first=True)
        .astype(np.float32)
    )
    q[q[:, 0] < 0] *= -1
    return q.reshape(*shape, 4)


def save(name, q, phase, valid, spacing, meta, data_root, **optional):
    out = Path(data_root) / "processed"
    out.mkdir(parents=True, exist_ok=True)
    valid = np.asarray(valid, dtype=bool) & np.all(np.isfinite(q), axis=-1)
    phase = np.asarray(phase, dtype=np.int16)
    phase[~valid] = 0
    q[~valid] = [1, 0, 0, 0]
    max_error = 0.0
    for z in range(q.shape[0]):
        norms = np.linalg.norm(q[z][valid[z]], axis=-1)
        if norms.size:
            max_error = max(max_error, float(np.max(np.abs(norms - 1))))
    assert max_error < 1e-5
    path = out / f"{name}.npz"
    np.savez_compressed(
        path,
        q=q,
        phase=phase,
        valid=valid,
        spacing_um=np.asarray(spacing, dtype=np.float64),
        **optional,
    )
    meta.update(CONVENTION)
    meta.update(
        {
            "dataset_id": name,
            "shape_zyx": list(phase.shape),
            "spacing_um_zyx": list(spacing),
            "specimens": 1,
            "valid_voxels": int(valid.sum()),
            "total_voxels": int(valid.size),
            "phase_counts": {
                str(k): int(v) for k, v in zip(*np.unique(phase, return_counts=True))
            },
            "array_order": "zyx",
            "source_slice_indices": list(range(phase.shape[0])),
            "npz_sha256": digest(path),
            "normalization_max_error": max_error,
        }
    )
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(path, meta["shape_zyx"], meta["phase_counts"], flush=True)


def in100(data_root):
    data_root = Path(data_root)
    path = data_root / "raw/in100_processed/Small_IN100.dream3d"
    if not path.exists():
        with tarfile.open(path.parent / "Small_IN100_dream3d_v3.tar.gz") as archive:
            archive.extract("Small_IN100.dream3d", path.parent, filter="data")
    with h5py.File(path) as h:
        c = h["DataStructure/DataContainer/CellData"]
        eu = c["EulerAngles"][()]
        q = euler_to_q(eu)
        phase = c["Phases"][..., 0]
        ci = c["Confidence Index"][..., 0]
        valid = (phase > 0) & (ci >= 0) & np.all(np.isfinite(eu), axis=-1)
        source = json.loads((path.parent / "source.json").read_text())
        meta = {
            "material": "Small IN100 nickel superalloy",
            "source": source,
            "input_file_sha256": digest(path),
            "input_hdf5_container": "DataStructure/DataContainer",
            "phase_symmetry": {"1": "cubic"},
            "phase_names": {"1": "Nickel FCC"},
            "validity_rule": "source phase>0 and source CI>=0 and finite Euler angles; no additional confidence or IQ threshold",
            "registration": "UNREGISTERED imported stack; dense preregistration must be explicit before using as aligned reference",
            "source_processing": "Official simplnx v3 file main DataContainer only. Auxiliary mirrored/cropped test containers excluded. Import contains upstream coordinate transforms; original source archive is retained.",
            "spatial_extent_origin_xyz_um": h["DataStructure/DataContainer"]
            .attrs["_ORIGIN"]
            .tolist(),
            "caveats": [
                "CI=0 retained as indexed low-confidence pixels; no imputation or grain segmentation",
                "Full-stack preregistration would make this an ideal aligned-reference benchmark, not a sparse registration validation.",
            ],
        }
        meta.update(
            {
                "source_slice_numbers": list(range(117, 0, -1)),
                "source_to_canonical_spatial_mapping": "canonical[z,y,x] = raw_h5ebsd[slice117-z][y,188-x]. Mapping established by the original source audit across all 4444713 voxels.",
                "upstream_orientation_transform": "source EulerTransformation90deg about Z; imported Phi1=(rawPhi1-pi/2)mod2pi, Phi/Phi2 unchanged (maximum numerical difference1.15e-5rad). Source SampleTransformation180deg about Y reverses X and Z.",
            }
        )
        save(
            "in100",
            q,
            phase,
            valid,
            [0.25, 0.25, 0.25],
            meta,
            data_root,
            confidence=ci.astype(np.float32),
            image_quality=c["Image Quality"][..., 0].astype(np.float32),
        )


def ti64(data_root):
    data_root = Path(data_root)
    path = data_root / "raw/ti64/Ti_64_equiaxed_6_15_2017_final.dream3d"
    with h5py.File(path) as h:
        c = h["DataContainers/ImageDataContainer/CellData"]

        eu = c["EulerAngles"][::10, ::20, ::20]
        reference = c["Quaternions"][::10, ::20, ::20]
        reference = reference[..., [3, 0, 1, 2]]
        reference[..., 1:] *= -1
        dots = np.abs(np.sum(euler_to_q(eu) * reference, axis=-1))
        assert np.max(1 - dots) < 2e-6
        q = c["Quaternions"][()][..., [3, 0, 1, 2]]
        q[..., 1:] *= -1
        q[q[..., 0] < 0] *= -1
        q /= np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-20)
        phase = c["Phases"][..., 0]
        valid = (phase > 0) & np.all(np.isfinite(q), axis=-1)
        meta = {
            "material": "Equiaxed Ti-6Al-4V",
            "source": json.loads((path.parent / "source.json").read_text()),
            "phase_symmetry": {"1": "hexagonal", "2": "cubic"},
            "phase_names": {"1": "Titanium alpha HCP", "2": "Titanium beta BCC"},
            "validity_rule": "publisher processed phase>0 and finite nonzero unit quaternion; publisher fills removed defects",
            "registration": "publisher full-stack DREAM3D AlignSectionsMisorientation (2deg) before sparse sampling",
            "source_processing": "Author processed volume: CI threshold0.01, orientation alignment2deg, segmentation2deg, minimum feature size125, minimum2 neighbors, bad-point fill, crop XYZ[0:318,25:402,0:121]. Stored voxel Quaternions retained, not grain-mean AverageEulers.",
            "quaternion_euler_validation_max_one_minus_abs_dot": float(
                np.max(1 - dots)
            ),
            "caveats": [
                "Dense publisher registration and cleanup may use all source slices. Evaluation is conditional on this registered processed reference.",
                "Not a pure-HCP single-phase dataset; both alpha and beta phases retained.",
            ],
        }
        save(
            "ti64",
            q.astype(np.float32),
            phase,
            valid,
            [1.0, 0.6, 0.6],
            meta,
            data_root,
            feature_id=c["FeatureIds"][..., 0].astype(np.int32),
            confidence=c["Confidence Index"][..., 0].astype(np.float32),
            publisher_mask=c["Mask"][..., 0].astype(bool),
        )


def zrnb(data_root):
    data_root = Path(data_root)
    path = data_root / "raw/zrnb/ctf.zip"
    with zipfile.ZipFile(path) as z:
        names = [
            n
            for n in z.namelist()
            if n.endswith(".ctf")
            and not Path(n).name.startswith("._")
            and not n.startswith("__MACOSX/")
        ]
        names.sort(key=lambda s: int(re.search(r"Slice (\d+)", s).group(1)))
        assert len(names) == 441
        q = np.empty((441, 333, 333, 4), dtype=np.float32)
        phase = np.empty((441, 333, 333), dtype=np.int16)
        bc = np.empty_like(phase, dtype=np.uint16)
        error = np.empty_like(phase, dtype=np.uint8)
        mad = np.empty_like(phase, dtype=np.float32)
        for i, name in enumerate(names):
            lines = z.read(name).decode().splitlines()
            header = next(
                k for k, line in enumerate(lines) if line.startswith("Phase\tX\tY")
            )
            a = np.loadtxt(
                io.StringIO("\n".join(lines[header + 1 :])), dtype=np.float32
            )
            assert a.shape == (333 * 333, 11), (name, a.shape)
            assert np.allclose(
                a[:, 1].reshape(333, 333)[0], np.arange(333) * 0.15, atol=1e-4
            )
            assert np.allclose(
                a[:, 2].reshape(333, 333)[:, 0], np.arange(333) * 0.15, atol=1e-4
            )
            q[i] = euler_to_q(a[:, 5:8].reshape(333, 333, 3), degrees=True)
            phase[i] = a[:, 0].reshape(333, 333)
            error[i] = a[:, 4].reshape(333, 333)
            mad[i] = a[:, 8].reshape(333, 333)
            bc[i] = a[:, 9].reshape(333, 333)
            if i % 50 == 0:
                print("zrnb parsed", i + 1, "/441", flush=True)
        valid = (phase > 0) & (error == 0)
        meta = {
            "material": "Zircaloy-4 + 7 wt% Nb",
            "source": json.loads((path.parent / "source.json").read_text()),
            "archive_md5_verified": "e74d10f6a7addadd060ea873bcc365e7",
            "phase_symmetry": {"1": "hexagonal", "2": "cubic"},
            "phase_names": {"1": "alpha Zr HCP", "2": "beta Zr BCC"},
            "validity_rule": "source phase>0 and CTF error=0 and finite Euler angles; no confidence or band-contrast threshold",
            "registration": "UNREGISTERED raw CTF serial stack; any dense preregistration must be explicit",
            "source_processing": "CTF degrees converted to Bunge ZXZ active quaternion; native row ordering (X fastest, Y next), naturally ordered Slice1..441. No segmentation, filling, cropping, resampling, or sample-axis rotation.",
            "ctf_source_names": names,
            "source_slice_numbers": list(range(1, 442)),
            "sample_coordinate_axes": "CTF CS0; metadata X=RD, Y=TD, Z=ND; AcqE1=AcqE2=AcqE3=0",
            "license": "CC BY 4.0",
            "caveats": [
                "CTF indexing failures retained as invalid; phase0 is not a material phase.",
                "Raw slice drift must be measured; full-stack preregistration would only support an aligned-reference benchmark.",
            ],
        }
        save(
            "zrnb",
            q,
            phase,
            valid,
            [0.1, 0.15, 0.15],
            meta,
            data_root,
            band_contrast=bc,
            indexing_error=error,
            mean_angular_deviation=mad,
        )


def in718(data_root):
    data_root = Path(data_root)
    folder = data_root / "raw/in718_mirror"
    source = json.loads((folder / "source.json").read_text())
    arrays = {
        key: np.memmap(
            folder / f"{key}.bin", dtype=v["dtype"], mode="r", shape=tuple(v["shape"])
        )
        for key, v in source["arrays"].items()
    }
    for key, v in source["arrays"].items():
        assert digest(folder / f"{key}.bin") == v["sha256"]
    q = np.empty((526, 420, 549, 4), dtype=np.float32)
    for z in range(526):
        q[z] = euler_to_q(arrays["EulerAngles"][z])
    phase = np.asarray(arrays["Phases"], dtype=np.int16)
    feature = arrays["FeatureIds"]
    valid = (phase > 0) & (feature > 0)
    meta = {
        "material": "Inconel 718",
        "source": source,
        "phase_symmetry": {"1": "cubic"},
        "phase_names": {"1": "Ni FCC"},
        "validity_rule": "source phase>0 and publisher FeatureIds>0 and finite Euler; feature0 denotes outside/unassigned cells and is not used as a fitting target",
        "registration": "author full-stack distortion correction, alignment, cleanup and crop before sparse sampling",
        "source_processing": "Embedded source pipeline imports DI_stack.h5ebsd original sections3..528 (526 sections), then distortion correction, 5deg section alignment, 2deg grain segmentation, cleanup/fill, crop XYZ[28:577,0:420,0:526]. Final EulerAngles has author6.015deg sample-reference rotation; quaternion recomputed from final EulerAngles because stored Quaternions predates that transform. No further filtering/resampling/cropping applied here.",
        "source_slice_numbers": list(range(3, 529)),
        "caveats": [
            "Public UCSB author-hosted processed variant named718_final_yield.dream3d, with extra SlipBands/FeatureIDsClean arrays. Same specimen/import chain and dimensions as Dryad record, but not byte-identical verified against the inaccessible Dryad zip.",
            "Entire HDF5 file not downloaded; each retained byte range, source ETag, array shape/type and SHA256 preserved.",
            "Dense reference preprocessing may use all source slices; experiments only support aligned-reference reconstruction.",
        ],
    }
    save("in718", q, phase, valid, [1.0, 1.0, 1.0], meta, data_root, feature_id=feature)


def prepare(name, data_root="data"):
    from .download import download
    from .registration import register
    from .regions import build_crops

    data_root = Path(data_root)
    source = datasets()[name]
    folder = data_root / "raw" / source["raw_directory"]
    manifest = folder / "source.json"
    if name == "in718":
        for field, array in source["arrays"].items():
            verify(folder / f"{field}.bin", array["length_bytes"], array["sha256"])
    else:
        verify(folder / source["filename"], source["size_bytes"], source["sha256"])
    if not manifest.exists():
        download(name, data_root)
    output = data_root / "processed" / f"{name}.npz"
    if not output.exists():
        globals()[name](data_root)
    elif not output.with_suffix(".json").exists():
        raise FileNotFoundError(output.with_suffix(".json"))
    else:
        metadata = json.loads(output.with_suffix(".json").read_text())
        if digest(output) != metadata["npz_sha256"]:
            raise ValueError(f"{output}: processed data checksum mismatch")
    if name in ("in100", "zrnb"):
        register(name, data_root)
    build_crops(name, data_root)
