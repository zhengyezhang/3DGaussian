import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from material_gaussians.analysis import catalog, phase_metrics
from material_gaussians.data import load_region
from material_gaussians.experiment import run_experiment
from material_gaussians.metrics import evaluate
from material_gaussians.models import build_model, model_config
from material_gaussians.models.sgf import GaussianOrientationField
from material_gaussians.orientation import (
    crystal_symmetry_quaternions,
    misorientation_deg,
    quat_mul,
)
from material_gaussians.sampling import observation_mask


def synthetic_volume():
    z, y, x = np.indices((9, 10, 12))
    phase = np.where(x < 5 + z / 4, 1, 2).astype(np.int16)
    angles = np.stack((x * 0.1, y * 0.15, z * 0.2 + 20 * (phase == 2)), axis=-1)
    q = Rotation.from_euler("ZXZ", angles.reshape(-1, 3), degrees=True).as_quat(
        scalar_first=True
    )
    return dict(
        q=q.reshape(9, 10, 12, 4).astype(np.float32),
        phase=phase,
        valid=np.ones(phase.shape, bool),
        spacing_um=np.array([1.0, 0.6, 0.6]),
    )


FAST = dict(
    fit_epochs=None,
    iterations=2,
    batch_size=32,
    boundary_batch_size=8,
    seed_stride=3,
    candidates=6,
    orientation_sample_limit=100,
    max_train_points=1000,
    log_every=100,
    threads=1,
    transport_levels=1,
    transport_iterations=1,
)


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.ref = synthetic_volume()
        self.syms = {1: "cubic", 2: "hexagonal"}
        self.obs = observation_mask(9)
        self.z = np.arange(9, dtype=float)

    def fit(self, method, variant="full"):
        config = model_config(method, variant, {**FAST, "phase_symmetry": self.syms})
        model = build_model(method, variant)
        model.fit(
            self.ref["q"][self.obs].copy(),
            self.ref["phase"][self.obs].copy(),
            self.ref["valid"][self.obs].copy(),
            self.z[self.obs],
            self.ref["spacing_um"][1:].copy(),
            config,
            0,
        )
        return model

    def test_sampling_budgets(self):
        regular = observation_mask(97)
        np.testing.assert_array_equal(regular, np.arange(0, 97, 4))
        self.assertEqual(len(observation_mask(97, pattern="irregular")), 25)
        self.assertEqual(len(observation_mask(97, pattern="block")), 22)
        self.assertEqual(len(observation_mask(97, pattern="dispersed")), 22)
        self.assertEqual(
            set(regular) - set(observation_mask(97, pattern="block")), {44, 48, 52}
        )

    def test_crystal_symmetry(self):
        q = self.ref["q"].reshape(-1, 4)[:20]
        for symmetry in ("cubic", "hexagonal"):
            equivalent = -quat_mul(q, crystal_symmetry_quaternions(symmetry)[3])
            self.assertLess(
                float(np.max(misorientation_deg(q, equivalent, symmetry))), 1e-5
            )

    def test_model_outputs(self):
        for method in ("nearest", "sgf", "cgf", "tgs", "gsr"):
            with self.subTest(method=method):
                pred = self.fit(method).predict(self.z)
                self.assertEqual(pred["q"].shape, self.ref["q"].shape)
                self.assertTrue(np.isfinite(pred["q"]).all())
                np.testing.assert_allclose(
                    np.linalg.norm(pred["q"], axis=-1), 1, atol=1e-5
                )
                self.assertTrue(set(np.unique(pred["phase"])) <= {1, 2})

    def test_cgf_paired_field(self):
        model = self.fit("cgf")
        full = model.predict(self.z, chunk_size=2048)
        model.query = types.MethodType(GaussianOrientationField.query, model)
        coarse = model.predict(self.z, chunk_size=2048)
        np.testing.assert_array_equal(full["phase"], coarse["phase"])
        np.testing.assert_array_equal(full["valid"], coarse["valid"])

    def test_gsr_observed_sections(self):
        prediction = self.fit("gsr").predict(self.z)
        np.testing.assert_array_equal(
            prediction["phase"][self.obs], self.ref["phase"][self.obs]
        )
        np.testing.assert_allclose(
            prediction["q"][self.obs], self.ref["q"][self.obs], atol=1e-6
        )

    def test_exact_prediction_metrics(self):
        metrics, _ = evaluate(self.ref, self.ref, self.obs, self.syms)
        self.assertLess(metrics["angle_penalized_mean_deg"], 1e-4)
        self.assertEqual(metrics["grain_ari"], 1)
        self.assertEqual(metrics["phase_accuracy"], 1)

    def test_physical_grain_geometry(self):
        labels = np.ones((2, 3, 4), dtype=np.int32)
        domain = np.ones(labels.shape, bool)
        spacing = np.array([2.0, 3.0, 4.0])
        result = catalog(labels, labels, domain, spacing)
        self.assertEqual(result["volume_um3"][0], 24 * 24)
        expected = np.sort((np.array(labels.shape) * spacing) ** 2 / 12)[::-1]
        actual = [
            result[key][0]
            for key in ("lambda_major_um2", "lambda_middle_um2", "lambda_minor_um2")
        ]
        np.testing.assert_allclose(actual, expected)
        self.assertTrue(result["touches_roi_boundary"][0])

    def test_face_connectivity(self):
        phase = np.zeros((3, 3, 3), bool)
        phase[:, 1, 1] = True
        value = phase_metrics(phase, np.ones_like(phase), np.ones(3))
        self.assertTrue(value["spans_z"])
        self.assertFalse(value["spans_x"])
        diagonal = np.zeros_like(phase)
        diagonal[0, 0, 0] = diagonal[1, 1, 1] = True
        value = phase_metrics(diagonal, np.ones_like(phase), np.ones(3))
        self.assertEqual(value["largest_fraction_of_phase"], 0.5)

    def test_no_hidden_input(self):
        model = self.fit("gsr")
        before = model.predict(self.z)
        self.ref["q"][1:4] = [0, 1, 0, 0]
        after = model.predict(self.z)
        np.testing.assert_array_equal(before["q"], after["q"])

    def test_end_to_end_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            crops = root / "data" / "crops"
            crops.mkdir(parents=True)
            source = crops / "ti64_test.npz"
            np.savez_compressed(source, **self.ref)
            source.with_suffix(".json").write_text(
                json.dumps(
                    dict(
                        phase_symmetry=self.syms,
                        crop_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    )
                )
            )
            loaded, _ = load_region("ti64", "test", root / "data")
            np.testing.assert_array_equal(loaded["q"], self.ref["q"])
            options = dict(
                dataset="ti64",
                method="gsr",
                data_root=root / "data",
                output=root / "results",
                overrides=FAST,
            )
            out = run_experiment(**options)
            self.assertTrue((out / "metrics.json").exists())
            self.assertEqual(run_experiment(**options), out)
            with self.assertRaisesRegex(RuntimeError, "changed"):
                run_experiment(
                    **{**options, "overrides": {**FAST, "grain_min_match": 0.2}}
                )


if __name__ == "__main__":
    unittest.main()
