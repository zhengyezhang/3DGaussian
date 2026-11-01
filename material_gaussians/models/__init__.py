import json
from pathlib import Path

METHODS = ("nearest", "slerp", "siren", "sgf", "cgf", "tgs", "gsr")
VARIANTS = {
    "nearest": ("full",),
    "slerp": ("full",),
    "siren": ("full",),
    "sgf": ("full", "initial", "isotropic", "mean", "symmetry_mean", "no_boundary"),
    "cgf": ("full", "paired", "laplace"),
    "tgs": ("full", "no_motion", "no_endpoint_mixing", "laplace"),
    "gsr": ("full", "no_matching", "no_smoothing", "distance", "calibrated_distance"),
}


def model_config(method, variant="full", overrides=None):
    if method not in VARIANTS or variant not in VARIANTS[method]:
        raise ValueError(f"Unsupported method/variant: {method}/{variant}")
    folder = Path(__file__).resolve().parents[2] / "configs"
    config = json.loads((folder / "base.json").read_text())
    if method in ("tgs", "gsr"):
        config.update(json.loads((folder / f"{method}.json").read_text()))
    changes = {
        "initial": dict(iterations=0, fit_epochs=0),
        "isotropic": dict(anisotropic=False),
        "mean": dict(decode="arithmetic"),
        "symmetry_mean": dict(decode="symmetrymean"),
        "no_boundary": dict(boundary_weight=0.0),
        "no_motion": dict(transport_motion_scale=0.0),
        "no_endpoint_mixing": dict(transport_merge_gate_deg=0.0),
        "no_matching": dict(grain_min_match=1.1),
        "no_smoothing": dict(grain_q_sigma_pixels=0.0),
        "distance": dict(grain_shape="distance"),
        "calibrated_distance": dict(grain_shape="distance", distance_scale=1.0),
    }
    config.update(changes.get(variant, {}))
    config.update(overrides or {})
    return config


def build_model(method, variant="full"):
    if method not in VARIANTS or variant not in VARIANTS[method]:
        raise ValueError(f"Unsupported method/variant: {method}/{variant}")
    from .baselines import SliceInterpolation, SirenOrientationField
    from .cgf import GaussianMaterialField
    from .sgf import GaussianOrientationField
    from .tgs import GaussianTransportField
    from .gsr import GaussianGrainField
    from .controls import (
        LaplaceMaterialField,
        LaplaceTransportField,
        CalibratedDistanceField,
    )

    if method in ("nearest", "slerp"):
        return SliceInterpolation(method)
    if variant == "laplace":
        return {"cgf": LaplaceMaterialField, "tgs": LaplaceTransportField}[method]()
    if variant == "calibrated_distance":
        return CalibratedDistanceField()
    return {
        "siren": SirenOrientationField,
        "sgf": GaussianOrientationField,
        "cgf": GaussianMaterialField,
        "tgs": GaussianTransportField,
        "gsr": GaussianGrainField,
    }[method]()
