# CGF · TGS · GSR

Sparse-section 3D reconstruction of crystal orientation and phase.

## Environment

Python 3.12 and pip are required; Git is needed for external baselines. Main experiments run on CPU. 

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data

| Dataset | Download | ID |
|---|---|---|
| IN100 | [DREAM3D](https://github.com/BlueQuartzSoftware/simplnx/releases/download/Data_Archive/Small_IN100_dream3d_v3.tar.gz) | `in100` |
| IN718 | [UCSB](https://bisque2.ece.ucsb.edu/blob_service/00-C8rwSY7hvRF3HNpckTRM77) | `in718` |
| Ti–6Al–4V | [UCSB](https://bisque2.ece.ucsb.edu/blob_service/00-oiu4R8ixJjcEYe7k5dKvt3) | `ti64` |
| Zircaloy-4 + 7 wt% Nb | [Zenodo](https://zenodo.org/records/3785085/files/ctf.zip?download=1) | `zrnb` |

```bash
python -m material_gaussians.data download all
python -m material_gaussians.data prepare all
```

Replace `all` with a dataset ID to process one dataset. These commands create:

```text
data/
├── raw/
│   ├── in100_processed/Small_IN100_dream3d_v3.tar.gz
│   ├── in718_mirror/{EulerAngles,Phases,FeatureIds}.bin
│   ├── ti64/Ti_64_equiaxed_6_15_2017_final.dream3d
│   └── zrnb/ctf.zip
├── processed/
│   ├── in100_registered.npz
│   ├── in718.npz
│   ├── ti64.npz
│   └── zrnb_registered.npz
└── crops/
    └── {dataset}_{test,validation,audit,fresh,compact}.npz
```

Each NPZ has a matching JSON file. Preparation includes reference-stack registration for IN100 and Zircaloy/Nb, followed by cropping.

## Main experiments

Run all four materials at 2×, 4× and 8× regular sampling with CGF, TGS, GSR, Nearest and SGF:

```bash
bash scripts/run_main.sh
python -m material_gaussians summary
```

Run a single method at the default 4× sampling:

```bash
python -m material_gaussians run --dataset ti64 --method cgf
```

Predictions and metrics are saved under `results/<dataset>/<method>/<condition>/`; the combined table is `results/summary.csv`. Completed runs are resumed. Use a new `--tag` or `--output` when changing parameters.

## Ablations and additional experiments

```bash
python -m material_gaussians suite ablation
python -m material_gaussians calibrate
python -m material_gaussians suite kernels
for group in sampling spatial robustness baselines development; do
  python -m material_gaussians suite "$group"
done
python -m material_gaussians analyze results --anchor both
python -m material_gaussians summary
```

| Group | Experiment |
|---|---|
| `ablation` | CGF regression; TGS displacement and endpoint mixing; GSR matching, orientation smoothing and support; SGF representation controls |
| `calibrate` → `kernels` | Development-set distance calibration and spatial-kernel comparisons |
| `sampling` | Irregular, missing-block and equal-budget dispersed sampling |
| `spatial` | Spatial holdout |
| `robustness` | Observation-mask, registration and section-depth perturbations |
| `baselines` | SLERP and SIREN |
| `development` | Fifteen candidate configurations on development regions |

`analyze` evaluates grain geometry, phase connectivity and 2°/5°/10° thresholds, with and without observed-section anchoring. Parameters are stored in `configs/`; use `--config` to supply overrides. Experiment suites support `--datasets`, `--methods` and `--dry-run`.

## External baselines

Q-RBSA and PolyMicros comparisons use IN100 and IN718:

```bash
python -m pip install -r requirements-external.txt
python -m material_gaussians.external fetch
bash scripts/run_external.sh cpu all
```

Use `compact` instead of `all` for the main comparison only. Replace `cpu` with `mps` or `cuda` when available.
