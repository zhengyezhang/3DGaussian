set -eu
python -m material_gaussians suite ablation "$@"
python -m material_gaussians suite kernels "$@"
