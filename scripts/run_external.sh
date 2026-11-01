set -euo pipefail

device="${1:-cpu}"
group="${2:-compact}"

case "$group" in
  develop|all)
    python -m material_gaussians.external develop --device "$device"
    ;;
esac

case "$group" in
  compact|all)
    for dataset in in100 in718; do
      for method in nearest sgf cgf tgs gsr; do
        python -m material_gaussians.external run --dataset "$dataset" --method "$method" --device cpu
      done
      for method in qrbsa polymicros; do
        for seed in 0 1 2; do
          python -m material_gaussians.external run --dataset "$dataset" --method "$method" --device "$device" --seed "$seed"
        done
      done
      python -m material_gaussians.external summary --dataset "$dataset"
    done
    ;;
esac

case "$group" in
  native|all)
    for method in nearest sgf cgf tgs gsr; do
      python -m material_gaussians.external run --dataset in718 --split native128 --method "$method" --device cpu
    done
    python -m material_gaussians.external run --dataset in718 --split native128 --method polymicros --device "$device"
    python -m material_gaussians.external summary --dataset in718 --split native128
    ;;
esac

case "$group" in
  legacy|all)
    for dataset in in100 in718; do
      for method in nearest sgf cgf tgs gsr; do
        python -m material_gaussians.external run --dataset "$dataset" --method "$method" --regime endpoint --device cpu
      done
      python -m material_gaussians.external run --dataset "$dataset" --method qrbsa --regime endpoint --device "$device"
      python -m material_gaussians.external run --dataset "$dataset" --method polymicros --regime endpoint --device "$device"
      python -m material_gaussians.external run --dataset "$dataset" --method qrbsa --regime disjoint --device "$device"
      python -m material_gaussians.external summary --dataset "$dataset" --regime endpoint
      python -m material_gaussians.external summary --dataset "$dataset" --regime disjoint
    done
    ;;
  compact|native|develop) ;;
  *) printf 'Unknown group: %s\n' "$group" >&2; exit 2 ;;
esac
