# External comparisons

- [Q-RBSA](https://github.com/UCSB-VRL/Q-RBSA): Jangid et al., *High-Resolution 3D EBSD Map Generation Using an Efficient Quaternion Transformer Network*. Revision `2203058b2c6aa7d246448d7749f2fb2047c7cb12`.
- [PolyMicros](https://github.com/Zraxix/PolyMicros): Buzzy et al., *PolyMicros: Bootstrapping a Foundation Model for Polycrystalline Material Structure*. Revision `9fb97e16d01565f99c08491fa9c2efdd40872993`.

Upstream code and weights are downloaded separately into `third_party/`. Their authors retain ownership; consult the upstream repositories for usage terms. This package contains local adapters and a device-compatibility patch, not the upstream implementations.

```bash
pip install -r requirements-external.txt
python -m material_gaussians.external fetch
bash scripts/run_external.sh cpu compact
bash scripts/run_external.sh cpu develop
bash scripts/run_external.sh cpu native
bash scripts/run_external.sh cpu legacy
```

Replace `cpu` with `mps` or `cuda` for external networks. Core comparisons remain on CPU. `all` runs development, compact, native-size and earlier controls. Development evaluates 1,000/5,000/10,000 Q-RBSA updates along one trajectory per development region. The selected shared budget is saved to `results/external/qrbsa_budget.json`; without that file the frozen 10,000-update setting is used.

The equal-input comparison uses 16 planes, `0,4,...,60`, in each 64³ cube. Q-RBSA learns from acquired XY patches; this is an adaptation of its published training regime. PolyMicros uses the released cubic ROGSH checkpoint and the first-75%-of-updates conditioning schedule. Its IN718 evaluation shares the pretraining source. HCP phases are not recoded as cubic. Native-size IN718 uses 128³ and planes through 124.

`summary` intersects valid outputs across completed runs in the selected dataset/split/regime. Descriptor errors use that common support; angular scores retain all valid hidden reference voxels with a 90° invalid-output penalty. No angle is inferred from PolyMicros's non-unique ROGSH representation. Earlier `endpoint` runs include plane 63 in conditioning or Q-RBSA training, but not Q-RBSA inference; `disjoint` trains Q-RBSA on a separate dense block. These runs remain separate from the equal-input comparison.

The Q-RBSA activation patch bounds only the inactive Taylor branch. The optional 128³ execution changes split attention projections and compute high-resolution convolutions in slabs while retaining released equations, weights and precision.
