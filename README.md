# CGF · TGS · GSR

Sparse-section 3D reconstruction of crystal orientation and phase.

Python 3.12；以下命令均在解压后的 `material_gaussians/` 目录运行。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## 1. 数据下载

| 数据 | 下载地址 | 本包 ID |
|---|---|---|
| IN100 | [DREAM3D archive](https://github.com/BlueQuartzSoftware/simplnx/releases/download/Data_Archive/Small_IN100_dream3d_v3.tar.gz) | `in100` |
| IN718 | [UCSB author-processed volume](https://bisque2.ece.ucsb.edu/blob_service/00-C8rwSY7hvRF3HNpckTRM77) | `in718` |
| Ti–6Al–4V | [UCSB DREAM3D volume](https://bisque2.ece.ucsb.edu/blob_service/00-oiu4R8ixJjcEYe7k5dKvt3) | `ti64` |
| Zircaloy-4 + 7 wt% Nb | [Zenodo CTF archive](https://zenodo.org/records/3785085/files/ctf.zip?download=1) | `zrnb` |

```bash
python -m material_gaussians.data download all
python -m material_gaussians.data prepare all
```



## 2. 数据目录

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
    ├── in100_{test,validation,audit,fresh,compact}.npz
    ├── in718_{test,validation,audit,fresh,compact}.npz
    ├── ti64_{test,validation,audit,fresh,compact}.npz
    └── zrnb_{test,validation,audit,fresh,compact}.npz
```



## 3. 主实验

四材料、2×/4×/8×等间隔采样；方法为 CGF、TGS、GSR、Nearest、SGF，共 60 次运行：

```bash
bash scripts/run_main.sh
python -m material_gaussians summary
```

单独运行 Ti64 的 CGF，默认等间隔 4×、seed 0：

```bash
python -m material_gaussians run --dataset ti64 --method cgf
```

输出位于 `results/<dataset>/<method>/<condition>/`，包含 `manifest.json`、`prediction.npz`、`model_info.json`、`metrics.json` 和评价图数组；总表为 `results/summary.csv`。4×对应保留 `0,4,…,96`，共 25/97 层。`E90` 为错相/无效预测赋 90° 后的平均误差，并非第 90 百分位。

所有组支持 `--datasets ti64`、`--methods cgf tgs gsr`、`--dry-run`。同配置已完成结果会跳过；更改代码或参数后使用 `--tag rerun1` 或新的 `--output`。

## 4. 消融与补充实验

```bash
python -m material_gaussians suite ablation
python -m material_gaussians calibrate
python -m material_gaussians suite kernels
python -m material_gaussians suite sampling
python -m material_gaussians suite spatial
python -m material_gaussians suite repeats
python -m material_gaussians suite robustness
python -m material_gaussians suite baselines
python -m material_gaussians suite development
python -m material_gaussians summary
```

| 组 | 内容 |
|---|---|
| `ablation` | CGF 同一次拟合的回归开/关；TGS 无位移/无端点混合；GSR 无匹配/无区内取向平滑/距离支持；SGF 初始化、各向同性、两种均值解码、无边界项 |
| `calibrate` → `kernels` | 在开发区选择距离尺度 0.5/1/2；主区域与 `fresh` 上比较 Gaussian、Laplace、校准距离支持 |
| `sampling` | 不等间隔、中央区段缺失、等预算分散缺失 |
| `spatial` | `audit` 空间留出区域 |
| `repeats` | SGF/CGF 额外拟合 seed 1、2 |
| `robustness` | `fresh` 上三种不等间隔掩码、±1 像素偏移、±10% 深度扰动，拟合 seed 固定 0 |
| `baselines` | SLERP 2×/4×/8×与 SIREN 4× |
| `development` | 正文附录的 15 种候选配置 × 4 材料；仅使用 `validation` |

缺少重新校准的结果时，距离对照使用正文冻结尺度 `1.0`。`configs/base.json`、`tgs.json`、`gsr.json` 为固定主参数；`--config path/to/config.json` 可覆盖参数。`suite all` 运行上述所有模型组；校准和后处理仍使用各自命令。

对已保存预测做物理晶粒尺寸/形状、相连通性、2°/5°/10°敏感性及统一观测层覆盖控制：

```bash
python -m material_gaussians analyze results --anchor both
```

结果分别位于运行目录的 `analysis_native/` 和 `analysis_anchored/`，含总体统计、一对一晶粒匹配、未匹配数量及原始 CSV。可将 `results` 换成某个运行目录。

连接错误案例及等预算比较（先运行 `main`、`ablation`、`sampling`）：

```bash
python -m material_gaussians compare results/ti64/tgs/test_regular_4x_m0_s0_full results/ti64/gsr/test_regular_4x_m0_s0_full --output results/cases/ti64
python -m material_gaussians compare results/zrnb/tgs/test_regular_4x_m0_s0_no_motion results/zrnb/tgs/test_regular_4x_m0_s0_full --output results/cases/zrnb
for ds in in100 in718 ti64 zrnb; do
  for method in cgf tgs gsr; do
    python -m material_gaussians compare "results/$ds/$method/test_block_4x_m0_s0_full" "results/$ds/$method/test_dispersed_4x_m0_s0_full" --mode budget --output "results/budget/$ds/$method"
  done
done
```

Q-RBSA / PolyMicros 使用独立依赖与固定上游版本，仅用于立方晶系 IN100/IN718：

```bash
python -m pip install -r requirements-external.txt
python -m material_gaussians.external fetch
bash scripts/run_external.sh cpu all
```


