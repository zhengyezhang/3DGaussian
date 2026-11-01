from __future__ import annotations
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch

BASE = Path("third_party")


def qrbsa_model(scale: int, device: str = "cpu", symmetry: str = "cubic"):
    sys.path.insert(0, str(BASE / "Q-RBSA"))
    from model.qrbsa_1d import QRBSA_1D

    args = SimpleNamespace(n_resblocks=10, n_feats=128, n_colors=4, scale=scale)
    model = QRBSA_1D(args).to(device)
    model.qrbsa_symmetry = symmetry
    return model


def qrbsa_pixel_loss(symmetry: str):
    sys.path.insert(0, str(BASE / "Q-RBSA"))
    from .stable_activation import install

    install()
    from mat_sci_torch_quats.losses import ActAndLoss, Loss
    from mat_sci_torch_quats.symmetries import fcc_syms, hcp_syms

    syms = {"cubic": fcc_syms, "hcp": hcp_syms, "hexagonal": hcp_syms}[symmetry]
    return ActAndLoss("tanhc", Loss("rot_dist_approx", syms), quat_dim=1)


@torch.no_grad()
def qrbsa_fundamental_zone(q_wxyz: np.ndarray, symmetry: str):
    sys.path.insert(0, str(BASE / "Q-RBSA"))
    from mat_sci_torch_quats.quats import fz_reduce
    from mat_sci_torch_quats.symmetries import fcc_syms, hcp_syms

    syms = {"cubic": fcc_syms, "hcp": hcp_syms, "hexagonal": hcp_syms}[symmetry]
    q = np.asarray(q_wxyz, np.float32)
    shape = q.shape
    flat = q.reshape(-1, 4)
    pieces = []
    for start in range(0, len(flat), 32768):
        pieces.append(
            fz_reduce(
                torch.from_numpy(flat[start : start + 32768].copy()), syms
            ).numpy()
        )
    return np.concatenate(pieces).reshape(shape)


def fit_qrbsa_observed(
    observed_wxyz: np.ndarray,
    valid: np.ndarray,
    symmetry: str,
    scale: int = 4,
    device: str = "cpu",
    steps: int = 500,
    patch_size: int = 32,
    batch_size: int = 4,
    seed: int = 0,
    model=None,
    progress=None,
    transpose_patches: bool = True,
):
    q = np.asarray(observed_wxyz, np.float32)
    good = np.asarray(valid, bool)
    if q.shape[:-1] != good.shape or q.shape[-1] != 4:
        raise ValueError("Expected [K,Y,X,4] wxyz and [K,Y,X] valid")
    q = qrbsa_fundamental_zone(q, symmetry)
    p = min(patch_size, q.shape[1], q.shape[2])
    p = p // scale * scale
    if p < scale:
        raise ValueError("Observed plane is too small")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = qrbsa_model(scale, device, symmetry) if model is None else model.to(device)
    model.qrbsa_symmetry = symmetry
    model.train()
    loss_fn = qrbsa_pixel_loss(symmetry)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.99))
    history = []
    for step in range(steps):
        targets, masks = ([], [])
        for _ in range(batch_size):
            for attempt in range(100):
                k = rng.integers(q.shape[0])
                y = rng.integers(q.shape[1] - p + 1)
                x = rng.integers(q.shape[2] - p + 1)
                target = q[k, y : y + p, x : x + p].copy()
                mask = good[k, y : y + p, x : x + p].copy()
                if mask.mean() >= 0.5:
                    break
            else:
                raise ValueError("No patch with >=50% valid phase pixels")
            if transpose_patches and rng.random() < 0.5:
                target = target.transpose(1, 0, 2)
                mask = mask.T
            targets.append(np.roll(target, -1, axis=-1))
            masks.append(mask)
        target = torch.tensor(np.stack(targets), device=device).permute(0, 3, 1, 2)
        mask = torch.tensor(np.stack(masks), device=device)
        pred = model(target[:, :, ::scale, :])
        loss = loss_fn(pred, target)[mask].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
        if progress is not None:
            progress(step + 1, history[-1])
    model.eval()
    return (model, history)


@torch.no_grad()
def predict_qrbsa(
    model,
    observed_wxyz: np.ndarray,
    output_depth: int,
    batch_size: int = 1,
    symmetry: str | None = None,
):
    from mat_sci_torch_quats.quats import fz_reduce
    from mat_sci_torch_quats.symmetries import fcc_syms, hcp_syms

    symmetry = symmetry or model.qrbsa_symmetry
    syms = {"cubic": fcc_syms, "hcp": hcp_syms, "hexagonal": hcp_syms}[symmetry]
    q = np.roll(qrbsa_fundamental_zone(observed_wxyz, symmetry), -1, axis=-1)
    device = next(model.parameters()).device
    pieces = []
    for y in range(0, q.shape[1], batch_size):
        batch = torch.tensor(q[:, y : y + batch_size], device=device).permute(
            1, 3, 0, 2
        )
        pred = model(batch).permute(2, 0, 3, 1)[:output_depth]
        pred = torch.nn.functional.normalize(pred, dim=-1)
        pred = torch.roll(pred, 1, dims=-1)
        pred = fz_reduce(pred, syms)
        pieces.append(pred.cpu().numpy())
    return np.concatenate(pieces, axis=1)


def rogsh_converter():
    sys.path.insert(0, str(BASE / "PolyMicros"))
    from ROGSHUtils import ROGSH432

    return ROGSH432(str(BASE / "PolyMicros" / "GSHTrees") + "/")


def quaternion_to_rogsh(q_wxyz: np.ndarray, converter=None):
    from scipy.spatial.transform import Rotation

    q = np.asarray(q_wxyz)
    e = Rotation.from_quat(q.reshape(-1, 4), scalar_first=True).as_euler("ZXZ")
    gsh = converter or rogsh_converter()
    return gsh.ROGSH(e).reshape(q.shape[:-1] + (3,))


def load_polymicros(edge: int = 32, device: str = "cpu"):
    sys.path.insert(0, str(BASE / "PolyMicros"))
    from Unet import UnetND
    from diffusion import Diffusion

    cp = BASE / "checkpoints" / "PolyMicros"
    config = json.loads((cp / "config.json").read_text())
    if config.get("periodic"):
        raise ValueError("This adapter expects the released nonperiodic model")
    unet = UnetND(config["udim"], dim_mults=tuple(config["channels"]), channels=3)
    unet.sample_size, unet.in_channels, unet.dims = (edge, 3, 3)
    diffusion = Diffusion(
        unet,
        bins_max=config["sample_steps"],
        samples_path=str(BASE / "pilot" / "samples"),
    )
    ckpt = torch.load(
        cp / "PolyMicros.ckpt", map_location="cpu", weights_only=True, mmap=True
    )
    diffusion.load_state_dict(ckpt["state_dict"], strict=True)
    del ckpt
    return diffusion.to(device).eval()


@torch.inference_mode()
def sample_polymicros(
    model,
    observed_rogsh: np.ndarray,
    observed_mask: np.ndarray,
    steps: int = 100,
    seed: int = 0,
    schedule: str = "paper75",
):
    r = np.asarray(observed_rogsh, np.float32)
    m = np.asarray(observed_mask, bool)
    if r.shape[:-1] != m.shape or r.shape[-1] != 3 or len(set(m.shape)) != 1:
        raise ValueError("PolyMicros adapter requires a cubic [E,E,E,3] ROGSH field")
    if m.shape[0] != model.image_size:
        raise ValueError("Input edge differs from initialized model")
    values = np.zeros_like(r)
    values[m] = r[m]
    device = model.device
    values = torch.tensor(values, device=device).permute(3, 0, 1, 2)[None]
    mask = torch.tensor(m, device=device)[None, None].expand_as(values)
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    call = 0
    n_conditioned = int(np.ceil(0.75 * (steps - 1)))

    def condition(images):
        nonlocal call
        if schedule == "release" or call < n_conditioned:
            images[mask] = values[mask]
        call += 1
        return images

    if schedule not in {"paper75", "release"}:
        raise ValueError(schedule)
    result = model.sample(steps=steps, cond_fn=condition)
    if schedule == "release":
        result = model.sample(steps=steps, skip=int(0.75 * steps), images=result)
    if device.type == "mps":
        torch.mps.synchronize()
    return result[0].permute(1, 2, 3, 0).cpu().numpy()
