import torch


def stable_tanhc(x):
    eps = 0.05
    mask = (torch.abs(x) < eps).float()
    x_clip = torch.clamp(abs(x), min=eps)
    t = torch.clamp(x, min=-eps, max=eps)
    output_ts = 1 - t**2 / 3 + 2 * t**4 / 15 - 17 * t**6 / 315
    output_ht = torch.tanh(x_clip) / x_clip
    return mask * output_ts + (1 - mask) * output_ht


def install():
    import mat_sci_torch_quats.losses as losses

    losses.tanhc = stable_tanhc
