import types, math
from functools import partial
from einops import reduce
import torch
from torch.nn import functional as F


def forward(self, x):
    if x.device.type == "mps" and math.prod(x.shape[-3:]) >= self._v5_min_voxels:
        torch.mps.synchronize()
        torch.mps.empty_cache()
    weight = self.weight
    if self._v5_weight_standardized:
        eps = 1e-05 if x.dtype == torch.float32 else 0.001
        mean = reduce(weight, "o ... -> o 1 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1 1", partial(torch.var, unbiased=False))
        weight = (weight - mean) / (var + eps).rsqrt()
    if math.prod(x.shape[-3:]) < self._v5_min_voxels:
        return F.conv3d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
    z, y, w = x.shape[-3:]
    kz, ky, kx = self.kernel_size
    pz, py, px = self.padding
    oz, oy, ox = (z + 2 * pz - kz + 1, y + 2 * py - ky + 1, w + 2 * px - kx + 1)
    out = torch.empty(
        (x.shape[0], self.out_channels, oz, oy, ox), dtype=x.dtype, device=x.device
    )
    for a in range(0, oz, 16):
        b = min(a + 16, oz)
        lo = a - pz
        hi = b - pz + kz - 1
        tile = x[:, :, max(0, lo) : min(z, hi)]
        if lo < 0 or hi > z:
            tile = F.pad(tile, (0, 0, 0, 0, max(0, -lo), max(0, hi - z)))
        part = F.conv3d(
            tile, weight, self.bias, padding=(0, py, px), groups=self.groups
        )
        out[:, :, a:b] = part
        del part, tile
        if x.device.type == "mps":
            torch.mps.synchronize()
    return out


def install(model, min_voxels=128**3):
    from Unet import WeightStandardizedConv3d

    count = 0
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv3d):
            assert (
                layer.stride == (1, 1, 1)
                and layer.dilation == (1, 1, 1)
                and (layer.padding_mode == "zeros")
            )
            assert type(layer) in (torch.nn.Conv3d, WeightStandardizedConv3d)
            layer._v5_weight_standardized = isinstance(layer, WeightStandardizedConv3d)
            layer._v5_min_voxels = min_voxels
            layer.forward = types.MethodType(forward, layer)
            count += 1
    return count
