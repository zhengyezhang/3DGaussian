import types
import torch
from torch.nn import functional as F


def forward(self, x):
    b, _, h, w, d = x.shape
    weights = self.to_qkv.weight.chunk(3, dim=0)
    hidden = weights[0].shape[0]
    q = F.conv3d(x, weights[0]).reshape(b, self.heads, hidden // self.heads, -1)
    q = q.softmax(dim=-2) * self.scale
    k = F.conv3d(x, weights[1]).reshape(b, self.heads, hidden // self.heads, -1)
    k = k.softmax(dim=-1)
    v = F.conv3d(x, weights[2]).reshape(b, self.heads, hidden // self.heads, -1)
    context = torch.einsum("b h d n,b h e n->b h d e", k, v)
    del k, v
    out = torch.einsum("b h d e,b h d n->b h e n", context, q)
    del context, q
    out = out.reshape(b, hidden, h, w, d)
    return self.to_out(out)


def install(model):
    from Unet import LinearAttention

    count = 0
    for layer in model.modules():
        if isinstance(layer, LinearAttention):
            layer.forward = types.MethodType(forward, layer)
            count += 1
    return count
