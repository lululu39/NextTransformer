"""Row-wise tangent projection for the initial velocity, with FP32 reductions."""
import torch
import triton
import triton.language as tl


@triton.jit
def _tangent(H, U, G, Y, DH, DU, D: tl.constexpr, BLOCK: tl.constexpr, BACKWARD: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    ix = row.to(tl.int64) * D + col
    h = tl.load(H + ix, col < D, 0.).to(tl.float32)
    u = tl.load(U + ix, col < D, 0.).to(tl.float32)
    norm_sq = tl.sum(h*h, 0) / D
    denom = tl.maximum(norm_sq, 1e-12)
    radial = (tl.sum(h*u, 0) / D) / denom
    if BACKWARD:
        g = tl.load(G + ix, col < D, 0.).to(tl.float32)
        g_radial = (tl.sum(g*h, 0) / D) / denom
        du = g - g_radial*h
        dh = -radial*g - g_radial*u + tl.where(norm_sq >= 1e-12, 2.*radial*g_radial*h, 0.)
        tl.store(DU + ix, du, col < D)
        tl.store(DH + ix, dh, col < D)
    else:
        tl.store(Y + ix, u - radial*h, col < D)


class TangentProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, u):
        h, u = h.contiguous(), u.contiguous()
        d = h.shape[-1]
        out = torch.empty_like(h)
        _tangent[(h.numel() // d,)](h, u, None, out, None, None, d, triton.next_power_of_2(d), False)
        ctx.save_for_backward(h, u)
        return out

    @staticmethod
    def backward(ctx, grad):
        h, u = ctx.saved_tensors
        d = h.shape[-1]
        dh, du = torch.empty_like(h), torch.empty_like(u)
        _tangent[(h.numel() // d,)](h, u, grad.contiguous(), None, dh, du, d, triton.next_power_of_2(d), True)
        return dh, du
