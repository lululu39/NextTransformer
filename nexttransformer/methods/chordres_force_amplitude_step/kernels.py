"""Fused force magnitude, softplus position step, retraction and tangent write."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from nexttransformer.methods.chordres_token_controls.kernels import _positive


@triton.jit
def _amplitude_write(
    H,
    V,
    F,
    BIAS,
    WEIGHT,
    BETA,
    NH,
    NV,
    GH,
    GV,
    DH,
    DV,
    DF,
    DP,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    BACKWARD: tl.constexpr,
    TERMINAL: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    idx = row.to(tl.int64) * D + lane
    h = tl.load(H + idx, lane < D, 0).to(tl.float32)
    v = tl.load(V + idx, lane < D, 0).to(tl.float32)
    force = tl.load(F + idx, lane < D, 0).to(tl.float32)
    f2 = tl.sum(force * force, 0) / D
    feature = libdevice.log1p(f2)
    amplitude_weight = tl.load(WEIGHT)
    alpha = _positive(tl.load(BIAS) + amplitude_weight * feature)
    beta = 0.5 if TERMINAL else tl.sigmoid(tl.load(BETA))
    fi = tl.rsqrt(tl.maximum(f2, 1e-12))
    target = force * fi
    delta = target - h
    mixed = h + alpha * delta
    m2 = tl.sum(mixed * mixed, 0) / D
    mi = tl.rsqrt(tl.maximum(m2, 1e-12))
    nxt = mixed * mi
    chord = nxt - h
    velocity = beta * v + (1.0 - beta) * chord
    radial = tl.sum(velocity * nxt, 0) / D
    if BACKWARD:
        gh = tl.load(GH + idx, lane < D, 0).to(tl.float32)
        gv = tl.load(GV + idx, lane < D, 0).to(tl.float32)
        dot = tl.sum(gv * nxt, 0) / D
        g_velocity = gv - dot * nxt
        g_chord = (1.0 - beta) * g_velocity
        g_nxt = gh + g_chord - radial * gv - dot * velocity
        g_mixed = mi * (
            g_nxt - tl.where(m2 >= 1e-12, nxt * (tl.sum(g_nxt * nxt, 0) / D), 0.0)
        )
        g_delta = alpha * g_mixed
        g_v = beta * g_velocity
        g_force = fi * (
            g_delta
            - tl.where(f2 >= 1e-12, target * (tl.sum(g_delta * target, 0) / D), 0.0)
        )
        grad_alpha = tl.sum(g_mixed * delta, 0)
        grad_raw = grad_alpha * -libdevice.expm1(-alpha)
        g_force += (2.0 * grad_raw * amplitude_weight / (D * (1.0 + f2))) * force
        tl.store(DH + idx, g_mixed - g_delta - g_chord, lane < D)
        tl.store(DV + idx, g_v, lane < D)
        tl.store(DF + idx, g_force, lane < D)
        tl.store(DP + 3 * row, grad_raw)
        tl.store(DP + 3 * row + 1, grad_raw * feature)
        if TERMINAL:
            tl.store(DP + 3 * row + 2, 0.0)
        else:
            tl.store(
                DP + 3 * row + 2,
                tl.sum(g_velocity * (v - chord), 0) * beta * (1.0 - beta),
            )
    else:
        tl.store(NH + idx, nxt, lane < D)
        tl.store(NV + idx, velocity - radial * nxt, lane < D)


@triton.jit
def _reduce_controls(PARTIAL, OUT, ROWS, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    k = tl.program_id(1)
    row = tile * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(PARTIAL + row.to(tl.int64) * 3 + k, row < ROWS, 0.0)
    tl.store(OUT + tile * 3 + k, tl.sum(values, 0))


class AmplitudeWrite(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, h, v, force, bias, weight, beta):
        h, v, force = h.contiguous(), v.contiguous(), force.contiguous()
        d = h.shape[-1]
        rows = h.numel() // d
        bptr = bias if beta is None else beta
        nh, nv = torch.empty_like(h), torch.empty_like(v)
        _amplitude_write[(rows,)](
            h,
            v,
            force,
            bias,
            weight,
            bptr,
            nh,
            nv,
            None,
            None,
            None,
            None,
            None,
            None,
            d,
            triton.next_power_of_2(d),
            False,
            beta is None,
        )
        ctx.save_for_backward(h, v, force, bias, weight, bptr)
        ctx.terminal = beta is None
        return nh, nv

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gh, gv):
        h, v, force, bias, weight, bptr = ctx.saved_tensors
        d = h.shape[-1]
        rows = h.numel() // d
        dh, dv, df = torch.empty_like(h), torch.empty_like(v), torch.empty_like(force)
        partial = bias.new_empty((rows, 3))
        _amplitude_write[(rows,)](
            h,
            v,
            force,
            bias,
            weight,
            bptr,
            None,
            None,
            gh.contiguous(),
            gv.contiguous(),
            dh,
            dv,
            df,
            partial,
            d,
            triton.next_power_of_2(d),
            True,
            ctx.terminal,
        )
        while rows > 1:
            block = min(1024, triton.next_power_of_2(rows))
            tiles = triton.cdiv(rows, block)
            out = bias.new_empty((tiles, 3))
            _reduce_controls[(tiles, 3)](partial, out, rows, block)
            partial, rows = out, tiles
        q = partial.view(3)
        return dh, dv, df, q[0:1], q[1:2], None if ctx.terminal else q[2:3]
