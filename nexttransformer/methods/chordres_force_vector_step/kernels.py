"""Fused full-vector step projection, spherical retraction and tangent write."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from nexttransformer.methods.chordres_token_controls.kernels import _positive


@triton.jit
def _vector_write(
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
    weight = tl.load(WEIGHT + lane, lane < D, 0.0)
    alpha = _positive(tl.load(BIAS) + tl.sum(weight * force, 0))
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
        g_force += grad_raw * weight
        tl.store(DH + idx, g_mixed - g_delta - g_chord, lane < D)
        tl.store(DV + idx, g_v, lane < D)
        tl.store(DF + idx, g_force, lane < D)
        tl.store(DP + 2 * row, grad_raw)
        if TERMINAL:
            tl.store(DP + 2 * row + 1, 0.0)
        else:
            tl.store(
                DP + 2 * row + 1,
                tl.sum(g_velocity * (v - chord), 0) * beta * (1.0 - beta),
            )
    else:
        tl.store(NH + idx, nxt, lane < D)
        tl.store(NV + idx, velocity - radial * nxt, lane < D)


@triton.jit
def _parameter_partials(
    F, DP, OUT, ROWS, D: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr
):
    tile, split = tl.program_id(0), tl.program_id(1)
    row = split * BR + tl.arange(0, BR)
    col = tile * BC + tl.arange(0, BC)
    f = tl.load(
        F + row[:, None].to(tl.int64) * D + col[None, :],
        (row[:, None] < ROWS) & (col[None, :] < D),
        0.0,
    ).to(tl.float32)
    dz = tl.load(DP + 2 * row, row < ROWS, 0.0)
    dst = OUT + split.to(tl.int64) * (D + 2)
    tl.store(dst + col, tl.sum(dz[:, None] * f, 0), col < D)
    if tile == 0:
        dbeta = tl.load(DP + 2 * row + 1, row < ROWS, 0.0)
        tl.store(dst + D, tl.sum(dz, 0))
        tl.store(dst + D + 1, tl.sum(dbeta, 0))


@triton.jit
def _reduce_partials(
    X, OUT, ROWS, COLS: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr
):
    tile, split = tl.program_id(0), tl.program_id(1)
    row = split * BR + tl.arange(0, BR)
    col = tile * BC + tl.arange(0, BC)
    x = tl.load(
        X + row[:, None].to(tl.int64) * COLS + col[None, :],
        (row[:, None] < ROWS) & (col[None, :] < COLS),
        0.0,
    )
    tl.store(OUT + split.to(tl.int64) * COLS + col, tl.sum(x, 0), col < COLS)


class VectorWrite(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, h, v, force, bias, weight, beta):
        h, v, force = h.contiguous(), v.contiguous(), force.contiguous()
        d = h.shape[-1]
        rows = h.numel() // d
        bptr = bias if beta is None else beta
        nh, nv = torch.empty_like(h), torch.empty_like(v)
        _vector_write[(rows,)](
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
        partial = bias.new_empty((rows, 2))
        _vector_write[(rows,)](
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
        cols = d + 2
        splits = triton.cdiv(rows, 64)
        sums = weight.new_empty((splits, cols))
        _parameter_partials[(triton.cdiv(d, 64), splits)](
            force, partial, sums, rows, d, 64, 64
        )
        while splits > 1:
            block = min(triton.next_power_of_2(splits), 128)
            count = triton.cdiv(splits, block)
            out = weight.new_empty((count, cols))
            _reduce_partials[(triton.cdiv(cols, 32), count)](
                sums, out, splits, cols, block, 32
            )
            sums, splits = out, count
        result = sums.view(cols)
        return (
            dh,
            dv,
            df,
            result[d : d + 1],
            result[:d],
            None if ctx.terminal else result[d + 1 : d + 2],
        )
