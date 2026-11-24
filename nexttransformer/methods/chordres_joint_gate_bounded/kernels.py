"""Fused bounded joint-RMS gate reads and their VJPs. Chord writes consume the unscaled velocity."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from nexttransformer.methods.chordres_token_controls.kernels import (
    _finish_parameters,
    _project,
)
from nexttransformer.methods.chordres_token_controls_joint_rms.kernels import (
    _gate_partials,
)
from . import GATE_EPS, INITIAL_ALPHA


@triton.jit
def _bounded_step(x, initial: tl.constexpr):
    pivot = tl.maximum(x, 0.0)
    numerator = initial * libdevice.exp(x - pivot)
    complement = (1.0 - initial) * libdevice.exp(-pivot)
    return libdevice.div_rn(numerator, numerator + complement)


@triton.jit
def _token_read_forward(
    H,
    V,
    W,
    MU,
    STEPS,
    BETA,
    GAINS,
    Q,
    C,
    ROWS,
    D: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    EPS: tl.constexpr,
    ALPHA0: tl.constexpr,
):
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    lane = tl.arange(0, BD)
    ix = row[:, None].to(tl.int64) * D + lane[None, :]
    mask = (row[:, None] < ROWS) & (lane[None, :] < D)
    h = tl.load(H + ix, mask, 0.0).to(tl.float32)
    v = tl.load(V + ix, mask, 0.0).to(tl.float32)
    gate_inv = tl.rsqrt(tl.sum(h * h + v * v, 1) / (2 * D) + EPS)
    raw_mu = tl.load(MU) + gate_inv * _project(h, v, W, lane, D, 0)
    mu = 3.0 * libdevice.tanh(raw_mu / 3.0)
    ap = _bounded_step(
        tl.load(STEPS) + gate_inv * _project(h, v, W, lane, D, 1), ALPHA0
    )
    al = _bounded_step(
        tl.load(STEPS + 1) + gate_inv * _project(h, v, W, lane, D, 2), ALPHA0
    )
    if K == 6:
        beta = tl.sigmoid(tl.load(BETA) + gate_inv * _project(h, v, W, lane, D, 3))
        gp = 2.0 * tl.sigmoid(tl.load(GAINS) + gate_inv * _project(h, v, W, lane, D, 4))
        gt = 2.0 * tl.sigmoid(
            tl.load(GAINS + 1) + gate_inv * _project(h, v, W, lane, D, 5)
        )
    else:
        beta, gp, gt = 0.5, 0.5, 0.5
    dest = C + row.to(tl.int64) * 6
    tl.store(dest, mu, row < ROWS)
    tl.store(dest + 1, ap, row < ROWS)
    tl.store(dest + 2, al, row < ROWS)
    tl.store(dest + 3, beta, row < ROWS)
    tl.store(dest + 4, gp, row < ROWS)
    tl.store(dest + 5, gt, row < ROWS)
    off = mu[:, None] * v
    inv = tl.rsqrt(1.0 + tl.sum(off * off, 1) / D)
    q = ((h + off) * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    scale = 1.0 / tl.sqrt(tl.sum(q * q, 1) / D + 1.1920928955078125e-7)
    tl.store(Q + ix, q * scale[:, None], mask)


@triton.jit
def _token_read_backward(
    H,
    V,
    W,
    C,
    DQ,
    DC,
    DH,
    DV,
    DZ,
    ROWS,
    D: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    lane = tl.arange(0, BD)
    ix = row[:, None].to(tl.int64) * D + lane[None, :]
    mask = (row[:, None] < ROWS) & (lane[None, :] < D)
    h = tl.load(H + ix, mask, 0.0).to(tl.float32)
    v = tl.load(V + ix, mask, 0.0).to(tl.float32)
    gate_inv = tl.rsqrt(tl.sum(h * h + v * v, 1) / (2 * D) + EPS)
    offc = row.to(tl.int64) * 6
    mu = tl.load(C + offc, row < ROWS, 0.0)
    off = mu[:, None] * v
    inv = tl.rsqrt(1.0 + tl.sum(off * off, 1) / D)
    raw = h + off
    q = (raw * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    scale = 1.0 / tl.sqrt(tl.sum(q * q, 1) / D + 1.1920928955078125e-7)
    normed = q * scale[:, None]
    dy = tl.load(DQ + ix, mask, 0.0).to(tl.float32)
    dq = (
        ((dy - normed * (tl.sum(dy * normed, 1) / D)[:, None]) * scale[:, None])
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    direct = dq * inv[:, None]
    coeff = tl.sum(dq * raw, 1) / D * inv * inv * inv
    d_off = direct - off * coeff[:, None]
    dh = direct
    dv = mu[:, None] * d_off
    grad_gate_h = tl.full((BT, BD), 0.0, tl.float32)
    grad_gate_v = tl.full((BT, BD), 0.0, tl.float32)
    query_mu_grad = tl.sum(v * d_off, 1)
    grad_row = DZ + row.to(tl.int64) * (K + 1)
    tl.store(grad_row + K, gate_inv, row < ROWS)
    for k in tl.static_range(K):
        g = tl.load(DC + offc + k, row < ROWS, 0.0)
        if k == 0:
            # Both the packed control and the physical lookahead depend on raw_mu.
            g = (g + query_mu_grad) * (1.0 - (mu / 3.0) * (mu / 3.0))
        elif k < 4:
            control = tl.load(C + offc + k, row < ROWS, 0.0)
            g *= control * (1.0 - control)
        else:
            gamma = tl.load(C + offc + k, row < ROWS, 0.0)
            g *= gamma * (1.0 - 0.5 * gamma)
        tl.store(grad_row + k, g, row < ROWS)
        wh = tl.load(W + 2 * k * D + lane, lane < D, 0.0)
        wv = tl.load(W + (2 * k + 1) * D + lane, lane < D, 0.0)
        grad_gate_h += g[:, None] * wh[None, :]
        grad_gate_v += g[:, None] * wv[None, :]
    # Both states share the radial component of the same 2D-dimensional VJP.
    gate_h = h * gate_inv[:, None]
    gate_v = v * gate_inv[:, None]
    radial = tl.sum(grad_gate_h * gate_h + grad_gate_v * gate_v, 1) / (2 * D)
    dh += gate_inv[:, None] * (grad_gate_h - radial[:, None] * gate_h)
    dv += gate_inv[:, None] * (grad_gate_v - radial[:, None] * gate_v)
    tl.store(DH + ix, dh, mask)
    tl.store(DV + ix, dv, mask)


class TokenRead(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, h, v, weight, mu, steps, beta, gains):
        h, v = h.contiguous(), v.contiguous()
        dim = h.shape[-1]
        count = weight.shape[0]
        rows = h.numel() // dim
        query = torch.empty_like(h)
        controls = weight.new_empty((*h.shape[:-1], 6))
        _token_read_forward[(triton.cdiv(rows, 4),)](
            h,
            v,
            weight,
            mu,
            steps,
            beta if beta is not None else mu,
            gains if gains is not None else mu,
            query,
            controls,
            rows,
            dim,
            count,
            4,
            triton.next_power_of_2(dim),
            GATE_EPS,
            INITIAL_ALPHA,
        )
        ctx.save_for_backward(h, v, weight, controls)
        return query, controls

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gquery, gcontrols):
        h, v, weight, controls = ctx.saved_tensors
        dim = h.shape[-1]
        count = weight.shape[0]
        rows = h.numel() // dim
        dh, dv = torch.empty_like(h), torch.empty_like(v)
        dz = weight.new_empty((rows, count + 1))
        tile = min(4, max(1, 4096 // triton.next_power_of_2(dim)))
        _token_read_backward[(triton.cdiv(rows, tile),)](
            h,
            v,
            weight,
            controls,
            gquery.contiguous(),
            gcontrols.contiguous(),
            dh,
            dv,
            dz,
            rows,
            dim,
            count,
            tile,
            triton.next_power_of_2(dim),
            GATE_EPS,
        )
        dw = torch.empty_like(weight)
        dm = weight.new_empty(1)
        ds = weight.new_empty(2)
        db = weight.new_empty(1) if count == 6 else None
        dg = weight.new_empty(2) if count == 6 else None
        chunk = 256
        cols = 64
        splits = triton.cdiv(rows, chunk)
        pw = weight.new_empty((splits, count, 2, dim))
        pb = weight.new_empty((splits, count))
        _gate_partials[(triton.cdiv(dim, cols), 2, splits)](
            h, v, dz, pw, pb, rows, dim, count, chunk, 32, cols
        )
        _finish_parameters[(triton.cdiv(dim, cols), 2, count)](
            pw,
            pb,
            dw,
            dm,
            ds,
            db if db is not None else dm,
            dg if dg is not None else dm,
            splits,
            dim,
            count,
            triton.next_power_of_2(splits),
            cols,
        )
        return dh, dv, dw, dm, ds, db, dg
