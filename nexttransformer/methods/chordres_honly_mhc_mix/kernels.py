"""Fused position-only coefficient projection, physical lookahead, and their gradients."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from nexttransformer.methods.chordres_honly_free.kernels import (
    _project,
    _gate_partials,
)
from nexttransformer.methods.chordres_token_controls.kernels import _positive
from . import GATE_EPS


@triton.jit
def _scale_group(k):
    # Preview (mu), position (two alphas), velocity (beta and two gammas).
    return tl.where(k == 0, 0, tl.where(k <= 2, 1, 2))


@triton.jit
def _token_read_forward(
    H,
    V,
    W,
    SCALES,
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
):
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    lane = tl.arange(0, BD)
    ix = row[:, None].to(tl.int64) * D + lane[None, :]
    mask = (row[:, None] < ROWS) & (lane[None, :] < D)
    h = tl.load(H + ix, mask, 0.0).to(tl.float32)
    v = tl.load(V + ix, mask, 0.0).to(tl.float32)
    gate_inv = tl.rsqrt(tl.sum(h * h, 1) / D + EPS)
    mu = tl.load(MU) + (tl.load(SCALES + 0) * gate_inv) * _project(h, W, lane, D, 0)
    ap = _positive(tl.load(STEPS) + (tl.load(SCALES + 1) * gate_inv) * _project(h, W, lane, D, 1))
    al = _positive(tl.load(STEPS + 1) + (tl.load(SCALES + 1) * gate_inv) * _project(h, W, lane, D, 2))
    if K == 6:
        beta = tl.sigmoid(tl.load(BETA) + (tl.load(SCALES + 2) * gate_inv) * _project(h, W, lane, D, 3))
        gp = tl.load(GAINS) + (tl.load(SCALES + 2) * gate_inv) * _project(h, W, lane, D, 4)
        gt = tl.load(GAINS + 1) + (tl.load(SCALES + 2) * gate_inv) * _project(h, W, lane, D, 5)
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
    SCALES,
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
    gate_inv = tl.rsqrt(tl.sum(h * h, 1) / D + EPS)
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
    query_mu_grad = tl.sum(v * d_off, 1)
    grad_row = DZ + row.to(tl.int64) * (K + 1)
    # Partials compute dL/d(effective weight), with the original
    # inverse RMS. Final reduction differentiates effective W[k] = s[group] * W[k].
    tl.store(grad_row + K, gate_inv, row < ROWS)
    for k in tl.static_range(K):
        g = tl.load(DC + offc + k, row < ROWS, 0.0)
        if k == 0:
            g += query_mu_grad
        elif k == 1 or k == 2:
            alpha = tl.load(C + offc + k, row < ROWS, 0.0)
            g *= -libdevice.expm1(-alpha)
        elif k == 3:
            beta = tl.load(C + offc + k, row < ROWS, 0.0)
            g *= beta * (1.0 - beta)
        tl.store(grad_row + k, g, row < ROWS)
        wh = tl.load(W + k * D + lane, lane < D, 0.0)
        grad_gate_h += (g * tl.load(SCALES + _scale_group(k)))[:, None] * wh[None, :]
    gate_h = h * gate_inv[:, None]
    radial = tl.sum(grad_gate_h * gate_h, 1) / D
    dh += gate_inv[:, None] * (grad_gate_h - radial[:, None] * gate_h)
    tl.store(DH + ix, dh, mask)
    tl.store(DV + ix, dv, mask)


@triton.jit
def _finish_parameters(
    PW,
    PB,
    W,
    SCALES,
    SCALE_PARTS,
    DW,
    DM,
    DS,
    DB,
    DG,
    SPLITS,
    D: tl.constexpr,
    K: tl.constexpr,
    BS: tl.constexpr,
    BC: tl.constexpr,
):
    tile = tl.program_id(0)
    k = tl.program_id(1)
    col = tile * BC + tl.arange(0, BC)
    split = tl.arange(0, BS)
    ptrs = (split[:, None].to(tl.int64) * K + k) * D + col[None, :]
    part = tl.load(PW + ptrs, (split[:, None] < SPLITS) & (col[None, :] < D), 0.0)
    effective_dw = tl.sum(part, 0)
    weight = tl.load(W + k * D + col, col < D, 0.)
    gain = tl.load(SCALES + _scale_group(k))
    tl.store(DW + k * D + col, gain * effective_dw, col < D)
    tl.store(SCALE_PARTS + k * tl.cdiv(D, BC) + tile, tl.sum(weight * effective_dw, 0))
    if tile == 0:
        val = tl.sum(tl.load(PB + split * K + k, split < SPLITS, 0.0), 0)
        if k == 0:
            tl.store(DM, val)
        elif k <= 2:
            tl.store(DS + k - 1, val)
        elif K == 6:
            if k == 3:
                tl.store(DB, val)
            else:
                tl.store(DG + k - 4, val)


@triton.jit
def _finish_scales(PART, OUT, N: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    k = lane // N
    valid = (lane < K * N) & (_scale_group(k) == group)
    tl.store(OUT + group, tl.sum(tl.load(PART + lane, valid, 0.), 0))


class TokenRead(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, h, v, weight, mu, steps, beta, gains, scales):
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
            scales,
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
        )
        ctx.save_for_backward(h, v, weight, controls, scales)
        return query, controls

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gquery, gcontrols):
        h, v, weight, controls, scales = ctx.saved_tensors
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
            scales,
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
        pw = weight.new_empty((splits, count, dim))
        pb = weight.new_empty((splits, count))
        _gate_partials[(triton.cdiv(dim, cols), splits)](
            h, dz, pw, pb, rows, dim, count, chunk, 32, cols
        )
        tiles = triton.cdiv(dim, cols)
        scale_parts = weight.new_empty((count, tiles))
        dscales = torch.empty_like(scales)
        _finish_parameters[(tiles, count)](
            pw,
            pb,
            weight,
            scales,
            scale_parts,
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
        _finish_scales[(scales.numel(),)](
            scale_parts, dscales, tiles, count, triton.next_power_of_2(tiles * count)
        )
        return dh, dv, dw, dm, ds, db, dg, dscales
