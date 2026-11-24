"""Fused per-token control projection, read and tangent-space write."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _token_write(
    H,
    V,
    F,
    P,
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
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    idx = row.to(tl.int64) * D + lane
    h = tl.load(H + idx, lane < D, 0).to(tl.float32)
    v = tl.load(V + idx, lane < D, 0).to(tl.float32)
    force = tl.load(F + idx, lane < D, 0).to(tl.float32)
    perp = tl.load(P + 6 * row + 1)
    gain = tl.load(P + 6 * row + 2) - perp
    beta = tl.load(P + 6 * row + 3)
    gamma_par = tl.load(P + 6 * row + 4)
    gamma_perp = tl.load(P + 6 * row + 5)
    gamma_diff = gamma_par - gamma_perp
    f2 = tl.sum(force * force, 0) / D
    fi = tl.rsqrt(tl.maximum(f2, 1e-12))
    target = force * fi
    delta = target - h
    v2 = tl.sum(v * v, 0) / D
    den = tl.maximum(v2, 1e-12)
    along = tl.sum(delta * v, 0) / D / den
    mixed = h + perp * delta + (gain * along) * v
    m2 = tl.sum(mixed * mixed, 0) / D
    mi = tl.rsqrt(tl.maximum(m2, 1e-12))
    nxt = mixed * mi
    chord = nxt - h
    # Project both vectors in FP32 before rounding the new position to BF16.
    v_radial = tl.sum(v * nxt, 0) / D
    chord_radial = tl.sum(chord * nxt, 0) / D
    u = v - v_radial * nxt
    d = chord - chord_radial * nxt
    u2 = tl.sum(u * u, 0) / D
    u_den = tl.maximum(u2, 1e-12)
    parallel = (tl.sum(d * u, 0) / D) / u_den
    # Project the common displacement term once, in the same order as the baseline.
    mix = beta * v + gamma_perp * chord
    radial = tl.sum(mix * nxt, 0) / D
    correction = gamma_diff * parallel
    if BACKWARD:
        gh = tl.load(GH + idx, lane < D, 0).to(tl.float32)
        gv = tl.load(GV + idx, lane < D, 0).to(tl.float32)
        # VJP of beta*u + gamma_perp*d + (gamma_par-gamma_perp)*parallel*u.
        coeff = gamma_diff * (tl.sum(gv * u, 0) / D) / u_den
        gd = gamma_perp * gv + coeff * u
        gu = (beta + correction) * gv + coeff * (
            d - tl.where(u2 >= 1e-12, 2.0 * parallel * u, 0.0)
        )
        gu_radial = tl.sum(gu * nxt, 0) / D
        gd_radial = tl.sum(gd * nxt, 0) / D
        gv_direct = gu - gu_radial * nxt
        g_chord = gd - gd_radial * nxt
        g_nxt = (
            gh
            + g_chord
            - v_radial * gu
            - gu_radial * v
            - chord_radial * gd
            - gd_radial * chord
        )
        g_mixed = mi * (
            g_nxt - tl.where(m2 >= 1e-12, nxt * (tl.sum(g_nxt * nxt, 0) / D), 0.0)
        )
        g_dot = gain * (tl.sum(g_mixed * v, 0) / D) / den
        g_delta = perp * g_mixed + g_dot * v
        g_v = gv_direct + (gain * along) * g_mixed
        g_v += g_dot * (delta - tl.where(v2 >= 1e-12, 2.0 * along * v, 0.0))
        g_force = fi * (
            g_delta
            - tl.where(f2 >= 1e-12, target * (tl.sum(g_delta * target, 0) / D), 0.0)
        )
        tl.store(DH + idx, g_mixed - g_delta - g_chord, lane < D)
        tl.store(DV + idx, g_v, lane < D)
        tl.store(DF + idx, g_force, lane < D)
        tl.store(DP + 6 * row, 0.0)
        tl.store(DP + 6 * row + 1, tl.sum(g_mixed * (delta - along * v), 0))
        tl.store(DP + 6 * row + 2, tl.sum(g_mixed * (along * v), 0))
        tl.store(DP + 6 * row + 3, tl.sum(gv * u, 0))
        parallel_vec = parallel * u
        tl.store(DP + 6 * row + 4, tl.sum(gv * parallel_vec, 0))
        tl.store(DP + 6 * row + 5, tl.sum(gv * (d - parallel_vec), 0))
    else:
        tl.store(NH + idx, nxt, lane < D)
        tl.store(NV + idx, (mix - radial * nxt) + correction * u, lane < D)


class TokenWrite(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, h, v, force, controls):
        h, v, force = h.contiguous(), v.contiguous(), force.contiguous()
        width = h.size(-1)
        rows = h.numel() // width
        nh, nv = torch.empty_like(h), torch.empty_like(v)
        _token_write[(rows,)](
            h,
            v,
            force,
            controls,
            nh,
            nv,
            None,
            None,
            None,
            None,
            None,
            None,
            width,
            triton.next_power_of_2(width),
            False,
        )
        ctx.save_for_backward(h, v, force, controls)
        return nh, nv

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gh, gv):
        h, v, force, controls = ctx.saved_tensors
        width = h.size(-1)
        rows = h.numel() // width
        dh, dv, df = torch.empty_like(h), torch.empty_like(v), torch.empty_like(force)
        dc = torch.empty_like(controls)
        _token_write[(rows,)](
            h,
            v,
            force,
            controls,
            None,
            None,
            gh.contiguous(),
            gv.contiguous(),
            dh,
            dv,
            df,
            dc,
            width,
            triton.next_power_of_2(width),
            True,
        )
        return dh, dv, df, dc


@triton.jit
def _project(h, v, W, lane, D: tl.constexpr, k: tl.constexpr):
    wh = tl.load(W + (2 * k) * D + lane, lane < D, 0.0)
    wv = tl.load(W + (2 * k + 1) * D + lane, lane < D, 0.0)
    return tl.sum(h * wh[None, :] + v * wv[None, :], 1)


@triton.jit
def _positive(x):
    # Same softplus evaluation order as the scalar reference.
    shift = tl.maximum(x, 0.0)
    return shift + libdevice.log(libdevice.exp(-shift) + libdevice.exp(x - shift))


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
):
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    lane = tl.arange(0, BD)
    ix = row[:, None].to(tl.int64) * D + lane[None, :]
    mask = (row[:, None] < ROWS) & (lane[None, :] < D)
    h = tl.load(H + ix, mask, 0.0).to(tl.float32)
    v = tl.load(V + ix, mask, 0.0).to(tl.float32)
    mu = tl.load(MU) + _project(h, v, W, lane, D, 0)
    ap = _positive(tl.load(STEPS) + _project(h, v, W, lane, D, 1))
    al = _positive(tl.load(STEPS + 1) + _project(h, v, W, lane, D, 2))
    if K == 6:
        beta = tl.sigmoid(tl.load(BETA) + _project(h, v, W, lane, D, 3))
        gp = tl.load(GAINS) + _project(h, v, W, lane, D, 4)
        gt = tl.load(GAINS + 1) + _project(h, v, W, lane, D, 5)
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
):
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    lane = tl.arange(0, BD)
    ix = row[:, None].to(tl.int64) * D + lane[None, :]
    mask = (row[:, None] < ROWS) & (lane[None, :] < D)
    h = tl.load(H + ix, mask, 0.0).to(tl.float32)
    v = tl.load(V + ix, mask, 0.0).to(tl.float32)
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
    query_mu_grad = tl.sum(v * d_off, 1)
    for k in tl.static_range(K):
        g = tl.load(DC + offc + k, row < ROWS, 0.0)
        if k == 0:
            g += query_mu_grad
        elif k == 1 or k == 2:
            alpha = tl.load(C + offc + k, row < ROWS, 0.0)
            g *= -libdevice.expm1(-alpha)
        elif k == 3:
            beta = tl.load(C + offc + 3, row < ROWS, 0.5)
            g *= beta * (1.0 - beta)
        tl.store(DZ + row.to(tl.int64) * K + k, g, row < ROWS)
        wh = tl.load(W + 2 * k * D + lane, lane < D, 0.0)
        wv = tl.load(W + (2 * k + 1) * D + lane, lane < D, 0.0)
        dh += g[:, None] * wh[None, :]
        dv += g[:, None] * wv[None, :]
    tl.store(DH + ix, dh, mask)
    tl.store(DV + ix, dv, mask)


@triton.jit
def _parameter_partials(
    H,
    V,
    DZ,
    PW,
    PB,
    ROWS,
    D: tl.constexpr,
    K: tl.constexpr,
    CHUNK: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    tile = tl.program_id(0)
    stream = tl.program_id(1)
    split = tl.program_id(2)
    col = tile * BC + tl.arange(0, BC)
    controls = tl.arange(0, 16)
    sums = tl.full((16, BC), 0, tl.float32)
    biases = tl.full((16,), 0, tl.float32)
    for offset in tl.static_range(0, CHUNK, BR):
        row = split * CHUNK + offset + tl.arange(0, BR)
        ptrs = row[:, None].to(tl.int64) * D + col[None, :]
        valid = (row[:, None] < ROWS) & (col[None, :] < D)
        if stream == 0:
            x = tl.load(H + ptrs, valid, 0)
        else:
            x = tl.load(V + ptrs, valid, 0)
        z = tl.load(
            DZ + row[:, None].to(tl.int64) * K + controls[None, :],
            (row[:, None] < ROWS) & (controls[None, :] < K),
            0.0,
        )
        # Two BF16 limbs retain small FP32 control gradients in tensor-core dot products.
        hi = z.to(tl.bfloat16)
        lo = (z - hi.to(tl.float32)).to(tl.bfloat16)
        sums = tl.dot(tl.trans(hi), x, sums)
        sums = tl.dot(tl.trans(lo), x, sums)
        if tile == 0 and stream == 0:
            biases += tl.sum(z, 0)
    ptrs = ((split * K + controls[:, None].to(tl.int64)) * 2 + stream) * D + col[
        None, :
    ]
    tl.store(PW + ptrs, sums, (controls[:, None] < K) & (col[None, :] < D))
    if tile == 0 and stream == 0:
        tl.store(PB + split * K + controls, biases, controls < K)


@triton.jit
def _finish_parameters(
    PW,
    PB,
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
    stream = tl.program_id(1)
    k = tl.program_id(2)
    col = tile * BC + tl.arange(0, BC)
    split = tl.arange(0, BS)
    ptrs = ((split[:, None].to(tl.int64) * K + k) * 2 + stream) * D + col[None, :]
    part = tl.load(PW + ptrs, (split[:, None] < SPLITS) & (col[None, :] < D), 0.0)
    tl.store(DW + (2 * k + stream) * D + col, tl.sum(part, 0), col < D)
    if tile == 0 and stream == 0:
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
        dz = weight.new_empty((rows, count))
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
        _parameter_partials[(triton.cdiv(dim, cols), 2, splits)](
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
