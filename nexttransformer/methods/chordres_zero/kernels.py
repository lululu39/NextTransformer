"""Fused scalar project-only Chord kernels, adapted from the FLAME ChordRes implementation.

The read preserves the BF16 boundary before NextTransformer's weightless RMSNorm.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _read_norm(H, V, P, Y, DY, DH, DV, DMU,
               ROWS, D: tl.constexpr, EPS: tl.constexpr,
               BT: tl.constexpr, BD: tl.constexpr, BACKWARD: tl.constexpr):
    tile = tl.program_id(0)
    rows = tile * BT + tl.arange(0, BT)
    cols = tl.arange(0, BD)
    indices = rows[:, None].to(tl.int64) * D + cols[None, :]
    mask = (rows[:, None] < ROWS) & (cols[None, :] < D)
    h = tl.load(H + indices, mask, 0).to(tl.float32)
    v = tl.load(V + indices, mask, 0).to(tl.float32)
    mu = tl.load(P)
    offset = mu * v
    inv = tl.rsqrt(1. + tl.sum(offset * offset, 1) / D)
    raw = h + offset
    # Preserve the BF16 interface between the two original autograd nodes.
    q = (raw * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    rstd = 1. / tl.sqrt(tl.sum(q * q, 1) / D + EPS)
    normalized = q * rstd[:, None]

    if BACKWARD:
        dy = tl.load(DY + indices, mask, 0).to(tl.float32)
        radial = tl.sum(normalized * dy, 1) / D
        dq = ((dy - normalized * radial[:, None]) * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
        direct = dq * inv[:, None]
        coeff = tl.sum(dq * raw, 1) / D * inv * inv * inv
        d_offset = direct - offset * coeff[:, None]
        tl.store(DH + indices, direct, mask)
        tl.store(DV + indices, mu * d_offset, mask)
        tl.store(DMU + rows, tl.sum(v * d_offset, 1), rows < ROWS)
    else:
        tl.store(Y + indices, normalized, mask)


class ReadNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, v, controls):
        h, v = h.contiguous(), v.contiguous()
        width = h.size(-1)
        rows = h.numel() // width
        out = torch.empty_like(h, dtype=torch.bfloat16)
        _read_norm[(rows,)](h, v, controls, out, None, None, None, None,
                            rows, width, torch.finfo(torch.float32).eps, 1, triton.next_power_of_2(width), False)
        ctx.save_for_backward(h, v, controls)
        return out

    @staticmethod
    def backward(ctx, grad):
        h, v, controls = ctx.saved_tensors
        width = h.size(-1)
        rows = h.numel() // width
        tile_rows = min(8, max(1, 4096 // triton.next_power_of_2(width)))
        tiles = triton.cdiv(rows, tile_rows)
        dh, dv = torch.empty_like(h), torch.empty_like(v)
        partial_mu = controls.new_empty(rows)
        _read_norm[(tiles,)](h, v, controls, None, grad.contiguous(), dh, dv,
                             partial_mu, rows, width, torch.finfo(torch.float32).eps,
                             tile_rows, triton.next_power_of_2(width), True)
        dcontrols = torch.zeros_like(controls)
        dcontrols[0] = partial_mu.sum()
        return dh, dv, dcontrols


@triton.jit
def _write(H, V, F, P, NH, NV, GH, GV, DH, DV, DF, DP,
           D: tl.constexpr, BLOCK: tl.constexpr, BACKWARD: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    idx = row.to(tl.int64) * D + lane
    h = tl.load(H + idx, lane < D, 0).to(tl.float32)
    v = tl.load(V + idx, lane < D, 0).to(tl.float32)
    force = tl.load(F + idx, lane < D, 0).to(tl.float32)
    perp = tl.load(P + 1)
    gain = tl.load(P + 2) - perp
    beta = tl.load(P + 3)
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
    velocity = beta * v + (1. - beta) * chord
    radial = tl.sum(velocity * nxt, 0) / D
    if BACKWARD:
        gh = tl.load(GH + idx, lane < D, 0).to(tl.float32)
        gv = tl.load(GV + idx, lane < D, 0).to(tl.float32)
        dot = tl.sum(gv * nxt, 0) / D
        g_velocity = gv - dot * nxt
        g_chord = (1. - beta) * g_velocity
        g_nxt = gh + g_chord - radial * gv - dot * velocity
        g_mixed = mi * (g_nxt - tl.where(m2 >= 1e-12, nxt * (tl.sum(g_nxt * nxt, 0) / D), 0.))
        g_dot = gain * (tl.sum(g_mixed * v, 0) / D) / den
        g_delta = perp * g_mixed + g_dot * v
        g_v = beta * g_velocity + (gain * along) * g_mixed
        g_v += g_dot * (delta - tl.where(v2 >= 1e-12, 2. * along * v, 0.))
        g_force = fi * (g_delta - tl.where(f2 >= 1e-12, target * (tl.sum(g_delta * target, 0) / D), 0.))
        tl.store(DH + idx, g_mixed - g_delta - g_chord, lane < D)
        tl.store(DV + idx, g_v, lane < D)
        tl.store(DF + idx, g_force, lane < D)
        tl.store(DP + 3 * row, tl.sum(g_mixed * (delta - along * v), 0))
        tl.store(DP + 3 * row + 1, tl.sum(g_mixed * (along * v), 0))
        tl.store(DP + 3 * row + 2, tl.sum(g_velocity * (v - chord), 0))
    else:
        tl.store(NH + idx, nxt, lane < D)
        tl.store(NV + idx, velocity - radial * nxt, lane < D)


class ChordWrite(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, h, v, force, p, projection=None, *projection_inputs):
        h, v = h.contiguous(), v.contiguous()
        d = h.size(-1)
        rows = h.numel() // d
        nh, nv = torch.empty_like(h), torch.empty_like(v)
        _write[(rows,)](h, v, force, p, nh, nv, None, None, None, None, None, None,
                        d, triton.next_power_of_2(d), False)
        ctx.projection = projection
        ctx.save_for_backward(h, v, p, *(projection_inputs if projection is not None else (force,)))
        return nh, nv

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gh, gv):
        h, v, p, *source = ctx.saved_tensors
        if ctx.projection is None:
            force, = source
        else:
            # The original projection's autograd node receives df below. Only
            # its numerical output is needed to evaluate the Chord derivative.
            with torch.no_grad():
                force = ctx.projection(*source).contiguous()
        d = h.size(-1)
        rows = h.numel() // d
        dh, dv, df = torch.empty_like(h), torch.empty_like(v), torch.empty_like(force)
        partial = p.new_empty(rows, 3)
        _write[(rows,)](h, v, force, p, None, None, gh.contiguous(), gv.contiguous(), dh, dv, df, partial,
                         d, triton.next_power_of_2(d), True)
        dp = torch.zeros_like(p)
        dp[1:] = partial.sum(0)
        return (dh, dv, df, dp) + (None,) * (len(ctx.needs_input_grad) - 4)
