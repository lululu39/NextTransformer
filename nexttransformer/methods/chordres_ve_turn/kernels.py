"""Fused VE attention write with the saved-force Chord position and an undamped tangent turn."""
import torch
import triton
import triton.language as tl


@triton.jit
def _turn_write(H, V, F, P, NH, NV, GH, GV, DH, DV, DF, DP,
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
    # Project both vectors in FP32 before rounding the new position to BF16.
    v_radial = tl.sum(v * nxt, 0) / D
    chord_radial = tl.sum(chord * nxt, 0) / D
    u = v - v_radial * nxt
    d = chord - chord_radial * nxt
    u2 = tl.sum(u * u, 0) / D
    u_den = tl.maximum(u2, 1e-12)
    parallel = (tl.sum(d * u, 0) / D) / u_den
    carry = beta * (1. - parallel)
    if BACKWARD:
        gh = tl.load(GH + idx, lane < D, 0).to(tl.float32)
        gv = tl.load(GV + idx, lane < D, 0).to(tl.float32)
        # VJP of d + beta*(1 - <u,d>/<u,u>)*u, including the denominator floor.
        coeff = -beta * (tl.sum(gv * u, 0) / D) / u_den
        gd = gv + coeff * u
        gu = carry * gv + coeff * (d - tl.where(u2 >= 1e-12, 2. * parallel * u, 0.))
        gu_radial = tl.sum(gu * nxt, 0) / D
        gd_radial = tl.sum(gd * nxt, 0) / D
        gv_direct = gu - gu_radial * nxt
        g_chord = gd - gd_radial * nxt
        g_nxt = gh + g_chord - v_radial * gu - gu_radial * v - chord_radial * gd - gd_radial * chord
        g_mixed = mi * (g_nxt - tl.where(m2 >= 1e-12, nxt * (tl.sum(g_nxt * nxt, 0) / D), 0.))
        g_dot = gain * (tl.sum(g_mixed * v, 0) / D) / den
        g_delta = perp * g_mixed + g_dot * v
        g_v = gv_direct + (gain * along) * g_mixed
        g_v += g_dot * (delta - tl.where(v2 >= 1e-12, 2. * along * v, 0.))
        g_force = fi * (g_delta - tl.where(f2 >= 1e-12, target * (tl.sum(g_delta * target, 0) / D), 0.))
        tl.store(DH + idx, g_mixed - g_delta - g_chord, lane < D)
        tl.store(DV + idx, g_v, lane < D)
        tl.store(DF + idx, g_force, lane < D)
        tl.store(DP + 3 * row, tl.sum(g_mixed * (delta - along * v), 0))
        tl.store(DP + 3 * row + 1, tl.sum(g_mixed * (along * v), 0))
        tl.store(DP + 3 * row + 2, tl.sum(gv * ((1. - parallel) * u), 0))
    else:
        tl.store(NH + idx, nxt, lane < D)
        tl.store(NV + idx, d + carry * u, lane < D)


class TurnWrite(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, h, v, force, controls):
        h, v, force = h.contiguous(), v.contiguous(), force.contiguous()
        width = h.size(-1)
        rows = h.numel() // width
        nh, nv = torch.empty_like(h), torch.empty_like(v)
        _turn_write[(rows,)](h, v, force, controls, nh, nv, None, None, None, None, None, None,
                             width, triton.next_power_of_2(width), False)
        ctx.save_for_backward(h, v, force, controls)
        return nh, nv

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gh, gv):
        h, v, force, controls = ctx.saved_tensors
        width = h.size(-1)
        rows = h.numel() // width
        dh, dv, df = torch.empty_like(h), torch.empty_like(v), torch.empty_like(force)
        partial = controls.new_empty(rows, 3)
        _turn_write[(rows,)](h, v, force, controls, None, None, gh.contiguous(), gv.contiguous(),
                             dh, dv, df, partial, width, triton.next_power_of_2(width), True)
        dc = torch.zeros_like(controls)
        dc[1:] = partial.sum(0)
        return dh, dv, df, dc
