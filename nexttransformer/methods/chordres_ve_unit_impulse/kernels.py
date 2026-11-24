"""Fused saved-force Chord write with a raw value-embedding velocity impulse."""
import torch
import triton
import triton.language as tl


@triton.jit
def _impulse_write(H, V, F, P, E, NH, NV, GH, GV, DH, DV, DF, DP, DE,
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
    ve = tl.load(E + idx, lane < D, 0).to(tl.float32)
    velocity = velocity + ve
    radial = tl.sum(velocity * nxt, 0) / D
    if BACKWARD:
        gh = tl.load(GH + idx, lane < D, 0).to(tl.float32)
        gv = tl.load(GV + idx, lane < D, 0).to(tl.float32)
        dot = tl.sum(gv * nxt, 0) / D
        g_velocity = gv - dot * nxt
        tl.store(DE + idx, g_velocity, lane < D)
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


class ImpulseWrite(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, v, force, controls, ve):
        h, v, force, ve = h.contiguous(), v.contiguous(), force.contiguous(), ve.contiguous()
        if ve.shape != h.shape:
            raise ValueError("The direct VE impulse requires a full-width value embedding")
        d = h.size(-1)
        rows = h.numel() // d
        nh, nv = torch.empty_like(h), torch.empty_like(v)
        _impulse_write[(rows,)](h, v, force, controls, ve, nh, nv, None, None, None, None,
                                None, None, None, d, triton.next_power_of_2(d), False)
        ctx.save_for_backward(h, v, force, controls, ve)
        return nh, nv

    @staticmethod
    def backward(ctx, gh, gv):
        h, v, force, controls, ve = ctx.saved_tensors
        d = h.size(-1)
        rows = h.numel() // d
        dh, dv, df, de = map(torch.empty_like, (h, v, force, ve))
        partial = controls.new_empty(rows, 3)
        _impulse_write[(rows,)](h, v, force, controls, ve, None, None, gh.contiguous(), gv.contiguous(),
                                dh, dv, df, partial, de, d, triton.next_power_of_2(d), True)
        dc = torch.zeros_like(controls)
        dc[1:] = partial.sum(0)
        return dh, dv, df, dc, de
