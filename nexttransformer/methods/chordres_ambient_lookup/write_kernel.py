"""Fused update-based Chord write and backward with a soft RMS force denominator."""
import torch
import triton
import triton.language as tl

from nexttransformer.methods.chordres_update_soft_force import FORCE_TAU


@triton.jit
def _ambient_write(
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
    TAU_SQ: tl.constexpr,
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
    fi = tl.rsqrt(f2 + TAU_SQ)
    delta = force * fi
    v2 = tl.sum(v * v, 0) / D
    den = tl.maximum(v2, 1e-12)
    along = tl.sum(delta * v, 0) / D / den
    mixed = h + perp * delta + (gain * along) * v
    m2 = tl.sum(mixed * mixed, 0) / D
    mi = tl.rsqrt(tl.maximum(m2, 1e-12))
    nxt = mixed * mi
    chord = nxt - h
    # The same velocity norm serves both splits; the radial component is retained.
    parallel = (tl.sum(chord * v, 0) / D) / den
    correction = gamma_diff * parallel
    if BACKWARD:
        gh = tl.load(GH + idx, lane < D, 0).to(tl.float32)
        gv = tl.load(GV + idx, lane < D, 0).to(tl.float32)
        coeff = gamma_diff * (tl.sum(gv * v, 0) / D) / den
        g_chord = gamma_perp * gv + coeff * v
        gv_direct = (beta + correction) * gv + coeff * (
            chord - tl.where(v2 >= 1e-12, 2.0 * parallel * v, 0.0)
        )
        g_nxt = gh + g_chord
        g_mixed = mi * (
            g_nxt - tl.where(m2 >= 1e-12, nxt * (tl.sum(g_nxt * nxt, 0) / D), 0.0)
        )
        g_dot = gain * (tl.sum(g_mixed * v, 0) / D) / den
        g_delta = perp * g_mixed + g_dot * v
        g_v = gv_direct + (gain * along) * g_mixed
        g_v += g_dot * (delta - tl.where(v2 >= 1e-12, 2.0 * along * v, 0.0))
        # The update direction depends only on force. Differentiate its smooth RMS
        # denominator at every magnitude, including zero.
        g_force = fi * (g_delta - delta * (tl.sum(g_delta * delta, 0) / D))
        tl.store(DH + idx, g_mixed - g_chord, lane < D)
        tl.store(DV + idx, g_v, lane < D)
        tl.store(DF + idx, g_force, lane < D)
        tl.store(DP + 6 * row, 0.0)
        tl.store(DP + 6 * row + 1, tl.sum(g_mixed * (delta - along * v), 0))
        tl.store(DP + 6 * row + 2, tl.sum(g_mixed * (along * v), 0))
        tl.store(DP + 6 * row + 3, tl.sum(gv * v, 0))
        parallel_vec = parallel * v
        tl.store(DP + 6 * row + 4, tl.sum(gv * parallel_vec, 0))
        tl.store(DP + 6 * row + 5, tl.sum(gv * (chord - parallel_vec), 0))
    else:
        tl.store(NH + idx, nxt, lane < D)
        tl.store(NV + idx, beta * v + gamma_perp * chord + correction * v, lane < D)


class AmbientWrite(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, h, v, force, controls):
        h, v, force = h.contiguous(), v.contiguous(), force.contiguous()
        width = h.size(-1)
        rows = h.numel() // width
        nh, nv = torch.empty_like(h), torch.empty_like(v)
        _ambient_write[(rows,)](
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
            FORCE_TAU**2,
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
        _ambient_write[(rows,)](
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
            FORCE_TAU**2,
        )
        return dh, dv, df, dc
