"""Static Chord coefficients with an unnormalized force target and unit-RMS position."""

import torch
from nexttransformer.methods.chordres_velocity_gains import VelocityGainChordRes
from nexttransformer.methods.chordres_zero import (
    frozen as frozen,
    reference_read,
    unit_rms,
)


def reference_raw_write(h, v, force, controls):
    state_dtype = h.dtype
    h, v = h.to(controls.dtype), v.to(controls.dtype)
    _, alpha_perp, alpha_par, beta, gamma_par, gamma_perp = controls.unbind()
    delta = force.to(h.dtype) - h
    along = (delta * v).mean(-1, keepdim=True) / v.square().mean(
        -1, keepdim=True
    ).clamp_min(1e-12)
    nxt = unit_rms(h + alpha_perp * delta + ((alpha_par - alpha_perp) * along) * v)
    chord = nxt - h

    def project(x):
        return x - (x * nxt).mean(-1, keepdim=True) * nxt

    u, d = project(v), project(chord)
    parallel = (u * d).mean(-1, keepdim=True) / u.square().mean(
        -1, keepdim=True
    ).clamp_min(1e-12)
    # This arrangement recovers the original write at gamma_par = gamma_perp = 1-beta.
    velocity = (
        project(beta * v + gamma_perp * chord)
        + ((gamma_par - gamma_perp) * parallel) * u
    )
    return nxt.to(state_dtype), velocity.to(state_dtype)


class RawForceChord(VelocityGainChordRes):
    def _ordinary_hidden(self, idx):
        x = self._embed_reference(idx)
        h = unit_rms(x).to(x.dtype)
        v = torch.zeros_like(h)
        read, write = reference_read, reference_raw_write
        if h.is_cuda and h.dtype == torch.bfloat16:
            from nexttransformer.methods.chordres_zero.kernels import ReadNorm
            from .kernels import RawForceWrite

            read, write = ReadNorm.apply, RawForceWrite.apply

        cos_sin = self.cos[:, : idx.size(1)], self.sin[:, : idx.size(1)]
        for i, layer in enumerate(self.transformer.h):
            ve = (
                self.value_embeds[str(i)](idx).to(x.dtype)
                if str(i) in self.value_embeds
                else None
            )
            window = self._window(i, self.config.n_layer)
            attn = self.residual_maps[2 * i]()
            force = layer.attn(read(h, v, attn), ve, cos_sin, window, None)
            self._record(
                "block",
                physical=i,
                virtual=i,
                window=window[0],
                tokens=idx.numel(),
                sequence_len=idx.size(1),
                ve=ve is not None,
                mixer="softmax",
                residual=self.config.method,
            )
            h, v = write(h, v, force, attn)
            mlp = self.residual_maps[2 * i + 1]()
            force = layer.mlp(read(h, v, mlp))
            h, v = write(h, v, force, mlp)
        return h, None


model = RawForceChord
