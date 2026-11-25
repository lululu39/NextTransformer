"""Update-based soft-normalized branch forces with the native-zero Chord mixing controls."""
import torch

from nexttransformer.methods.chordres_honly_mhc_mix import HOnlyChord
from nexttransformer.methods.chordres_zero import frozen as frozen, unit_rms


FORCE_TAU = 0.05


def token_reference_write(h, v, force, controls):
    state_dtype = h.dtype
    h, v = h.to(controls.dtype), v.to(controls.dtype)
    _, alpha_perp, alpha_par, beta, gamma_par, gamma_perp = controls.unsqueeze(
        -1
    ).unbind(-2)
    force = force.to(controls.dtype)
    delta = force * (force.square().mean(-1, keepdim=True) + FORCE_TAU**2).rsqrt()
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


class SoftUpdateChord(HOnlyChord):
    def _ordinary_hidden(self, idx):
        x = self._embed_reference(idx)
        h = unit_rms(x).to(x.dtype)
        v = torch.zeros_like(h)
        write = token_reference_write
        if h.is_cuda and h.dtype == torch.bfloat16:
            from .kernels import SoftUpdateWrite

            write = SoftUpdateWrite.apply
        cos_sin = self.cos[:, : idx.size(1)], self.sin[:, : idx.size(1)]
        for i, layer in enumerate(self.transformer.h):
            ve = (
                self.value_embeds[str(i)](idx).to(x.dtype)
                if str(i) in self.value_embeds
                else None
            )
            window = self._window(i, self.config.n_layer)
            query, controls = self.residual_maps[2 * i](h, v)
            force = layer.attn(query, ve, cos_sin, window, None)
            self._record(
                "block",
                physical=i,
                virtual=i,
                window=window[0],
                tokens=idx.numel(),
                ve=ve is not None,
                sequence_len=idx.size(1),
                mixer="softmax",
                residual=self.config.method,
            )
            h, v = write(h, v, force, controls)
            query, controls = self.residual_maps[2 * i + 1](h, v)
            force = layer.mlp(query)
            h, v = write(h, v, force, controls)
        return h, None



model = SoftUpdateChord
