"""Independent old-velocity retention and displacement gains on the new tangent plane."""
import torch
from torch import nn

from nexttransformer.methods.chordres_zero import ChordRes, ChordSite, frozen, reference_read, unit_rms


class VelocityGainSite(ChordSite):
    def __init__(self, terminal):
        super().__init__(terminal)
        # The ordinary head reads h only, so no last-MLP velocity gains are trainable.
        self.velocity_gains = None if terminal else nn.Parameter(torch.full((2,), .5))

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        if getattr(self, "velocity_gains", None) is not None:
            self.velocity_gains.fill_(.5)

    def forward(self):
        p = super().forward()
        gains = torch.full_like(self.steps, .5) if self.velocity_gains is None else self.velocity_gains
        return torch.cat((p, gains))


def reference_gain_write(h, v, force, controls):
    state_dtype = h.dtype
    h, v = h.to(controls.dtype), v.to(controls.dtype)
    _, alpha_perp, alpha_par, beta, gamma_par, gamma_perp = controls.unbind()
    delta = unit_rms(force) - h
    along = (delta*v).mean(-1,keepdim=True) / v.square().mean(-1,keepdim=True).clamp_min(1e-12)
    nxt = unit_rms(h + alpha_perp*delta + ((alpha_par-alpha_perp)*along)*v)
    chord = nxt - h
    def project(x):
        return x - (x*nxt).mean(-1,keepdim=True)*nxt

    u, d = project(v), project(chord)
    parallel = (u*d).mean(-1,keepdim=True) / u.square().mean(-1,keepdim=True).clamp_min(1e-12)
    # This arrangement recovers the original write at gamma_par = gamma_perp = 1-beta.
    velocity = project(beta*v + gamma_perp*chord) + ((gamma_par-gamma_perp)*parallel)*u
    return nxt.to(state_dtype), velocity.to(state_dtype)


class VelocityGainChordRes(ChordRes):
    def __init__(self, config):
        super().__init__(config)
        sites = 2*config.n_layer
        self.residual_maps = nn.ModuleList(VelocityGainSite(i==sites-1) for i in range(sites))

    def _ordinary_hidden(self, idx):
        x = self._embed_reference(idx)
        h = unit_rms(x).to(x.dtype)
        v = torch.zeros_like(h)
        read, write = reference_read, reference_gain_write
        if h.is_cuda and h.dtype == torch.bfloat16:
            from nexttransformer.methods.chordres_zero.kernels import ReadNorm
            from .kernels import VelocityGainWrite
            read, write = ReadNorm.apply, VelocityGainWrite.apply

        cos_sin = self.cos[:, :idx.size(1)], self.sin[:, :idx.size(1)]
        for i, layer in enumerate(self.transformer.h):
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            window = self._window(i, self.config.n_layer)
            attn = self.residual_maps[2 * i]()
            force = layer.attn(read(h, v, attn), ve, cos_sin, window, None)
            self._record("block", physical=i, virtual=i, window=window[0], tokens=idx.numel(),
                         sequence_len=idx.size(1), ve=ve is not None, mixer="softmax", residual=self.config.method)
            h, v = write(h, v, force, attn)
            mlp = self.residual_maps[2 * i + 1]()
            force = layer.mlp(read(h, v, mlp))
            h, v = write(h, v, force, mlp)
        return h, None


model = VelocityGainChordRes
