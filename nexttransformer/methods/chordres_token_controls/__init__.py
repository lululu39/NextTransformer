"""Use a zero-start, per-site linear map of the current (position, velocity) to raw controls."""

import torch
from torch import nn
from torch.nn import functional as F

from nexttransformer.methods.chordres_zero import ChordRes, frozen as frozen, unit_rms
from nexttransformer.methods.chordres_velocity_gains import VelocityGainSite
from nexttransformer.models.backbone import norm


def token_reference_read(h, v, weight, mu, steps, beta, gains):
    dtype = weight.dtype
    z = F.linear(h.to(dtype), weight[:, 0]) + F.linear(v.to(dtype), weight[:, 1])
    m = z[..., :1] + mu
    raw = z[..., 1:3] + steps
    shift = raw.detach().clamp_min(0)
    alpha = shift + ((-shift).exp() + (raw - shift).exp()).log()
    if beta is None:
        retention = torch.full_like(m, 0.5)
        gamma = torch.full_like(alpha, 0.5)
    else:
        retention = (z[..., 3:4] + beta).sigmoid()
        gamma = z[..., 4:] + gains
    controls = torch.cat((m, alpha, retention, gamma), -1)
    offset = m * v.to(dtype)
    read = (h.to(dtype) + offset) * (1 + offset.square().mean(-1, keepdim=True)).rsqrt()
    return norm(read.to(h.dtype)), controls


class TokenSite(VelocityGainSite):
    def __init__(self, width, terminal):
        super().__init__(terminal)
        self.token_weight = nn.Parameter(torch.zeros(3 if terminal else 6, 2, width))

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "token_weight"):
            self.token_weight.zero_()

    def forward(self, h, v):
        args = (
            h,
            v,
            self.token_weight,
            self.mu,
            self.steps,
            self.beta,
            self.velocity_gains,
        )
        if h.is_cuda and h.dtype == torch.bfloat16:
            from .kernels import TokenRead

            return TokenRead.apply(*args)
        return token_reference_read(*args)


def token_reference_write(h, v, force, controls):
    state_dtype = h.dtype
    h, v = h.to(controls.dtype), v.to(controls.dtype)
    _, alpha_perp, alpha_par, beta, gamma_par, gamma_perp = controls.unsqueeze(
        -1
    ).unbind(-2)
    delta = unit_rms(force) - h
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


class TokenControlledChord(ChordRes):
    def __init__(self, config):
        super().__init__(config)
        sites = 2 * config.n_layer
        self.residual_maps = nn.ModuleList(
            TokenSite(config.width, i == sites - 1) for i in range(sites)
        )

    def _ordinary_hidden(self, idx):
        x = self._embed_reference(idx)
        h = unit_rms(x).to(x.dtype)
        v = torch.zeros_like(h)
        write = token_reference_write
        if h.is_cuda and h.dtype == torch.bfloat16:
            from .kernels import TokenWrite

            write = TokenWrite.apply
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


model = TokenControlledChord
