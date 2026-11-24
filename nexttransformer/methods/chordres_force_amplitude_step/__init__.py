"""Single-step spherical Chord with token dependence only through branch magnitude."""

import math

import torch
from torch import nn
from nexttransformer.methods.chordres_zero import (
    ChordRes,
    frozen as frozen,
    reference_read,
    unit_rms,
)


def reference_write(h, v, force, bias, amplitude_weight, raw_beta):
    dtype = h.dtype
    h, v, force = h.to(bias.dtype), v.to(bias.dtype), force.to(bias.dtype)
    power = force.square().mean(-1, keepdim=True)
    raw = bias + amplitude_weight * (1 + power).log()
    pivot = raw.detach().clamp_min(0)
    alpha = pivot + ((-pivot).exp() + (raw - pivot).exp()).log()
    beta = torch.full_like(bias, 0.5) if raw_beta is None else raw_beta.sigmoid()
    delta = force * power.clamp_min(1e-12).rsqrt() - h
    nxt = unit_rms(h + alpha * delta)
    mix = beta * v + (1 - beta) * (nxt - h)
    return nxt.to(dtype), (mix - (mix * nxt).mean(-1, keepdim=True) * nxt).to(dtype)


class AmplitudeSite(nn.Module):
    def __init__(self, terminal):
        super().__init__()
        self.mu = nn.Parameter(torch.empty(1))
        self.step = nn.Parameter(torch.empty(1))
        self.amplitude_weight = nn.Parameter(torch.empty(1))
        self.beta = None if terminal else nn.Parameter(torch.empty(1))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.mu.zero_()
        self.step.fill_(math.log(math.expm1(0.05)))
        self.amplitude_weight.zero_()
        if self.beta is not None:
            self.beta.zero_()

    def forward(self, h, v, force):
        args = h, v, force, self.step, self.amplitude_weight, self.beta
        if h.is_cuda and h.dtype == torch.bfloat16:
            from .kernels import AmplitudeWrite

            return AmplitudeWrite.apply(*args)
        return reference_write(*args)


class AmplitudeChord(ChordRes):
    def __init__(self, config):
        super().__init__(config)
        count = 2 * config.n_layer
        self.residual_maps = nn.ModuleList(
            AmplitudeSite(i == count - 1) for i in range(count)
        )

    def _ordinary_hidden(self, idx):
        x = self._embed_reference(idx)
        h = unit_rms(x).to(x.dtype)
        v = torch.zeros_like(h)
        read = reference_read
        if h.is_cuda and h.dtype == torch.bfloat16:
            from nexttransformer.methods.chordres_zero.kernels import ReadNorm

            read = ReadNorm.apply
        cos_sin = self.cos[:, : idx.size(1)], self.sin[:, : idx.size(1)]
        for i, layer in enumerate(self.transformer.h):
            ve = (
                self.value_embeds[str(i)](idx).to(x.dtype)
                if str(i) in self.value_embeds
                else None
            )
            window = self._window(i, self.config.n_layer)
            site = self.residual_maps[2 * i]
            query = read(h, v, site.mu)
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
            h, v = site(h, v, force)
            site = self.residual_maps[2 * i + 1]
            query = read(h, v, site.mu)
            force = layer.mlp(query)
            h, v = site(h, v, force)
        return h, None


model = AmplitudeChord
