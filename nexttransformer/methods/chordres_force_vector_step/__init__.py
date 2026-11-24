"""Single-step spherical Chord with position steps predicted from the full branch vector."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from nexttransformer.methods.chordres_force_amplitude_step import AmplitudeChord
from nexttransformer.methods.chordres_zero import (
    ChordRes,
    frozen as frozen,
    unit_rms,
)


def reference_write(h, v, force, bias, force_weight, raw_beta):
    dtype = h.dtype
    h, v, force = h.to(bias.dtype), v.to(bias.dtype), force.to(bias.dtype)
    power = force.square().mean(-1, keepdim=True)
    raw = F.linear(force, force_weight.unsqueeze(0), bias)
    pivot = raw.detach().clamp_min(0)
    alpha = pivot + ((-pivot).exp() + (raw - pivot).exp()).log()
    beta = torch.full_like(bias, 0.5) if raw_beta is None else raw_beta.sigmoid()
    delta = force * power.clamp_min(1e-12).rsqrt() - h
    nxt = unit_rms(h + alpha * delta)
    mix = beta * v + (1 - beta) * (nxt - h)
    return nxt.to(dtype), (mix - (mix * nxt).mean(-1, keepdim=True) * nxt).to(dtype)


class VectorSite(nn.Module):
    def __init__(self, width, terminal):
        super().__init__()
        self.mu = nn.Parameter(torch.empty(1))
        self.step = nn.Parameter(torch.empty(1))
        self.force_weight = nn.Parameter(torch.empty(width))
        self.beta = None if terminal else nn.Parameter(torch.empty(1))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.mu.zero_()
        self.step.fill_(math.log(math.expm1(0.05)))
        self.force_weight.zero_()
        if self.beta is not None:
            self.beta.zero_()

    def forward(self, h, v, force):
        args = h, v, force, self.step, self.force_weight, self.beta
        if h.is_cuda and h.dtype == torch.bfloat16:
            from .kernels import VectorWrite

            return VectorWrite.apply(*args)
        return reference_write(*args)


class VectorChord(AmplitudeChord):
    def __init__(self, config):
        ChordRes.__init__(self, config)
        count = 2 * config.n_layer
        self.residual_maps = nn.ModuleList(
            VectorSite(config.width, i == count - 1) for i in range(count)
        )


model = VectorChord
