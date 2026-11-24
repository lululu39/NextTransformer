"""Position-only Chord with one unsplit position step and one velocity mixing coefficient."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from nexttransformer.methods.chordres_zero import (
    ChordRes,
    frozen as frozen,
    unit_rms,
)
from nexttransformer.models.backbone import norm
from nexttransformer.methods.chordres_token_controls_joint_rms import GATE_EPS


def token_reference_read(h, v, weight, mu, step, beta):
    dtype = weight.dtype
    position = h.to(dtype)
    inverse_rms = (position.square().mean(-1, keepdim=True) + GATE_EPS).rsqrt()
    z = F.linear(position, weight) * inverse_rms
    m = z[..., :1] + mu
    raw_steps = z[..., 1:2] + step
    pivot = raw_steps.detach().clamp_min(0)
    alpha = pivot + ((-pivot).exp() + (raw_steps - pivot).exp()).log()
    if beta is None:
        retention = torch.full_like(m, 0.5)
    else:
        retention = (z[..., 2:3] + beta).sigmoid()
    controls = torch.cat((m, alpha, retention), -1)
    offset = m * v.to(dtype)
    read = (h.to(dtype) + offset) * (1 + offset.square().mean(-1, keepdim=True)).rsqrt()
    return norm(read.to(h.dtype)), controls


class HOnlyGateSite(nn.Module):
    def __init__(self, width, terminal):
        super().__init__()
        self.mu = nn.Parameter(torch.empty(1))
        self.step = nn.Parameter(torch.empty(1))
        self.beta = None if terminal else nn.Parameter(torch.empty(1))
        self.token_weight = nn.Parameter(torch.empty(2 if terminal else 3, width))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.mu.zero_()
        self.step.fill_(math.log(math.expm1(0.05)))
        if self.beta is not None:
            self.beta.zero_()
        self.token_weight.zero_()

    def forward(self, h, v):
        args = (
            h,
            v,
            self.token_weight,
            self.mu,
            self.step,
            self.beta,
        )
        if h.is_cuda and h.dtype == torch.bfloat16:
            from .kernels import TokenRead

            return TokenRead.apply(*args)
        return token_reference_read(*args)


def token_reference_write(h, v, force, controls):
    dtype = h.dtype
    h, v = h.to(controls.dtype), v.to(controls.dtype)
    _, alpha, beta = controls.unsqueeze(-1).unbind(-2)
    delta = unit_rms(force) - h
    nxt = unit_rms(h + alpha * delta)
    mix = beta * v + (1 - beta) * (nxt - h)
    return nxt.to(dtype), (mix - (mix * nxt).mean(-1, keepdim=True) * nxt).to(dtype)


class HOnlyChord(ChordRes):
    def __init__(self, config):
        super().__init__(config)
        count = 2 * config.n_layer
        self.residual_maps = nn.ModuleList(
            HOnlyGateSite(config.width, i == count - 1) for i in range(count)
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


model = HOnlyChord
