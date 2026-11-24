"""Token-conditioned Chord with joint position-velocity RMS normalization at the gate input."""

import torch
from torch import nn
from torch.nn import functional as F

from nexttransformer.methods.chordres_token_controls import (
    TokenControlledChord,
    TokenSite,
)
from nexttransformer.methods.chordres_zero import ChordRes, frozen as frozen
from nexttransformer.models.backbone import norm

GATE_EPS = 1e-6


def token_reference_read(h, v, weight, mu, steps, beta, gains):
    dtype = weight.dtype
    position, velocity = h.to(dtype), v.to(dtype)
    mean_square = (position.square() + velocity.square()).mean(-1, keepdim=True) * 0.5
    gate_inv = (mean_square + GATE_EPS).rsqrt()
    z = (F.linear(position, weight[:, 0]) + F.linear(velocity, weight[:, 1])) * gate_inv
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


class JointGateSite(TokenSite):
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


class JointGateChord(TokenControlledChord):
    def __init__(self, config):
        ChordRes.__init__(self, config)
        count = 2 * config.n_layer
        self.residual_maps = nn.ModuleList(
            JointGateSite(config.width, i == count - 1) for i in range(count)
        )


model = JointGateChord
