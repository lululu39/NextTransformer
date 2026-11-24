"""Joint-RMS token-conditioned Chord with bounded lookahead, position and velocity gains."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from nexttransformer.methods.chordres_token_controls import (
    TokenControlledChord,
    TokenSite,
)
from nexttransformer.methods.chordres_zero import ChordRes, frozen as frozen
from nexttransformer.models.backbone import norm
from nexttransformer.methods.chordres_token_controls_joint_rms import GATE_EPS


# The original FP32 softplus bias evaluates to this precise value at a nominal step of 0.05.
INITIAL_ALPHA = 0.049999985843896866


def token_reference_read(h, v, weight, mu, steps, beta, gains):
    dtype = weight.dtype
    position, velocity = h.to(dtype), v.to(dtype)
    mean_square = (position.square() + velocity.square()).mean(-1, keepdim=True) * 0.5
    gate_inv = (mean_square + GATE_EPS).rsqrt()
    z = (F.linear(position, weight[:, 0]) + F.linear(velocity, weight[:, 1])) * gate_inv
    m = 3.0 * torch.tanh((z[..., :1] + mu) / 3.0)
    # Stable sigmoid in odds form, centered at the previous effective FP32 step.
    raw_alpha = z[..., 1:3] + steps
    pivot = raw_alpha.detach().clamp_min(0)
    numerator = INITIAL_ALPHA * (raw_alpha - pivot).exp()
    complement = (1.0 - INITIAL_ALPHA) * (-pivot).exp()
    alpha = numerator / (numerator + complement)
    if beta is None:
        retention = torch.full_like(m, 0.5)
        gamma = torch.full_like(alpha, 0.5)
    else:
        retention = (z[..., 3:4] + beta).sigmoid()
        gamma = 2.0 * (z[..., 4:] + gains).sigmoid()
    controls = torch.cat((m, alpha, retention, gamma), -1)
    offset = m * v.to(dtype)
    read = (h.to(dtype) + offset) * (1 + offset.square().mean(-1, keepdim=True)).rsqrt()
    return norm(read.to(h.dtype)), controls


class BoundedGateSite(TokenSite):
    def __init__(self, width, terminal):
        super().__init__(width, terminal)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        self.steps.zero_()
        if getattr(self, "velocity_gains", None) is not None:
            self.velocity_gains.fill_(-math.log(3.0))

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


class BoundedGateChord(TokenControlledChord):
    def __init__(self, config):
        ChordRes.__init__(self, config)
        count = 2 * config.n_layer
        self.residual_maps = nn.ModuleList(
            BoundedGateSite(config.width, i == count - 1) for i in range(count)
        )


model = BoundedGateChord
