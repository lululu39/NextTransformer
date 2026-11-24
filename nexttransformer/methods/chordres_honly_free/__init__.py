"""Tokenwise affine controls from normalized position with physical Chord velocity."""

import torch
from torch import nn
from torch.nn import functional as F

from nexttransformer.methods.chordres_joint_gate_bounded import INITIAL_ALPHA
from nexttransformer.methods.chordres_token_controls import TokenControlledChord
from nexttransformer.methods.chordres_zero import ChordRes, frozen as frozen
from nexttransformer.methods.chordres_velocity_gains import VelocityGainSite
from nexttransformer.models.backbone import norm
from nexttransformer.methods.chordres_token_controls_joint_rms import GATE_EPS


def token_reference_read(h, v, weight, mu, steps, beta, gains):
    dtype = weight.dtype
    position = h.to(dtype)
    inverse_rms = (position.square().mean(-1, keepdim=True) + GATE_EPS).rsqrt()
    z = F.linear(position, weight) * inverse_rms
    m = z[..., :1] + mu
    alpha = z[..., 1:3] + steps
    if beta is None:
        retention = torch.full_like(m, 0.5)
        gamma = torch.full_like(alpha, 0.5)
    else:
        retention = z[..., 3:4] + beta
        gamma = z[..., 4:6] + gains
    controls = torch.cat((m, alpha, retention, gamma), -1)
    offset = m * v.to(dtype)
    read = (h.to(dtype) + offset) * (1 + offset.square().mean(-1, keepdim=True)).rsqrt()
    return norm(read.to(h.dtype)), controls


class HOnlyGateSite(VelocityGainSite):
    def __init__(self, width, terminal):
        super().__init__(terminal)
        self.token_weight = nn.Parameter(torch.zeros(3 if terminal else 6, width))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        self.steps.fill_(INITIAL_ALPHA)
        if self.beta is not None:
            self.beta.fill_(0.5)
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


class HOnlyChord(TokenControlledChord):
    def __init__(self, config):
        ChordRes.__init__(self, config)
        count = 2 * config.n_layer
        self.residual_maps = nn.ModuleList(
            HOnlyGateSite(config.width, i == count - 1) for i in range(count)
        )


model = HOnlyChord
