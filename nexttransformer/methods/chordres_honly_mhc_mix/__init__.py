"""Position-only token controls using the static six-coefficient Chord parameterization."""

import torch
from torch import nn
from torch.nn import functional as F

from nexttransformer.methods.chordres_token_controls import TokenControlledChord
from nexttransformer.methods.chordres_zero import ChordRes, frozen as frozen
from nexttransformer.methods.chordres_honly_scalar_recipe import HOnlyGateSite as BaseHOnlyGateSite
from nexttransformer.models.backbone import norm
from nexttransformer.methods.chordres_token_controls_joint_rms import GATE_EPS


INITIAL_SCALE = 0.01


def token_reference_read(h, v, weight, mu, steps, beta, gains, scales):
    dtype = weight.dtype
    position = h.to(dtype)
    inverse_rms = (position.square().mean(-1, keepdim=True) + GATE_EPS).rsqrt()
    z = F.linear(position, weight) * inverse_rms
    m = z[..., :1] * scales[0] + mu
    raw_steps = z[..., 1:3] * scales[1] + steps
    pivot = raw_steps.detach().clamp_min(0)
    alpha = pivot + ((-pivot).exp() + (raw_steps - pivot).exp()).log()
    if beta is None:
        retention = torch.full_like(m, 0.5)
        gamma = torch.full_like(alpha, 0.5)
    else:
        retention = (z[..., 3:4] * scales[2] + beta).sigmoid()
        gamma = z[..., 4:6] * scales[2] + gains
    controls = torch.cat((m, alpha, retention, gamma), -1)
    offset = m * v.to(dtype)
    read = (h.to(dtype) + offset) * (1 + offset.square().mean(-1, keepdim=True)).rsqrt()
    return norm(read.to(h.dtype)), controls


class HOnlyGateSite(BaseHOnlyGateSite):
    def __init__(self, width, terminal):
        super().__init__(width, terminal)
        self.mix_scales = nn.Parameter(torch.full((2 if terminal else 3,), INITIAL_SCALE))

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "mix_scales"):
            self.mix_scales.fill_(INITIAL_SCALE)

    def forward(self, h, v):
        args = (
            h,
            v,
            self.token_weight,
            self.mu,
            self.steps,
            self.beta,
            self.velocity_gains,
            self.mix_scales,
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
