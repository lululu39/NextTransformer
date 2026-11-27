"""Static six-coefficient control for the tangent, soft-force token-lookup Chord."""
import torch
from torch import nn

from nexttransformer.methods.chordres_soft_force_token_velocity import TokenVelocityChord, frozen as frozen
from nexttransformer.methods.chordres_velocity_gains import VelocityGainSite
from nexttransformer.methods.chordres_zero import reference_read


class StaticSixSite(VelocityGainSite):
    def forward(self, h, v):
        controls = super().forward()
        if h.is_cuda and h.dtype == torch.bfloat16:
            from nexttransformer.methods.chordres_zero.kernels import ReadNorm
            query = ReadNorm.apply(h, v, controls)
        else:
            query = reference_read(h, v, controls)
        # SoftUpdateWrite reads six consecutive FP32 controls per token.
        return query, controls.expand(*h.shape[:-1], 6).contiguous()


class StaticSixLookupChord(TokenVelocityChord):
    def __init__(self, config):
        super().__init__(config)
        sites = 2 * config.n_layer
        self.residual_maps = nn.ModuleList(StaticSixSite(i == sites - 1) for i in range(sites))


model = StaticSixLookupChord
