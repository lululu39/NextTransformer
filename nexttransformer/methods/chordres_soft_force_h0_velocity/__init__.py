"""Initialize velocity once from the starting Chord position after token smear."""
import torch
from torch import nn

from nexttransformer.methods.chordres_update_soft_force import SoftUpdateChord, token_reference_write
from nexttransformer.methods.chordres_zero import frozen as frozen, unit_rms
from nexttransformer.models.backbone import Linear


def tangent_projection(h, u):
    dtype = torch.promote_types(h.dtype, torch.float32)
    hf, uf = h.to(dtype), u.to(dtype)
    norm_sq = hf.square().mean(dim=-1, keepdim=True).clamp_min(1e-12)
    radial = (uf * hf).mean(dim=-1, keepdim=True) / norm_sq
    return (uf - radial * hf).to(h.dtype)


class InitialVelocity(Linear):
    def reset_parameters(self):
        # Also used after meta construction. Preserve the paired backbone RNG.
        nn.init.zeros_(self.weight)

    def forward(self, h):
        u = super().forward(h)
        if h.is_cuda and h.dtype == torch.bfloat16:
            from .kernels import TangentProjection
            return TangentProjection.apply(h, u)
        return tangent_projection(h, u)


class H0VelocityChord(SoftUpdateChord):
    def __init__(self, config):
        super().__init__(config)
        self.initial_velocity_projection = InitialVelocity(config.width, config.width, bias=False)

    def _ordinary_hidden(self, idx):
        x = self._embed_reference(idx)
        h = unit_rms(x).to(x.dtype)
        v = self.initial_velocity_projection(h)
        write = token_reference_write
        if h.is_cuda and h.dtype == torch.bfloat16:
            from nexttransformer.methods.chordres_update_soft_force.kernels import SoftUpdateWrite

            write = SoftUpdateWrite.apply
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


model = H0VelocityChord
