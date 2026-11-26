"""Token-indexed initial tangent velocity with the update-based soft-force Chord."""
import torch
from torch import nn
from torch.nn import functional as F

from nexttransformer.methods.chordres_soft_force_h0_velocity import tangent_projection
from nexttransformer.methods.chordres_update_soft_force import SoftUpdateChord, token_reference_write
from nexttransformer.methods.chordres_zero import frozen as frozen, unit_rms


class TokenVelocityChord(SoftUpdateChord):
    def __init__(self, config):
        super().__init__(config)
        # Registration on wte gives the independent table the native embedding dtype and optimizer.
        wte = self.transformer.wte
        wte.register_parameter("velocity_weight", nn.Parameter(torch.empty_like(wte.weight)))

    @torch.no_grad()
    def init_weights(self):
        super().init_weights()
        self.transformer.wte.velocity_weight.zero_()

    def _ordinary_hidden(self, idx):
        x = self._embed_reference(idx)
        h = unit_rms(x).to(x.dtype)
        table = self.transformer.wte.velocity_weight
        raw = F.embedding(idx, table)
        if h.is_cuda and h.dtype == torch.bfloat16:
            from nexttransformer.methods.chordres_soft_force_h0_velocity.kernels import TangentProjection
            v = TangentProjection.apply(h, raw)
        else:
            v = tangent_projection(h, raw)
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


model = TokenVelocityChord
