"""ChordRes with a token-only tangent initial velocity and saved BF16 forces."""
import torch
from torch import nn
from torch.nn import functional as F

from nexttransformer.methods.chordres_zero import ChordRes, frozen, reference_read, reference_write, unit_rms


class TokenVelocityChordRes(ChordRes):
    def __init__(self, config):
        super().__init__(config)
        # Both token tables use the native embedding AdamW group and storage dtype.
        wte = self.transformer.wte
        wte.register_parameter("velocity_weight", nn.Parameter(torch.empty_like(wte.weight)))

    @torch.no_grad()
    def init_weights(self):
        super().init_weights()
        weight = self.transformer.wte.velocity_weight
        # Independent draws keep every shared tensor and the training RNG matched.
        generator = torch.Generator(device=weight.device).manual_seed(torch.initial_seed() + 2)
        weight.normal_(mean=0., std=.02, generator=generator)

    def initial_state(self, idx, x):
        h = unit_rms(x)
        raw = F.embedding(idx, self.transformer.wte.velocity_weight).float()
        v = raw - (h * raw).mean(dim=-1, keepdim=True) * h
        return h.to(x.dtype), v.to(x.dtype)

    def _ordinary_hidden(self, idx):
        x = self._embed_reference(idx)
        h, v = self.initial_state(idx, x)
        read, write = reference_read, reference_write
        if h.is_cuda and h.dtype == torch.bfloat16:
            from nexttransformer.methods.chordres_zero.kernels import ChordWrite, ReadNorm
            read, write = ReadNorm.apply, ChordWrite.apply

        cos_sin = self.cos[:, :idx.size(1)], self.sin[:, :idx.size(1)]
        for i, layer in enumerate(self.transformer.h):
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            window = self._window(i, self.config.n_layer)
            attn = self.residual_maps[2 * i]()
            force = layer.attn(read(h, v, attn), ve, cos_sin, window, None)
            self._record("block", physical=i, virtual=i, window=window[0], tokens=idx.numel(),
                         sequence_len=idx.size(1), ve=ve is not None, mixer="softmax", residual=self.config.method)
            h, v = write(h, v, force, attn)
            mlp = self.residual_maps[2 * i + 1]()
            force = layer.mlp(read(h, v, mlp))
            h, v = write(h, v, force, mlp)
        return h, None


model = TokenVelocityChordRes
