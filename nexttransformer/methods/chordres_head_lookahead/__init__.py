"""Saved-force ChordRes from rest with a learned lookahead in the LM readout."""
import torch

from nexttransformer.methods.chordres_zero import ChordRes, frozen, reference_read, reference_write, unit_rms


class HeadLookahead(torch.nn.Module):
    """The parent readout applies the single final RMSNorm and the existing LM head."""
    def __init__(self):
        super().__init__()
        self.rho = torch.nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def reset_parameters(self):
        self.rho.zero_()

    def forward(self, h, v):
        dtype = torch.promote_types(h.dtype, torch.float32)
        return (h.to(dtype) + self.rho * v.to(dtype)).to(h.dtype)


class HeadLookaheadChordRes(ChordRes):
    def __init__(self, config):
        super().__init__(config)
        # Keep the 2L branch sites (including the fixed final retention) identical.
        # The final read belongs to the same auxiliary AdamW group as their controls.
        self.residual_maps.append(HeadLookahead())

    def _ordinary_hidden(self, idx):
        x = self._embed_reference(idx)
        h = unit_rms(x).to(x.dtype)
        v = torch.zeros_like(h)
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
        return self.residual_maps[-1](h, v), None


model = HeadLookaheadChordRes
