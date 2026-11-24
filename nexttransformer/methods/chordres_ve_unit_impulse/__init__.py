"""Static ChordRes adding each VE site's raw token vector directly to velocity."""
import torch

from nexttransformer.methods.chordres_zero import ChordRes, frozen, reference_read, reference_write, unit_rms


def reference_impulse_write(h, v, force, controls, ve):
    dtype = h.dtype
    h, v = h.to(controls.dtype), v.to(controls.dtype)
    _, perp, par, beta = controls.unbind()
    delta = unit_rms(force) - h
    along = (delta * v).mean(-1, keepdim=True) / v.square().mean(-1, keepdim=True).clamp_min(1e-12)
    nxt = unit_rms(h + perp * delta + ((par - perp) * along) * v)
    velocity = beta * v + (1 - beta) * (nxt - h) + ve.to(controls.dtype)
    tangent = velocity - (velocity * nxt).mean(-1, keepdim=True) * nxt
    return nxt.to(dtype), tangent.to(dtype)


class UnitVEImpulseChordRes(ChordRes):
    def _ordinary_hidden(self, idx):
        x = self._embed_reference(idx)
        h = unit_rms(x).to(x.dtype)
        v = torch.zeros_like(h)
        read, write, impulse = reference_read, reference_write, reference_impulse_write
        if h.is_cuda and h.dtype == torch.bfloat16:
            from nexttransformer.methods.chordres_zero.kernels import ChordWrite, ReadNorm
            from .kernels import ImpulseWrite
            read, write, impulse = ReadNorm.apply, ChordWrite.apply, ImpulseWrite.apply

        cos_sin = self.cos[:, :idx.size(1)], self.sin[:, :idx.size(1)]
        for i, layer in enumerate(self.transformer.h):
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            window = self._window(i, self.config.n_layer)
            attn = self.residual_maps[2 * i]()
            force = layer.attn(read(h, v, attn), ve, cos_sin, window, None)
            self._record("block", physical=i, virtual=i, window=window[0], tokens=idx.numel(),
                         sequence_len=idx.size(1), ve=ve is not None, mixer="softmax", residual=self.config.method)
            if ve is None:
                h, v = write(h, v, force, attn)
            else:
                h, v = impulse(h, v, force, attn, ve)
            mlp = self.residual_maps[2 * i + 1]()
            force = layer.mlp(read(h, v, mlp))
            h, v = write(h, v, force, mlp)
        return h, None


model = UnitVEImpulseChordRes
