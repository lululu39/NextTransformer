"""Project-only ChordRes from rest on the Standard Residual full-attention shell."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from nexttransformer.models.backbone import Linear, norm
from nexttransformer.models.lm import ReferenceLM


def unit_rms(x):
    x = x.to(torch.promote_types(x.dtype, torch.float32))
    return x * x.square().mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()


def reference_read(h, v, controls):
    offset = controls[0] * v.to(controls.dtype)
    preview = (h.to(controls.dtype) + offset) * (1 + offset.square().mean(-1, keepdim=True)).rsqrt()
    return norm(preview.to(h.dtype))


def reference_write(h, v, force, controls):
    dtype = h.dtype
    h, v = h.to(controls.dtype), v.to(controls.dtype)
    _, perp, par, beta = controls.unbind()
    delta = unit_rms(force) - h
    along = (delta * v).mean(-1, keepdim=True) / v.square().mean(-1, keepdim=True).clamp_min(1e-12)
    nxt = unit_rms(h + perp * delta + ((par - perp) * along) * v)
    mix = beta * v + (1 - beta) * (nxt - h)
    return nxt.to(dtype), (mix - (mix * nxt).mean(-1, keepdim=True) * nxt).to(dtype)


def reference_projected_write(h, v, controls, inner, weight):
    return reference_write(h, v, F.linear(inner, weight), controls)


class ChordSite(nn.Module):
    def __init__(self, terminal):
        super().__init__()
        self.mu = nn.Parameter(torch.empty(1))
        self.steps = nn.Parameter(torch.empty(2))
        self.beta = None if terminal else nn.Parameter(torch.empty(1))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.mu.zero_()
        self.steps.fill_(math.log(math.expm1(.05)))
        if self.beta is not None:
            self.beta.zero_()

    def forward(self):
        # Stable softplus expressed with primitives covered by the cost ledger.
        shift = self.steps.detach().clamp_min(0)
        steps = shift + ((-shift).exp() + (self.steps - shift).exp()).log()
        beta = torch.full_like(self.mu, .5) if self.beta is None else self.beta.sigmoid()
        return torch.cat((self.mu, steps, beta))


class ForceProjection(Linear):
    """Defer the final GEMM to the Chord write so backward can reconstruct its force."""
    def forward(self, x):
        return x, self.weight.to(x.dtype)


def block(block, config, layer):
    for branch in (block.attn, block.mlp):
        proj = branch.c_proj
        branch.c_proj = ForceProjection(proj.in_features, proj.out_features, bias=False)
    return block


def _embedding_reset_by_shell():
    """The shell initializes embeddings once with the same RNG stream as residual_standard."""


class ChordRes(ReferenceLM):
    def __init__(self, config):
        super().__init__(config)
        sites = 2 * config.n_layer
        self.residual_maps = nn.ModuleList(ChordSite(i == sites - 1) for i in range(sites))
        # build_model resets plugin modules once more before the common init_weights.
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                module.reset_parameters = _embedding_reset_by_shell

    def _ordinary_hidden(self, idx):
        x = self._embed_reference(idx)
        h = unit_rms(x).to(x.dtype)
        v = torch.zeros_like(h)
        fast = h.is_cuda and h.dtype == torch.bfloat16
        if fast:
            from .kernels import ReadNorm, ChordWrite
        cos_sin = self.cos[:, :idx.size(1)], self.sin[:, :idx.size(1)]
        for i, layer in enumerate(self.transformer.h):
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            window = self._window(i, self.config.n_layer)
            for j in range(2):
                site = 2 * i + j
                controls = self.residual_maps[site]()
                query = ReadNorm.apply(h, v, controls) if fast else reference_read(h, v, controls)
                if j == 0:
                    inner, weight = layer.attn(query, ve, cos_sin, window, None)
                    self._record("block", physical=i, virtual=i, window=window[0], tokens=idx.numel(),
                                 sequence_len=idx.size(1), ve=ve is not None, mixer="softmax", residual="chordres_zero")
                else:
                    inner, weight = layer.mlp(query)
                if fast:
                    force = F.linear(inner, weight)
                    h, v = ChordWrite.apply(h, v, force, controls, F.linear, inner, weight)
                elif self.training and torch.is_grad_enabled():
                    # Account for replaying the force projection during backward.
                    h, v = checkpoint(reference_projected_write, h, v, controls, inner, weight,
                                      use_reentrant=False, preserve_rng_state=False)
                else:
                    h, v = reference_projected_write(h, v, controls, inner, weight)
        return h, None


model = ChordRes
frozen = ("resid_lambdas", "x0_lambdas", "backout_lambda")
