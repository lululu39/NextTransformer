"""Nanochat four-stream mHC router and learned readout on the NextTransformer shell."""

import math
import torch
from torch import nn
from torch.nn import functional as F

from nexttransformer.models.backbone import norm
from nexttransformer.models.lm import ReferenceLM
from nexttransformer.models.residual import coefficient_precision, sinkhorn_knopp


def projected_logits(x, weight):
    flat = x.flatten(-2)
    normalized = F.rms_norm(flat, (flat.size(-1),), eps=torch.finfo(torch.float32).eps)
    return coefficient_precision(F.linear(normalized, weight.to(flat.dtype)))


class Router(nn.Module):
    def __init__(self, width, streams, index, iterations, head=False):
        super().__init__()
        self.streams, self.index, self.iterations, self.head = streams, index, iterations, head
        outputs = streams if head else 2 * streams + streams**2
        self.projection = nn.Parameter(torch.empty(outputs, width * streams))
        self.bias = nn.Parameter(torch.empty(outputs))
        self.scales = nn.Parameter(torch.empty(1 if head else 3))

    @torch.no_grad()
    def reset_parameters(self):
        n = self.streams
        # A private, site-specific draw preserves the ordinary backbone RNG.
        rng = torch.Generator(device=self.projection.device).manual_seed(torch.initial_seed() + self.index + 1)
        self.projection.normal_(std=self.projection.size(-1)**-.5, generator=rng)
        self.scales.fill_(.01)
        self.bias.zero_()
        self.bias[:n].fill_(-math.log(n-1))
        if not self.head:
            self.bias[2*n:].view(n,n).copy_(torch.eye(n, device=self.bias.device))

    def forward(self, x):
        n = self.streams
        logits = projected_logits(x, self.projection)
        pre = (logits[..., :n]*self.scales[0] + self.bias[:n]).sigmoid()
        branch = torch.einsum("btn,btnd->btd", pre.to(x.dtype), x)
        if self.head:
            return branch
        post = 2 * (logits[..., n:2*n]*self.scales[1] + self.bias[n:2*n]).sigmoid()
        mix_logits = (logits[..., 2*n:]*self.scales[2] + self.bias[2*n:]).unflatten(-1, (n,n))
        # Nanochat rows are input streams. Transposing makes its row-then-column
        # normalization use the native column-then-row Sinkhorn implementation.
        mix = sinkhorn_knopp(mix_logits.transpose(-1,-2), self.iterations)
        residual = torch.einsum("btmn,btnd->btmd", mix.to(x.dtype), x)
        return branch, residual, post


def shell_initializes_embedding():
    """The native shell initializes this table after plugin components are reset."""


class NanochatMHC(ReferenceLM):
    def __init__(self, config):
        super().__init__(config)
        n, iters = config.option("streams"), config.option("iterations")
        sites = 2 * config.n_layer
        self.residual_maps = nn.ModuleList(Router(config.width,n,k,iters,head=(k==sites)) for k in range(sites+1))
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                module.reset_parameters = shell_initializes_embedding

    def _ordinary_hidden(self, idx):
        x = self._embed_reference(idx)
        n = self.config.option("streams")
        streams = x.unsqueeze(-2).expand(-1,-1,n,-1)
        for i, layer in enumerate(self.transformer.h):
            for sublayer in range(2):
                k = 2*i+sublayer
                branch, residual, post = self.residual_maps[k](streams)
                update = self._attention_branch(branch,idx,i) if sublayer==0 else layer.mlp(norm(branch))
                streams = residual + post.to(update.dtype).unsqueeze(-1)*update.unsqueeze(-2)
                self._record("residual_map", sublayer=k, streams=n, tokens=idx.numel())
        return self.residual_maps[-1](streams), None


model = NanochatMHC
frozen = ("resid_lambdas", "x0_lambdas", "backout_lambda")
