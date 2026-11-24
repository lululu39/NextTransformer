"""Standard Pre-Norm residuals with the input smear and context-projected values."""
from torch import nn

from nexttransformer.models.lm import ReferenceLM


class StandardWithoutVE(ReferenceLM):
    def __init__(self, config):
        super().__init__(config)
        self.value_embeds = nn.ModuleDict()
        for block in self.transformer.h:
            block.attn.ve_gate = None

    def _ordinary_hidden(self, idx):
        h = self._embed_reference(idx)
        length = idx.size(1)
        cos_sin = self.cos[:, :length], self.sin[:, :length]
        for i, block in enumerate(self.transformer.h):
            window = self._window(i, self.config.n_layer)
            h = block(h, None, cos_sin, window, None)
            self._record("block", physical=i, virtual=i, window=window[0], tokens=idx.numel(),
                         ve=False, mixer="softmax", residual=self.config.method)
        return h, None


model = StandardWithoutVE
frozen = ("resid_lambdas", "x0_lambdas", "backout_lambda")


def check(config):
    if config.option("attention_value_embeddings"):
        raise ValueError("This ablation requires attention_value_embeddings=false")
