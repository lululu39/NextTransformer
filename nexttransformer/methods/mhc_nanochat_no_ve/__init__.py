"""Nanochat-style four-stream mHC with context-projected values and native input smear."""
from torch import nn

from nexttransformer.methods.mhc_nanochat_readout import NanochatMHC, frozen


class NoValueEmbeddingMHC(NanochatMHC):
    def __init__(self, config):
        super().__init__(config)
        self.value_embeds = nn.ModuleDict()
        for block in self.transformer.h:
            block.attn.ve_gate = None


model = NoValueEmbeddingMHC


def check(config):
    if config.option("attention_value_embeddings"):
        raise ValueError("This mHC comparison requires attention_value_embeddings=false")
