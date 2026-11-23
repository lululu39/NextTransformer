"""Ablation of token-indexed attention values only, with the input smear preserved."""
from torch import nn
from nexttransformer.methods.chordres_zero_save_force import SavedForceChordRes, frozen


class ChordResWithoutValueEmbeddings(SavedForceChordRes):
    def __init__(self, config):
        super().__init__(config)
        self.value_embeds = nn.ModuleDict()
        for layer in self.transformer.h:
            layer.attn.ve_gate = None


model = ChordResWithoutValueEmbeddings


def check(config):
    if config.option("attention_value_embeddings"):
        raise ValueError("This ablation requires attention_value_embeddings=false")
