"""Fixed-zero tangent Chord with the attention value tables and gates removed."""
from torch import nn

from nexttransformer.methods.chordres_update_soft_force import SoftUpdateChord, frozen as frozen


class ZeroStartChordWithoutVE(SoftUpdateChord):
    def __init__(self, config):
        super().__init__(config)
        self.value_embeds = nn.ModuleDict()
        for block in self.transformer.h:
            block.attn.ve_gate = None


model = ZeroStartChordWithoutVE
