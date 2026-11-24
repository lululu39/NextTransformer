"""Remove attention VE from the public native mHC; inherit its full forward unchanged."""
from dataclasses import replace

import torch
from torch import nn

from nexttransformer.models.lm import ReferenceLM, build_model


def is_ve_parameter(name):
    return name.startswith("value_embeds.") or ".ve_gate." in name


class OfficialMHCWithoutVE(ReferenceLM):
    def __init__(self, config):
        # Select the built-in mHC router, ordinary attention branches and sum readout.
        super().__init__(replace(config, method="mhc", options=()))
        self.value_embeds = nn.ModuleDict()
        for layer in self.transformer.h:
            layer.attn.ve_gate = None

    @torch.no_grad()
    def init_weights(self):
        # Initialize nonpersistent RoPE buffers and the native embedding dtypes.
        super().init_weights()
        # Omitting VE gates changes the RNG consumed by native module resets.
        # Copy the exact native draw for every retained weight, then discard the donor.
        native = build_model(self.config, self.transformer.wte.weight.device, seed=torch.initial_seed())
        shared = {k: v for k, v in native.state_dict().items() if not is_ve_parameter(k)}
        self.load_state_dict(shared, strict=True)


model = OfficialMHCWithoutVE


def check(config):
    if config.option("attention_value_embeddings"):
        raise ValueError("This fixed ablation requires attention_value_embeddings=false")
