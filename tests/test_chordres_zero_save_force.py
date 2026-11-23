from dataclasses import replace

import torch

from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model


def test_saved_forces_match_replay_with_one_projection_per_sublayer():
    config = debug_config("chordres_zero_save_force")
    saved = build_model(config)
    replay = build_model(replace(config, method="chordres_zero"))
    for key, value in saved.state_dict().items():
        torch.testing.assert_close(value, replay.state_dict()[key], rtol=0, atol=0)

    with torch.no_grad():
        for layer in saved.transformer.h:
            layer.attn.c_proj.weight.normal_(std=.03)
            layer.mlp.c_proj.weight.normal_(std=.03)
        for site in saved.residual_maps:
            site.mu.fill_(.4)
            site.steps[1].add_(.3)
    replay.load_state_dict(saved.state_dict())
    calls = []
    for layer in saved.transformer.h:
        for branch in (layer.attn, layer.mlp):
            branch.c_proj.register_forward_hook(lambda module, args, result: calls.append(module))
    ids = torch.randint(config.vocab_size, (2, config.sequence_len))
    losses = [m(ids, ids) for m in (saved, replay)]
    for loss in losses:
        loss.backward()
    assert len(calls) == len(set(calls)) == 2 * config.n_layer
    torch.testing.assert_close(*losses, rtol=0, atol=0)
    other = dict(replay.named_parameters())
    for name, param in saved.named_parameters():
        if param.requires_grad:
            torch.testing.assert_close(param.grad, other[name].grad, rtol=1e-5, atol=1e-6)
