from dataclasses import replace

import torch

from launch.train_chord_control_lr import CONTROL_PEAK_LR, set_control_lr
from launch.train_matched_normal import build_matched_model
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import build_optimizer


def test_only_value_embeddings_and_gates_are_removed():
    config = debug_config("chordres_no_ve_save_force")
    control_config = replace(config, method="chordres_zero_save_force", options=())
    no_ve = build_matched_model(build_model, config, "cpu", 42)
    control = build_matched_model(build_model, control_config, "cpu", 42)
    params = dict(no_ve.named_parameters())
    old = dict(control.named_parameters())
    assert {name for name in old if name not in params} == {
        name for name in old if name.startswith("value_embeds.") or ".ve_gate." in name
    }
    for name, param in params.items():
        assert torch.equal(param, old[name]) and param.requires_grad == old[name].requires_grad
    assert not no_ve.value_embeds and all(block.attn.ve_gate is None for block in no_ve.transformer.h)
    opt = build_optimizer(no_ve)
    set_control_lr(no_ve, opt)
    group = next(g for g in opt.param_groups if "residual_maps.0.mu" in g["parameter_names"])
    assert group["initial_lr"] == CONTROL_PEAK_LR
    with torch.no_grad():
        for model in (no_ve, control):
            model.smear_lambda.fill_(.25)
            for site in model.residual_maps:
                site.mu.fill_(.3)
                site.steps[1].add_(.2)
        for table in control.value_embeds.values():
            table.weight.zero_()
    ids = torch.randint(config.vocab_size, (2, config.sequence_len))
    targets = ids.roll(-1,1)
    torch.testing.assert_close(no_ve(ids), control(ids), rtol=0, atol=0)
    for model in (no_ve, control):
        model(ids,targets).backward()
    for name, param in params.items():
        if param.requires_grad:
            torch.testing.assert_close(param.grad, old[name].grad, rtol=0, atol=0)
    assert no_ve.smear_gate.weight.grad.abs().sum() > 0
