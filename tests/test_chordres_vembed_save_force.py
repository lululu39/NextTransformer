from dataclasses import replace

import torch

from launch.train_matched_normal import initialize_projections
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import parameter_groups


def test_velocity_table_is_the_only_new_weight_and_receives_task_gradients():
    config = debug_config("chordres_vembed_save_force")
    control = initialize_projections(build_model(replace(config, method="chordres_zero_save_force")), 42)
    model = initialize_projections(build_model(config), 42)
    original = dict(control.named_parameters())
    params = dict(model.named_parameters())
    name = "transformer.wte.velocity_weight"
    assert params.keys() - original.keys() == {name}
    for key, param in original.items():
        torch.testing.assert_close(params[key], param, rtol=0, atol=0)
    embed_group = next(g for g in parameter_groups(model) if name in g["parameter_names"])
    assert embed_group["kind"] == "adamw" and "transformer.wte.weight" in embed_group["parameter_names"]
    table = params[name]
    assert table.dtype == model.transformer.wte.weight.dtype
    ids = torch.randint(config.vocab_size, (2, config.sequence_len))
    h, v = model.initial_state(ids, model._embed_reference(ids))
    assert v.float().square().mean().sqrt() > .01
    assert (h.float()*v.float()).mean(-1).abs().max() < .001
    with torch.no_grad():
        saved = table.clone()
        table.zero_()
        torch.testing.assert_close(model(ids), control(ids), rtol=0, atol=0)
        table.copy_(saved)
        model.residual_maps[0].mu.fill_(.4)
    model(ids, ids).backward()
    assert table.grad.isfinite().all() and table.grad.norm() > 0
