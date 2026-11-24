from dataclasses import replace

import torch

from launch.train_matched_normal import build_matched_model
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model


def test_standard_no_ve_is_the_paired_additive_baseline():
    cfg = debug_config("residual_standard_no_ve")
    model = build_matched_model(build_model, cfg, "cpu", 42)
    full = build_matched_model(build_model, replace(cfg, method="residual_standard", options=()), "cpu", 42)
    chord = build_matched_model(build_model, replace(cfg, method="chordres_no_ve_save_force"), "cpu", 42)
    params = dict(model.named_parameters())
    refs = dict(full.named_parameters())
    chord_params = dict(chord.named_parameters())
    assert set(refs) - set(params) == {n for n in refs if n.startswith("value_embeds.") or ".ve_gate." in n}
    assert set(params) == {n for n in chord_params if not n.startswith("residual_maps.")}
    for name, param in params.items():
        assert torch.equal(param, refs[name]) and torch.equal(param, chord_params[name])
        assert param.requires_grad == refs[name].requires_grad == chord_params[name].requires_grad
    assert not model.value_embeds and all(layer.attn.ve_gate is None for layer in model.transformer.h)
    assert not any(p.requires_grad for p in (model.resid_lambdas,model.x0_lambdas,model.backout_lambda))
    ids = torch.randint(cfg.vocab_size, (2, cfg.sequence_len))
    labels = ids.roll(-1,1)
    with torch.no_grad():
        for table in full.value_embeds.values():
            table.weight.zero_()
        for m in (model,full):
            m.smear_lambda.fill_(.3)
    torch.testing.assert_close(model(ids),full(ids),rtol=0,atol=0)
    for m in (model,full):
        m(ids,labels).backward()
    for name,param in params.items():
        if param.requires_grad:
            torch.testing.assert_close(param.grad,refs[name].grad,rtol=0,atol=0)
    assert model.smear_gate.weight.grad.abs().sum() > 0
