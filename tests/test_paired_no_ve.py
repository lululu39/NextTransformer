"""Check native seeded VE ablations on the same backbone tensors."""
from dataclasses import replace

import pytest
import torch

from launch.train_paired_no_ve import PARENTS, build_paired, is_ve_weight
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model


@pytest.mark.parametrize("method", tuple(PARENTS))
def test_paired_initialization_and_forward(method):
    config = debug_config(method)
    check_pair(config, "cpu")


def check_pair(config, device):
    reference = replace(config, method=PARENTS[config.method], options=())
    model = build_paired(build_model, config, device, seed=42)
    control = build_model(reference, device, seed=42)
    params = dict(model.named_parameters())
    controls = dict(control.named_parameters())
    assert not model.value_embeds and all(b.attn.ve_gate is None for b in model.transformer.h)
    assert params.keys() == {n for n in controls if not is_ve_weight(n)}
    assert model.smear_lambda.requires_grad and model.smear_gate.weight.requires_grad
    for name, value in params.items():
        assert torch.equal(value, controls[name]) and value.requires_grad == controls[name].requires_grad
    # Exercise the residual paths beyond the trivial native-zero output projections.
    with torch.no_grad():
        for net_block, ref_block in zip(model.transformer.h, control.transformer.h, strict=True):
            for a, b in [(net_block.attn, ref_block.attn), (net_block.mlp, ref_block.mlp)]:
                a.c_proj.weight.normal_(std=.02)
                b.c_proj.weight.copy_(a.c_proj.weight)
        for emb in control.value_embeds.values():
            emb.weight.zero_()
        model.smear_lambda.fill_(.3)
        control.smear_lambda.copy_(model.smear_lambda)
    ids = torch.arange(config.sequence_len, device=device).view(1, -1) % config.vocab_size
    x,y=model(ids),control(ids)
    torch.testing.assert_close(x,y,rtol=0,atol=0)
    x.float().square().mean().backward()
    y.float().square().mean().backward()
    for name, value in params.items():
        other=controls[name].grad
        if value.requires_grad:
            assert other is not None and value.grad is not None
            torch.testing.assert_close(value.grad,other,rtol=.01,atol=.001)


def test_segment_boundary_preserves_native_schedule():
    from types import SimpleNamespace
    from launch.train_paired_no_ve import install
    opts = SimpleNamespace(smoke_steps=0, max_wall_hours=0.)
    steps = []
    def native_schedule(optimizer, step, config):
        steps.append((step, config.steps))
        return 1.0
    trainer = SimpleNamespace(parse=lambda argv: opts, build_model=build_model, set_schedule=native_schedule)
    install(trainer)
    assert trainer.parse([]) is opts
    schedule = SimpleNamespace(steps=46875)
    assert trainer.set_schedule(None,15998,schedule)==1.0 and opts.max_wall_hours==0.
    assert trainer.set_schedule(None,15999,schedule)==1.0 and opts.max_wall_hours>0
    assert steps==[(15998,46875),(15999,46875)]
