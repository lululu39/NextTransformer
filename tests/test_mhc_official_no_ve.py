from dataclasses import replace

import torch

from nexttransformer.methods.mhc_official_no_ve import is_ve_parameter
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import ReferenceLM, build_model
from nexttransformer.models.residual import ManifoldHyperConnection
from nexttransformer.training.optimizer import parameter_groups


def test_official_mhc_ablation_changes_only_value_embeddings():
    cfg = debug_config("mhc_official_no_ve")
    ablated = build_model(cfg, seed=42)
    official = build_model(replace(cfg, method="mhc", options=()), seed=42)
    assert type(ablated).forward is ReferenceLM.forward
    assert type(ablated)._residual_hidden is ReferenceLM._residual_hidden
    assert ablated.config == official.config
    assert all(type(site) is ManifoldHyperConnection for site in ablated.residual_maps)
    assert len(ablated.residual_maps) == 2*cfg.n_layer
    assert not ablated.value_embeds and all(b.attn.ve_gate is None for b in ablated.transformer.h)

    ours, full = dict(ablated.named_parameters()), dict(official.named_parameters())
    assert set(ours) == {k for k in full if not is_ve_parameter(k)}
    for k, p in ours.items():
        assert torch.equal(p, full[k]) and p.requires_grad == full[k].requires_grad
    for x,y in zip(ablated.transformer.h,official.transformer.h,strict=True):
        assert torch.count_nonzero(x.attn.c_proj.weight)==torch.count_nonzero(y.mlp.c_proj.weight)==0
    def settings(model):
        return {n:{key:v for key,v in g.items() if key not in ("params","parameter_names")}
                for g in parameter_groups(model) for n in g["parameter_names"]}
    g1,g2=settings(ablated),settings(official)
    assert all(g1[k]==g2[k] for k in g1)
    # Exercise the nonlinear routing and residual branches away from their zero start.
    with torch.no_grad():
        for name,p in ours.items():
            if name.endswith('c_proj.weight') or name.endswith('.projection'):
                p.normal_(std=.03)
                full[name].copy_(p)
        ablated.smear_lambda.fill_(.3); official.smear_lambda.fill_(.3)
        for ve in official.value_embeds.values():
            ve.weight.zero_()
    ids=torch.randint(cfg.vocab_size,(2,cfg.sequence_len)); target=ids.roll(-1,1)
    torch.testing.assert_close(ablated(ids),official(ids),rtol=0,atol=0)
    for model in (ablated,official):
        model(ids,target).backward()
    for name,p in ours.items():
        if p.requires_grad:
            torch.testing.assert_close(p.grad,full[name].grad,rtol=0,atol=0)
    assert ablated.residual_maps[0].projection.grad.norm()>0
