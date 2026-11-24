from dataclasses import replace
import torch

from launch.train_matched_normal import build_matched_model
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import parameter_groups


def test_only_attention_ve_is_removed_from_the_nanochat_mhc():
    cfg = debug_config("mhc_nanochat_no_ve")
    opts = {n: v for n, v in cfg.options if n != "attention_value_embeddings"}
    full = build_matched_model(build_model, replace(cfg, method="mhc_nanochat_readout", options=opts), "cpu", 42)
    reduced = build_matched_model(build_model, cfg, "cpu", 42)
    standard = build_matched_model(build_model, replace(cfg, method="residual_standard_no_ve",
                                  options={"attention_value_embeddings": False}), "cpu", 42)
    ours = dict(reduced.named_parameters())
    complete = dict(full.named_parameters())
    assert complete.keys()-ours.keys() == {n for n in complete if n.startswith("value_embeds.") or ".ve_gate." in n}
    for name,p in ours.items():
        assert torch.equal(p,complete[name])
    for name,p in standard.named_parameters():
        assert torch.equal(p,ours[name])
    assert not reduced.value_embeds and all(layer.attn.ve_gate is None for layer in reduced.transformer.h)
    assert len(reduced.residual_maps) == 2 * cfg.n_layer + 1 and reduced.residual_maps[-1].head
    groups = parameter_groups(reduced)
    group = next(g for g in groups if 'residual_maps.0.projection' in g['parameter_names'])
    assert group['kind']=='adamw' and group['weight_decay']==0.
    with torch.no_grad():
        for emb in full.value_embeds.values():
            emb.weight.zero_()
        reduced.smear_lambda.fill_(.25)
        full.smear_lambda.fill_(.25)
    ids=torch.randint(cfg.vocab_size,(2,cfg.sequence_len))
    torch.testing.assert_close(reduced(ids),full(ids),rtol=0,atol=0)
    for model in (reduced,full):
        model(ids,ids.roll(-1,1)).backward()
    for n,p in ours.items():
        if p.requires_grad:
            torch.testing.assert_close(p.grad,complete[n].grad,rtol=0,atol=0)
    assert reduced.residual_maps[-1].projection.grad.norm() > 0
