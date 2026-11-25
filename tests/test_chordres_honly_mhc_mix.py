from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from launch.train_chord_control_lr import CONTROL_PEAK_LR, set_control_lr
from nexttransformer.methods.chordres_honly_mhc_mix import HOnlyGateSite
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import build_optimizer


def test_native_zero_and_optimizer_groups():
    cfg = debug_config("chordres_honly_mhc_mix")
    model = build_model(cfg, seed=42)
    original = build_model(replace(cfg, method="chordres_velocity_gains"), seed=42)
    ours = dict(model.named_parameters())
    for name, param in original.named_parameters():
        assert torch.equal(ours[name], param)
    for layer in model.transformer.h:
        for branch in (layer.attn,layer.mlp):
            assert branch.c_proj.weight.count_nonzero() == 0
    for site in model.residual_maps:
        assert torch.equal(site.mix_scales,torch.full_like(site.mix_scales,.01))
        assert site.token_weight.count_nonzero() == 0
    ids=torch.arange(cfg.sequence_len).view(1,-1)
    with torch.no_grad():
        torch.testing.assert_close(model(ids),original(ids),rtol=0,atol=0)
    opt = build_optimizer(model)
    set_control_lr(model, opt, None)
    static, dynamic = [g for g in opt.param_groups if g['parameter_names'][0].startswith('residual_maps.')]
    assert static['initial_lr'] == CONTROL_PEAK_LR
    assert dynamic['initial_lr'] == pytest.approx(.000848528137423857)
    assert set(dynamic['params']) == {p for site in model.residual_maps for p in (site.token_weight,site.mix_scales)}
    assert all(not n.endswith(('.mix_scales','.token_weight')) for n in static['parameter_names'])
    all_params = [p for g in opt.param_groups for p in g['params']]
    assert len(all_params) == len(set(all_params)) == sum(p.requires_grad for p in model.parameters())


def expected(site, h, v, terminal):
    normalized = F.rms_norm(h.float(),(h.shape[-1],),eps=1e-6)
    proj = F.linear(normalized,site.token_weight)
    groups = [0,1,1] if terminal else [0,1,1,2,2,2]
    biases = [site.mu,site.steps]
    if not terminal:
        biases.extend((site.beta,site.velocity_gains))
    z = torch.cat(biases) + proj * site.mix_scales[groups]
    tail = torch.full_like(z,.5) if terminal else torch.cat((z[...,3:4].sigmoid(),z[...,4:]),-1)
    controls = torch.cat((z[...,:1],F.softplus(z[...,1:3]),tail),-1)
    offset=controls[...,:1]*v.float()
    read=(h.float()+offset)*(1+offset.square().mean(-1,keepdim=True)).rsqrt()
    return F.rms_norm(read.bfloat16(),(h.shape[-1],)), controls


@pytest.mark.skipif(not torch.cuda.is_available(),reason="requires the fused BF16 gates")
def test_cuda_oracle():
    tf32=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        for width,terminal in ((1536,False),(96,True)):
            site=HOnlyGateSite(width,terminal).cuda()
            # Include a remainder in both the tile and reduction dimensions.
            h,v=[torch.randn(1,263,width,device="cuda").bfloat16().requires_grad_() for _ in range(2)]
            for starting in (True,False):
                if not starting:
                    with torch.no_grad():
                        site.token_weight.normal_(std=.4/width**.5)
                        scales=[0.,-.12] if terminal else [0.,-.12,.18]
                        site.mix_scales.copy_(torch.tensor(scales,device="cuda"))
                        site.mu.fill_(.2)
                args=[h,v,*site.parameters()]
                q,c=site(h,v)
                rq,rc=expected(site,h,v,terminal)
                torch.testing.assert_close(c,rc,atol=2e-6,rtol=1e-5)
                assert (q.float()-rq.float()).norm()/rq.float().norm()<.004
                for dq in (torch.zeros_like(q),torch.randn_like(q)):
                    dc=torch.randn_like(c)
                    got=torch.autograd.grad((q,c),args,(dq,dc),retain_graph=True)
                    want=torch.autograd.grad((rq,rc),args,(dq,dc),retain_graph=True)
                    for p,a,b in zip(args,got,want,strict=True):
                        err=(a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-8)
                        assert torch.isfinite(a).all() and err<.015,(a.shape,err)
                        if p is site.mix_scales:
                            if starting: assert a.count_nonzero()==0
                            else: assert torch.all(a!=0)
    finally:
        torch.backends.cuda.matmul.allow_tf32=tf32
