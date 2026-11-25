from dataclasses import replace

import pytest
import torch

from launch.train_chord_control_lr import set_control_lr
from nexttransformer.methods.chordres_zero import unit_rms
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import build_optimizer


def fp64_projection(h, u):
    h, u = h.double(), u.double()
    return u - h * (h*u).sum(-1,keepdim=True) / h.square().sum(-1,keepdim=True).clamp_min(1e-12*h.size(-1))


def test_h0_input_and_paired_start():
    cfg=debug_config("chordres_soft_force_h0_velocity")
    model=build_model(cfg,seed=42)
    parent=build_model(replace(cfg,method="chordres_update_soft_force"),seed=42)
    ours,base=dict(model.named_parameters()),dict(parent.named_parameters())
    key="initial_velocity_projection.weight"
    assert set(ours)-set(base)=={key} and set(base)<=set(ours)
    assert all(torch.equal(p,ours[n]) and p.requires_grad==ours[n].requires_grad for n,p in base.items())
    proj=model.initial_velocity_projection
    assert proj.bias is None and proj.weight.shape==(cfg.width,cfg.width) and proj.weight.count_nonzero()==0
    ids=torch.arange(1,cfg.sequence_len+1).remainder(cfg.vocab_size).unsqueeze(0)
    with torch.no_grad():
        assert torch.equal(model(ids),parent(ids))
    def layout(m):
        opt=build_optimizer(m)
        set_control_lr(m,opt,None)
        return {name:{k:v for k,v in g.items() if k not in ('params','parameter_names')}
                for g in opt.param_groups for name in g['parameter_names']}
    old,new=layout(parent),layout(model)
    assert all(old[k]==new[k] for k in old) and new[key]==new['transformer.h.0.attn.c_proj.weight']
    assert new[key]['kind']=='muon'

    with torch.no_grad():
        proj.weight.normal_(std=.5/cfg.width**.5)
        model.smear_lambda.fill_(1.)
        embedded=model._embed_reference(ids)
        h0=unit_rms(embedded).to(embedded.dtype)
        u=torch.nn.functional.linear(h0,proj.weight.to(h0.dtype))
        v0=fp64_projection(h0,u).to(h0.dtype)
    observed=[]
    def record(site,inputs):
        observed.append(tuple(t.detach() for t in inputs))
    with model.residual_maps[0].register_forward_pre_hook(record):
        model(ids)
    first_h,first_v=observed[0]
    assert torch.equal(first_h,h0) and first_v.norm()>0
    torch.testing.assert_close(first_v,v0,rtol=.008,atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="requires the BF16 tangent kernel")
def test_cuda_oracle():
    from nexttransformer.methods.chordres_soft_force_h0_velocity.kernels import TangentProjection
    dev=torch.device('cuda',torch.cuda.current_device())
    gen=torch.Generator(device=dev).manual_seed(2901+dev.index)
    def sample():return torch.randn((19,1536),device=dev,generator=gen)
    h,u=sample(),sample()
    h[0].zero_();h[1]*=1e-8;u[2].zero_();h[3]*=3.
    h,u=[z.bfloat16().requires_grad_() for z in (h,u)]
    x=TangentProjection.apply(h,u)
    hx,ux=[t.detach().double().requires_grad_() for t in (h,u)]
    y=fp64_projection(hx,ux);dy=sample().bfloat16()
    grad=torch.autograd.grad(x,(h,u),dy)
    wanted=torch.autograd.grad(y,(hx,ux),dy.double())
    for actual,ref in zip((x,*grad),(y,*wanted),strict=True):
        error=(actual.double()-ref).norm(dim=-1)/ref.norm(dim=-1).clamp_min(1e-8)
        assert torch.isfinite(actual).all() and error.max()<.006,(actual.shape,error)
    cos=torch.nn.functional.cosine_similarity(h[4:].float(),x[4:].float(),dim=-1)
    assert cos.abs().max()<.001
