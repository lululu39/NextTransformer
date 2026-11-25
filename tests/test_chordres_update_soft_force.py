from dataclasses import replace

import pytest
import torch

from launch.train_chord_control_lr import set_control_lr
from nexttransformer.methods.chordres_update_soft_force import token_reference_write
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import build_optimizer


def definition(h, v, f, c):
    def project(x, direction):
        den = direction.square().mean(-1, keepdim=True).clamp_min(1e-12)
        return (x * direction).mean(-1, keepdim=True) / den * direction
    def tangent(x, unit):
        return x - (x * unit).mean(-1, keepdim=True) * unit

    delta = f / torch.sqrt(f.square().mean(-1, keepdim=True) + .05**2)
    parallel = project(delta, v)
    mu, perp, par, beta, gp, gt = c.unsqueeze(-1).unbind(-2)
    position = h + par * parallel + perp * (delta - parallel)
    nxt = position / position.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    carried = tangent(v, nxt)
    chord = tangent(nxt - h, nxt)
    along = project(chord, carried)
    return nxt, beta * carried + gp * along + gt * (chord - along)


def test_paired_parameters_and_reference():
    cfg = debug_config("chordres_update_soft_force")
    model = build_model(cfg, seed=42)
    control = build_model(replace(cfg, method="chordres_honly_mhc_mix"), seed=42)
    a,b = dict(model.named_parameters()),dict(control.named_parameters())
    assert a.keys()==b.keys()
    assert all(torch.equal(p,b[k]) and p.requires_grad==b[k].requires_grad for k,p in a.items())
    def groups(m):
        opt=build_optimizer(m)
        set_control_lr(m,opt,None)
        return [{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups]
    assert groups(model)==groups(control)
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=.02)
            block.mlp.c_proj.weight.normal_(std=.02)
    ids=torch.arange(cfg.sequence_len).reshape(1,-1)
    loss=model(ids,ids)
    loss.backward()
    assert torch.isfinite(loss) and model.transformer.h[0].attn.c_proj.weight.grad.norm()>0
    h,v,f=torch.randn(3,4,17,dtype=torch.double)
    c=torch.randn(4,6,dtype=torch.double)
    a,b=token_reference_write(h,v,f,c)
    x,y=definition(h,v,f,c)
    torch.testing.assert_close(a,x)
    torch.testing.assert_close(b,y)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="requires the Triton BF16 write")
def test_cuda_oracle():
    from nexttransformer.methods.chordres_update_soft_force.kernels import SoftUpdateWrite
    dev=torch.device('cuda',torch.cuda.current_device())
    gen=torch.Generator(device=dev).manual_seed(701+dev.index)
    def sample(shape):
        return torch.randn(shape,device=dev,generator=gen)
    width=1536
    h=sample((12,width))
    h=(h/h.square().mean(-1,keepdim=True).sqrt()).bfloat16().requires_grad_()
    v=sample(h.shape)
    v[0].zero_()
    v=v.bfloat16().requires_grad_()
    f=sample(h.shape)*torch.tensor([0.,1e-8,.001,.01,.05,.1,.25,1.,4.,.03,.045,.055],device=dev)[:,None]
    f=f.bfloat16().requires_grad_()
    c=sample((12,6))*.15+torch.tensor([0.,.17,.09,.6,.4,.72],device=dev)
    c.requires_grad_()
    args=(h,v,f,c)
    y=SoftUpdateWrite.apply(*args)
    exact=[x.detach().double().requires_grad_() for x in args]
    ref=definition(*exact)
    dy=tuple(sample(x.shape).bfloat16() for x in y)
    grad=torch.autograd.grad(y,args,dy)
    correct=torch.autograd.grad(ref,exact,tuple(x.double() for x in dy))
    for a,b in zip((*y,*grad),(*ref,*correct),strict=True):
        # Check each force magnitude independently, including the zero row.
        err=(a.double()-b).norm(dim=-1)/b.norm(dim=-1).clamp_min(1e-7)
        assert torch.isfinite(a).all() and err.max()<.012,(a.shape,err)
    # A unit-RMS h with exactly representable entries gives the analytic
    # force-to-position VJP at the neutral, zero-initialized site.
    h0=torch.ones((1,width),device=dev,dtype=torch.bfloat16)
    f0=torch.zeros_like(h0,requires_grad=True)
    c0=torch.tensor([[0.,.05,.05,.5,.5,.5]],device=dev)
    nxt,nv=SoftUpdateWrite.apply(h0,torch.zeros_like(h0),f0,c0)
    assert torch.equal(nxt,h0) and nv.count_nonzero()==0
    probe=sample(h0.shape).bfloat16()
    df,=torch.autograd.grad(nxt,f0,probe)
    want=(probe.float()-probe.float().mean(-1,keepdim=True)).bfloat16()
    torch.testing.assert_close(df,want,rtol=.004,atol=1e-5)
