from dataclasses import replace

import pytest
import torch

from launch.train_chord_control_lr import CONTROL_PEAK_LR, set_control_lr
from launch.train_matched_normal import initialize_projections
from nexttransformer.methods.chordres_velocity_gains import reference_gain_write
from nexttransformer.methods.chordres_zero import reference_write, unit_rms
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import build_optimizer


def literal_write(h, v, force, p):
    # Independent definition: baseline position and three tangent-space terms.
    q, _ = reference_write(h, v, force, p[:4])
    u = v - (v*q).mean(-1, keepdim=True)*q
    chord = q-h
    d = chord - (chord*q).mean(-1, keepdim=True)*q
    proj = (u*d).mean(-1, keepdim=True) / u.square().mean(-1, keepdim=True).clamp_min(1e-12) * u
    return q, p[3]*u + p[4]*proj + p[5]*(d-proj)


def test_formula_and_zero_start():
    torch.manual_seed(93)
    h=unit_rms(torch.randn(3,16,dtype=torch.float64));v=torch.randn_like(h);v[0].zero_()
    f=torch.randn_like(v);p=torch.tensor([.2,.13,.24,.65,-.15,.8],dtype=h.dtype)
    actual=reference_gain_write(h,v,f,p)
    for a,b in zip(actual,literal_write(h,v,f,p),strict=True):
        torch.testing.assert_close(a,b)
    q,w=actual
    torch.testing.assert_close((q*w).mean(-1),torch.zeros(3,dtype=q.dtype),atol=1e-12,rtol=0)
    chord=q[0]-h[0]; d=chord-(chord*q[0]).mean()*q[0]
    torch.testing.assert_close(w[0],p[5]*d)
    p[4:]=1-p[3]
    q,w=reference_gain_write(h,v,f,p)
    baseline=reference_write(h,v,f,p[:4])
    torch.testing.assert_close(q,baseline[0],rtol=0,atol=0)
    torch.testing.assert_close(w,baseline[1],rtol=0,atol=0)


def test_only_new_gains_change_the_model():
    cfg=debug_config("chordres_velocity_gains")
    model=initialize_projections(build_model(cfg,seed=42),42)
    ref=initialize_projections(build_model(replace(cfg,method="chordres_zero_save_force"),seed=42),42)
    params=dict(model.named_parameters());shared=dict(ref.named_parameters())
    assert set(params)-set(shared)=={f"residual_maps.{i}.velocity_gains" for i in range(2*cfg.n_layer-1)}
    for n in shared:torch.testing.assert_close(params[n],shared[n],rtol=0,atol=0)
    assert model.residual_maps[-1].velocity_gains is None and model.residual_maps[-1].beta is None
    gains=[m.velocity_gains for m in model.residual_maps[:-1]]
    assert all(torch.equal(g,torch.full_like(g,.5)) for g in gains)
    opt=build_optimizer(model);set_control_lr(model,opt)
    group=next(g for g in opt.param_groups if any(p is gains[0] for p in g["params"]))
    assert group["kind"]=="adamw" and group["initial_lr"]==CONTROL_PEAK_LR and group["weight_decay"]==0
    ids=torch.randint(cfg.vocab_size,(2,cfg.sequence_len))
    torch.testing.assert_close(model(ids),ref(ids),rtol=0,atol=0)
    with torch.no_grad():
        for m in model.residual_maps:m.mu.fill_(.4)
    model(ids,ids.roll(-1,1)).backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters() if p.requires_grad)
    assert gains[0].grad[0].item()==0 and gains[0].grad[1].abs().item()>0
    assert torch.stack([g.grad for g in gains[1:]]).abs().sum(0).min()>0


@pytest.mark.skipif(not torch.cuda.is_available(),reason="Triton paired gradient check")
def test_cuda_oracle():
    from nexttransformer.methods.chordres_zero.kernels import ChordWrite
    from nexttransformer.methods.chordres_velocity_gains.kernels import VelocityGainWrite
    rank=torch.cuda.current_device()
    gen=torch.Generator(device='cuda').manual_seed(301+rank)
    h=unit_rms(torch.randn(8,1536,device='cuda',generator=gen))
    v=torch.randn(h.shape,device='cuda',generator=gen)
    v-=h*(v*h).mean(-1,keepdim=True);v[0].zero_();v[1]*=1e-7
    f=torch.randn(h.shape,device='cuda',generator=gen)
    leaves=[x.detach().bfloat16().requires_grad_() for x in (h,v,f)]
    p=torch.tensor([.4,.13,.24,.5,.5,.5],device='cuda')
    got=VelocityGainWrite.apply(*leaves,p)
    old=ChordWrite.apply(*leaves,p[:4])
    for a,b in zip(got,old,strict=True):assert torch.equal(a,b)
    p=torch.tensor([.4,.13,.24,.65,.27,.85],device='cuda',requires_grad=True)
    got=VelocityGainWrite.apply(*leaves,p)
    exact=[x.detach().double().requires_grad_() for x in (*leaves,p)]
    want=literal_write(*exact)
    probes=[torch.randn(a.shape,device='cuda',generator=gen).bfloat16() for a in got]
    actual_grad=torch.autograd.grad(got,(*leaves,p),probes)
    oracle_grad=torch.autograd.grad(want,exact,[g.double() for g in probes])
    for a,b in zip((*got,*actual_grad),(*want,*oracle_grad),strict=True):
        err=(a.double()-b).norm()/b.norm().clamp_min(1e-10)
        assert err<.016,err
    torch.testing.assert_close(actual_grad[-1].double(),oracle_grad[-1],rtol=.015,atol=.003)
