from unittest.mock import patch

import pytest
import torch

from launch.train_matched_normal import initialize_projections
from nexttransformer.methods.chordres_ve_turn import reference_turn_write
from nexttransformer.methods.chordres_zero import reference_write, unit_rms
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model


def definition(h, v, force, p):
    nxt, _ = reference_write(h, v, force, p)
    u = v - (v*nxt).mean(-1, keepdim=True)*nxt
    chord = nxt - h
    d = chord - (chord*nxt).mean(-1, keepdim=True)*nxt
    par = (d*u).mean(-1,keepdim=True) / u.square().mean(-1,keepdim=True).clamp_min(1e-12) * u
    return nxt, p[3]*u + (1-p[3])*par + (d-par)


def test_perpendicular_displacement_and_same_geometry():
    torch.manual_seed(917)
    h=unit_rms(torch.randn(3,16,dtype=torch.float64))
    v=torch.randn_like(h)
    v-=h*(v*h).mean(-1,keepdim=True)
    v[0].zero_()
    force=torch.randn_like(h)
    p=torch.tensor([.4,.13,.24,.7],dtype=h.dtype)
    for beta in (.0,.5,1.):
        p[3]=beta
        nxt,nv=reference_turn_write(h,v,force,p)
        base_h,base_v=reference_write(h,v,force,p)
        torch.testing.assert_close(nxt,base_h,rtol=0,atol=0)
        q=nxt-h
        u=v-(v*nxt).mean(-1,keepdim=True)*nxt
        d=q-(q*nxt).mean(-1,keepdim=True)*nxt
        par=(d*u).mean(-1,keepdim=True)/u.square().mean(-1,keepdim=True).clamp_min(1e-12)*u
        torch.testing.assert_close(nv-base_v,beta*(d-par))
        torch.testing.assert_close(nv[0],d[0])
        torch.testing.assert_close((nv*nxt).mean(-1),torch.zeros(3,dtype=h.dtype),atol=1e-12,rtol=0)


def test_only_ve_attention_sites_change_velocity_write():
    import nexttransformer.methods.chordres_ve_turn as method
    config=debug_config("chordres_ve_turn")
    model=initialize_projections(build_model(config),42)
    ids=torch.randint(config.vocab_size,(2,config.sequence_len))
    events=[]
    def wrapped(fn,name):
        def write(*args):
            events.append(name)
            return fn(*args)
        return write
    with patch.object(method,"reference_turn_write",wrapped(reference_turn_write,"turn")), patch.object(method,"reference_write",wrapped(reference_write,"ordinary")):
        model(ids,ids).backward()
    expected=[]
    for i in range(config.n_layer):
        expected += ["turn" if str(i) in model.value_embeds else "ordinary","ordinary"]
    assert events==expected
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters() if p.requires_grad)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA write kernel")
def test_cuda_oracle():
    from nexttransformer.methods.chordres_zero.kernels import ChordWrite
    from nexttransformer.methods.chordres_ve_turn.kernels import TurnWrite
    rank=torch.cuda.current_device()
    gen=torch.Generator(device='cuda').manual_seed(301+rank)
    h=unit_rms(torch.randn(8,1536,device='cuda',generator=gen))
    v=torch.randn(h.shape,device='cuda',generator=gen)
    v-=h*(v*h).mean(-1,keepdim=True);v[0].zero_();v[1]*=1e-7
    f=torch.randn(h.shape,device='cuda',generator=gen)
    leaves=[a.detach().bfloat16().requires_grad_() for a in (h,v,f)]
    controls=torch.tensor([.4,.13,.24,.65],device='cuda',requires_grad=True)
    actual=TurnWrite.apply(*leaves,controls)
    base=ChordWrite.apply(*leaves,controls)
    assert torch.equal(actual[0],base[0])
    # A separate expression and FP64 autograd model the exact real-valued write.
    exact=[x.detach().double().requires_grad_() for x in (*leaves,controls)]
    oracle=definition(*exact)
    directions=[torch.randn(x.shape,device='cuda',generator=gen).bfloat16() for x in actual]
    grad=torch.autograd.grad(actual,(*leaves,controls),directions)
    want=torch.autograd.grad(oracle,exact,[g.double() for g in directions])
    for a,b in zip((*actual,*grad),(*oracle,*want),strict=True):
        err=(a.double()-b).norm()/b.norm().clamp_min(1e-10)
        assert err < .016,err
    torch.testing.assert_close(grad[-1].double(),want[-1],rtol=.015,atol=.003)
    assert torch.isfinite(actual[1]).all()
