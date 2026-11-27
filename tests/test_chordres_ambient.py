"""Velocity geometry regressions specific to ambient, nonorthogonal states."""
import pytest
import torch
from torch.nn import functional as F

from nexttransformer.methods.chordres_ambient_lookup import AmbientGateSite, token_reference_write
from nexttransformer.methods.chordres_ambient_lookup.write_kernel import AmbientWrite


def test_radial_component_survives():
    h = torch.ones(2, 32)
    v = 2 * h
    controls = torch.tensor([[0., .05, .05, .75, .5, .5]]).expand(2, -1)
    new_h, new_v = token_reference_write(h, v, torch.zeros_like(h), controls)
    torch.testing.assert_close(new_h, h)
    torch.testing.assert_close(new_v, .75 * v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA BF16")
def test_ambient_gpu_oracle():
    # Non-tangent and zero rows distinguish ambient from the tangent shortcut.
    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(1024 + dev.index)
    sample = lambda shape: torch.randn(shape, device=dev, generator=gen)
    for d, terminal in ((1536, False), (96, True)):
        site = AmbientGateSite(d, terminal).to(dev)
        with torch.no_grad():
            site.mu.fill_(.45)
            site.token_weight.copy_(sample(site.token_weight.shape) * (.25 / d**.5))
            site.mix_scales.fill_(.12)
        h = F.rms_norm(sample((259,d)), (d,)).bfloat16().requires_grad_()
        v = (1.5*h.detach().float() + .3*sample(h.shape)).bfloat16()
        v[0].zero_()
        v.requires_grad_()
        q, controls = site(h,v)
        # Independent six-control definition and the true RMS of the full lookahead.
        bias = [site.mu, site.steps] if terminal else [site.mu,site.steps,site.beta,site.velocity_gains]
        groups = [0,1,1] if terminal else [0,1,1,2,2,2]
        z = F.linear(F.rms_norm(h.float(),(d,),eps=1e-6),site.token_weight)
        z = z * site.mix_scales[groups] + torch.cat(bias)
        tail = torch.full_like(z,.5) if terminal else torch.cat((z[:,3:4].sigmoid(),z[:,4:]),-1)
        ref_c = torch.cat((z[:,:1], F.softplus(z[:,1:3]), tail), -1)
        raw = h.float() + z[:,:1] * v.float()
        ref_q = F.rms_norm((raw * raw.square().mean(-1,keepdim=True).clamp_min(1e-12).rsqrt()).bfloat16(),
                           (d,), eps=torch.finfo(torch.float32).eps)
        torch.testing.assert_close(controls,ref_c,atol=2e-6,rtol=1e-5)
        assert (q.float()-ref_q.float()).norm()/ref_q.float().norm() < .004
        weights = [h,v,*site.parameters()]
        upstream = (sample(q.shape).bfloat16(),sample(controls.shape))
        actual = torch.autograd.grad((q,controls),weights,upstream)
        expected = torch.autograd.grad((ref_q,ref_c),weights,upstream)
        for x,y in zip(actual,expected,strict=True):
            assert (x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-7) < .018

    # Compare both output states and all four input VJPs to an independent FP64 split.
    d = 1536
    h = F.rms_norm(sample((13,d)),(d,)).bfloat16()
    v = (1.25*h.float()+.3*sample(h.shape)).bfloat16()
    f = sample(h.shape).bfloat16()
    f[0].zero_(); v[1].zero_()
    c = sample((13,6))
    args = [x.requires_grad_() for x in (h,v,f,c)]
    ref = [x.detach().double().requires_grad_() for x in args]
    h64,v64,f64,c64 = ref
    def split(x, axis):
        par = (x*axis).sum(-1,keepdim=True)/axis.square().sum(-1,keepdim=True).clamp_min(d*1e-12)*axis
        return par,x-par
    delta = f64 / (f64.square().mean(-1,keepdim=True)+.05**2).sqrt()
    par,perp=split(delta,v64)
    u=h64+c64[:,1:2]*perp+c64[:,2:3]*par
    new_h=u/u.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
    par,perp=split(new_h-h64,v64)
    new_v=c64[:,3:4]*v64+c64[:,4:5]*par+c64[:,5:6]*perp
    out=AmbientWrite.apply(*args)
    grad=tuple(sample(h.shape).bfloat16() for _ in range(2))
    expected=torch.autograd.grad((new_h,new_v),ref,tuple(x.double() for x in grad))
    actual=torch.autograd.grad(out,args,grad)
    for x,y in zip((*out,*actual),(new_h,new_v,*expected),strict=True):
        rel=(x.double()-y).norm(dim=-1)/y.norm(dim=-1).clamp_min(1e-7)
        assert torch.isfinite(x).all() and rel.max()<.015
