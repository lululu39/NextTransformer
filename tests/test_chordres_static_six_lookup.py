"""A bias-only Chord site must match the learned-dynamic site at zero projection."""
import pytest
import torch

from nexttransformer.methods.chordres_honly_mhc_mix import HOnlyGateSite
from nexttransformer.methods.chordres_static_six_lookup import StaticSixSite


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compares the BF16 training kernels")
def test_static_six_matches_token_independent_dynamic_gate():
    device = torch.device("cuda", torch.cuda.current_device())
    rng = torch.Generator(device=device).manual_seed(610 + device.index)
    for terminal in (False, True):
        width = 1536
        site = StaticSixSite(terminal).to(device)
        dynamic = HOnlyGateSite(width, terminal).to(device)
        with torch.no_grad():
            for name, value in site.named_parameters():
                # Distinguish parallel/perpendicular activations and all velocity gains.
                value.copy_(torch.randn(value.shape, device=device, generator=rng) * .2)
                dict(dynamic.named_parameters())[name].copy_(value)
        h, v = [torch.randn(2, 9, width, device=device, generator=rng).bfloat16().requires_grad_() for _ in range(2)]
        q, c = site(h, v)
        other_q, other_c = dynamic(h, v)
        torch.testing.assert_close(q, other_q, rtol=0, atol=0)
        torch.testing.assert_close(c, other_c, rtol=2e-6, atol=2e-6)
        assert c.is_contiguous() and torch.equal(c[0,0], c[-1,-1])
        dq = torch.randn(q.shape, device=device, generator=rng).bfloat16()
        dc = torch.randn(c.shape, device=device, generator=rng)
        shared = list(dict(site.named_parameters()))
        static_grads = torch.autograd.grad((q,c),[h,v,*site.parameters()],(dq,dc))
        dynamic_grads = torch.autograd.grad((other_q,other_c),[h,v,*[dict(dynamic.named_parameters())[n] for n in shared]],(dq,dc))
        for x,y in zip(static_grads,dynamic_grads,strict=True):
            torch.testing.assert_close(x.float(),y.float(),atol=.002,rtol=.004)
