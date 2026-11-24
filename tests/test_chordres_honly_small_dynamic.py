import pytest
import torch
from torch.nn import functional as F

from launch.train_chord_control_lr import CONTROL_PEAK_LR, set_control_lr
from nexttransformer.methods.chordres_honly_small_dynamic import HOnlyGateSite
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import build_optimizer


def test_group_ownership_and_schedule():
    from nexttransformer.models.native.config import NativeTrainConfig
    from nexttransformer.native.optimizer import set_schedule
    model = build_model(debug_config("chordres_honly_small_dynamic"))
    opt = build_optimizer(model)
    def contents(group):
        return dict(zip(group["parameter_names"], group["params"], strict=True))
    others = [(group, contents(group), group['initial_lr']) for group in opt.param_groups
              if not any(n.startswith('residual_maps.') for n in group['parameter_names'])]
    set_control_lr(model, opt)
    expected = {p for p in model.parameters() if p.requires_grad}
    seen = [p for group in opt.param_groups for p in group['params']]
    assert len(seen) == len(set(seen)) and set(seen) == expected
    static, gate = [g for g in opt.param_groups if g['parameter_names'][0].startswith('residual_maps.')]
    assert all(not n.endswith('.token_weight') for n in contents(static))
    assert set(gate['params']) == {s.token_weight for s in model.residual_maps}
    assert all((g['betas'], g['weight_decay'], g['eps']) == ((.9,.95),0.,1e-10) for g in (static, gate))
    assert static['initial_lr'] == CONTROL_PEAK_LR
    assert gate['initial_lr'] == pytest.approx(.000848528137423857)
    for group, saved, lr in others:
        assert contents(group).keys() == saved.keys()
        assert all(contents(group)[n] is p for n, p in saved.items())
        assert group['initial_lr'] == lr
    set_schedule(opt, 0, NativeTrainConfig())
    assert static['lr']/gate['lr'] == pytest.approx(static['initial_lr']/gate['initial_lr'])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires BF16 Triton")
def test_cuda_oracle():
    before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for width, terminal in ((1536, False), (96, True)):
            site = HOnlyGateSite(width, terminal).cuda()
            h, v = [torch.randn(2, 23, width, device="cuda").bfloat16().requires_grad_() for _ in range(2)]
            assert site.token_weight.count_nonzero() == 0
            with torch.no_grad():
                site.token_weight.normal_(std=.7 / width**.5)
                site.mu.fill_(.15)
            leaves = [x for x in (h,v,site.token_weight,site.mu,site.steps,site.beta,site.velocity_gains) if x is not None]
            q, c = site(h, v)
            x = F.rms_norm(h.float(), (width,), eps=1e-6)
            bias = torch.cat(leaves[3:])
            z = bias + .01 * F.linear(x, site.token_weight)
            tail = torch.full_like(z, .5) if terminal else torch.cat((z[...,3:4].sigmoid(), z[...,4:6]), dim=-1)
            expected = torch.cat((z[...,:1],F.softplus(z[...,1:3]),tail),dim=-1)
            offset = expected[...,:1]*v.float()
            read = (h.float()+offset) * (1+offset.square().mean(-1,keepdim=True)).rsqrt()
            reference_q = F.rms_norm(read.bfloat16(), (width,))
            torch.testing.assert_close(c,expected,rtol=1e-5,atol=2e-6)
            assert (q.float()-reference_q.float()).norm()/reference_q.float().norm() < .004
            # The control-only VJP isolates the small position path, which a
            # much larger direct query gradient could otherwise conceal.
            for dq in (torch.zeros_like(q),torch.randn_like(q)):
                dc = torch.randn_like(c)
                got = torch.autograd.grad((q,c),leaves,(dq,dc),retain_graph=True)
                want = torch.autograd.grad((reference_q,expected),leaves,(dq,dc),retain_graph=True)
                for a,b in zip(got,want,strict=True):
                    error = (a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-8)
                    assert torch.isfinite(a).all() and error < .015, (a.shape, error)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before
