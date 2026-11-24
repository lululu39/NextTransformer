from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from launch.train_matched_normal import initialize_projections
from nexttransformer.methods.chordres_honly_scalar_recipe import GATE_EPS, HOnlyGateSite
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model


def oracle(h, v, site):
    x = F.rms_norm(h.float(), (h.size(-1),), eps=GATE_EPS)
    values = F.linear(x, site.token_weight)
    bias = [site.mu, site.steps]
    if site.beta is not None:
        bias.extend((site.beta, site.velocity_gains))
    raw = values + torch.cat(bias)
    tail = (
        torch.full_like(raw, 0.5)
        if site.beta is None
        else torch.cat((raw[..., 3:4].sigmoid(), raw[..., 4:6]), -1)
    )
    controls = torch.cat((raw[..., :1], F.softplus(raw[..., 1:3]), tail), -1)
    offset = controls[..., :1] * v.float()
    read = (h.float() + offset) * (1 + offset.square().mean(-1, keepdim=True)).rsqrt()
    return F.rms_norm(read.to(h.dtype), (h.size(-1),)), controls


def test_matched_start_and_scalar_constraints():
    cfg = debug_config("chordres_honly_scalar_recipe")
    model = initialize_projections(build_model(cfg, seed=42), 42)
    old = initialize_projections(
        build_model(replace(cfg, method="chordres_velocity_gains"), seed=42), 42
    )
    ours, theirs = dict(model.named_parameters()), dict(old.named_parameters())
    extra = {name for name in ours if name.endswith("token_weight")}
    assert ours.keys() == theirs.keys() | extra
    assert all(torch.count_nonzero(ours[name]) == 0 for name in extra)
    assert all(torch.equal(ours[name], p) for name, p in theirs.items())
    ids = torch.arange(cfg.sequence_len).unsqueeze(0)
    torch.testing.assert_close(model(ids), old(ids), rtol=0, atol=0)
    site = model.residual_maps[0]
    h = torch.randn(2, 6, cfg.width)
    v = torch.randn_like(h)
    with torch.no_grad():
        site.token_weight.normal_(std=0.01)
        for sign in (-1.0, 1.0):
            for bias in (site.mu, site.steps, site.beta, site.velocity_gains):
                bias.fill_(8 * sign)
            q, c = site(h, v)
            q2, c2 = site(h, 3 * v)
            assert torch.equal(c, c2)
            assert (c[..., [0, 4, 5]] * sign > 3).all()
            assert ((c[..., 3] > 0) & (c[..., 3] < 1)).all()
            if sign > 0:
                assert (c[..., 1:3] > 3).all()
            else:
                assert ((c[..., 1:3] > 0) & (c[..., 1:3] < 0.01)).all()
            assert not torch.equal(q, q2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires BF16 Triton")
def test_cuda_oracle():
    torch.backends.cuda.matmul.allow_tf32 = False
    gen = torch.Generator(device="cuda").manual_seed(706 + torch.cuda.current_device())

    def rand(shape):
        return torch.randn(shape, generator=gen, device="cuda")

    for width, terminal in ((1536, False), (96, True)):
        site = HOnlyGateSite(width, terminal).cuda()
        h = rand((1, 271, width))
        h[:, 1:32] *= 0.05
        v = rand(h.shape)
        v[:, 0] = 0
        v[:, 32:64] *= 16
        h, v = [a.bfloat16().requires_grad_() for a in (h, v)]
        from nexttransformer.methods.chordres_velocity_gains import VelocityGainSite

        static = VelocityGainSite(terminal).cuda()()
        _, initial = site(h, v)
        assert torch.equal(initial, static.expand_as(initial))
        with torch.no_grad():
            site.token_weight.copy_(rand(site.token_weight.shape) * (0.45 / width**0.5))
            site.mu.fill_(0.25)
        q, c = site(h, v)
        rq, rc = oracle(h, v, site)
        q2, c2 = site(h, -3 * v)
        assert torch.equal(c, c2) and not torch.equal(q, q2)
        assert (
            torch.autograd.grad(c.sum(), v, retain_graph=True)[0].count_nonzero() == 0
        )
        if terminal:
            assert (c[..., 3:] == 0.5).all()
        torch.testing.assert_close(c, rc, atol=2e-6, rtol=1e-5)
        assert (q.float() - rq.float()).norm() / rq.float().norm() < 0.003
        leaves = [
            p
            for p in (
                h,
                v,
                site.token_weight,
                site.mu,
                site.steps,
                site.beta,
                site.velocity_gains,
            )
            if p is not None
        ]
        probes = [rand(t.shape).to(t.dtype) for t in (q, c)]
        grads = torch.autograd.grad((q, c), leaves, probes)
        exact = torch.autograd.grad((rq, rc), leaves, probes)
        for a, b in zip(grads, exact, strict=True):
            error = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-7)
            assert torch.isfinite(a).all() and error < 0.012, (a.shape, float(error))
