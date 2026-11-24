from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from launch.train_matched_normal import initialize_projections
from nexttransformer.methods.chordres_honly_single_step import (
    GATE_EPS,
    HOnlyGateSite,
    token_reference_write,
)
from nexttransformer.methods.chordres_zero import ChordSite
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model


def test_single_velocity_control():
    cfg = debug_config("chordres_honly_single_step")
    model = initialize_projections(build_model(cfg, seed=42), 42)
    ref = initialize_projections(
        build_model(replace(cfg, method="chordres_zero_save_force"), seed=42), 42
    )
    ours, theirs = dict(model.named_parameters()), dict(ref.named_parameters())
    gates = {n for n in ours if n.endswith("token_weight")}
    for key, p in ours.items():
        if key in gates:
            assert p.count_nonzero() == 0
        elif key.endswith(".step"):
            prior = theirs[key + "s"]
            assert p.shape == (1,) and torch.equal(prior, p.expand_as(prior))
        else:
            assert torch.equal(p, theirs[key])
    assert not any("velocity_gains" in n for n in ours)
    ids = torch.arange(cfg.sequence_len).unsqueeze(0)
    torch.testing.assert_close(model(ids), ref(ids), rtol=0, atol=0)
    h = torch.randn(2, 7, cfg.width)
    v = torch.randn_like(h)
    with torch.no_grad():
        site = model.residual_maps[0]
        site.token_weight.normal_(std=0.02)
        q, c = site(h, v)
        q2, c2 = site(h, v * 3)
        assert (
            c.shape == (*h.shape[:-1], 3)
            and torch.equal(c, c2)
            and not torch.equal(q, q2)
        )
        assert (c[..., 1] > 0).all() and ((c[..., 2] > 0) & (c[..., 2] < 1)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton BF16")
def test_cuda_oracle():
    from nexttransformer.methods.chordres_honly_single_step.kernels import TokenWrite
    from nexttransformer.methods.chordres_zero.kernels import ChordWrite

    torch.backends.cuda.matmul.allow_tf32 = False
    g = torch.Generator(device="cuda").manual_seed(891 + torch.cuda.current_device())

    def rand(shape):
        return torch.randn(shape, generator=g, device="cuda")

    for width, terminal in ((1536, False), (96, True)):
        site = HOnlyGateSite(width, terminal).cuda()
        h = F.rms_norm(rand((1, 271, width)), (width,)).bfloat16().requires_grad_()
        v = rand(h.shape)
        v[:, 0] = 0
        v[:, 1:20] *= 0.01
        v = v.bfloat16().requires_grad_()
        f = rand(h.shape).bfloat16().requires_grad_()
        q, c = site(h, v)
        base = ChordSite(terminal).cuda()()
        assert torch.equal(c, base[[0, 1, 3]].expand_as(c))
        for a, b in zip(
            TokenWrite.apply(h, v, f, c), ChordWrite.apply(h, v, f, base), strict=True
        ):
            assert torch.equal(a, b)
        with torch.no_grad():
            site.token_weight.copy_(rand(site.token_weight.shape) * (0.3 / width**0.5))
            site.mu.fill_(0.3)
        q, c = site(h, v)
        oh, ov = TokenWrite.apply(h, v, f, c)
        assert torch.equal(site(h, -2 * v)[1], c)
        assert (
            torch.autograd.grad(c.sum(), v, retain_graph=True)[0].count_nonzero() == 0
        )
        z = F.linear(F.rms_norm(h.float(), (width,), eps=GATE_EPS), site.token_weight)
        mu = z[..., :1] + site.mu
        a = F.softplus(z[..., 1:2] + site.step)
        beta = (
            torch.full_like(mu, 0.5)
            if terminal
            else (z[..., 2:3] + site.beta).sigmoid()
        )
        ec = torch.cat([mu, a, beta], dim=-1)
        off = mu * v.float()
        eq = F.rms_norm(
            (
                (h.float() + off) * (1 + off.square().mean(-1, keepdim=True)).rsqrt()
            ).bfloat16(),
            (width,),
        )
        eh, ev = token_reference_write(h, v, f, ec)
        outs = (q, c, oh, ov)
        refs = (eq, ec, eh, ev)
        torch.testing.assert_close(c, ec, atol=1e-6, rtol=1e-5)
        for a, b in zip((q, oh, ov), (eq, eh, ev), strict=True):
            assert (a.float() - b.float()).norm() / b.float().norm().clamp_min(
                1e-6
            ) < 0.003
        probes = [rand(t.shape).to(t.dtype) for t in outs]
        leaves = [
            t
            for t in (h, v, f, site.token_weight, site.mu, site.step, site.beta)
            if t is not None
        ]
        got = torch.autograd.grad(outs, leaves, probes)
        expected = torch.autograd.grad(refs, leaves, probes)
        for a, b in zip(got, expected, strict=True):
            diff = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-6)
            assert torch.isfinite(a).all() and diff < 0.012, (
                width,
                a.shape,
                float(diff),
            )
