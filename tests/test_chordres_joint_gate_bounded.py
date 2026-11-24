from dataclasses import replace
import pytest
import torch
from nexttransformer.methods.chordres_joint_gate_bounded import (
    BoundedGateSite,
    GATE_EPS,
    token_reference_read,
)
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model
from launch.train_matched_normal import initialize_projections


def independent(h, v, site):
    x = torch.cat((h.float(), v.float()), -1)
    x = torch.nn.functional.rms_norm(x, (x.shape[-1],), eps=GATE_EPS)
    z = torch.nn.functional.linear(x, site.token_weight.flatten(1))
    mu = 3 * ((z[..., :1] + site.mu) / 3).tanh()
    alpha = (z[..., 1:3] + site.steps).sigmoid()
    beta = (
        torch.full_like(mu, 0.5)
        if site.beta is None
        else (z[..., 3:4] + site.beta).sigmoid()
    )
    gamma = (
        torch.full_like(alpha, 0.5)
        if site.beta is None
        else 2 * (z[..., 4:] + site.velocity_gains).sigmoid()
    )
    off = mu * v.float()
    q = ((h.float() + off) * (1 + off.square().mean(-1, keepdim=True)).rsqrt()).to(
        h.dtype
    )
    return torch.nn.functional.rms_norm(q, (h.size(-1),)), torch.cat(
        [mu, alpha, beta, gamma], -1
    )


def test_initialization_and_ranges():
    cfg = debug_config("chordres_joint_gate_bounded")
    new = initialize_projections(build_model(cfg, seed=42), 42)
    old = initialize_projections(
        build_model(replace(cfg, method="chordres_token_controls_joint_rms"), seed=42),
        42,
    )
    a, b = dict(new.named_parameters()), dict(old.named_parameters())
    assert a.keys() == b.keys()
    for name, val in a.items():
        assert val.requires_grad == b[name].requires_grad
        if not name.endswith((".steps", ".velocity_gains")):
            assert torch.equal(val, b[name]), name
    h, v = torch.randn(13, cfg.width), torch.randn(13, cfg.width)
    for site in (new.residual_maps[0], new.residual_maps[-1]):
        q, c = site(h, v)
        torch.testing.assert_close(
            c, torch.tensor([0, 0.05, 0.05, 0.5, 0.5, 0.5]).expand_as(c)
        )
        with torch.no_grad():
            site.token_weight.normal_(std=10)
        _, c = site(h, v)
        refq, ref = independent(h, v, site)
        torch.testing.assert_close(c, ref, atol=2e-5, rtol=2e-5)
        assert (c.abs() <= torch.tensor([3.0, 1.0, 1.0, 1.0, 2.0, 2.0])).all()
        assert (c[..., 1:] >= 0).all()
        if site.beta is None:
            assert (c[..., 3:] == 0.5).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires BF16 Triton")
def test_cuda_oracle():
    from nexttransformer.methods.chordres_joint_gate_bounded.kernels import TokenRead

    torch.backends.cuda.matmul.allow_tf32 = False
    g = torch.Generator(device="cuda").manual_seed(428 + torch.cuda.current_device())

    def rand(shape):
        return torch.randn(shape, generator=g, device="cuda")

    for dim, terminal in ((1536, False), (96, True)):
        site = BoundedGateSite(dim, terminal).cuda()
        h = rand((1, 271, dim)).bfloat16().requires_grad_()
        v = rand(h.shape)
        v[:, 0] = 0
        v[:, 1:32] *= 0.001
        v[:, 32:64] *= 32
        v = v.bfloat16().requires_grad_()
        args = (site.token_weight, site.mu, site.steps, site.beta, site.velocity_gains)
        _, init = TokenRead.apply(h, v, *args)
        torch.testing.assert_close(
            init,
            torch.tensor([0, 0.05, 0.05, 0.5, 0.5, 0.5], device="cuda").expand_as(init),
        )
        for sigma in (0.5, 12.0):
            with torch.no_grad():
                site.token_weight.copy_(
                    rand(site.token_weight.shape) * (sigma / dim**0.5)
                )
            out = TokenRead.apply(h, v, *args)
            ref = independent(h, v, site)
            torch.testing.assert_close(out[1], ref[1], atol=3e-5, rtol=3e-5)
            assert (out[1][..., 0].abs() <= 3).all() and (out[1][..., 1:] >= 0).all()
            assert (out[1][..., 1:4] <= 1).all() and (out[1][..., 4:] <= 2).all()
            assert (out[0].float() - ref[0].float()).norm() / ref[
                0
            ].float().norm() < 0.003
            probes = [rand(t.shape).to(t.dtype) for t in out]
            leaves = [t for t in (h, v, *args) if t is not None]
            actual = torch.autograd.grad(out, leaves, probes)
            expected = torch.autograd.grad(ref, leaves, probes)
            for a, b in zip(actual, expected, strict=True):
                error = (a.float() - b.float()).norm() / b.float().norm().clamp_min(
                    1e-7
                )
                assert a.isfinite().all() and error < 0.012, (
                    dim,
                    sigma,
                    a.shape,
                    float(error),
                )
