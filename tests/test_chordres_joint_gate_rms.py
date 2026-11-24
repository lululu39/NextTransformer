from dataclasses import replace

import pytest
import torch

from launch.train_matched_normal import initialize_projections
from nexttransformer.methods.chordres_token_controls import (
    token_reference_read as raw_read,
)
from nexttransformer.methods.chordres_token_controls_joint_rms import (
    GATE_EPS,
    JointGateSite,
    token_reference_read,
)
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model


def test_model_and_joint_rms():
    cfg = debug_config("chordres_token_controls_joint_rms")
    model = initialize_projections(build_model(cfg, seed=42), 42)
    ref = initialize_projections(
        build_model(replace(cfg, method="chordres_token_controls_vnorm"), seed=42), 42
    )
    ours, theirs = dict(model.named_parameters()), dict(ref.named_parameters())
    assert ours.keys() == theirs.keys()
    assert all(
        torch.equal(p, theirs[n]) and p.requires_grad == theirs[n].requires_grad
        for n, p in ours.items()
    )
    ids = torch.arange(cfg.sequence_len).unsqueeze(0)
    torch.testing.assert_close(model(ids), ref(ids), rtol=0, atol=0)
    site = JointGateSite(32, False).double()
    with torch.no_grad():
        site.token_weight.normal_(std=0.03)
        site.mu.fill_(0.4)
    h, v = (
        torch.randn(9, 32, dtype=torch.float64),
        torch.randn(9, 32, dtype=torch.float64),
    )
    v[0] = 0
    args = (site.token_weight, site.mu, site.steps, site.beta, site.velocity_gains)
    q, c = token_reference_read(h, v, *args)
    joint = torch.cat((h, v), -1).requires_grad_()
    normalized = torch.nn.functional.rms_norm(joint, (64,), eps=GATE_EPS)
    hn, vn = normalized.chunk(2, -1)
    _, expected = raw_read(hn, vn, *args)
    torch.testing.assert_close(c, expected)
    upstream = torch.randn_like(c)
    joint2 = joint.detach().clone().requires_grad_()
    _, actual = token_reference_read(*joint2.chunk(2, -1), *args)
    torch.testing.assert_close(
        torch.autograd.grad(actual, joint2, upstream)[0],
        torch.autograd.grad(expected, joint, upstream)[0],
    )
    # The physical lookahead must see the original magnitude.
    offset = c[..., :1] * v
    preview = (h + offset) * (1 + offset.square().mean(-1, keepdim=True)).rsqrt()
    torch.testing.assert_close(q, torch.nn.functional.rms_norm(preview, (32,)))
    _, both_scaled = token_reference_read(32 * h, 32 * v, *args)
    torch.testing.assert_close(c, both_scaled, atol=2e-6, rtol=1e-5)
    _, velocity_scaled = token_reference_read(h, 32 * v, *args)
    assert not torch.allclose(c[1:], velocity_scaled[1:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton CUDA oracle")
def test_cuda_oracle():
    from nexttransformer.methods.chordres_token_controls_joint_rms.kernels import (
        TokenRead,
    )
    from nexttransformer.methods.chordres_token_controls.kernels import (
        TokenRead as RawRead,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    gen = torch.Generator(device="cuda").manual_seed(473 + torch.cuda.current_device())

    def rand(shape):
        return torch.randn(shape, generator=gen, device="cuda")

    for width, terminal in ((1536, False), (96, True)):
        site = JointGateSite(width, terminal).cuda()
        h = rand((1, 271, width)).bfloat16().requires_grad_()
        v = rand(h.shape)
        v[:, 0] = 0
        v[:, 1:32] *= 1e-4
        v[:, 32:64] *= 0.01
        v[:, 64:96] *= 32
        v = v.bfloat16().requires_grad_()
        args = (site.token_weight, site.mu, site.steps, site.beta, site.velocity_gains)
        for a, b in zip(
            TokenRead.apply(h, v, *args), RawRead.apply(h, v, *args), strict=True
        ):
            assert torch.equal(a, b)
        with torch.no_grad():
            site.token_weight.copy_(rand(site.token_weight.shape) * (0.1 / width**0.5))
            site.mu.fill_(0.3)
        out = TokenRead.apply(h, v, *args)
        ref = token_reference_read(h, v, *args)
        torch.testing.assert_close(out[1], ref[1], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(
            out[0].float(), ref[0].float(), atol=0.032, rtol=0.012
        )
        probes = [rand(t.shape).to(t.dtype) for t in out]
        leaves = [t for t in (h, v, *args) if t is not None]
        got = torch.autograd.grad(out, leaves, probes)
        expected = torch.autograd.grad(ref, leaves, probes)
        for a, b in zip(got, expected, strict=True):
            assert torch.isfinite(a).all()
            diff = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-7)
            assert diff < 0.012, (width, a.shape, float(diff))
        for sample in (
            slice(0, 1),
            slice(1, 32),
            slice(32, 64),
            slice(64, 96),
            slice(96, None),
        ):
            av, bv = got[1][:, sample].float(), expected[1][:, sample].float()
            assert (av - bv).norm() / bv.norm().clamp_min(1e-7) < 0.012
