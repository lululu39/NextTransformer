from dataclasses import replace

import pytest
import torch
from launch.train_matched_normal import initialize_projections
from nexttransformer.methods.chordres_raw_force import reference_raw_write
from nexttransformer.methods.chordres_velocity_gains import reference_gain_write
from nexttransformer.methods.chordres_zero import unit_rms
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model


def oracle(h, v, force, p):
    # Independent definition using explicit projections on the two tangent components.
    dt = h.dtype
    h, v, f = h.to(p.dtype), v.to(p.dtype), force.to(p.dtype)

    def dot(a, b):
        return (a * b).mean(-1, keepdim=True)

    def parallel(a, b):
        return dot(a, b) / dot(b, b).clamp_min(1e-12) * b

    d = f - h
    dpara = parallel(d, v)
    pos = unit_rms(h + p[2] * dpara + p[1] * (d - dpara))
    u = v - dot(v, pos) * pos
    chord = pos - h
    disp = chord - dot(chord, pos) * pos
    dp = parallel(disp, u)
    vel = p[3] * u + p[4] * dp + p[5] * (disp - dp)
    return pos.to(dt), vel.to(dt)


def test_matched_parameters_and_force_scaling():
    cfg = debug_config("chordres_raw_force")
    model = initialize_projections(build_model(cfg, seed=42), 42)
    ref = initialize_projections(
        build_model(replace(cfg, method="chordres_velocity_gains"), seed=42), 42
    )
    a, b = dict(model.named_parameters()), dict(ref.named_parameters())
    assert a.keys() == b.keys()
    for name, weight in a.items():
        assert weight.requires_grad == b[name].requires_grad and torch.equal(
            weight, b[name]
        ), name
    h = unit_rms(torch.randn(7, 32, dtype=torch.float64))
    v = torch.zeros_like(h)
    f = unit_rms(torch.randn_like(h))
    p = torch.tensor([0.0, 0.2, 0.2, 0.5, 0.5, 0.5], dtype=h.dtype)
    torch.testing.assert_close(
        reference_raw_write(h, v, f, p)[0], reference_gain_write(h, v, f, p)[0]
    )
    pos, vel = reference_raw_write(h, v, 8 * f, p)
    assert not torch.allclose(pos, reference_raw_write(h, v, f, p)[0])
    torch.testing.assert_close(pos.square().mean(-1), torch.ones(7, dtype=h.dtype))
    torch.testing.assert_close(
        (pos * vel).mean(-1), torch.zeros(7, dtype=h.dtype), atol=1e-12, rtol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 Triton oracle")
def test_cuda_oracle():
    from nexttransformer.methods.chordres_raw_force.kernels import RawForceWrite

    gen = torch.Generator(device="cuda").manual_seed(248 + torch.cuda.current_device())

    def rand(shape):
        return torch.randn(shape, generator=gen, device="cuda")

    for width in (96, 1536):
        h = unit_rms(rand((1, 137, width)))
        v = rand(h.shape)
        v -= h * (v * h).mean(-1, keepdim=True)
        v[:, 0] = 0
        f = rand(h.shape)
        f[:, 0] = 0
        f[:, 1:32] *= 0.01
        f[:, 32:64] *= 8
        h, v, f = [t.bfloat16().requires_grad_() for t in (h, v, f)]
        p = torch.tensor(
            [0.15, 0.1, 0.23, 0.68, -0.13, 1.21], device="cuda", requires_grad=True
        )
        actual = RawForceWrite.apply(h, v, f, p)
        leaves = [t.detach().double().requires_grad_() for t in (h, v, f, p)]
        expected = oracle(*leaves)
        probes = [rand(q.shape).bfloat16() for q in actual]
        grads = torch.autograd.grad(actual, (h, v, f, p), probes)
        exact = torch.autograd.grad(expected, leaves, [q.double() for q in probes])
        for a, b in zip((*actual, *grads), (*expected, *exact), strict=True):
            diff = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-7)
            assert torch.isfinite(a).all() and diff < 0.006, (a.shape, float(diff))
        nh, nv = actual
        eh, ev = expected
        torch.testing.assert_close(
            eh.square().mean(-1), torch.ones_like(eh[..., 0]), rtol=0, atol=1e-12
        )
        torch.testing.assert_close(
            (eh * ev).mean(-1), torch.zeros_like(eh[..., 0]), rtol=0, atol=1e-12
        )
        # Both returned states are rounded separately to BF16. Bound their geometric
        # errors by the unit roundoff of each coordinate and the row velocity RMS.
        u = torch.finfo(torch.bfloat16).eps / 2
        roundoff = 2 * u + u * u + 2e-6
        assert (nh.float().square().mean(-1) - 1).abs().max() < roundoff
        tangent_error = (nh.float() * nv.float()).mean(-1).abs()
        assert (tangent_error <= roundoff * ev.square().mean(-1).sqrt() + 1e-7).all()
