from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from nexttransformer.methods.chordres_force_amplitude_step import AmplitudeSite
from nexttransformer.methods.chordres_zero import unit_rms
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model


def independent(h, v, f, b, w, beta):
    h, v, f = h.double(), v.double(), f.double()
    power = f.square().mean(-1, keepdim=True)
    a = F.softplus(b.double() + w.double() * torch.log1p(power))
    t = 0.5 if beta is None else beta.double().sigmoid()
    nh = unit_rms((1 - a) * h + a * unit_rms(f))
    mix = t * v + (1 - t) * (nh - h)
    return nh, mix - (nh * mix).mean(-1, keepdim=True) * nh


def test_amplitude_only_and_zero_start():
    cfg = debug_config("chordres_force_amplitude_step")
    model = build_model(cfg, seed=42)
    ref = build_model(replace(cfg, method="chordres_honly_single_step"), seed=42)
    for site in model.residual_maps:
        assert (
            site.mu.numel() == site.step.numel() == site.amplitude_weight.numel() == 1
        )
        assert site.amplitude_weight.item() == 0.0
    assert model.residual_maps[-1].beta is None
    assert not any("token_weight" in n for n, _ in model.named_parameters())
    ids = torch.arange(cfg.sequence_len).unsqueeze(0)
    torch.testing.assert_close(model(ids), ref(ids), rtol=0, atol=0)
    s = AmplitudeSite(False).double()
    h = torch.ones(2, 8, dtype=torch.float64)
    v = torch.zeros_like(h)
    f = torch.tensor([1.0, -1.0] * 4, dtype=h.dtype).expand_as(h)
    torch.testing.assert_close(s(h, v, f)[0], s(h, v, 8 * f)[0])
    with torch.no_grad():
        s.amplitude_weight.fill_(0.6)
    assert not torch.allclose(s(h, v, f)[0], s(h, v, 8 * f)[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 Triton write")
def test_cuda_oracle():
    from nexttransformer.methods.chordres_force_amplitude_step.kernels import (
        AmplitudeWrite,
    )
    from nexttransformer.methods.chordres_honly_single_step.kernels import TokenWrite
    from nexttransformer.methods.chordres_zero import ChordSite

    g = torch.Generator(device="cuda").manual_seed(127 + torch.cuda.current_device())

    def rand(shape):
        return torch.randn(shape, generator=g, device="cuda")

    for width, terminal, rows in ((1536, False, 1033), (96, True, 73), (96, False, 1)):
        site = AmplitudeSite(terminal).cuda()
        h = unit_rms(rand((1, rows, width)))
        v = rand(h.shape)
        v -= h * (h * v).mean(-1, keepdim=True)
        f = rand(h.shape)
        if rows > 1:
            f[:, 0] = 0
        f[:, 1:8] *= 1e-4
        f[:, 8:16] *= 16
        h, v, f = [t.bfloat16().requires_grad_() for t in (h, v, f)]
        q = (
            ChordSite(terminal)
            .cuda()()[[0, 1, 3]]
            .expand(*h.shape[:-1], 3)
            .contiguous()
        )
        for a, b in zip(site(h, v, f), TokenWrite.apply(h, v, f, q), strict=True):
            assert torch.equal(a, b)
        with torch.no_grad():
            site.amplitude_weight.fill_(0.37)
            site.step.fill_(-1.1)
        params = (site.step, site.amplitude_weight, site.beta)
        got = AmplitudeWrite.apply(h, v, f, *params)
        inputs = [t for t in (h, v, f, *params) if t is not None]
        exact = [t.detach().double().requires_grad_() for t in inputs]
        want = independent(*exact, None) if terminal else independent(*exact)
        probes = [rand(h.shape).bfloat16() for _ in got]
        ga = torch.autograd.grad(got, inputs, probes)
        gb = torch.autograd.grad(want, exact, [p.double() for p in probes])
        for a, b in zip((*got, *ga), (*want, *gb), strict=True):
            diff = (a.double() - b).norm() / b.norm().clamp_min(1.0e-7)
            assert a.isfinite().all() and diff < 0.009, (
                width,
                rows,
                a.shape,
                float(diff),
            )
        torch.testing.assert_close(ga[4].double(), gb[4], atol=0.003, rtol=0.012)

        if rows == 1:
            zero = torch.zeros_like(f, requires_grad=True)
            nh, nv = AmplitudeWrite.apply(h, v, zero, *params)
            db, dw = torch.autograd.grad(
                (nh, nv), (site.step, site.amplitude_weight), probes
            )
            # For zero force the positive scalar multiplying h cancels in the retraction.
            torch.testing.assert_close(db, torch.zeros_like(db), rtol=0, atol=1e-5)
            assert dw.count_nonzero() == 0
