from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from nexttransformer.methods.chordres_force_vector_step import VectorSite
from nexttransformer.methods.chordres_zero import unit_rms
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model


def oracle(h, v, f, b, w, beta):
    h, v, f = h.double(), v.double(), f.double()
    a = F.softplus(
        b.double() + torch.einsum("...d,d->...", f, w.double()).unsqueeze(-1)
    )
    retention = 0.5 if beta is None else beta.double().sigmoid()
    nxt = unit_rms((1 - a) * h + a * unit_rms(f))
    mix = retention * v + (1 - retention) * (nxt - h)
    return nxt, mix - (mix * nxt).mean(-1, keepdim=True) * nxt


def test_zero_start_and_equal_rms_features():
    cfg = debug_config("chordres_force_vector_step")
    model = build_model(cfg, seed=42)
    amplitude = build_model(
        replace(cfg, method="chordres_force_amplitude_step"), seed=42
    )
    assert all(
        s.force_weight.shape == (cfg.width,) and s.force_weight.count_nonzero() == 0
        for s in model.residual_maps
    )
    assert model.residual_maps[-1].beta is None
    ids = torch.arange(cfg.sequence_len).unsqueeze(0)
    torch.testing.assert_close(model(ids), amplitude(ids), rtol=0, atol=0)
    site = VectorSite(4, False).double()
    h = torch.tensor([[1.0, -1.0, 1.0, -1.0]], dtype=torch.float64)
    f = torch.tensor([[1.0, 1.0, -1.0, -1.0]], dtype=h.dtype)
    v = torch.zeros_like(h)
    with torch.no_grad():
        site.force_weight.copy_(f[0] * 0.5)
    positive, _ = site(h, v, f)
    negative, _ = site(h, v, -f)
    # Equal RMS and zero initial alignment with h, but different projected force logits.
    assert (positive * f).mean() > 10 * (negative * -f).mean()
    torch.testing.assert_close(
        positive, oracle(h, v, f, site.step, site.force_weight, site.beta)[0]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 fused CUDA oracle")
def test_cuda_oracle():
    from nexttransformer.methods.chordres_force_vector_step.kernels import VectorWrite
    from nexttransformer.methods.chordres_honly_single_step.kernels import TokenWrite
    from nexttransformer.methods.chordres_zero import ChordSite

    gen = torch.Generator(device="cuda").manual_seed(127 + torch.cuda.current_device())

    def rand(shape):
        return torch.randn(shape, generator=gen, device="cuda")

    for width, terminal, rows in ((1536, False, 1033), (96, True, 73), (96, False, 1)):
        s = VectorSite(width, terminal).cuda()
        h = unit_rms(rand((1, rows, width)))
        v = rand(h.shape)
        v -= h * (h * v).mean(-1, keepdim=True)
        f = rand(h.shape)
        if rows > 1:
            f[:, 0] = 0
            f[:, 1:8] *= 1e-4
            f[:, 8:16] *= 5
        h, v, f = [t.bfloat16().requires_grad_() for t in (h, v, f)]
        ctrl = (
            ChordSite(terminal)
            .cuda()()[[0, 1, 3]]
            .expand(*h.shape[:-1], 3)
            .contiguous()
        )
        for a, b in zip(s(h, v, f), TokenWrite.apply(h, v, f, ctrl), strict=True):
            assert torch.equal(a, b)
        with torch.no_grad():
            s.force_weight.copy_(rand((width,)) * (0.32 / width**0.5))
            s.step.fill_(-1.2)
        pars = (s.step, s.force_weight, s.beta)
        got = VectorWrite.apply(h, v, f, *pars)
        leaves = [q for q in (h, v, f, *pars) if q is not None]
        exact = [q.detach().double().requires_grad_() for q in leaves]
        want = oracle(*exact, None) if terminal else oracle(*exact)
        probes = [rand(h.shape).bfloat16() for _ in got]
        ga = torch.autograd.grad(got, leaves, probes)
        gb = torch.autograd.grad(want, exact, [p.double() for p in probes])
        for a, b in zip((*got, *ga), (*want, *gb), strict=True):
            assert a.isfinite().all()
            diff = (a.double() - b).norm()
            assert diff < 0.009 * b.norm() + 3e-5, (
                width,
                rows,
                a.shape,
                float(diff),
                float(b.norm()),
            )
        assert ga[4].shape == (width,) and ga[4].abs().max() > 0
