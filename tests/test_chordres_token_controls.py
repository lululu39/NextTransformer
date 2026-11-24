from dataclasses import replace

import pytest
import torch

from launch.train_chord_control_lr import CONTROL_PEAK_LR, set_control_lr
from launch.train_matched_normal import initialize_projections
from nexttransformer.methods.chordres_token_controls import (
    TokenSite,
    token_reference_read,
    token_reference_write,
)
from nexttransformer.methods.chordres_velocity_gains import VelocityGainSite
from nexttransformer.methods.chordres_zero import unit_rms
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import build_optimizer


def test_initial_model_and_optimizer():
    cfg = debug_config("chordres_token_controls")
    model = initialize_projections(build_model(cfg, seed=42), 42)
    ref = initialize_projections(
        build_model(replace(cfg, method="chordres_velocity_gains"), seed=42), 42
    )
    ours, theirs = dict(model.named_parameters()), dict(ref.named_parameters())
    weights = {f"residual_maps.{i}.token_weight" for i in range(2 * cfg.n_layer)}
    assert ours.keys() == theirs.keys() | weights
    assert all(torch.equal(ours[name], p) for name, p in theirs.items())
    assert all(ours[name].count_nonzero() == 0 for name in weights)
    ids = torch.arange(cfg.sequence_len).unsqueeze(0)
    torch.testing.assert_close(model(ids), ref(ids), rtol=0, atol=0)
    opt = build_optimizer(model)
    set_control_lr(model, opt)
    controls = next(
        group
        for group in opt.param_groups
        if weights.intersection(group["parameter_names"])
    )
    assert weights <= set(controls["parameter_names"])
    assert controls["initial_lr"] == CONTROL_PEAK_LR and controls["weight_decay"] == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires Triton BF16 kernels"
)
def test_cuda_oracle():
    from nexttransformer.methods.chordres_token_controls.kernels import (
        TokenRead,
        TokenWrite,
    )
    from nexttransformer.methods.chordres_velocity_gains.kernels import (
        VelocityGainWrite,
    )
    from nexttransformer.methods.chordres_zero.kernels import ReadNorm

    gen = torch.Generator(device="cuda").manual_seed(101 + torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = False
    for width, terminal in ((1536, False), (96, True)):

        def normal(shape):
            return torch.randn(shape, device="cuda", generator=gen)

        h = unit_rms(normal((1, 271, width)))
        v = normal(h.shape)
        v -= h * (h * v).mean(-1, keepdim=True)
        v[:, 0].zero_()
        h, v, force = [t.bfloat16().requires_grad_() for t in (h, v, normal(h.shape))]
        site = TokenSite(width, terminal).cuda()
        params = [
            site.token_weight,
            site.mu,
            site.steps,
            site.beta,
            site.velocity_gains,
        ]
        q, c = TokenRead.apply(h, v, *params)
        static = VelocityGainSite(terminal).cuda()()
        assert torch.equal(c, static.expand_as(c)), (c[0, 0], static)
        assert torch.equal(q, ReadNorm.apply(h, v, static))
        for a, b in zip(
            TokenWrite.apply(h, v, force, c),
            VelocityGainWrite.apply(h, v, force, static),
            strict=True,
        ):
            assert torch.equal(a, b)
        with torch.no_grad():
            site.token_weight.copy_(
                normal(site.token_weight.shape) * (0.12 / width**0.5)
            )
            site.mu.fill_(0.25)
            site.steps.copy_(torch.tensor([-1.3, -2.4], device="cuda"))
            if not terminal:
                site.velocity_gains.copy_(torch.tensor([0.41, 0.89], device="cuda"))
        q, c = TokenRead.apply(h, v, *params)
        x, u = TokenWrite.apply(h, v, force, c)
        rq, rc = token_reference_read(h, v, *params)
        rx, ru = token_reference_write(h, v, force, rc)
        assert c[..., :3].reshape(-1, 3).std(0).min() > 0
        if terminal:
            assert (c[..., 3:] == 0.5).all()
        probes = [normal(t.shape).to(t.dtype) for t in (q, c, x, u)]
        inputs = [p for p in (h, v, force, *params) if p is not None]
        grad = torch.autograd.grad((q, c, x, u), inputs, probes)
        ref_grad = torch.autograd.grad((rq, rc, rx, ru), inputs, probes)
        torch.testing.assert_close(c, rc, atol=1e-6, rtol=1e-5)
        for name, a, b in zip(
            [
                "q",
                "h_next",
                "v_next",
                "dh",
                "dv",
                "df",
                "dw",
                "dmu",
                "dalpha",
                "dbeta",
                "dgamma",
            ],
            (q, x, u, *grad),
            (rq, rx, ru, *ref_grad),
        ):
            assert torch.isfinite(a).all()
            error = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-6)
            assert error < 0.012, (width, name, float(error))
