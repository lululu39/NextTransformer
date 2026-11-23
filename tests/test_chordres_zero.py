from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from nexttransformer.methods.chordres_zero import ChordSite, reference_read, reference_write, unit_rms
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import parameter_groups


def test_shared_initialization_and_zero_velocity_model():
    cfg = debug_config("chordres_zero")
    chord = build_model(cfg)
    standard = build_model(replace(cfg, method="residual_standard", options=()))
    actual = dict(chord.named_parameters())
    for name, param in standard.named_parameters():
        torch.testing.assert_close(actual[name], param, rtol=0, atol=0)
        assert actual[name].requires_grad == param.requires_grad
    assert not any("velocity" in name for name in actual)
    assert sum(p.numel() for p in chord.residual_maps.parameters()) == 8 * cfg.layers - 1
    groups = parameter_groups(chord)
    for group in groups:
        if any(name.startswith("residual_maps.") for name in group["parameter_names"]):
            assert group["kind"] == "adamw" and group["weight_decay"] == 0

    ids = torch.randint(cfg.vocab_size, (2, cfg.sequence_len))
    initial = chord(ids)
    loss = chord(ids, ids)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(initial).all()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in actual.values() if p.requires_grad)
    assert chord.transformer.h[0].attn.c_proj.weight.grad.abs().max() > 0
    # The fixed shell initializes both output projections to zero. Exercise the moving state too.
    with torch.no_grad():
        for layer in chord.transformer.h:
            layer.attn.c_proj.weight.normal_(std=.03)
            layer.mlp.c_proj.weight.normal_(std=.03)
        for site in chord.residual_maps:
            site.mu.fill_(.4)
            site.steps[1].add_(.3)
        full = chord(ids)
        prefix = chord(ids[:, :-2])
        torch.testing.assert_close(full[:, :-2], prefix, atol=1e-5, rtol=1e-5)
        saved = chord.state_dict()
        reloaded = build_model(cfg)
        reloaded.load_state_dict(saved, strict=True)
        torch.testing.assert_close(reloaded(ids), full, atol=0, rtol=0)


def test_geometry_at_rest_and_in_motion():
    h = unit_rms(torch.randn(2, 7, 32, dtype=torch.float64))
    site = ChordSite(False).double()
    controls = site()
    one, velocity = reference_write(h, torch.zeros_like(h), torch.randn_like(h), controls)
    torch.testing.assert_close(one.square().mean(-1), torch.ones_like(one[..., 0]))
    torch.testing.assert_close((one * velocity).mean(-1), torch.zeros_like(one[..., 0]))
    assert velocity.abs().sum() > 0
    torch.testing.assert_close(controls, torch.tensor([0, .05, .05, .5], dtype=h.dtype), atol=1e-8, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Triton parity")
def test_cuda_branches_and_rematerialized_gradients():
    from nexttransformer.methods.chordres_zero.kernels import ChordWrite, ReadNorm
    torch.manual_seed(42)
    width = 1536
    h = unit_rms(torch.randn(1, 19, width, device="cuda"))
    v = torch.randn_like(h)
    v = v - h * (v * h).mean(-1, keepdim=True)
    v[:, :3].zero_()
    inner = torch.randn(1, 19, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(width, 256, device="cuda", dtype=torch.bfloat16) * .015
    controls = torch.tensor([.29, .13, .24, .7], device="cuda")
    leaves = [x.detach().requires_grad_() for x in (h.bfloat16(), v.bfloat16(), controls, inner, weight)]
    h, v, controls, inner, weight = leaves
    force = F.linear(inner, weight)
    actual = (ReadNorm.apply(h, v, controls), *ChordWrite.apply(h, v, force, controls, F.linear, inner, weight))
    expected = (reference_read(h, v, controls), *reference_write(h, v, F.linear(inner, weight), controls))
    directions = tuple(torch.randn_like(x) for x in actual)
    grad = torch.autograd.grad(actual, leaves, directions)
    oracle = torch.autograd.grad(expected, leaves, directions)
    for a, b in zip((*actual, *grad), (*expected, *oracle)):
        error = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)
        assert error < .016, error
