from dataclasses import replace

import pytest
import torch

from launch.train_matched_normal import initialize_projections
from nexttransformer.methods.chordres_zero import reference_write, unit_rms
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model


def test_routing_and_value_table_gradients():
    config = debug_config("chordres_ve_unit_impulse")
    fixed = initialize_projections(build_model(config), 42)
    shared = initialize_projections(build_model(replace(config, method="chordres_ve_impulse")), 42)
    fixed_params, shared_params = dict(fixed.named_parameters()), dict(shared.named_parameters())
    assert shared_params.keys() - fixed_params.keys() == {"residual_maps.0.ve_scale"}
    assert fixed_params.keys() <= shared_params.keys()
    with torch.no_grad():
        shared.residual_maps[0].ve_scale.fill_(1.)
        for name, p in fixed_params.items():
            torch.testing.assert_close(p, shared_params[name], atol=0, rtol=0)
        for model in (fixed, shared):
            for site in model.residual_maps:
                site.mu.fill_(.3)
    ids = torch.randint(config.vocab_size, (2, config.sequence_len))
    a, b = fixed(ids, ids), shared(ids, ids)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    a.backward()
    b.backward()
    assert len(fixed.value_embeds) > 0
    for layer, table in fixed.value_embeds.items():
        assert table.weight.grad.abs().max() > 0
        torch.testing.assert_close(table.weight.grad, shared.value_embeds[layer].weight.grad, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the fused BF16 write")
def test_cuda_oracle():
    from nexttransformer.methods.chordres_ve_unit_impulse.kernels import ImpulseWrite
    from nexttransformer.methods.chordres_zero.kernels import ChordWrite

    gen = torch.Generator(device="cuda").manual_seed(750 + torch.cuda.current_device())
    def sample():
        return torch.randn((12, 1536), device="cuda", generator=gen)

    h = unit_rms(sample())
    v = sample()
    v -= h * (h * v).mean(-1, keepdim=True)
    v[0].zero_()
    h, v, f, e = [x.bfloat16().requires_grad_() for x in (h, v, sample(), sample())]
    controls = torch.tensor([.2, .13, .21, .65], device="cuda", requires_grad=True)
    args = (h, v, f, controls, e)
    out = ImpulseWrite.apply(*args)
    ordinary = ChordWrite.apply(h, v, f, controls)
    zero_ve = ImpulseWrite.apply(h, v, f, controls, torch.zeros_like(e))
    assert torch.equal(out[0], ordinary[0]) and not torch.equal(out[1], ordinary[1])
    assert torch.equal(zero_ve[1], ordinary[1])

    exact = [t.detach().double().requires_grad_() for t in args]
    h_next, tangent = reference_write(*exact[:4])
    embedding = exact[4]
    oracle = (h_next, tangent + (embedding - h_next * (embedding * h_next).mean(-1, keepdim=True)))
    weights = (sample().bfloat16(), sample().bfloat16())
    grads = torch.autograd.grad(out, args, weights)
    expected = torch.autograd.grad(oracle, exact, tuple(w.double() for w in weights))
    for actual, ref in zip((*out, *grads), (*oracle, *expected), strict=True):
        rel = (actual.double() - ref).norm() / ref.norm().clamp_min(1e-10)
        assert rel < .009, rel
    assert grads[-1].abs().max() > 0
