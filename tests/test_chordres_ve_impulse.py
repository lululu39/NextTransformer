from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from launch.train_matched_normal import initialize_projections
from nexttransformer.methods import chordres_ve_impulse as method
from nexttransformer.methods.chordres_zero import reference_write, unit_rms
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model


def definition(h, v, force, controls, embedding, scale):
    # Project the two contributions separately in double precision.
    nxt, base = reference_write(h, v, force, controls)
    tangent_embedding = embedding - (embedding * nxt).mean(-1, keepdim=True) * nxt
    return nxt, base + scale * tangent_embedding


def test_raw_lookup_routing_and_shared_scale():
    cfg = debug_config("chordres_ve_impulse")
    model = initialize_projections(build_model(cfg), 42)
    scale = model.residual_maps[0].ve_scale
    assert scale.item() == 0
    assert [name for name, _ in model.named_parameters() if "ve_scale" in name] == ["residual_maps.0.ve_scale"]
    # The initial equal position steps and zero lookahead mask velocity at readout.
    # Make subsequent reads use velocity to test the full loss-to-gain gradient.
    with torch.no_grad():
        for site in model.residual_maps:
            site.mu.fill_(0.25)
    ids = torch.randint(cfg.vocab_size, (2, cfg.sequence_len))
    lookups, writes = [], []
    original = method.reference_impulse_write

    def collect(module, inputs, output):
        lookups.append(output.detach().clone())

    def check_write(h, v, force, controls, embedding, gain):
        assert gain is scale and len(lookups) == len(writes) + 1
        torch.testing.assert_close(embedding, lookups[-1].to(embedding.dtype), atol=0, rtol=0)
        writes.append(embedding)
        return original(h, v, force, controls, embedding, gain)

    with ExitStack() as stack:
        for table in model.value_embeds.values():
            stack.enter_context(table.register_forward_hook(collect))
        stack.enter_context(patch.object(method, "reference_impulse_write", check_write))
        model(ids, ids).backward()
    assert len(writes) == len(model.value_embeds) > 0
    assert scale.grad.isfinite().all() and scale.grad.abs().item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the BF16 Triton write")
def test_cuda_oracle():
    from nexttransformer.methods.chordres_ve_impulse.kernels import ImpulseWrite
    from nexttransformer.methods.chordres_zero.kernels import ChordWrite

    gen = torch.Generator(device="cuda").manual_seed(582 + torch.cuda.current_device())
    def sample():
        return torch.randn((8, 1536), generator=gen, device="cuda")
    h = unit_rms(sample())
    v = sample()
    v -= (v*h).mean(-1, keepdim=True) * h
    v[0].zero_()
    h, v, f, e = [t.bfloat16().requires_grad_() for t in (h, v, sample(), sample())]
    p = torch.tensor([.1, .12, .21, .7], device="cuda", requires_grad=True)
    weights = (sample().bfloat16(), sample().bfloat16())
    for value in (0., -.18, .12):
        gain = torch.tensor([value], device="cuda", requires_grad=True)
        args = (h, v, f, p, e, gain)
        actual = ImpulseWrite.apply(*args)
        base = ChordWrite.apply(h, v, f, p)
        assert torch.equal(actual[0], base[0])
        if value == 0:
            assert torch.equal(actual[1], base[1])
        exact_args = [t.detach().double().requires_grad_() for t in args]
        oracle = definition(*exact_args)
        grads = torch.autograd.grad(actual, args, weights)
        expected = torch.autograd.grad(oracle, exact_args, tuple(x.double() for x in weights))
        for fused, ref in zip((*actual, *grads), (*oracle, *expected), strict=True):
            rel = (fused.double() - ref).norm() / ref.norm().clamp_min(1e-10)
            assert rel < .009, rel
        assert grads[-1].abs().item() > 0
        if value == 0:
            assert torch.count_nonzero(grads[-2]) == 0
