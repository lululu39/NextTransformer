from dataclasses import replace
import math

import torch
from torch.nn import functional as F

from launch.train_matched_normal import initialize_projections
from nexttransformer.methods.mhc_nanochat_readout import Router
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import parameter_groups


def test_router_and_head_match_nanochat_input_to_output_equations():
    torch.manual_seed(7)
    for head in (False, True):
        router = Router(8, 4, 0, 20, head=head).double()
        router.reset_parameters()
        torch.testing.assert_close(router.bias[:4], torch.full((4,), -math.log(3), dtype=torch.float64))
        with torch.no_grad():
            router.scales.fill_(.7)  # Exercise asymmetric, input-dependent matrices and their gradients.
        x = torch.randn(2, 5, 4, 8, dtype=torch.float64, requires_grad=True)
        raw = x.transpose(-1,-2).contiguous()
        flat = raw.flatten(-2)
        # Equivalent permutation to nanochat's interleave_phi for [batch,time,width,streams].
        phi = router.projection.unflatten(-1, (4,8)).transpose(-1,-2).flatten(-2)
        projected = F.linear(flat * (flat.square().mean(-1,keepdim=True) + torch.finfo(torch.float32).eps).rsqrt(), phi)
        read = (projected[...,:4] * router.scales[0] + router.bias[:4]).sigmoid()
        expected_read = (raw * read.unsqueeze(-2)).sum(-1)
        got = router(x)
        if head:
            outputs, targets = (got,), (expected_read,)
        else:
            post = 2*(projected[...,4:8] * router.scales[1] + router.bias[4:8]).sigmoid()
            logmix = (projected[...,8:] * router.scales[2] + router.bias[8:]).unflatten(-1,(4,4))
            for _ in range(20):
                logmix = logmix - torch.logsumexp(logmix, -1, keepdim=True)
                logmix = logmix - torch.logsumexp(logmix, -2, keepdim=True)
            expected_residual = torch.einsum("btdn,btnm->btdm", raw, logmix.exp()).transpose(-1,-2)
            outputs, targets = got, (expected_read, expected_residual, post)
        probes = tuple(torch.randn_like(t) for t in targets)
        inputs = (x, *router.parameters())
        actual_grads = torch.autograd.grad(outputs, inputs, probes)
        expected_grads = torch.autograd.grad(targets, inputs, probes)
        for a, b in zip((*outputs, *actual_grads), (*targets, *expected_grads)):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)


def test_shared_backbone_initialization_and_learned_readout():
    cfg = debug_config("mhc_nanochat_readout")
    model = initialize_projections(build_model(cfg, seed=42), 42)
    official = build_model(replace(cfg, method="mhc", options=()), seed=42)
    # The official method uses the shell's usual zero projections; apply matched draws
    # to precisely the same-shaped attention/FFN matrices for this backbone comparison.
    rng = torch.Generator().manual_seed(43)
    with torch.no_grad():
        for layer in official.transformer.h:
            for proj in (layer.attn.c_proj, layer.mlp.c_proj):
                proj.weight.normal_(std=.02/math.sqrt(2*cfg.n_layer), generator=rng)
    params = dict(official.named_parameters())
    for name, param in model.named_parameters():
        if not name.startswith("residual_maps."):
            torch.testing.assert_close(param, params[name], rtol=0, atol=0)
    group = next(g for g in parameter_groups(model) if any(n.startswith("residual_maps.") for n in g["parameter_names"]))
    head_name = f"residual_maps.{2*cfg.n_layer}.projection"
    assert head_name in group["parameter_names"] and group["kind"]=="adamw" and group["weight_decay"]==0.
    ids = torch.randint(cfg.vocab_size, (2,cfg.sequence_len))
    model(ids,ids).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert model.residual_maps[-1].projection.grad.norm() > 0
