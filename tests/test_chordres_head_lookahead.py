from dataclasses import replace

import torch

from launch.train_chord_control_lr import CONTROL_PEAK_LR, set_control_lr
from launch.train_matched_normal import build_matched_model
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import build_optimizer


def test_terminal_read_starts_as_exact_control_and_learns_from_velocity():
    cfg = debug_config("chordres_head_lookahead")
    control = build_matched_model(build_model, replace(cfg,method="chordres_zero_save_force"), "cpu", 42)
    trained = build_matched_model(build_model, cfg, "cpu", 42)
    params = dict(trained.named_parameters())
    reference = dict(control.named_parameters())
    rho = trained.residual_maps[-1].rho
    site_index = 2*cfg.n_layer
    assert params.keys() - reference.keys() == {f"residual_maps.{site_index}.rho"}
    assert trained.residual_maps[-2].beta is None and torch.count_nonzero(rho) == 0
    assert {k for k in trained.value_embeds} == set(control.value_embeds) and len(trained.value_embeds)>0
    for name,p in reference.items():
        torch.testing.assert_close(params[name],p,rtol=0,atol=0)
    opt = build_optimizer(trained)
    set_control_lr(trained,opt)
    group = next(g for g in opt.param_groups if any(p is rho for p in g["params"]))
    assert group["kind"]=="adamw" and group["initial_lr"]==CONTROL_PEAK_LR and group["weight_decay"]==0.
    ids = torch.randint(cfg.vocab_size,(2,cfg.sequence_len))
    targets = ids.roll(-1,1)
    arg = {}
    def capture(module,args):
        arg["h"],arg["v"]=args
    hook=trained.residual_maps[-1].register_forward_pre_hook(capture)
    torch.testing.assert_close(trained(ids),control(ids),rtol=0,atol=0)
    loss1,loss2=trained(ids,targets),control(ids,targets)
    torch.testing.assert_close(loss1,loss2,rtol=0,atol=0)
    loss1.backward();loss2.backward()
    for name,p in reference.items():
        if p.requires_grad:
            torch.testing.assert_close(params[name].grad,p.grad,rtol=1e-5,atol=1e-7)
    assert torch.isfinite(rho.grad).all() and rho.grad.abs().item()>0
    hook.remove()
    with torch.no_grad():
        rho.fill_(.3)
        actual=trained(ids); h,v=arg["h"],arg["v"]
        expected=trained._readout((h+.3*v).to(h.dtype),None)
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
        assert (actual-control(ids)).abs().max()>0
