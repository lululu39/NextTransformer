"""Matched-normal ChordRes from rest with the 50M nanochat connector peak LR."""
from launch import train_matched_normal

CONTROL_PEAK_LR = 0.0024494897427831783
METHOD = "chordres_zero_save_force"
SOFT_FORCE_METHODS = ("chordres_update_soft_force", "chordres_soft_force_h0_velocity")
LEARNED_MIX_METHODS = ("chordres_honly_mhc_mix", *SOFT_FORCE_METHODS)


def set_control_lr(model, optimizer, peak_lr=CONTROL_PEAK_LR):
    if model.config.method not in (METHOD, *LEARNED_MIX_METHODS, "chordres_honly_small_dynamic", "chordres_ve_unit_impulse", "chordres_ve_impulse", "chordres_force_vector_step", "chordres_force_amplitude_step", "chordres_honly_single_step", "chordres_honly_scalar_recipe", "chordres_honly_free", "chordres_joint_gate_bounded", "chordres_token_controls_joint_rms", "chordres_token_controls_vnorm", "chordres_token_controls", "chordres_raw_force", "chordres_velocity_gains", "chordres_ve_turn", train_matched_normal.NO_VE_METHOD, train_matched_normal.HEAD_LOOKAHEAD_METHOD):
        raise ValueError("This research recipe requires a saved-force ChordRes model starting from rest")
    controls = {id(p) for p in model.residual_maps.parameters()}
    groups = [g for g in optimizer.param_groups if any(id(p) in controls for p in g["params"])]
    if len(groups) != 1 or {id(p) for p in groups[0]["params"]} != controls:
        raise ValueError("Expected a dedicated group for exactly the Chord control parameters")
    group = groups[0]
    if group["kind"] != "adamw" or group["weight_decay"] != 0.:
        raise ValueError("Unexpected Chord control optimizer group")
    if model.config.method in LEARNED_MIX_METHODS:
        peak_lr = CONTROL_PEAK_LR
    if model.config.method in ("chordres_honly_small_dynamic", *LEARNED_MIX_METHODS):
        # Keep the projection at the shell's native auxiliary rate. Raise only
        # the static biases to the peak used by the paired static Chord run.
        if peak_lr != CONTROL_PEAK_LR:
            raise ValueError("The small dynamic correction uses the high static Chord learning rate")
        projections = []
        biases = []
        for name, parameter in zip(group["parameter_names"], group["params"], strict=True):
            (projections if name.endswith((".token_weight", ".mix_scales")) else biases).append((name, parameter))
        if not projections or not biases:
            raise ValueError("Both static Chord biases and token projections are required")
        projection_group = dict(group, parameter_names=[n for n, _ in projections],
                                params=[p for _, p in projections])
        group["parameter_names"] = [n for n, _ in biases]
        group["params"] = [p for _, p in biases]
        optimizer.add_param_group(projection_group)
    if peak_lr is not None:
        group["lr"] = group["initial_lr"] = peak_lr
    return group["initial_lr"]


def install(trainer, *, native_control_lr=False, matched_normal=True):
    if matched_normal:
        train_matched_normal.install(trainer)
    build, identity, tracker = trainer.build_optimizer, trainer.software_identity, trainer.ReferenceTracker

    append = trainer.append_event
    ve_impulse_scale = None
    head_rho = None
    gain_model = None
    control_peak_lr = None if native_control_lr else CONTROL_PEAK_LR

    def build_optimizer(model, train):
        nonlocal head_rho, gain_model, control_peak_lr, ve_impulse_scale
        if model.config.method == "chordres_ve_impulse":
            ve_impulse_scale = model.residual_maps[0].ve_scale
        if model.config.method in ("chordres_velocity_gains", "chordres_raw_force"):
            gain_model = model
        if model.config.method == train_matched_normal.HEAD_LOOKAHEAD_METHOD:
            head_rho = model.residual_maps[-1].rho
        optimizer = build(model, train)
        control_peak_lr = set_control_lr(model, optimizer, None if native_control_lr else CONTROL_PEAK_LR)
        return optimizer

    def software(config, sparse_backend):
        record = identity(config, sparse_backend)
        record["chord_control_peak_lr"] = control_peak_lr
        if not matched_normal:
            record["branch_output_initialization"] = "native_zero"
        return record

    class ControlTracker(tracker):
        def __init__(self, args, rank, config, train, state, source):
            super().__init__(args, rank, config, train, state, source)
            if self.run is not None:
                self.run.config.update({"chord_control_peak_lr": control_peak_lr,
                                        "zero_velocity_reference": "1879c1a2"})
                if not matched_normal:
                    self.run.config.update({"branch_output_initialization": "native_zero"})
                if config.method in ("chordres_token_controls", "chordres_token_controls_vnorm", "chordres_token_controls_joint_rms", "chordres_joint_gate_bounded"):
                    self.run.config.update({"token_control_input": "pre-sublayer position and velocity",
                                            "token_control_weight_init": 0., "static_control_run": "a710a057",
                                            "token_control_terminal_outputs": 3, "head_lookahead": False})
                if config.method in LEARNED_MIX_METHODS:
                    from nexttransformer.methods.chordres_honly_mhc_mix import INITIAL_SCALE
                    self.run.config.update({"dynamic_scale_init": INITIAL_SCALE, "dynamic_scale_trainable": True,
                                            "modulation_scale_groups": ["lookahead", "position", "velocity"],
                                            "projection_and_scale_peak_lr": train.adamw_lr * (train.global_batch_tokens / 524288) ** .5,
                                            "token_projection_init": 0., "gate_input": "RMSNorm(h)",
                                            "fixed_scale_reference": "faea35e6", "static_three_gain_reference": "a710a057"})
                if config.method in SOFT_FORCE_METHODS:
                    from nexttransformer.methods.chordres_update_soft_force import FORCE_TAU
                    self.run.config.update({"position_force_role": "update", "force_rms_tau": FORCE_TAU,
                                            "target_hard_reference": "45be3fd3"})
                if config.method == "chordres_soft_force_h0_velocity":
                    self.run.config.update({"initial_velocity_source": "tangent projection of W_v h0",
                                            "initial_velocity_weight_init": 0., "zero_v0_reference": "2574b7dc"})
                if config.method == "chordres_honly_small_dynamic":
                    from nexttransformer.methods.chordres_honly_small_dynamic import DYNAMIC_SCALE
                    self.run.config.update({"token_projection_scale": DYNAMIC_SCALE,
                                            "token_projection_peak_lr": train.adamw_lr * (train.global_batch_tokens / 524288) ** .5,
                                            "token_projection_init": 0., "gate_input": "RMSNorm(h)",
                                            "static_three_gain_reference": "a710a057", "unscaled_gate_reference": "e266c628"})
                if config.method == "chordres_force_vector_step":
                    self.run.config.update({"position_step": "softplus(b + dot(w,F))",
                                            "step_input": "full pre-normalization branch output vector",
                                            "force_vector_weight_init": 0., "force_direction": "unit RMS",
                                            "velocity_write": "static beta * old + (1-beta) * displacement",
                                            "lookahead": "static mu"})
                if config.method == "chordres_force_amplitude_step":
                    self.run.config.update({"position_step": "softplus(b + w * log(1 + mean(F^2)))",
                                            "amplitude_weight_init": 0., "force_direction": "unit RMS",
                                            "velocity_write": "static beta * old + (1-beta) * displacement",
                                            "lookahead": "static mu"})
                if config.method == "chordres_ve_unit_impulse":
                    self.run.config.update({"ve_impulse_gain": 1.0, "ve_impulse_learnable": False,
                                            "learned_gain_reference": "6645790a", "ve_impulse_reference": "13fb5703"})
                if config.method == "chordres_ve_impulse":
                    self.run.config.update({"ve_impulse_reference": "13fb5703", "ve_impulse_scale_init": 0.,
                                            "ve_impulse_scale_count": 1, "ve_impulse_input": "raw same-layer same-token lookup"})
                if config.method == "chordres_honly_single_step":
                    self.run.config.update({"token_control_input": "normalized pre-sublayer position only",
                                            "gate_rms_eps": 1e-6, "velocity_write": "beta * old + (1-beta) * displacement",
                                            "control_activation": "direct mu, softplus alpha, sigmoid beta",
                                            "honly_three_gain_reference": "e266c628", "head_lookahead": False})
                if config.method in ("chordres_honly_free", "chordres_honly_scalar_recipe"):
                    from nexttransformer.methods.chordres_honly_free import GATE_EPS
                    self.run.config.update({"token_control_input": "normalized pre-sublayer position only",
                                            "gate_rms_eps": GATE_EPS, "control_activation": ("identity" if config.method == "chordres_honly_free" else "scalar recipe: direct mu/gamma, softplus alpha, sigmoid beta"),
                                            "token_control_weight_init": 0., "token_control_terminal_outputs": 3,
                                            "bounded_joint_reference": "ae5f6592", "head_lookahead": False})
                if config.method == "chordres_joint_gate_bounded":
                    from nexttransformer.methods.chordres_joint_gate_bounded import GATE_EPS
                    self.run.config.update({"gate_normalization": "joint position and velocity", "gate_rms_eps": GATE_EPS,
                                            "control_bounds": {"mu": [-3, 3], "alpha": [0, 1], "beta": [0, 1], "gamma": [0, 2]},
                                            "unbounded_joint_gate_reference": "9fda8c33"})
                if config.method == "chordres_token_controls_joint_rms":
                    from nexttransformer.methods.chordres_token_controls_joint_rms import GATE_EPS
                    self.run.config.update({"gate_normalization": "joint position and velocity", "gate_rms_eps": GATE_EPS,
                                            "velocity_only_reference": "57313ecf"})
                if config.method == "chordres_token_controls_vnorm":
                    from nexttransformer.methods.chordres_token_controls_vnorm import GATE_EPS
                    self.run.config.update({"velocity_gate_rms_eps": GATE_EPS, "raw_velocity_reference": "81a3c5bc"})
                if config.method in ("chordres_velocity_gains", "chordres_raw_force"):
                    self.run.config.update({"velocity_gains": "direct trainable parallel and perpendicular gains",
                                            "gamma_parallel_init": .5, "gamma_perp_init": .5,
                                            "velocity_rule_reference": "13fb5703", "head_lookahead": False})
                if config.method == "chordres_raw_force":
                    self.run.config.update({"force_normalization": "none", "normalized_force_reference": "a710a057"})
                if config.method == "chordres_ve_turn":
                    self.run.config.update({"full_perpendicular_velocity_at": "VE-enabled attention only",
                                            "matched_velocity_rule_reference": "13fb5703",
                                            "attention_value_embeddings": True, "input_smear": True})
                if config.method == train_matched_normal.HEAD_LOOKAHEAD_METHOD:
                    self.run.config.update({"head_lookahead": True, "readout_rho_init": 0.,
                                            "head_off_same_lr_reference": "13fb5703",
                                            "attention_value_embeddings": True, "input_smear": True})
                if config.method == train_matched_normal.NO_VE_METHOD:
                    self.run.config.update({"attention_value_embeddings": False, "input_smear": True,
                                            "ve_on_same_lr_reference": "13fb5703"})

    def append_event(output, event):
        # Native rank-zero events subsequently feed both the JSONL and W&B.
        if ve_impulse_scale is not None and event["event"] in ("train", "validation"):
            event["ve_impulse_lambda"] = ve_impulse_scale.detach().item()
        if head_rho is not None and event["event"] == "train":
            event["head_rho"] = head_rho.detach().item()
        if gain_model is not None and event["event"] == "validation":
            import torch
            from statistics import median
            from nexttransformer.runtime.files import atomic_json
            sites = list(gain_model.residual_maps)[:-1]
            with torch.no_grad():
                controls = torch.stack([torch.cat((s.beta.sigmoid(), s.velocity_gains)) for s in sites]).cpu().tolist()
            rows = []
            for i, (beta, par, perp) in enumerate(controls):
                family = "mlp" if i % 2 else ("ve_attention" if str(i // 2) in gain_model.value_embeds else "attention")
                rows.append(dict(block=i//2+1, site=i, family=family, beta=beta, gamma_parallel=par, gamma_perpendicular=perp))
            atomic_json(output / f"velocity_gains_{event['step']:08d}.json", rows)
            for family in ("ve_attention", "attention", "mlp"):
                values = [row for row in rows if row["family"] == family]
                for key in ("beta", "gamma_parallel", "gamma_perpendicular"):
                    event[f"{family}_{key}_median"] = median(row[key] for row in values)
        return append(output, event)

    from launch.chord_token_metrics import wrap_validation
    trainer.evaluate_bpb = wrap_validation(trainer.evaluate_bpb)
    trainer.append_event = append_event
    trainer.build_optimizer = build_optimizer
    trainer.software_identity = software
    trainer.ReferenceTracker = ControlTracker


if __name__ == "__main__":
    from nexttransformer.runtime.site import bootstrap_runtime
    bootstrap_runtime()
    import nexttransformer.train as trainer
    install(trainer)
    trainer.main()
