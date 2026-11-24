"""Matched-normal ChordRes from rest with the 50M nanochat connector peak LR."""
from launch import train_matched_normal

CONTROL_PEAK_LR = 0.0024494897427831783
METHOD = "chordres_zero_save_force"


def set_control_lr(model, optimizer, peak_lr=CONTROL_PEAK_LR):
    if model.config.method not in (METHOD, "chordres_force_amplitude_step", "chordres_honly_single_step", "chordres_honly_scalar_recipe", "chordres_honly_free", "chordres_joint_gate_bounded", "chordres_token_controls_joint_rms", "chordres_token_controls_vnorm", "chordres_token_controls", "chordres_raw_force", "chordres_velocity_gains", "chordres_ve_turn", train_matched_normal.NO_VE_METHOD, train_matched_normal.HEAD_LOOKAHEAD_METHOD):
        raise ValueError("This research recipe requires a saved-force ChordRes model starting from rest")
    controls = {id(p) for p in model.residual_maps.parameters()}
    groups = [g for g in optimizer.param_groups if any(id(p) in controls for p in g["params"])]
    if len(groups) != 1 or {id(p) for p in groups[0]["params"]} != controls:
        raise ValueError("Expected a dedicated group for exactly the Chord control parameters")
    group = groups[0]
    if group["kind"] != "adamw" or group["weight_decay"] != 0.:
        raise ValueError("Unexpected Chord control optimizer group")
    if peak_lr is not None:
        group["lr"] = group["initial_lr"] = peak_lr
    return group["initial_lr"]


def install(trainer, *, native_control_lr=False, matched_normal=True):
    if matched_normal:
        train_matched_normal.install(trainer)
    build, identity, tracker = trainer.build_optimizer, trainer.software_identity, trainer.ReferenceTracker

    append = trainer.append_event
    head_rho = None
    gain_model = None
    control_peak_lr = None if native_control_lr else CONTROL_PEAK_LR

    def build_optimizer(model, train):
        nonlocal head_rho, gain_model, control_peak_lr
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
                if config.method == "chordres_force_amplitude_step":
                    self.run.config.update({"position_step": "softplus(b + w * log(1 + mean(F^2)))",
                                            "amplitude_weight_init": 0., "force_direction": "unit RMS",
                                            "velocity_write": "static beta * old + (1-beta) * displacement",
                                            "lookahead": "static mu"})
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
