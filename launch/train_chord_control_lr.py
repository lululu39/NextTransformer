"""Matched-normal ChordRes from rest with the 50M nanochat connector peak LR."""
from launch import train_matched_normal

CONTROL_PEAK_LR = 0.0024494897427831783
METHOD = "chordres_zero_save_force"


def set_control_lr(model, optimizer):
    if model.config.method not in (METHOD, "chordres_ve_turn", train_matched_normal.NO_VE_METHOD, train_matched_normal.HEAD_LOOKAHEAD_METHOD):
        raise ValueError("This research recipe requires a saved-force ChordRes model starting from rest")
    controls = {id(p) for p in model.residual_maps.parameters()}
    groups = [g for g in optimizer.param_groups if any(id(p) in controls for p in g["params"])]
    if len(groups) != 1 or {id(p) for p in groups[0]["params"]} != controls:
        raise ValueError("Expected a dedicated group for exactly the Chord control parameters")
    group = groups[0]
    if group["kind"] != "adamw" or group["weight_decay"] != 0.:
        raise ValueError("Unexpected Chord control optimizer group")
    group["lr"] = group["initial_lr"] = CONTROL_PEAK_LR


def install(trainer):
    train_matched_normal.install(trainer)
    build, identity, tracker = trainer.build_optimizer, trainer.software_identity, trainer.ReferenceTracker

    append = trainer.append_event
    head_rho = None

    def build_optimizer(model, train):
        nonlocal head_rho
        if model.config.method == train_matched_normal.HEAD_LOOKAHEAD_METHOD:
            head_rho = model.residual_maps[-1].rho
        optimizer = build(model, train)
        set_control_lr(model, optimizer)
        return optimizer

    def software(config, sparse_backend):
        record = identity(config, sparse_backend)
        record["chord_control_peak_lr"] = CONTROL_PEAK_LR
        return record

    class ControlTracker(tracker):
        def __init__(self, args, rank, config, train, state, source):
            super().__init__(args, rank, config, train, state, source)
            if self.run is not None:
                self.run.config.update({"chord_control_peak_lr": CONTROL_PEAK_LR,
                                        "zero_velocity_reference": "1879c1a2"})
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
        return append(output, event)

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
