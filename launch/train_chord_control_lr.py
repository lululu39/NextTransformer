"""Matched-normal ChordRes from rest with the 50M nanochat connector peak LR."""
from launch import train_matched_normal

CONTROL_PEAK_LR = 0.0024494897427831783
METHOD = "chordres_zero_save_force"


def set_control_lr(model, optimizer):
    if model.config.method not in (METHOD, train_matched_normal.NO_VE_METHOD):
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

    def build_optimizer(model, train):
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
                if config.method == train_matched_normal.NO_VE_METHOD:
                    self.run.config.update({"attention_value_embeddings": False, "input_smear": True,
                                            "ve_on_same_lr_reference": "13fb5703"})

    trainer.build_optimizer = build_optimizer
    trainer.software_identity = software
    trainer.ReferenceTracker = ControlTracker


if __name__ == "__main__":
    from nexttransformer.runtime.site import bootstrap_runtime
    bootstrap_runtime()
    import nexttransformer.train as trainer
    install(trainer)
    trainer.main()
