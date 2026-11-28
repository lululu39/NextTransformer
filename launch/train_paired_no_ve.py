"""Train either native-zero attention-VE ablation on a prefix of the original schedule."""
from dataclasses import replace

PARENTS = {
    "residual_standard_no_ve": "residual_standard",
    "chordres_soft_force_zero_no_ve": "chordres_update_soft_force",
}


def is_ve_weight(name):
    return name.startswith("value_embeds.") or ".ve_gate." in name


def build_paired(build, config, device, seed):
    reference = replace(config, method=PARENTS[config.method], options=())
    model = build(config, device, seed=seed)
    # A different number of initialization draws shifts shared backbone weights.
    # Build the seeded VE-on control last to leave the native RNG in its paired state.
    parent = build(reference, device, seed=seed)
    retained = {k: v for k, v in parent.state_dict().items() if not is_ve_weight(k)}
    model.load_state_dict(retained, strict=True)
    return model


def install(trainer, stop_step=16_000):
    native_parse = trainer.parse
    native_build = trainer.build_model
    native_schedule = trainer.set_schedule
    options = None

    def parse(argv=None):
        nonlocal options
        options = native_parse(argv)
        return options

    def build(config, device, seed):
        return build_paired(native_build, config, device, seed)

    def schedule(optimizer, step, config):
        if not options.smoke_steps and step + 1 == stop_step:
            # Activate the native segment-end path on all ranks. It performs due
            # validation and CORE, saves a resumable checkpoint and closes W&B.
            options.max_wall_hours = 1e-12
        return native_schedule(optimizer, step, config)

    trainer.parse = parse
    trainer.build_model = build
    trainer.set_schedule = schedule


if __name__ == "__main__":
    from nexttransformer.runtime.site import bootstrap_runtime
    bootstrap_runtime()
    import nexttransformer.train as trainer
    install(trainer)
    trainer.main()
