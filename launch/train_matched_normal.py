"""Research comparison of residual models with matched random output projections.

Launch like nexttransformer.train, using a supported residual configuration. The ordinary model, cost proof, optimizer and schedule are reused. The shared
shell's projection initialization is changed only through this research entry point.
"""
import math
from dataclasses import replace

NO_VE_METHOD = "chordres_no_ve_save_force"

HEAD_LOOKAHEAD_METHOD = "chordres_head_lookahead"

NO_VE_CONTROLS = {
    NO_VE_METHOD: "chordres_zero_save_force",
    "residual_standard_no_ve": "residual_standard",
    "mhc_nanochat_no_ve": "mhc_nanochat_readout",
}

METHODS = {"chordres_honly_single_step", "chordres_honly_scalar_recipe", "chordres_honly_free", "chordres_joint_gate_bounded", "chordres_token_controls_joint_rms", "chordres_token_controls_vnorm", "chordres_token_controls", "chordres_velocity_gains", "chordres_ve_turn", *NO_VE_CONTROLS, HEAD_LOOKAHEAD_METHOD, "mhc_nanochat_readout", "residual_standard", "chordres_zero_save_force", "chordres_vembed_save_force"}


def initialization(seed, depth):
    return dict(residual_projection_init="normal", projection_std=.02 / math.sqrt(2 * depth),
                projection_seed=seed + 1, weight_seed=seed)


def initialize_projections(model, seed):
    import torch
    if model.config.method not in METHODS:
        raise ValueError(f"Unsupported method: {model.config.method}")
    setup = initialization(seed, model.config.n_layer)
    generator = torch.Generator(device=model.transformer.wte.weight.device)
    generator.manual_seed(setup["projection_seed"])
    with torch.no_grad():
        for block in model.transformer.h:
            for projection in (block.attn.c_proj, block.mlp.c_proj):
                projection.weight.normal_(std=setup["projection_std"], generator=generator)
    return model


def build_matched_model(build, config, device, seed):
    model = build(config, device, seed=seed)
    if config.method in NO_VE_CONTROLS:
        # Removing VE gates changes the native initialization RNG sequence. Seed
        # the shared tensors from the VE-on control and preserve its final RNG state.
        control_options = {name: value for name, value in config.options if name != "attention_value_embeddings"}
        control_config = replace(config, method=NO_VE_CONTROLS[config.method], options=control_options)
        control = build(control_config, device, seed=seed)
        shared = {name: weight for name, weight in control.state_dict().items()
                  if not name.startswith("value_embeds.") and ".ve_gate." not in name}
        model.load_state_dict(shared, strict=True)
    return initialize_projections(model, seed)


def install(trainer):
    parse, build, identity = trainer.parse, trainer.build_model, trainer.software_identity
    tracker = trainer.ReferenceTracker

    def research_args(argv=None):
        args = parse(argv)
        if args.formal:
            raise ValueError("Random residual output initialization requires --research")
        return args

    def build_matched(config, device, seed):
        return build_matched_model(build, config, device, seed)

    def software(config, sparse_backend):
        record = identity(config, sparse_backend)
        record["research_initialization"] = initialization(trainer.Protocol().seed, config.n_layer)
        return record

    class ResearchTracker(tracker):
        def __init__(self, args, rank, config, train, state, source):
            super().__init__(args, rank, config, train, state, source)
            if self.run is not None:
                self.run.config.update({"research_initialization": initialization(train.seed, config.n_layer),
                                        "training_mode": "research"})

    trainer.parse = research_args
    trainer.build_model = build_matched
    trainer.software_identity = software
    trainer.ReferenceTracker = ResearchTracker


if __name__ == "__main__":
    from nexttransformer.runtime.site import bootstrap_runtime
    bootstrap_runtime()
    import nexttransformer.train as trainer
    install(trainer)
    trainer.main()
