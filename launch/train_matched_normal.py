"""Research comparison of residual models with matched random output projections.

Launch like nexttransformer.train, using a supported residual configuration. The ordinary model, cost proof, optimizer and schedule are reused. The shared
shell's projection initialization is changed only through this research entry point.
"""
import math

METHODS = {"residual_standard", "chordres_zero_save_force", "chordres_vembed_save_force"}


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


def install(trainer):
    parse, build, identity = trainer.parse, trainer.build_model, trainer.software_identity
    tracker = trainer.ReferenceTracker

    def research_args(argv=None):
        args = parse(argv)
        if args.formal:
            raise ValueError("Random residual output initialization requires --research")
        return args

    def build_matched(config, device, seed):
        return initialize_projections(build(config, device, seed=seed), seed)

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
