"""Chord research launcher using native zero branch outputs and native auxiliary AdamW."""
from launch import train_chord_control_lr


def install(trainer):
    train_chord_control_lr.install(trainer, native_control_lr=True, matched_normal=False)


if __name__ == "__main__":
    from nexttransformer.runtime.site import bootstrap_runtime

    bootstrap_runtime()
    import nexttransformer.train as trainer

    install(trainer)
    trainer.main()
