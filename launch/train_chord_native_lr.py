"""Matched-normal ChordRes with the native NextTransformer auxiliary AdamW rate."""
from launch import train_chord_control_lr


def install(trainer):
    train_chord_control_lr.install(trainer, native_control_lr=True)


if __name__ == "__main__":
    from nexttransformer.runtime.site import bootstrap_runtime

    bootstrap_runtime()
    import nexttransformer.train as trainer

    install(trainer)
    trainer.main()
