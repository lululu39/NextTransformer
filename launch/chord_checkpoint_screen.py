"""Apply a matched-BPB Chord screen at native validation boundaries."""

import torch
import torch.distributed as dist
from nexttransformer.runtime.files import atomic_json


def decision(step, gap, previous_gap):
    if step == 2000:
        return "promote" if gap < 0 else "stop"
    if previous_gap is not None and previous_gap > 0 and gap > 0 and gap >= previous_gap:
        return "stop"
    return "continue"


def install(trainer, get_options, references):
    evaluate = trainer.evaluate_bpb
    previous_gap = None
    history = {}

    def measured(model, *args, **kwargs):
        nonlocal previous_gap
        result = evaluate(model, *args, **kwargs)
        options = get_options()
        step = model.optimizer_step + 1
        if (options.smoke_steps or options.formal or kwargs.get("split", "val") != "val"
                or step == 1 or step > 2000 or str(step) not in references):
            return result
        stop = False
        if kwargs["rank"] == 0:
            ref = references[str(step)]
            bpb = result["bpb"]
            gap = bpb - ref["bpb"]
            action = decision(step, gap, previous_gap)
            previous_gap = gap
            history[str(step)] = dict(bpb=bpb, best_previous_chord=ref["bpb"],
                                      reference_run=ref["run"], gap=gap, decision=action)
            atomic_json(options.output / "screen_history.json", history)
            result.update(best_previous_chord_bpb=ref["bpb"], gap_to_best_chord=gap)
            stop = action == "stop"
        signal = torch.tensor(int(stop), device=kwargs["device"])
        if dist.is_initialized():
            dist.broadcast(signal, 0)
        if signal.item():
            # The native trainer commits the validation and resumable checkpoint at this update.
            options.max_wall_hours = 1e-12
        return result

    trainer.evaluate_bpb = measured
