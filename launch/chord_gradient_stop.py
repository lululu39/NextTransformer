"""Stop this research run before an optimizer update if the DDP gradient explodes."""

from dataclasses import asdict
import json
from pathlib import Path

import torch
import torch.distributed as dist
from nexttransformer.runtime.files import atomic_json


class GradientStop(Exception):
    pass


def install(model, output: Path, threshold: float = 1e4):
    native_clip = torch.nn.utils.clip_grad_norm_
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    last_batch = None
    last_high = None

    def remember_batch(module, args):
        nonlocal last_batch
        if module.training and torch.is_grad_enabled() and len(args) >= 2:
            last_batch = tuple(t.detach() for t in args[:2])

    if rank == 0:
        model.register_forward_pre_hook(remember_batch)

    def checked(parameters, max_norm, *args, **kwargs):
        nonlocal last_high
        step = model.optimizer_step + 1
        try:
            norm = native_clip(parameters, max_norm, *args, **kwargs)
            value = norm.item()
            stop = int(value >= threshold and last_high == step - 1)
        except RuntimeError as error:
            if "non-finite" not in str(error):
                raise
            value = "nonfinite"
            stop = 2

        last_high = step if value == "nonfinite" or value >= threshold else None
        device = next(model.parameters()).device
        decision = torch.tensor(stop, device=device)
        if distributed:
            dist.all_reduce(decision, op=dist.ReduceOp.MAX)
        stop = decision.item()
        if rank == 0:
            with (output / "gradient_trace.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "grad_norm": value}) + "\n")
        if not stop:
            return norm
        if rank == 0:
            directory = output / f"gradient_stop_before_{step:08d}"
            directory.mkdir(exist_ok=True)
            temporary = directory / "model.pt.tmp"
            torch.save(
                {"model": model.state_dict(), "model_config": asdict(model.config)},
                temporary,
            )
            temporary.replace(directory / "model.pt")
            if last_batch is not None:
                torch.save(last_batch, directory / "rank0_last_microbatch.pt")
            record = {
                "before_update": step,
                "completed_updates": step - 1,
                "threshold": threshold,
                "reason": "nonfinite" if stop == 2 else "two_consecutive_high_norms",
                "last_grad_norm_rank0": value,
                "checkpoint": directory.name,
            }
            atomic_json(directory / "complete.json", record)
            atomic_json(output / "gradient_stop.json", record)
        if distributed:
            dist.barrier()
        raise GradientStop(
            "Saved pre-update diagnostic weights after gradient explosion"
        )

    torch.nn.utils.clip_grad_norm_ = checked
