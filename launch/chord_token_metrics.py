"""Sample cross-token coefficient dispersion once per validation on rank zero."""
import torch


def wrap_validation(evaluate):
    def measured(model, *args, **kwargs):
        if model.config.method not in ("chordres_token_controls", "chordres_token_controls_vnorm", "chordres_token_controls_joint_rms", "chordres_joint_gate_bounded", "chordres_honly_free", "chordres_honly_scalar_recipe", "chordres_honly_single_step") or kwargs["rank"] != 0:
            return evaluate(model, *args, **kwargs)

        dispersions = []
        handles = []

        def sample(site, inputs, output):
            controls = output[1].detach().reshape(-1, output[1].shape[-1])
            dispersions.append(controls.std(dim=0, unbiased=False))
            handles[len(dispersions) - 1].remove()

        handles.extend(site.register_forward_hook(sample) for site in model.residual_maps)
        try:
            result = evaluate(model, *args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        if len(dispersions) != len(model.residual_maps):
            raise RuntimeError("Expected one conditioning sample from every Chord site")
        # The terminal MLP has a position read and write but no consumed velocity.
        spread = torch.stack(dispersions)
        if model.config.method == "chordres_honly_single_step":
            values = torch.cat((spread[:, :2].mean(0), spread[:-1, 2:].mean(0))).tolist()
            names = ("mu", "alpha", "beta")
        else:
            values = torch.cat((spread[:, :3].mean(0), spread[:-1, 3:].mean(0))).tolist()
            names = ("mu", "alpha_perp", "alpha_parallel", "beta", "gamma_parallel", "gamma_perp")
        result.update({f"token_std_{name}": value for name, value in zip(names, values, strict=True)})
        return result

    return measured
