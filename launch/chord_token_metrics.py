"""Sample cross-token coefficient dispersion once per validation on rank zero."""
import torch


def wrap_validation(evaluate):
    def measured(model, *args, **kwargs):
        amplitude = model.config.method == "chordres_force_amplitude_step"
        vector = model.config.method == "chordres_force_vector_step"
        if kwargs["rank"] != 0 or (not (amplitude or vector) and model.config.method not in ("chordres_token_controls", "chordres_token_controls_vnorm", "chordres_token_controls_joint_rms", "chordres_joint_gate_bounded", "chordres_honly_free", "chordres_honly_small_dynamic", "chordres_honly_mhc_mix", "chordres_update_soft_force", "chordres_honly_scalar_recipe", "chordres_honly_single_step")):
            return evaluate(model, *args, **kwargs)

        dispersions = []
        handles = []

        def sample(site, inputs, output):
            if vector:
                force = inputs[2].detach().float()
                w = site.force_weight.detach()
                logit = torch.nn.functional.linear(force, w.unsqueeze(0), site.step.detach())
                alpha = torch.nn.functional.softplus(logit)
                dispersions.append(torch.stack((alpha.mean(), alpha.std(unbiased=False), w.norm())))
            elif amplitude:
                force = inputs[2].detach().float()
                power = force.square().mean(-1)
                weight = site.amplitude_weight.detach().reshape(())
                alpha = torch.nn.functional.softplus(site.step.detach() + weight * torch.log1p(power))
                dispersions.append(torch.stack((alpha.mean(), alpha.std(unbiased=False), weight, weight.abs())))
            else:
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
        if vector:
            result.update(dict(zip(("mean_alpha","token_std_alpha","force_projection_l2"),
                                   spread.mean(0).tolist(),strict=True)))
            return result
        if amplitude:
            values=spread.mean(0).tolist()
            names=("mean_alpha","token_std_alpha","amplitude_weight","abs_amplitude_weight")
            result.update(dict(zip(names, values, strict=True)))
            return result
        if model.config.method == "chordres_honly_single_step":
            values = torch.cat((spread[:, :2].mean(0), spread[:-1, 2:].mean(0))).tolist()
            names = ("mu", "alpha", "beta")
        else:
            values = torch.cat((spread[:, :3].mean(0), spread[:-1, 3:].mean(0))).tolist()
            names = ("mu", "alpha_perp", "alpha_parallel", "beta", "gamma_parallel", "gamma_perp")
        result.update({f"token_std_{name}": value for name, value in zip(names, values, strict=True)})
        if model.config.method in ("chordres_honly_mhc_mix", "chordres_update_soft_force"):
            scales = [site.mix_scales.detach() for site in model.residual_maps]
            summaries = torch.stack((torch.stack([s[0] for s in scales]).mean(),
                                     torch.stack([s[1] for s in scales]).mean(),
                                     torch.stack([s[2] for s in scales[:-1]]).mean()))
            result.update(dict(zip(("mix_scale_read_mean", "mix_scale_position_mean", "mix_scale_velocity_mean"),
                                   summaries.tolist(), strict=True)))
        return result

    return measured
