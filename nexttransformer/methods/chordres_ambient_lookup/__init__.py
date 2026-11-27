"""Six h-conditioned controls, spherical positions and ambient velocity."""
import torch
from torch import nn
from torch.nn import functional as F

from nexttransformer.methods.chordres_honly_mhc_mix import HOnlyGateSite, GATE_EPS
from nexttransformer.methods.chordres_update_soft_force import FORCE_TAU
from nexttransformer.methods.chordres_zero import ChordRes, frozen as frozen, unit_rms
from nexttransformer.models.backbone import Linear, norm


def token_reference_read(h, v, weight, mu, steps, beta, gains, scales):
    dtype = weight.dtype
    position = h.to(dtype)
    inverse_rms = (position.square().mean(-1, keepdim=True) + GATE_EPS).rsqrt()
    z = F.linear(position, weight) * inverse_rms
    m = z[..., :1] * scales[0] + mu
    raw_steps = z[..., 1:3] * scales[1] + steps
    pivot = raw_steps.detach().clamp_min(0)
    alpha = pivot + ((-pivot).exp() + (raw_steps - pivot).exp()).log()
    if beta is None:
        retention = torch.full_like(m, 0.5)
        gamma = torch.full_like(alpha, 0.5)
    else:
        retention = (z[..., 3:4] * scales[2] + beta).sigmoid()
        gamma = z[..., 4:6] * scales[2] + gains
    controls = torch.cat((m, alpha, retention, gamma), -1)
    raw = position + m * v.to(dtype)
    read = unit_rms(raw).to(h.dtype)
    return norm(read), controls


def token_reference_write(h, v, force, controls):
    state_dtype = h.dtype
    h, v, force = (x.to(controls.dtype) for x in (h, v, force))
    _, perp, par, beta, gamma_par, gamma_perp = controls.unsqueeze(-1).unbind(-2)
    delta = force * (force.square().mean(-1, keepdim=True) + FORCE_TAU**2).rsqrt()
    denom = v.square().mean(-1, keepdim=True).clamp_min(1e-12)
    along = (delta * v).mean(-1, keepdim=True) / denom
    new_h = unit_rms(h + perp * delta + ((par - perp) * along) * v)
    chord = new_h - h
    parallel = (chord * v).mean(-1, keepdim=True) / denom
    new_v = beta * v + gamma_perp * chord + ((gamma_par - gamma_perp) * parallel) * v
    return new_h.to(state_dtype), new_v.to(state_dtype)


class AmbientGateSite(HOnlyGateSite):
    def forward(self, h, v):
        args = (h, v, self.token_weight, self.mu, self.steps,
                self.beta, self.velocity_gains, self.mix_scales)
        if h.is_cuda and h.dtype == torch.bfloat16:
            from nexttransformer.methods.chordres_ambient_lookup.read_kernel import AmbientRead
            return AmbientRead.apply(*args)
        return token_reference_read(*args)


class InitialVelocityLinear(Linear):
    def reset_parameters(self):
        # Match the zero-start position and backbone RNG of the lookup and zero modes.
        nn.init.zeros_(self.weight)


class AmbientChord(ChordRes):
    initial_velocity_mode = "lookup"

    def __init__(self, config):
        super().__init__(config)
        count = 2 * config.n_layer
        self.residual_maps = nn.ModuleList(AmbientGateSite(config.width, i == count - 1) for i in range(count))
        if self.initial_velocity_mode == "lookup":
            wte = self.transformer.wte
            wte.register_parameter("velocity_weight", nn.Parameter(torch.empty_like(wte.weight)))
        elif self.initial_velocity_mode == "projected_h0":
            self.initial_velocity_projection = InitialVelocityLinear(config.width, config.width, bias=False)

    @torch.no_grad()
    def init_weights(self):
        super().init_weights()
        if self.initial_velocity_mode == "lookup":
            self.transformer.wte.velocity_weight.zero_()

    def initial_velocity(self, idx, h):
        if self.initial_velocity_mode == "lookup":
            return F.embedding(idx, self.transformer.wte.velocity_weight).to(h.dtype)
        if self.initial_velocity_mode == "projected_h0":
            return self.initial_velocity_projection(h)
        return torch.zeros_like(h)

    def _ordinary_hidden(self, idx):
        x = self._embed_reference(idx)
        h = unit_rms(x).to(x.dtype)
        v = self.initial_velocity(idx, h)
        write = token_reference_write
        if h.is_cuda and h.dtype == torch.bfloat16:
            from nexttransformer.methods.chordres_ambient_lookup.write_kernel import AmbientWrite

            write = AmbientWrite.apply
        cos_sin = self.cos[:, : idx.size(1)], self.sin[:, : idx.size(1)]
        for i, layer in enumerate(self.transformer.h):
            ve = (
                self.value_embeds[str(i)](idx).to(x.dtype)
                if str(i) in self.value_embeds
                else None
            )
            window = self._window(i, self.config.n_layer)
            query, controls = self.residual_maps[2 * i](h, v)
            force = layer.attn(query, ve, cos_sin, window, None)
            self._record(
                "block",
                physical=i,
                virtual=i,
                window=window[0],
                tokens=idx.numel(),
                ve=ve is not None,
                sequence_len=idx.size(1),
                mixer="softmax",
                residual=self.config.method,
            )
            h, v = write(h, v, force, controls)
            query, controls = self.residual_maps[2 * i + 1](h, v)
            force = layer.mlp(query)
            h, v = write(h, v, force, controls)
        return h, None


model = AmbientChord
