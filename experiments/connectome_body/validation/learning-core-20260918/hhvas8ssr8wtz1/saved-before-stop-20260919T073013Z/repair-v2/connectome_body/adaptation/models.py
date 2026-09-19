"""Stateless E/D; persistent learned-policy state resides only in the substrate."""

from __future__ import annotations

import torch
from torch import nn

from ..policy import parameter_count
from .ports import StructuralPorts
from .substrates import RateConfig, RateSubstrate

INTERVENTIONS = ("none", "lesion", "reset", "state_shuffle", "port_disruption", "silence")


class GenericAdapter(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        budget=20000,
        channels=16,
        support=256,
        graph=None,
        variant="real",
        rate=RateConfig(),
        port_seed=0,
        backend="cpu",
    ):
        super().__init__()
        if variant not in (
            "real",
            "degree_shuffled",
            "matched_random",
            "no_edges",
            "adapter_only",
            "trainable_gru",
        ):
            raise ValueError("Unknown adaptation substrate")
        self.variant, self.budget = variant, budget
        self.obs_dim, self.action_dim, self.channels = obs_dim, action_dim, channels
        self.substrate = self.ports = self.gru = None
        if variant == "trainable_gru":
            h = 1

            def cost(width):
                return 3 * width * (obs_dim + width + 2) + action_dim * (width + 1)

            while cost(h + 1) <= budget:
                h += 1
            if cost(h) > budget:
                raise ValueError("Budget too small for a conventional GRU")
            self.encoder = nn.Identity()
            self.gru = nn.GRUCell(obs_dim, h)
            self.decoder = nn.Linear(h, action_dim)
            self.state_dim, self.width = h, h
        else:
            port_count = 0
            if variant != "adapter_only":
                if graph is None:
                    raise ValueError("A real graph is required; no fallback")
                self.substrate = RateSubstrate(graph, variant, rate, port_seed, backend)
                self.ports = StructuralPorts(self.substrate.features, channels, support, port_seed)
                port_count = parameter_count(self.ports)
            width = (budget - port_count - channels - action_dim) // (
                obs_dim + 2 * channels + action_dim + 2
            )
            if width < 1:
                raise ValueError("Budget too small for E/D plus structural queries")
            self.width = width
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, width), nn.Tanh(), nn.Linear(width, channels)
            )
            self.decoder = nn.Sequential(
                nn.Linear(channels, width), nn.Tanh(), nn.Linear(width, action_dim)
            )
            self.state_dim = graph.n if self.substrate is not None else 0
        # Moderate decoder initialization preserves measurable input gradients.
        for module in (self.encoder, self.decoder):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        self.num_parameters = parameter_count(self)
        if self.num_parameters > budget:
            raise AssertionError("Adapter exceeded its declared parameter ceiling")
        self.parameter_report = {
            "budget": budget,
            "actual": self.num_parameters,
            "hidden_width": self.width,
            "port_queries": parameter_count(self.ports) if self.ports is not None else 0,
            "encoder": parameter_count(self.encoder),
            "decoder": parameter_count(self.decoder),
            "trainable_recurrence": parameter_count(self.gru) if self.gru is not None else 0,
            "learned_connectome_weights": 0,
            "state_dimension": self.state_dim,
        }

    def trainable_parameters(self):
        return parameter_count(self)

    def reset(self, batch_size):
        return next(self.parameters()).new_zeros((batch_size, self.state_dim))

    def context(self):
        return self.ports.compile() if self.ports is not None else None

    def forward(self, observation, state, context=None, intervention="none", reset=None):
        if intervention not in INTERVENTIONS:
            raise ValueError("Unknown acute intervention")
        if reset is not None:
            state = state * (~reset.bool()).to(state.dtype).unsqueeze(-1)
        if intervention == "reset":
            state = torch.zeros_like(state)
        z = self.encoder(observation)
        if self.substrate is not None:
            context = self.context() if context is None else context
            if intervention == "state_shuffle":
                # Only output activity is permuted; input populations stay disjoint.
                state = state[:, self.ports.output_permutation]
            drive = self.ports.inject(z, context)
            state = self.substrate.step(state, drive, lesion=intervention == "lesion")
            z = self.ports.read(state, context, disrupted=intervention == "port_disruption")
        elif self.gru is not None:
            if intervention == "lesion":
                state = torch.zeros_like(state)
            elif intervention == "state_shuffle":
                state = torch.flip(state, [-1])
            state = self.gru(z, state)
            z = state
        if intervention == "silence":
            z = torch.zeros_like(z)
        return torch.tanh(self.decoder(z)), state
