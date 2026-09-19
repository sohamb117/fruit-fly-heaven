"""One observation/state/action contract across neural substrates and plasticity regimes."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace

import torch
from torch import nn

from ..adaptation.ports import StructuralPorts
from ..graphs import Graph
from ..util import seed_for
from .adapters import AnatomicalPorts, allocate_maps, count_parameters
from .config import ControllerConfig
from .dynamics import SparseRateCore
from .preparation import prepare_substrate


@dataclass
class ControllerContext:
    ports: object = None
    edge_values: torch.Tensor | None = None


class Controller(nn.Module):
    def __init__(self, obs_dim, action_dim, config: ControllerConfig, graph=None):
        super().__init__()
        config.validate()
        self.config, self.obs_dim, self.action_dim = config, obs_dim, action_dim
        self.ports = self.core = self.readout = None
        self.state_dim, self.topology = 0, None
        self.graph_fingerprint = None
        self.feature_report = self.initialization_report = None
        substrate, adapter = config.substrate, config.adapter
        started = time.monotonic()
        self.preparation_report = None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed_for(config.seed, "compatibility-controller"))
            if substrate.kind == "connectome":
                graph = Graph.load(substrate.graph) if graph is None else graph
                self.graph_fingerprint = graph.fingerprint
                prepared = prepare_substrate(graph, substrate)
                self.preparation_report = prepared.cache_report
                self.topology = prepared.topology
                t = self.topology
                features, self.feature_report = prepared.features, prepared.feature_report
                self.ports = (
                    AnatomicalPorts(graph, adapter.anatomical_mapping, adapter.channels)
                    if adapter.family == "anatomical"
                    else StructuralPorts(
                        features, adapter.channels, adapter.support, substrate.seed
                    )
                )
                magnitude, signs = prepared.magnitudes, prepared.signs
                self.initialization_report = prepared.initialization_report
                self.core = SparseRateCore(
                    t.n,
                    t.src,
                    t.dst,
                    magnitude,
                    signs,
                    substrate.dynamics,
                    plastic=substrate.plasticity in ("substrate", "joint"),
                )
                self.state_dim = t.n
            map_config = adapter
            if substrate.kind in ("rnn", "gru") and substrate.total_budget is not None:
                minimum_core = self._recurrent_cost(1)
                map_config = replace(
                    adapter, budget=min(adapter.budget, substrate.total_budget - minimum_core)
                )
            self.encoder, self.decoder, self.adapter_report = allocate_maps(
                obs_dim,
                action_dim,
                map_config,
                count_parameters(self.ports) if self.ports is not None else 0,
                config.seed,
            )
            if substrate.kind in ("rnn", "gru"):
                if substrate.hidden_size is not None:
                    hidden = substrate.hidden_size
                else:
                    available = substrate.total_budget - self.adapter_report["allocated"]
                    hidden = 1
                    while self._recurrent_cost(hidden + 1) <= available:
                        hidden += 1
                    if self._recurrent_cost(hidden) > available:
                        raise ValueError("Recurrent control cannot fit the total parameter ceiling")
                kind = nn.RNNCell if substrate.kind == "rnn" else nn.GRUCell
                self.core = kind(adapter.channels, hidden)
                self.readout = nn.Linear(hidden, adapter.channels)
                self.state_dim = hidden
        self._configure_plasticity()
        self.parameter_report = {
            "adapter": self.adapter_report,
            "allocated_parameters": count_parameters(self),
            "total_trainable_parameters": count_parameters(self, True),
            "encoder_trainable": count_parameters(self.encoder, True),
            "decoder_trainable": count_parameters(self.decoder, True),
            "port_trainable": count_parameters(self.ports, True) if self.ports is not None else 0,
            "substrate_trainable": count_parameters(self.core, True)
            if self.core is not None
            else 0,
            "readout_trainable": count_parameters(self.readout, True)
            if self.readout is not None
            else 0,
            "state_dimension": self.state_dim,
            "plasticity": substrate.plasticity,
            "topology_edges": self.topology.m if self.topology is not None else None,
            "declared_adapter_ceiling": adapter.budget,
            "declared_total_ceiling": substrate.total_budget,
        }
        if count_parameters(self, True) == 0:
            raise ValueError("This configuration has no trainable coefficients")
        self.construction_seconds = time.monotonic() - started
        if os.environ.get("CONNECTOME_PREPARATION_LOG") == "1":
            print(
                json.dumps(
                    {
                        "stage": "controller_ready",
                        "construction_seconds": self.construction_seconds,
                        "preparation": self.preparation_report,
                    }
                ),
                flush=True,
            )

    def _recurrent_cost(self, hidden):
        k = self.config.adapter.channels
        gates = 3 if self.config.substrate.kind == "gru" else 1
        return gates * hidden * (k + hidden + 2) + k * (hidden + 1)

    def _configure_plasticity(self):
        regime = self.config.substrate.plasticity
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for side in ("encoder", "decoder"):
            active = regime in ("adapters", "joint", side)
            for parameter in getattr(self, side).parameters():
                parameter.requires_grad_(active)
            if isinstance(self.ports, StructuralPorts):
                query = self.ports.input_queries if side == "encoder" else self.ports.output_queries
                query.requires_grad_(active)
        for component in (self.core, self.readout):
            if component is not None:
                for parameter in component.parameters():
                    parameter.requires_grad_(regime in ("substrate", "joint"))

    def trainable_parameters(self):
        return count_parameters(self, True)

    def reset(self, batch_size):
        return next(self.parameters()).new_zeros((batch_size, self.state_dim))

    def context(self):
        return ControllerContext(
            self.ports.compile() if self.ports is not None else None,
            self.core.edge_values() if isinstance(self.core, SparseRateCore) else None,
        )

    def forward(
        self,
        observation,
        state,
        context=None,
        *,
        reset=None,
        lesion_mask=None,
        remove_edges=False,
        squash=True,
        state_transform=None,
    ):
        if observation.ndim != 2 or observation.shape[-1] != self.obs_dim:
            raise ValueError("Observation must have shape [batch, observation_dim]")
        if state.shape != (len(observation), self.state_dim):
            raise ValueError("Wrong recurrent state shape")
        if reset is not None:
            state = state * (~reset.bool()).to(state).unsqueeze(-1)
        z = self.encoder(observation)
        if isinstance(self.core, SparseRateCore):
            context = self.context() if context is None else context
            state = self.core(
                state,
                self.ports.inject(z, context.ports),
                values=context.edge_values,
                lesion_mask=lesion_mask,
                remove_edges=remove_edges,
            )
            if state_transform is not None:
                state = state_transform(state)
            z = self.ports.read(state, context.ports)
        elif self.core is not None:
            if lesion_mask is not None or remove_edges:
                raise ValueError("Graph-edge lesions require a graph substrate")
            state = self.core(z, state)
            if state_transform is not None:
                state = state_transform(state)
            z = self.readout(state)
        logits = self.decoder(z)
        return (torch.tanh(logits) if squash else logits), state
