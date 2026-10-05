"""Cheap exact controller accounting for planning, without constructing a full connectome."""

from __future__ import annotations

from dataclasses import replace

import torch

from ..adaptation.features import FEATURE_NAMES
from .adapters import allocate_maps


def recurrent_cost(kind, channels, hidden):
    gates = 3 if kind == "gru" else 1
    return gates * hidden * (channels + hidden + 2) + channels * (hidden + 1)


def estimate_parameters(config, obs_dim, action_dim, *, neurons=None, edges=None):
    """Match Controller.parameter_report; never allocate O(N) states or O(M) weights."""
    config.validate()
    substrate, adapter = config.substrate, config.adapter
    graph = substrate.kind == "connectome"
    if graph and (neurons is None or edges is None):
        raise ValueError("Connectome accounting requires measured neuron and edge counts")
    if graph and adapter.family != "anatomical" and neurons < 2 * adapter.channels:
        raise ValueError("Generic structural ports require at least 2K neurons")
    ports = (
        2 * adapter.channels * len(FEATURE_NAMES) if graph and adapter.family != "anatomical" else 0
    )
    map_config = adapter
    if substrate.kind in ("rnn", "gru") and substrate.total_budget is not None:
        map_config = replace(
            adapter,
            budget=min(
                adapter.budget,
                substrate.total_budget - recurrent_cost(substrate.kind, adapter.channels, 1),
            ),
        )
    # Seed isolation also makes planning incapable of perturbing an in-process
    # learner. Meta tensors retain exact shapes without allocating large maps.
    with torch.random.fork_rng(), torch.device("meta"):
        _, _, report = allocate_maps(obs_dim, action_dim, map_config, ports, config.seed)
    regime = substrate.plasticity
    encoder = report["encoder_parameters"] if regime in ("adapters", "joint", "encoder") else 0
    decoder = report["decoder_parameters"] if regime in ("adapters", "joint", "decoder") else 0
    trainable_ports = (
        ports
        if regime in ("adapters", "joint")
        else ports // 2
        if regime in ("encoder", "decoder")
        else 0
    )
    core, readout, state_dim = 0, 0, neurons if graph else 0
    if graph and regime in ("substrate", "joint"):
        core = edges
    if substrate.kind in ("rnn", "gru"):
        if substrate.hidden_size is None:
            available = substrate.total_budget - report["allocated"]
            low, high = 1, 2
            while recurrent_cost(substrate.kind, adapter.channels, high) <= available:
                low, high = high, high * 2
            while high - low > 1:
                middle = (low + high) // 2
                if recurrent_cost(substrate.kind, adapter.channels, middle) <= available:
                    low = middle
                else:
                    high = middle
            state_dim = low
            if recurrent_cost(substrate.kind, adapter.channels, low) > available:
                raise ValueError("Recurrent control cannot fit the total parameter ceiling")
        else:
            state_dim = substrate.hidden_size
        readout = adapter.channels * (state_dim + 1)
        core = recurrent_cost(substrate.kind, adapter.channels, state_dim) - readout
    total = encoder + decoder + trainable_ports + core + readout
    if total == 0:
        raise ValueError("No trainable parameters in this condition")
    return {
        "adapter": report,
        "allocated_parameters": report["allocated"] + core + readout,
        "total_trainable_parameters": total,
        "encoder_trainable": encoder,
        "decoder_trainable": decoder,
        "port_trainable": trainable_ports,
        "substrate_trainable": core,
        "readout_trainable": readout,
        "state_dimension": state_dim,
        "plasticity": regime,
        "topology_edges": edges if graph else None,
        "declared_adapter_ceiling": adapter.budget,
        "declared_total_ceiling": substrate.total_budget,
    }


def effective_config_key(config, report):
    """Do not count unused capacity ceilings as distinct learned architectures."""
    value = config.to_dict()
    value["adapter"]["budget"] = report["adapter"]["allocated"]
    value["adapter"]["width"] = report["adapter"]["width"]
    value["adapter"]["rank"] = report["adapter"]["rank"]
    if not (report["adapter"]["encoder_depth"] or report["adapter"]["decoder_depth"]):
        value["adapter"]["depth"] = 1
    if config.substrate.kind in ("rnn", "gru"):
        value["substrate"]["hidden_size"] = report["state_dimension"]
        value["substrate"]["total_budget"] = None
    return value
