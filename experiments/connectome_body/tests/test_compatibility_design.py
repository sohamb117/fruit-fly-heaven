from dataclasses import replace

import pytest

from connectome_body.compatibility.config import (
    ADAPTER_FAMILIES,
    PLASTICITY_REGIMES,
    AdapterConfig,
    ControllerConfig,
    SubstrateConfig,
)
from connectome_body.compatibility.controller import Controller
from connectome_body.compatibility.design import effective_config_key, estimate_parameters
from connectome_body.graphs import make_fixture


@pytest.mark.parametrize("family", [f for f in ADAPTER_FAMILIES if f != "anatomical"])
@pytest.mark.parametrize("regime", PLASTICITY_REGIMES)
def test_planning_accounting_matches_executed_controller(tmp_path, family, regime):
    graph = make_fixture(tmp_path / "graph", 32, 2)
    config = ControllerConfig(
        AdapterConfig(family=family, channels=4, budget=1000),
        SubstrateConfig(graph=str(tmp_path / "graph"), plasticity=regime),
    )
    predicted = estimate_parameters(config, 8, 3, neurons=graph.n, edges=graph.m)
    actual = Controller(8, 3, config, graph).parameter_report
    assert predicted == actual


@pytest.mark.parametrize("kind", ["adapter_only", "rnn", "gru"])
def test_artificial_control_accounting_and_effective_capacity_deduplication(kind):
    config = ControllerConfig(
        AdapterConfig(family="linear_linear", channels=4, budget=1000),
        SubstrateConfig(
            kind=kind,
            plasticity="joint" if kind != "adapter_only" else "adapters",
            total_budget=1000 if kind != "adapter_only" else None,
        ),
    )
    predicted = estimate_parameters(config, 8, 3)
    assert predicted == Controller(8, 3, config).parameter_report
    larger = replace(config, adapter=replace(config.adapter, budget=4000))
    assert effective_config_key(config, predicted) == effective_config_key(
        larger, estimate_parameters(larger, 8, 3)
    )
