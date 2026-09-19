"""Serializable factors for the ten-experiment compatibility study."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

ADAPTER_FAMILIES = (
    "linear_linear",
    "mlp_linear",
    "linear_mlp",
    "mlp_mlp",
    "low_rank",
    "sparse_structured",
    "anatomical",
)
PLASTICITY_REGIMES = ("adapters", "substrate", "encoder", "decoder", "joint")
TOPOLOGIES = ("real", "community_rewired", "degree_rewired", "direction_shuffled", "random")
INITIALIZATIONS = ("biological", "random_magnitudes", "shuffled_magnitudes", "random_signed")


def _positive_integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class AdapterConfig:
    family: str = "mlp_mlp"
    budget: int = 5000
    channels: int = 16
    support: int = 256
    width: int | None = None
    depth: int = 1
    rank: int | None = None
    density: float = 0.25
    anatomical_mapping: str | None = None

    def validate(self):
        if self.family not in ADAPTER_FAMILIES:
            raise ValueError("Unknown stateless adapter family")
        for name in ("budget", "channels", "support", "depth"):
            _positive_integer(getattr(self, name), name)
        for name in ("width", "rank"):
            if getattr(self, name) is not None:
                _positive_integer(getattr(self, name), name)
        if not math.isfinite(self.density) or not 0 < self.density <= 1:
            raise ValueError("Sparse adapter density must be in (0, 1]")
        if (self.family == "anatomical") != bool(self.anatomical_mapping):
            raise ValueError("Only the anatomical oracle uses an explicit anatomical_mapping")
        if self.family != "low_rank" and self.rank is not None:
            raise ValueError("rank is a factor only for low_rank adapters")
        if self.family in ("linear_linear", "low_rank") and self.width is not None:
            raise ValueError("Linear and low-rank adapters do not have a hidden width")


@dataclass(frozen=True)
class DynamicsConfig:
    gain: float = 0.9
    tau_seconds: float = 0.002
    control_dt: float = 0.0002
    substeps: int = 2
    input_gain: float = 1.0
    sign_mode: str = "random_dale"
    inhibitory_fraction: float = 0.2
    weight_transform: str = "log1p"
    gradient_edge_chunk: int = 65536

    def validate(self):
        for name in ("tau_seconds", "control_dt", "input_gain"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.gain) or self.gain < 0:
            raise ValueError("gain must be nonnegative and finite")
        for name in ("substeps", "gradient_edge_chunk"):
            _positive_integer(getattr(self, name), name)
        if self.sign_mode not in ("random_dale", "annotated", "available"):
            raise ValueError("Unknown sign convention")
        if not 0 <= self.inhibitory_fraction <= 1:
            raise ValueError("Invalid inhibitory fraction")
        if self.weight_transform not in ("log1p", "count"):
            raise ValueError("Unknown synapse-count transform")


@dataclass(frozen=True)
class SubstrateConfig:
    kind: str = "connectome"
    graph: str | None = None
    topology: str = "real"
    initialization: str = "biological"
    plasticity: str = "adapters"
    seed: int = 0
    community_partition: str | None = None
    swap_attempts_per_edge: int = 10
    preserve_strengths: bool = False
    hidden_size: int | None = None
    total_budget: int | None = None
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)

    def validate(self):
        if self.kind not in ("connectome", "adapter_only", "rnn", "gru"):
            raise ValueError("Unknown neural substrate kind")
        if self.topology not in TOPOLOGIES or self.initialization not in INITIALIZATIONS:
            raise ValueError("Unknown topology or weight initialization")
        if self.plasticity not in PLASTICITY_REGIMES:
            raise ValueError("Unknown plasticity regime")
        self.dynamics.validate()
        _positive_integer(self.seed, "substrate seed", minimum=0)
        _positive_integer(self.swap_attempts_per_edge, "swap attempts")
        if type(self.preserve_strengths) is not bool:
            raise ValueError("preserve_strengths must be boolean")
        if self.kind == "connectome":
            if not self.graph:
                raise ValueError("Connectome conditions require a canonical graph; no fallback")
            if self.total_budget is not None or self.hidden_size is not None:
                raise ValueError("Graph state size is determined by the connectome")
        else:
            if self.graph or self.topology != "real" or self.initialization != "biological":
                raise ValueError("Brain-free/dense controls cannot silently ignore graph factors")
            if self.community_partition or self.preserve_strengths:
                raise ValueError("Graph null factors require a connectome")
        if self.topology == "community_rewired" and not self.community_partition:
            raise ValueError("Community rewiring requires a pinned partition")
        if self.topology != "community_rewired" and self.community_partition:
            raise ValueError("A community partition is only used by community_rewired")
        if self.kind == "adapter_only" and self.plasticity == "substrate":
            raise ValueError("An adapter-only controller has no substrate to train")
        if self.kind == "adapter_only" and (
            self.hidden_size is not None or self.total_budget is not None
        ):
            raise ValueError("Adapter-only capacity is specified by the adapter budget")
        if self.kind in ("rnn", "gru"):
            if self.plasticity != "joint":
                raise ValueError("Conventional learned RNN/GRU controls use joint training")
            if (self.total_budget is None) == (self.hidden_size is None):
                raise ValueError("Choose a total parameter ceiling or an explicit recurrent width")
        for name in ("hidden_size", "total_budget"):
            if getattr(self, name) is not None:
                _positive_integer(getattr(self, name), name)


@dataclass(frozen=True)
class ControllerConfig:
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    substrate: SubstrateConfig = field(default_factory=SubstrateConfig)
    seed: int = 0
    normalize_observations: bool = False

    def validate(self):
        self.adapter.validate()
        self.substrate.validate()
        _positive_integer(self.seed, "controller seed", minimum=0)
        if type(self.normalize_observations) is not bool:
            raise ValueError("normalize_observations must be boolean")
        if self.substrate.kind != "connectome" and self.adapter.family == "anatomical":
            raise ValueError("Anatomical mappings require an actual connectome")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["adapter"] = AdapterConfig(**value.get("adapter", {}))
        substrate = dict(value.get("substrate", {}))
        substrate["dynamics"] = DynamicsConfig(**substrate.get("dynamics", {}))
        value["substrate"] = SubstrateConfig(**substrate)
        result = cls(**value)
        result.validate()
        return result
