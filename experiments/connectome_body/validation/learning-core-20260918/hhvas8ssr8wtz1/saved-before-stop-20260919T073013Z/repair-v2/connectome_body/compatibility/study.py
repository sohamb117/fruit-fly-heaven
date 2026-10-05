"""Compile the complete paper into deduplicated training and evaluation jobs.

Compilation is read-only with respect to data and bodies. Missing biological
inputs remain explicit prerequisites; no smaller or synthetic graph is used as
a replacement. Planning never launches a learner or provisions compute.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
from collections import Counter
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..graphs import Graph
from ..util import atomic_json, digest_file, digest_json
from .adapters import AnatomicalPorts
from .bodies import BODY_TASKS, BodySpec, make_body
from .config import (
    ADAPTER_FAMILIES,
    INITIALIZATIONS,
    PLASTICITY_REGIMES,
    TOPOLOGIES,
    AdapterConfig,
    ControllerConfig,
    DynamicsConfig,
    SubstrateConfig,
)
from .design import effective_config_key, estimate_parameters
from .predictors import body_features
from .qualification import body_interface_identity
from .run_config import LearningConfig, RunSpec
from .training import source_identity

EXPERIMENTS = {
    "1": "Encoder-decoder architecture and performance-capacity Pareto frontier",
    "2": "Architecture x connectome x body/task; minimum width, depth, rank and capacity",
    "3": "Biology versus parameter- and compute-matched recurrence and no brain",
    "4": "Original, community, degree, direction-shuffled and N/M-random wiring",
    "5": "Initialization x permitted-edge plasticity, with rewired controls",
    "6": "Crossed connectome-body effects and matched-pair interaction",
    "7": "Connectome x task and task-relevant versus equal-size subgraphs",
    "8": "Post-training causal substrate interventions",
    "9": "Distribution shifts, zero-shot task transfer and fine-tuning",
    "10": "Nested structural prediction with whole pairs held out",
}


def load_study(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value.get("schema") != "connectome-compatibility-study-v1":
        raise ValueError("Unsupported paper study schema")
    if value.get("purpose") not in ("experiment", "smoke"):
        raise ValueError("Declare experiment or explicitly synthetic smoke purpose")
    required = {
        "graphs",
        "bodies",
        "adapter",
        "dynamics",
        "training",
        "seeds",
        "capacities",
        "experiments",
    }
    if not required <= value.keys() or set(value["experiments"]) != set(EXPERIMENTS):
        raise ValueError("A study must declare all ten experiments and their common factors")
    if not value["graphs"] or not value["bodies"]:
        raise ValueError("A study needs explicit graph and body catalogs")
    for field in ("seeds", "capacities"):
        items = value[field]
        if (
            not items
            or len(set(items)) != len(items)
            or any(type(x) is not int or x < (0 if field == "seeds" else 1) for x in items)
        ):
            raise ValueError(f"Invalid {field} grid")
    for key, item in value["bodies"].items():
        spec = BodySpec(**item["spec"])
        spec.validate()
        if (
            key != f"{spec.name}:{spec.task}"
            or not math.isfinite(item["control_dt"])
            or item["control_dt"] <= 0
        ):
            raise ValueError("Body catalog keys and physical decision intervals must be explicit")
    LearningConfig(**value["training"]).validate()
    if value["purpose"] == "experiment":
        declared = set(value["bodies"])
        complete = {f"{body}:{task}" for body, tasks in BODY_TASKS.items() for task in tasks}
        if declared != complete or {x["native_body"] for x in value["graphs"].values()} != set(
            BODY_TASKS
        ):
            raise ValueError("The primary paper must retain the full three-body, ten-task matrix")
    root = (path.parent / value.get("project_root", "../..")).resolve()
    return value, root


@lru_cache(maxsize=8192)
def _estimate(serialized, obs, actions, neurons, edges):
    return estimate_parameters(
        ControllerConfig.from_dict(json.loads(serialized)),
        obs,
        actions,
        neurons=neurons,
        edges=edges,
    )


class StudyCompiler:
    def __init__(self, path, output):
        self.path, self.output = Path(path).resolve(), Path(output).resolve()
        self.study, self.root = load_study(path)
        self.conditions, self.jobs, self.proposals = {}, {}, []
        self.graphs, self.bodies = {}, {}
        self.loaded_graphs, self.checked_oracles = {}, {}
        self.active_neurons = {}
        self.learning = LearningConfig(**self.study["training"])
        self.common_adapter = AdapterConfig(**self.study["adapter"])
        self.common_dynamics = DynamicsConfig(**self.study["dynamics"])
        self.code = source_identity()
        self.primary = []
        self._inspect_inputs()

    def resolve(self, path):
        return str((self.root / path).resolve()) if path is not None else None

    def _inspect_graph(self, label, declaration):
        path = Path(self.resolve(declaration["path"]))
        report = {
            "path": str(path),
            "native_body": declaration.get("native_body"),
            "declaration": declaration,
            "issues": [],
        }
        try:
            graph = Graph.load(path)
            if declaration.get("fingerprint") and graph.fingerprint != declaration["fingerprint"]:
                raise ValueError("Prepared graph differs from the registered release fingerprint")
            if self.study["purpose"] == "experiment" and graph.manifest["provenance"].get(
                "is_synthetic"
            ):
                raise ValueError("A synthetic graph cannot stand in for a biological substrate")
            for key in ("parent_graph", "selection_role", "task"):
                if (
                    declaration.get(key) is not None
                    and graph.manifest["provenance"].get(key) != declaration[key]
                ):
                    raise ValueError(f"Subgraph {key} differs from its preregistered selection")
            self.loaded_graphs[label] = graph
            active = np.zeros(graph.n, dtype=bool)
            active[graph.src] = active[graph.dst] = True
            self.active_neurons[label] = np.flatnonzero(active)
            report.update(
                fingerprint=graph.fingerprint,
                summary=graph.manifest["summary"],
                provenance=graph.manifest["provenance"],
            )
            report["descriptor_inputs"] = {
                kind: {
                    "path": self.resolve(declaration[kind]) if declaration.get(kind) else None,
                    "sha256": digest_file(self.resolve(declaration[kind]))
                    if declaration.get(kind) and Path(self.resolve(declaration[kind])).is_file()
                    else None,
                }
                for kind in ("communities", "anatomy")
            }
            if declaration.get("qualification"):
                review_path = Path(self.resolve(declaration["qualification"]))
                review = json.loads(review_path.read_text())
                if (
                    review.get("schema") != "connectome-use-qualification-v1"
                    or review.get("graph_fingerprint") != graph.fingerprint
                    or type(review.get("eligible_for_primary")) is not bool
                    or not review.get("scope")
                    or not review.get("reason")
                    or review.get("fingerprint")
                    != digest_json({k: v for k, v in review.items() if k != "fingerprint"})
                ):
                    raise ValueError(
                        "Graph use qualification is malformed or belongs to another graph"
                    )
                report["qualification"] = {
                    "path": str(review_path),
                    "sha256": digest_file(review_path),
                    "record": review,
                }
                if not review["eligible_for_primary"]:
                    report["issues"].append(
                        {
                            "input": f"graph_qualification:{label}",
                            "path": str(review_path),
                            "reason": review["reason"],
                        }
                    )
        except (OSError, ValueError, KeyError) as exc:
            report["issues"].append(
                {"input": f"graph:{label}", "path": str(path), "reason": str(exc)}
            )
        self.graphs[label] = report
        return report

    def _inspect_inputs(self):
        for label, item in self.study["graphs"].items():
            self._inspect_graph(label, item)
        for key, item in self.study["bodies"].items():
            data = copy.deepcopy(item["spec"])
            if data.get("model_manifest"):
                data["model_manifest"] = self.resolve(data["model_manifest"])
            spec = BodySpec(**data)
            report = {"spec": spec, "control_dt": item["control_dt"], "issues": []}
            body = None
            try:
                body = make_body(spec)
                if not math.isclose(body.control_dt, item["control_dt"]):
                    raise ValueError(
                        "Executed body decision interval differs from the registered value"
                    )
                if body.evidence == "software_fixture_only" and self.study["purpose"] != "smoke":
                    raise ValueError("A software fixture is not a primary biological body")
                if self.study["purpose"] == "experiment" and body.evidence in (
                    "published_worm_mechanics",
                    "simzfish_derived_3d_extension",
                ):
                    qualification_path = self.resolve(item.get("qualification"))
                    if not qualification_path or not Path(qualification_path).is_file():
                        raise ValueError(
                            "Record the native body mechanics qualification before training"
                        )
                    qualification = json.loads(Path(qualification_path).read_text())
                    if (
                        qualification.get("passed") is not True
                        or qualification.get("body") != spec.name
                        or qualification.get("model_manifest_sha256")
                        != digest_file(spec.model_manifest)
                        or qualification.get("body_interface_identity")
                        != body_interface_identity(spec.name)
                        or qualification.get("fingerprint")
                        != digest_json(
                            {k: v for k, v in qualification.items() if k != "fingerprint"}
                        )
                    ):
                        raise ValueError("Body mechanics qualification is failed, stale or changed")
                    report["qualification"] = {
                        "path": qualification_path,
                        "sha256": digest_file(qualification_path),
                        "scope": qualification["scope"],
                    }
                report.update(
                    observation_dim=body.obs_dim,
                    action_dim=body.action_dim,
                    fingerprint=body.fingerprint,
                    evidence=body.evidence,
                    action_names=body.action_names,
                    features=body_features(body),
                )
            except (OSError, ValueError, RuntimeError, ImportError) as exc:
                report["issues"].append(
                    {"input": f"body:{key}", "path": spec.model_manifest, "reason": str(exc)}
                )
            finally:
                if body is not None:
                    body.close()
            self.bodies[key] = report

    def _condition(
        self,
        experiment,
        connectome,
        body_key,
        budget,
        seed,
        *,
        family=None,
        topology="real",
        initialization="biological",
        plasticity="adapters",
        kind="connectome",
        channels=None,
        width=None,
        depth=None,
        rank=None,
        sign_mode=None,
        total_budget=None,
        hidden_size=None,
        matching=None,
        extra_issues=(),
        tags=(),
        learning=None,
        subgraph_role=None,
    ):
        body = self.bodies[body_key]
        graph = self.graphs.get(connectome) if kind == "connectome" else None
        issues = copy.deepcopy(
            body["issues"] + (graph["issues"] if graph else []) + list(extra_issues)
        )
        family = family or self.common_adapter.family
        k = channels or self.common_adapter.channels
        mapping, partition = None, None
        if family == "anatomical":
            mapping = self.resolve(
                (graph or {})
                .get("declaration", {})
                .get("anatomical_mappings", {})
                .get(
                    f"{body_key}:{k}",
                    (graph or {})
                    .get("declaration", {})
                    .get("anatomical_mappings", {})
                    .get(str(k), f"data/paper-annotations/{connectome}/ports-k{k}.json"),
                )
            )
            if kind != "connectome":
                issues.append(
                    {
                        "input": "anatomical_artificial_control",
                        "not_applicable": True,
                        "reason": "Anatomical oracle ports require biological neuron identities",
                    }
                )
            elif not Path(mapping).is_file():
                issues.append(
                    {
                        "input": "anatomical_mapping",
                        "path": mapping,
                        "reason": "Provide provenance-backed, graph-pinned disjoint sensory/motor populations",
                    }
                )
            elif connectome in self.loaded_graphs:
                oracle_key = (connectome, mapping, k, digest_file(mapping))
                if oracle_key not in self.checked_oracles:
                    try:
                        AnatomicalPorts(self.loaded_graphs[connectome], mapping, k)
                        self.checked_oracles[oracle_key] = None
                    except (ValueError, KeyError) as exc:
                        self.checked_oracles[oracle_key] = str(exc)
                if self.checked_oracles[oracle_key]:
                    issues.append(
                        {
                            "input": "anatomical_mapping",
                            "path": mapping,
                            "reason": self.checked_oracles[oracle_key],
                        }
                    )
        if topology == "community_rewired":
            partition = self.resolve(
                graph["declaration"].get(
                    "communities", f"data/paper-annotations/{connectome}/communities.json"
                )
            )
            if not Path(partition).is_file():
                issues.append(
                    {
                        "input": "community_partition",
                        "path": partition,
                        "reason": "Prepare and pin a nondegenerate community partition",
                    }
                )
            else:
                record = json.loads(Path(partition).read_text())
                labels = np.asarray(record.get("labels", []))
                active = self.active_neurons.get(connectome, [])
                if (
                    record.get("graph_fingerprint") != graph.get("fingerprint")
                    or labels.shape != (graph.get("summary", {}).get("neurons"),)
                    or labels.dtype.kind not in "iu"
                    or len(np.unique(labels[active])) < 2
                ):
                    issues.append(
                        {
                            "input": "community_partition",
                            "path": partition,
                            "reason": "Partition fingerprint mismatch or fewer than two active communities",
                        }
                    )
        adapter = replace(
            self.common_adapter,
            family=family,
            budget=budget,
            channels=k,
            width=width,
            depth=depth or self.common_adapter.depth,
            rank=rank,
            anatomical_mapping=mapping,
        )
        dynamics = replace(
            self.common_dynamics,
            control_dt=body["control_dt"],
            sign_mode=sign_mode or self.common_dynamics.sign_mode,
        )
        substrate = SubstrateConfig(
            kind=kind,
            graph=graph["path"] if graph else None,
            topology=topology,
            initialization=initialization,
            plasticity=plasticity,
            seed=seed,
            community_partition=partition,
            total_budget=total_budget,
            hidden_size=hidden_size,
            dynamics=dynamics,
            swap_attempts_per_edge=self.study.get("swap_attempts_per_edge", 10),
            preserve_strengths=self.study.get("preserve_null_strengths", True)
            if kind == "connectome" and topology in ("degree_rewired", "community_rewired")
            else False,
        )
        controller = ControllerConfig(adapter, substrate, seed)
        config = RunSpec(
            body["spec"],
            controller,
            learning or self.learning,
            train_seed=seed,
            device=self.study.get("device", "cpu"),
            threads=self.study.get("threads", 1),
            purpose=self.study["purpose"],
            metadata={
                "connectome": connectome if graph else kind,
                "native_body": graph.get("native_body") if graph else None,
                "subgraph_role": subgraph_role,
                "expected_body_fingerprint": body.get("fingerprint"),
                "expected_graph_fingerprint": graph.get("fingerprint") if graph else None,
                "expected_anatomy_sha256": digest_file(mapping)
                if mapping and Path(mapping).is_file()
                else None,
                "expected_community_sha256": digest_file(partition)
                if partition and Path(partition).is_file()
                else None,
            },
        )
        parameters = None
        status = "blocked_prerequisite" if issues else "ready"
        if any(item.get("not_applicable") for item in issues):
            status = "not_applicable"
        if not issues:
            try:
                config.validate()
                summary = graph["summary"] if graph else {}
                parameters = _estimate(
                    json.dumps(controller.to_dict(), sort_keys=True),
                    body["observation_dim"],
                    body["action_dim"],
                    summary.get("neurons"),
                    summary.get("edges"),
                )
            except (ValueError, TypeError) as exc:
                status = "infeasible_capacity"
                issues.append({"input": "controller_configuration", "reason": str(exc)})
        key = config.to_dict()
        key.pop("metadata")
        key["body"] = body.get("fingerprint", asdict(body["spec"]))
        if parameters:
            key["controller"] = effective_config_key(controller, parameters)
        if graph:
            key["controller"]["substrate"]["graph"] = graph.get("fingerprint", graph["path"])
        identity = digest_json(key)[:24]
        if identity not in self.conditions:
            self.conditions[identity] = {
                "id": identity,
                "status": status,
                "config": config.to_dict(),
                "parameters": parameters,
                "issues": issues,
                "experiments": [],
                "requested_adapter_budgets": [],
                "tags": [],
                "matches": [],
            }
        record = self.conditions[identity]
        for field, values in (
            ("experiments", [str(experiment)]),
            ("requested_adapter_budgets", [budget]),
            ("tags", tags),
        ):
            record[field] = sorted(set(record[field]) | set(values))
        if matching:
            existing = next(
                (
                    entry
                    for entry in record["matches"]
                    if (entry["reference"], entry["dimension"])
                    == (matching["reference"], matching["dimension"])
                ),
                None,
            )
            if existing is None:
                record["matches"].append(matching)
            else:
                existing.update(matching)
        self.proposals.append(
            {"experiment": str(experiment), "condition": identity, "requested_budget": budget}
        )
        return identity

    def _dense_control(self, experiment, reference, kind, *, compute=False):
        parent = self.conditions[reference]
        source = RunSpec.from_dict(parent["config"])
        issues, target, hidden = [], None, None
        adapter_budget = source.controller.adapter.budget
        match = {
            "reference": reference,
            "dimension": "training_step_seconds" if compute else "total_trainable_parameters",
        }
        if compute:
            calibration = self.study["experiments"]["3"].get("compute_calibration")
            matches = {}
            if calibration and Path(self.resolve(calibration)).is_file():
                record = json.loads(Path(self.resolve(calibration)).read_text())
                if (
                    record.get("schema") == "compatibility-compute-matches-v1"
                    and record.get("code_fingerprint") == self.code
                ):
                    matches = record["matches"]
            selected = matches.get(reference)
            if (
                selected
                and selected.get("status") == "matched"
                and selected.get("device") == source.device
                and selected["reference"]["controller"] == source.controller.to_dict()
                and selected["reference"]["learning"] == asdict(source.training)
            ):
                hidden = selected["hidden_size"]
                adapter_budget = selected["adapter_budget"]
                match["calibration"] = selected
            else:
                issues.append(
                    {
                        "input": "compute_match",
                        "reference": reference,
                        "reason": "Measure an RNN with the same hardware, precision, batch and BPTT length; FLOP guesses are not compute matching",
                    }
                )
                hidden = 1
        elif parent["parameters"] is None:
            issues.append(
                {
                    "input": "parameter_match",
                    "reference": reference,
                    "reason": "Resolve the biological controller dimensions and trainable count first",
                }
            )
            target = 1
        else:
            target = parent["parameters"]["total_trainable_parameters"]
            # A common rule reserves at least half the trainable ceiling for
            # recurrence. Linear maps naturally leave more. The same family and
            # bottleneck remain in place; actual allocation error is reported.
            body = self.bodies[f"{source.body.name}:{source.body.task}"]
            minimal_adapter = replace(
                source.controller.adapter,
                width=1
                if source.controller.adapter.family not in ("linear_linear", "low_rank")
                else None,
                rank=1 if source.controller.adapter.family == "low_rank" else None,
            )
            minimal_config = ControllerConfig(minimal_adapter, SubstrateConfig(kind="adapter_only"))
            try:
                minimum = estimate_parameters(
                    minimal_config, body["observation_dim"], body["action_dim"]
                )["adapter"]["allocated"]
            except ValueError:
                minimum = adapter_budget
            adapter_budget = min(adapter_budget, max(minimum, target // 2))
            match["target"] = target
        body_key = f"{source.body.name}:{source.body.task}"
        identity = self._condition(
            experiment,
            kind,
            body_key,
            adapter_budget,
            source.train_seed,
            kind=kind,
            family=source.controller.adapter.family,
            plasticity="joint",
            channels=source.controller.adapter.channels,
            total_budget=target,
            hidden_size=hidden,
            matching=match,
            extra_issues=issues,
            tags=("compute_matched" if compute else "parameter_matched",),
        )
        item = self.conditions[identity]
        if item["parameters"] and not compute:
            match["actual"] = item["parameters"]["total_trainable_parameters"]
            match["relative_error"] = (target - match["actual"]) / target
            for entry in item["matches"]:
                if (entry["reference"], entry["dimension"]) == (
                    match["reference"],
                    match["dimension"],
                ):
                    entry.update(match)
        return identity

    def _job(self, experiment, kind, reference, parameters, *, dependencies=()):
        value = {
            "kind": kind,
            "reference": reference,
            "parameters": parameters,
            "dependencies": sorted(set(dependencies) | ({reference} if reference else set())),
        }
        identity = digest_json(value)[:24]
        if identity not in self.jobs:
            inputs = {
                path: digest_file(path) if Path(path).is_file() else None
                for key, path in parameters.items()
                if key == "lesion_file"
            }
            if kind == "predict_pairs":
                inputs.update(
                    {
                        item["path"]: item["sha256"]
                        for graph in self.graphs.values()
                        for item in graph.get("descriptor_inputs", {}).values()
                        if item["path"] and item["sha256"]
                    }
                )
            self.jobs[identity] = {
                "id": identity,
                **value,
                "experiments": [],
                "input_files": inputs,
            }
        self.jobs[identity]["experiments"] = sorted(
            set(self.jobs[identity]["experiments"]) | {str(experiment)}
        )
        return identity

    def compile(self):
        graphs, bodies = list(self.study["graphs"]), list(self.bodies)
        capacities, seeds, experiments = (
            self.study["capacities"],
            self.study["seeds"],
            self.study["experiments"],
        )
        # Experiments 1 and 6 share the complete crossed architecture matrix.
        for c, b, p, seed, family, regime in itertools.product(
            graphs,
            bodies,
            capacities,
            seeds,
            ADAPTER_FAMILIES,
            experiments["1"].get("plasticity", ["adapters", "joint"]),
        ):
            identity = self._condition("1", c, b, p, seed, family=family, plasticity=regime)
            self.conditions[identity]["experiments"].append("6")
            if family == self.common_adapter.family:
                self.primary.append(identity)
            if family != "anatomical":
                self._condition(
                    "1",
                    c,
                    b,
                    p,
                    seed,
                    family=family,
                    topology="random",
                    plasticity=regime,
                    matching={
                        "reference": identity,
                        "dimension": "neurons_edges_sparsity_and_gain",
                    },
                )
                self._condition(
                    "1",
                    "adapter_only",
                    b,
                    p,
                    seed,
                    family=family,
                    kind="adapter_only",
                    matching={"reference": identity, "dimension": "adapter_budget_and_bottleneck"},
                )
                for kind in ("rnn", "gru"):
                    self._dense_control("1", identity, kind)
        # Experiment 2 varies one interface axis at a time in addition to P.
        for c, b, seed in itertools.product(graphs, bodies, seeds):
            for axis in experiments["2"]["sweeps"]:
                for value in axis["values"]:
                    options = {
                        axis["factor"]: value,
                        "family": axis["family"],
                        **axis.get("fixed", {}),
                    }
                    self._condition("2", c, b, experiments["2"]["ceiling"], seed, **options)
        # Mandatory artificial controls, including an independently measured
        # compute match. Multiple biological cells may share one baseline.
        for c, b, p, seed, regime in itertools.product(
            graphs,
            bodies,
            capacities,
            seeds,
            experiments["3"].get("plasticity", ["adapters", "joint"]),
        ):
            reference = self._condition("3", c, b, p, seed, plasticity=regime)
            for kind in ("rnn", "gru"):
                self._dense_control("3", reference, kind)
            self._dense_control("3", reference, "rnn", compute=True)
            self._condition(
                "3",
                c,
                b,
                p,
                seed,
                topology="random",
                plasticity=regime,
                matching={"reference": reference, "dimension": "neurons_edges_sparsity_and_gain"},
            )
            self._condition(
                "3",
                "adapter_only",
                b,
                p,
                seed,
                kind="adapter_only",
                matching={"reference": reference, "dimension": "adapter_budget_and_bottleneck"},
            )
            for topology in TOPOLOGIES:
                self._condition("4", c, b, p, seed, topology=topology, plasticity=regime)
        for c, b, p, seed, initialization, plasticity, topology in itertools.product(
            graphs,
            bodies,
            experiments["5"]["capacities"],
            seeds,
            INITIALIZATIONS,
            PLASTICITY_REGIMES,
            experiments["5"].get("topologies", ["real", "degree_rewired"]),
        ):
            self._condition(
                "5",
                c,
                b,
                p,
                seed,
                initialization=initialization,
                plasticity=plasticity,
                topology=topology,
                sign_mode="available",
                tags=("measured_signs_where_available",),
            )
        # Task specificity includes the whole-connectome matrix and explicit
        # relevant/equal-size comparison graph artifacts, never name heuristics.
        for c, b in itertools.product(graphs, bodies):
            roles = ["relevant", "random_selection"]
            irrelevant = self.study["graphs"][c].get("subgraphs", {}).get(b, {}).get("irrelevant")
            if irrelevant:
                roles.append("irrelevant")
            for role in roles:
                label = f"{c}:{b}:{role}"
                declared = self.study["graphs"][c].get("subgraphs", {}).get(b, {}).get(role)
                item = {
                    "path": declared or f"data/paper-subgraphs/{c}/{b.replace(':', '-')}/{role}",
                    "native_body": self.study["graphs"][c]["native_body"],
                    "parent_graph": self.graphs[c].get("fingerprint"),
                    "task": b,
                    "selection_role": role,
                }
                self._inspect_graph(label, item)
            relevant_label = f"{c}:{b}:relevant"
            for role in roles[1:]:
                label = f"{c}:{b}:{role}"
                if label in self.loaded_graphs and relevant_label in self.loaded_graphs:
                    a, other = self.loaded_graphs[relevant_label], self.loaded_graphs[label]
                    if a.n != other.n or np.intersect1d(a.node_ids, other.node_ids).size:
                        self.graphs[label]["issues"].append(
                            {
                                "input": "equal_size_subgraph",
                                "reason": "Controls must match neuron count and use disjoint selected populations",
                            }
                        )
            for p, seed in itertools.product(experiments["7"]["capacities"], seeds):
                whole = self._condition("7", c, b, p, seed)
                part = self._condition(
                    "7",
                    relevant_label,
                    b,
                    p,
                    seed,
                    subgraph_role="relevant",
                    matching={"reference": whole, "dimension": "whole_versus_task_relevant"},
                )
                self._condition(
                    "7",
                    relevant_label,
                    b,
                    p,
                    seed,
                    topology="random",
                    subgraph_role="randomized_relevant",
                    matching={"reference": part, "dimension": "neurons_and_edges"},
                )
                for role in roles[1:]:
                    self._condition(
                        "7",
                        f"{c}:{b}:{role}",
                        b,
                        p,
                        seed,
                        subgraph_role=role,
                        matching={"reference": part, "dimension": "neuron_count"},
                    )
        self.primary = sorted(set(self.primary))
        for reference in self.primary:
            source = RunSpec.from_dict(self.conditions[reference]["config"])
            if source.controller.adapter.budget not in experiments["8"]["capacities"]:
                continue
            mean = self._job(
                "8",
                "calibrate_mean",
                reference,
                {"split": "validation", "episodes": self.learning.eval_episodes},
            )
            for kind in (
                "none",
                "zero",
                "temporal_mean",
                "time_shuffle",
                "neuron_permutation",
                "reset",
                "random_graph",
            ):
                self._job(
                    "8",
                    "intervention",
                    reference,
                    {"kind": kind},
                    dependencies=[mean] if kind == "temporal_mean" else [],
                )
            c, b = source.metadata["connectome"], f"{source.body.name}:{source.body.task}"
            for role in ("relevant", "random_selection"):
                path = self.resolve(
                    self.study["graphs"][c]
                    .get("lesions", {})
                    .get(b, {})
                    .get(
                        role,
                        f"data/paper-subgraphs/{c}/{b.replace(':', '-')}/{role}-selection.json",
                    )
                )
                self._job(
                    "8",
                    "intervention",
                    reference,
                    {"kind": "targeted_lesion", "lesion_file": path, "label": role},
                )
        # Robustness covers biology, rewired and conventional controls; source
        # experience and additional fine-tuning interactions remain separate.
        candidates = [
            item["id"]
            for item in list(self.conditions.values())
            if "3" in item["experiments"]
            or (
                "4" in item["experiments"]
                and item["config"]["controller"]["substrate"]["topology"] == "degree_rewired"
            )
        ]
        for reference in sorted(set(candidates)):
            source = RunSpec.from_dict(self.conditions[reference]["config"])
            if source.controller.adapter.budget not in experiments["9"]["capacities"] and not any(
                self.conditions[m["reference"]]["config"]["controller"]["adapter"]["budget"]
                in experiments["9"]["capacities"]
                for m in self.conditions[reference]["matches"]
                if "reference" in m
            ):
                continue
            b = f"{source.body.name}:{source.body.task}"
            for perturbation in experiments["9"]["perturbations"]:
                if b not in perturbation["tasks"]:
                    continue
                self._job("9", "robustness", reference, {"perturbation": perturbation["spec"]})
            for pair in experiments["9"]["transfers"]:
                if pair["source"] != b or pair["target"] not in self.bodies:
                    continue
                target = self.bodies[pair["target"]]
                target_spec = asdict(target["spec"])
                self._job("9", "task_transfer", reference, {"body": target_spec})
                self._job(
                    "9",
                    "finetune",
                    reference,
                    {
                        "body": target_spec,
                        "interactions": experiments["9"]["fine_tune_interactions"],
                    },
                )
                # Scratch runs match the NEW task's experience budget, not the
                # sum of source and target experience. Both are reported.
                self._condition(
                    "9",
                    source.metadata["connectome"],
                    pair["target"],
                    source.controller.adapter.budget,
                    source.train_seed,
                    kind=source.controller.substrate.kind,
                    family=source.controller.adapter.family,
                    topology=source.controller.substrate.topology,
                    plasticity=source.controller.substrate.plasticity,
                    total_budget=source.controller.substrate.total_budget,
                    hidden_size=source.controller.substrate.hidden_size,
                    extra_issues=self.conditions[reference]["issues"],
                    learning=replace(
                        self.learning, interactions=experiments["9"]["fine_tune_interactions"]
                    ),
                    matching={"reference": reference, "dimension": "target_task_experience"},
                    tags=("transfer_scratch",),
                )
        for outcome in ("success", "final_score", "score_auc", "restricted_efficiency"):
            self._job("10", "predict_pairs", None, {"outcome": outcome}, dependencies=self.primary)
        return self._write()

    def _write(self):
        inputs = {
            "graphs": self.graphs,
            "bodies": {
                key: {**value, "spec": asdict(value["spec"])} for key, value in self.bodies.items()
            },
        }
        coverage = {}
        for experiment, title in EXPERIMENTS.items():
            conditions = [x for x in self.conditions.values() if experiment in x["experiments"]]
            jobs = [x for x in self.jobs.values() if experiment in x["experiments"]]
            coverage[experiment] = {
                "title": title,
                "unique_training_conditions": len(conditions),
                "training_statuses": dict(Counter(x["status"] for x in conditions)),
                "post_training_jobs": len(jobs),
                "job_kinds": dict(Counter(x["kind"] for x in jobs)),
            }
        for item in self.conditions.values():
            item["experiments"] = sorted(set(item["experiments"]), key=int)
            item["config"]["metadata"].update(
                experiments=item["experiments"],
                tags=item["tags"],
                requested_adapter_budgets=item["requested_adapter_budgets"],
                matches=item["matches"],
            )
        plan = {
            "schema": "compatibility-study-plan-v1",
            "project_root": str(self.root),
            "study": self.study,
            "study_sha256": digest_file(self.path),
            "code_fingerprint": self.code,
            "inputs": inputs,
            "conditions": list(self.conditions.values()),
            "jobs": list(self.jobs.values()),
            "coverage": coverage,
            "proposals": self.proposals,
            "notes": [
                "Compilation is not execution or evidence of a scientific effect.",
                "Parameter ceilings that yield identical controllers share one training run.",
                "Body and graph prerequisites are never silently replaced.",
                "Compute matching requires empirical calibration; no FLOP estimate is substituted.",
                "Biological signs are only claimed at measured annotation coverage; unknown signs are imputed explicitly.",
            ],
        }
        plan["fingerprint"] = digest_json(plan)
        path = self.output / "plan.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if previous.get("fingerprint") != plan["fingerprint"]:
                raise FileExistsError(
                    "Plan already exists with different inputs/code; choose a new plan directory"
                )
            return previous
        self.output.mkdir(parents=True, exist_ok=True)
        atomic_json(path, plan)
        atomic_json(self.output / "coverage.json", coverage)
        atomic_json(
            self.output / "prerequisites.json",
            [
                {
                    "condition": item["id"],
                    "experiments": item["experiments"],
                    "status": item["status"],
                    "issues": item["issues"],
                }
                for item in self.conditions.values()
                if item["status"] != "ready"
            ],
        )
        for item in self.conditions.values():
            if item["status"] == "ready":
                atomic_json(self.output / "conditions" / f"{item['id']}.json", item["config"])
        return plan


def compile_study(path, output):
    return StudyCompiler(path, output).compile()
