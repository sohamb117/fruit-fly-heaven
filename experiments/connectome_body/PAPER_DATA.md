# Biological input review — 2026-09-16

The study target is [experiments/GOAL.md](../GOAL.md). In particular, its
Experiment 6 needs a defensible connectome × body comparison. Successful parsing
or a functioning controller does not establish suitable biological coverage.

## Fish1 whole-volume export

Materialization **700** of `fish1_full`, segmentation `fish1_v250915`, is exported
locally with separate unjoined soma, physical-synapse and label tables. Every
table has a count-complete, unique-ID audit and checksummed Parquet pages.

| Quantity | Observed value |
|---|---:|
| Soma annotations | 187,052 |
| Physical axon-to-dendrite synapses | 29,474,316 |
| Synapse labels | 29,474,316 |
| Selected nonzero single-soma roots | 178,976 |
| Retained physical synapses between selected roots | 235,608 (0.799367%) |
| Unique directed edges in the induced graph | 157,014 |
| Isolated selected roots | 108,111 (60.4053%) |
| Synapses whose presynaptic root has any soma annotation | 548,823 |
| Synapses whose postsynaptic root has any soma annotation | 9,513,591 |
| Synapses with neither endpoint associated with a soma | 19,654,348 |
| Synapses with a zero root ID | 0 |

An independent scan uses uint64 endpoint arrays and sorted membership lookup,
separate from the importer's string lookup and SQLite aggregation. It reproduces
the retained-synapse count exactly. Three sampled roots' incoming and outgoing
counts also match live CAVE queries at v700 exactly. All physical synapses have
one exported label; the label join has no orphan target IDs. There are 12,493
label-versus-soma sign disagreements among retained, sign-comparable annotations.
The importer preserves the soma evidence and records the disagreement.

This supports a reconstruction limitation rather than a numeric-ID conversion
error. The original [Fish1 resource paper](https://vcg.seas.harvard.edu/publications/20250615-zbrain/paper)
describes the automatic reconstruction as fragmented, with soma-containing
fragments dominated by dendrites. Its initial segmentation was not a catalog of
fully reconstructed neurons. The current audit concerns the later v700 snapshot;
the earlier paper alone cannot establish that snapshot's coverage.

**Decision:** preserve the graph for explicitly scoped partial-segmentation work,
but block its primary whole-connectome compatibility conditions. The planner
reads a graph-pinned qualification record, so a downloaded graph cannot silently
turn this scientific prerequisite into a ready condition. A neuron-resolved
reconstruction, or a separately declared circuit-level design, is needed before
making the intended zebrafish compatibility claim. Excluding isolated nodes alone
would not restore the missing axonal connections.

Evidence:

- [Graph manifest](data/graphs/fish1/manifest.json)
- [Endpoint audit](data/raw/fish1-endpoint-audit-v700.json)
- [Live source check](data/raw/fish1-source-check-v700.json)
- [Primary-use qualification](data/paper-annotations/fish1/primary-qualification-v700.json)
- [Native-interface check](validation/paper-goal-study-20260916/fish1-interface-check.json)

The last check consists of two physical transitions and a backward pass for each
of three bodies × two plasticity regimes. It has zero optimizer updates and is
not evidence of learned control or biological superiority.

## Separate curated Fish1 circuit

The [published HMI analysis archive](https://storage.googleapis.com/fish1-release/paper_data/HMI_analysis.zip)
contains a 999-cell catalog with reconstruction and neurotransmitter annotations.
The separate importer selects exactly the source labels
`soma, dendrite(c), axon(c)` and
`soma, dendrite(reconstructed), axon(reconstructed)`, yielding **197 cells**.
Those labels are source evidence, not an independent revalidation of morphology
or exhaustive synaptic tracing.

Its induced graph has **293 directed pairs**, **14 isolated selected cells**, and
75.6345% source-annotated signs. Input and output lists lack common synapse IDs;
the importer takes binary edge presence rather than adding potentially repeated
contacts. Of 3,013 source contact annotations, 363 fall inside the selected
population, 1,392 connect outside it, and 1,258 have unidentified partners.
These are annotation counts, not a unique-synapse census.

This is an explicit secondary circuit option, stored at
[fish1-hmi-binary](data/graphs/fish1-hmi-binary/manifest.json). It is not inserted
into the primary Fish1 row or treated as a whole zebrafish nervous system.
Comparisons should declare its circuit selection and match subgraph size and
coverage. Binary weights also cannot test an advantage from biological synapse
multiplicities in GOAL Experiment 5.

## Sign and measurement coverage

BANC's current import has 78.4551% annotated source signs. MaleCNS and Cook's
current imports have none; their missing signs are explicitly imputed under the
shared rule. Fish1's soma-root graph has 22.8869% annotated signs. Generic topology
comparisons use the same seeded sign convention; the available-sign sensitivity
study retains each source's coverage limitation. Synapse counts, Cook serial-
section extents and HMI binary edges are distinct measurements, not equivalent
physical conductances.
