# Anatomical registration of flight mechanoreceptors

Read-only primary-source review, 2026-09-14. No sensor tuning or controller was implemented.

## Local inventory

`data/prepared/banc888/console/sensory-inputs.json` contains 449 `kind: rotation` rows across 34 distinct cell types: 328 haltere and 121 wing-base neurons. These prepared rows carry organ, side, modality, type, and annotation, but no measured strain axis, directional sign, or wingbeat phase. Their `tuning_status` explicitly treats numerical tuning and direction preference as priors.

## Wing crosswalk available now

[Lesser, Moussa, and Tuthill (2026), Appendix 1, table 1](https://elifesciences.org/articles/107867#app1) explicitly relates peripheral structures to MANC names used in the BANC annotations:

| Locally present names | Published peripheral category | Local wing-base rows |
| --- | --- | ---: |
| SApp04, SApp10, SApp11, SApp13, SApp14, SApp18, SApp19, SApp20, SApp21 | Small proximal campaniform sensilla | 77 |
| SNpp06, SNpp08, SNpp11, SNpp26 | Small distal campaniform sensilla | 17 |
| SNpp30, SNpp32 | Large campaniform sensilla | 23 |
| SNpp09 | Unidentified | 4 |

This gives anatomical categories for 15 type names and 117/121 wing-base rows. It is not a signed angular-velocity response map. SApp11 additionally labels **two haltere rows** locally; a type-name-only join must not overwrite neuron-level organ identity. More detailed peripheral identifications use FANC root IDs and genetic/light microscopy matches. Adjacent sensilla can have different central projections, and axons with similar morphology can originate from different fields.

The [released FANC annotation table](https://github.com/EllenLesser/Lesser_eLife_2025/blob/main/dfs/sn_table.csv) has `classification_system` clusters and broad `cell_type` structure categories. More detailed peripheral annotation and imaging are available through [Dryad](https://doi.org/10.5061/dryad.mgqnk99b5). Downloading the multi-gigabyte image collection is unnecessary for this review.

## Haltere crosswalk available now

[Dhawan, Huang, and Dickerson (2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12872070/) reconstructs FANC haltere afferents and relates central morphological clusters to sets of peripheral fields. Its [released `final_haltere_clusters.csv`](https://github.com/serene-da1/Dhawan-et-al-2025-/blob/main/Data/Annotation%20tables/final_haltere_clusters.csv) contains:

| FANC cluster | `campaniform_fields` | CSV rows |
| --- | --- | ---: |
| CL1 | dF3,dF2,vF1 | 88 |
| CL2 | dF3,dF2,vF1 | 79 |
| CL3 | dF2,vF2 | 71 |
| CL4a | vF1,vF2 | 67 |
| CL4b | unknown | 67 |

These are **cluster-level sets of possible peripheral origins**, not a resolved field for every neuron. The paper states that precise phase ranges are not yet determined and predicts diverse preferred phases within each subtype. Neither its examined annotation tables nor the paper provided a validated CL1–CL4b to our SApp/SNpp crosswalk. Do not interpret the repeated field set on each CSV row as evidence that the individual cell's sensillum location is known.

The [BANC paper's matching methods](https://www.nature.com/articles/s41586-026-10735-w) and [analysis pipeline](https://github.com/htem/bancpipeline) provide cross-connectome morphology comparisons, including `banc_fanc_1116_nblast.feather`. This is a possible route to further registration, with expert review rather than simply accepting each top match. Dhawan's atlas used FANC v840, so its IDs must also be reconciled with the v1116 matching resource. Cross-connectome matching establishes anatomy; it does not measure receptor tuning.

## What physiology adds, and what it does not

- [Verbe et al. (2024), Drosophila](https://pmc.ncbi.nlm.nih.gov/articles/PMC11338719/): field-level calcium imaging during tethered flight shows ongoing activity and modulation during visually driven and spontaneous maneuvers. The authors explicitly note that calcium-indicator kinetics cannot resolve individual sensillum spike timing. Visual-motion tuning is not a direct measurement of receptor tuning to body rotation.
- [Dickinson and Palka (1987), Drosophila](https://pmc.ncbi.nlm.nih.gov/articles/PMC6569114/): identified wing sensilla exhibit rapid or slow adaptation associated with central projection. This demonstrates physiological heterogeneity, not a three-axis signed angular-velocity map for the BANC types.
- [Yarger and Fox (2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6170812/): precise spike-phase shifts and recruitment thresholds were recorded in **Sarcophaga bullata**. The authors explicitly could not simultaneously identify the recorded axons' specific peripheral locations. These data cannot be assigned directly to named Drosophila BANC cells.
- [Fox, Fairhall, and Daniel (2010)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2840414/): the haltere encoding recordings used **Holorusia hespera**, a crane fly.
- [Sharma et al. (2026), Drosophila](https://authors.library.caltech.edu/records/r1ywc-zsa65): genetically silencing increasing numbers of haltere campaniform sensilla reduces yaw equilibrium responses. This establishes causal participation, without providing per-type strain axes or phase curves. This review checked the primary abstract/resource record, not its full experimental supplement.
- [Dinges et al. (2021)](https://pubmed.ncbi.nlm.nih.gov/32678470/): structural anatomy supplies sensillum locations and arrangement, not a connectome-registered dynamic transfer function.

## Practical conclusion

Published data support improving anatomical labels and identifying additional candidate matches. They do **not** currently justify assigning a compensatory roll/pitch/yaw axis or response sign to every central sensory type. Missing links are individual peripheral identity where unresolved, mechanical coupling from body/wing/haltere motion to local strain, and the cell's strain-to-spike recruitment/phase response. Deriving an input axis from the desired downstream motor correction would assume the feedback relationship under test.
