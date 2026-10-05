# Optional parameter search scales

The reviewed source patch adds optional `searchScale` to each parameter definition. It must be a finite number in `(0, 1]`; omission means exactly `1`. Parameter names, the 27 physical log coordinates, checkpoint schema, and the antithetic ES algorithm remain unchanged. Validation does not insert default fields into configuration objects.

The root agent applied the [two-source patch](search-scale.patch) after independent review. This directory preserves the original sources, reviewed staged copies, the exact 27-parameter configuration used for compatibility checks, and [validation evidence](validation.json). No database, body simulation, or neural execution was involved in this work.

## Update interpretation

Candidate coordinates are `theta +/- sigma * S * epsilon`, where `S` is diagonal with the declared scales. Aggregation still uses the *actual assigned* difference `(plus - minus)/(2*sigma)`, including candidate clipping. It deliberately does not divide the direction by the parameter's search scale.

Away from caps, this matches the same realized-displacement ES rule in normalized coordinates `y = inverse(S) * theta`, then maps the update back through `S`. Locally without clipping, the expected physical update is proportional to `S² * gradient(return)`. This change reduces exploration **and** learning speed along a scaled coordinate. It is not an inverse-covariance gradient estimator.

The existing `maximumUpdate` remains a cap in physical log coordinates. To reproduce that cap in normalized coordinates would require a per-coordinate cap `maximumUpdate / searchScale`; simply using the same cap in both coordinate systems would not be equivalent. Near parameter bounds, realized-displacement weighting retains the pre-existing clipped-direction heuristic and is not standard unclipped Gaussian ES.

The root selected the following v7 scales, with global sigma `0.25`:

| Coordinates | Scale | Candidate log standard deviation before clipping | Local expected update relative to scale 1 |
| --- | ---: | ---: | ---: |
| Power, deployment | 0.5 | 0.125 | 0.25 |
| Frequency | 0.15 | 0.0375 | 0.0225 |
| Remaining 24 steering gains | 1 | 0.25 | 1 |

The frequency reduction intentionally makes local frequency adaptation about 44 times smaller. These are optimizer priors, not a new physiological interpretation of the existing parameters. Extremely small positive floating-point scales are accepted by the stated contract and can underflow into effectively fixed coordinates; the selected ordinary scales are far from that limit.

## Namespace and compatibility

The Python coordinator seeds exploration from `configHash`; adding even an explicit scale of `1` changes the serialized configuration and therefore the real namespace, seeds, and Gaussian samples. The new experiment must use its new configuration hash and database namespace. Tests hold the hash constant only when isolating the arithmetic equivalence of explicit `1` and omission.

JavaScript and Python already use different RNG implementations. Their independent default streams remain unchanged, and their aggregation agrees exactly when given the same assigned candidates and returns. No cross-language equality of independently generated Gaussian streams is claimed.

The original snapshot checks compared complete generation records, noise, assigned jobs, hashes, and aggregate updates for 32 generations using the actual 27-parameter configuration; all were exactly equal when scales were omitted. Durable tests also pin old-implementation signatures from fixed small fixtures so they do not rely on historical files at test time.

## Tests

The following permanent suites were installed after the parent applied the source patch:

```sh
node --test web/test/training-search-scale.test.mjs
.venv/bin/python -B tests/test_training_search_scale.py
```

Both pass, 6 tests each. They cover omission signatures, explicit-one equivalence, scaled candidate displacement, clipping, cross-language aggregation on identical assignments, normalized-coordinate equivalence, the squared scale response, retained physical update caps, unchanged centers with equal returns, and invalid scales. The [test-only patch](search-scale-tests.patch) records their placement; these files are already installed and should not be applied again.

The earlier stage-only suites remain reproducible:

```sh
node --test reports/flight-development-v7-staging/tests/search-scale.test.mjs
.venv/bin/python -B reports/flight-development-v7-staging/tests/test_search_scale.py
```

They also passed 6 tests each. These checks establish optimizer arithmetic and compatibility, not flight improvement. Held-out flight conclusions and subsequent training belong to the separately pinned v7 experiment.
