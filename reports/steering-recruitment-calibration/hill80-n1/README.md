# Steering recruitment calibration

All 390 legacy native muscle states are byte-identical to the captured profiles. Nonsteering states are byte-identical under the Hill variant. The 100–280 ms force integral sets a single bilateral multiplier per type, applied to its bias and amplitude gains. No reward or body outcome was optimized.

| Type | Legacy mean force | Hill mean force | Multiplier | New gain | Per-side mean residual |
|---|---:|---:|---:|---:|---|
| b1_muscle | 0.024231 | 0.021983 | 1.1023 | 0.05511 | n/a / 0.00% |
| b2_muscle | 0.988055 | 0.695217 | 1.4212 | 0.07106 | 1.18% / -1.18% |
| b3_muscle | 0.411978 | 0.259301 | 1.5888 | 0.07944 | -4.87% / 18.18% |
| i1_muscle | 0.987533 | 0.664524 | 1.4861 | 0.07430 | 8.54% / -8.55% |
| i2_muscle | 0.987709 | 0.737747 | 1.3388 | 0.06694 | 0.59% / -0.59% |
| iii1_muscle | 0.000000 | 0.000000 | 1.0000 | 0.05000 | n/a / n/a |
| iii3_muscle | 0.020668 | 0.018881 | 1.0947 | 0.05473 | n/a / 0.00% |
| iii4_muscle | 0.987772 | 0.610795 | 1.6172 | 0.08086 | 5.10% / -5.10% |
| iv1_muscle | 0.988056 | 0.722795 | 1.3670 | 0.06835 | -0.65% / 0.65% |
| iv2_muscle | 0.210788 | 0.153324 | 1.3748 | 0.06874 | 1.80% / -1.33% |
| iv3_muscle | 0.599255 | 0.355791 | 1.6843 | 0.08421 | 14.12% / -6.89% |
| iv4_muscle | 0.987881 | 0.633537 | 1.5593 | 0.07797 | -2.06% / 2.06% |

Force is the native muscle output with the captured Fmax input. Aggregate mean is preserved before the fixed steering basis and command clipping. The candidate keeps power 1.5, deployment scale 1 and frequency scale 1. The Hill curve requires explicit model metadata; this is not a flight validation. Exact profiles, per-side variability and source identities are in the adjacent JSON artifacts.
