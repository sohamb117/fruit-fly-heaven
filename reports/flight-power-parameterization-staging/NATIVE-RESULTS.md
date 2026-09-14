# Native validation, 2026-09-14

These are diagnostic evaluations with live BANC, the original dynamic full body and the individual motor-event interface. No coordinator jobs were leased and no checkpoint parameters were updated. Each maintained-flight evaluation uses 0.5 s of unscored live warm-up followed by up to five scored seconds, ending early only on physical failure. Root restraint ends at release; the audit records no later root correction.

The new activation/amplitude profile at amplitude 0.90 reproduced the preceding common-0.90 experiment exactly: the complete trajectory and all 1,226 motor packets match. Physics digest: `47ebf3b68a2266884fe8d75aca48a7dd53a47ed4d81f0aa7158c38d93feed207`. The new optional body adapters with reduction disabled also reproduce that trajectory and motor history exactly.

| Amplitude | Seed | Longest qualified bout | Total qualified flight | First environment contact | Physical termination |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.90 | 1290888 | 0.602 s | 1.248 s | 1.838 s, wall | 1.950 s |
| 0.91 | 1290888 | 0.602 s | 1.188 s | 2.074 s | 2.076 s |
| 0.90 | 1190888 | 0.602 s | 1.140 s | 1.838 s | 1.838 s |

All three ended with `excessive_rotation`; none completed the five-second task. At amplitude 0.90 the first seed stays within the 45-degree attitude limit until after wall contact, but its vertical velocity repeatedly fails the qualification gate before contact. At 0.91 the first 45-degree excursion occurs at 1.208 s, before contact, so its later collision does not establish improved control.

The 0.91 physics digest differs (`d984a80a70d9febbdca1662bcba08a18c1b223af886751826d2bf68283fd0d23`), demonstrating that the migrated parameter reaches the mechanics. It does not demonstrate learning or monotonic aerodynamic response. These seeds are diagnostic seeds, not independent validation of a trained checkpoint.

Evidence: `full-equivalence-run1/result.json`, `full-amplitude-0p91-run1/{result,summary}.json`, `full-amplitude-0p90-seed1190888-run1/{result,summary}.json`, the preceding `../flight-common-power-staging/scale-0p90-run1/{result,summary}.json`, and `../flight-reduced-native-staging/full-adapter-equivalence-run1/result.json`.

The first reduced-body attempt stopped during the initial preview because telemetry required a removed claw actuator. It recorded zero trajectory rows and is an integration failure, not a flight result. Its original error is preserved in `../flight-reduced-native-staging/reduced-amplitude-0p90-run1/result.json`.
