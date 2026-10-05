# Spacious native saved-record analysis

Every original v1 score-state field and both saved physics digests verify exactly. Raw trajectory observations are copied and receive the scene profile from the pinned configuration before either scorer is called.

Original v1: longest 0.172 s; current 0.000 s; total 2.044 s.

Revised v2: longest 1.244 s; current 0.000 s; total 2.222 s. Reason excessive_rotation at 2.724 s; full-five-second success false.

First contact 2.652 s; first >45-degree tilt 2.71 s. This ordering is temporal evidence only. Root audit reports no additional writes after release; 250 warm-up root snapshots and the release state match the declared pose/zero velocity exactly.

Before the small-bowl control's first contact at 1.274 s, 885/887 complete raw rows match after excluding only observation ceiling/profile. All-prefix match: false; 887 motor packets match: true. See result.json for exact fields, first differences, extrema, scored windows and hashes.

Saved-record/pure-scorer analysis only: zero native or brain steps, no live edits, no optimizer work. One seed and an offline scorer change do not establish learning or generalization.
