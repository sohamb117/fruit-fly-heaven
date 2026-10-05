# Frozen version-1 parity oracle

`banc-haltere.js` and `virtual-haltere.js` are unmodified copies of the runtime immediately before the opt-in coupled-haltere change on 2026-09-15. Their relative import still resolves locally. They are test fixtures only, never served as runtime assets.

`orientation-prior.json` is the original 328-cell, four-angle exchangeable prior. `../haltere-identities-v888.json` is the actual prepared annotation/root-ID subset with source SHA256 records. The focused test compares current legacy outputs and diagnostics exactly against this oracle, while v2 tests use the self-contained versioned models. No ignored report file or full BANC binary is needed.
