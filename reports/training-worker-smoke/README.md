# Browser BANC training environment smoke

The actual worker initialized BANC v888 on **webgpu**, with native MuJoCo and WASM muscles, then evaluated one baseline parameter vector for **0.2 simulated seconds / 100 body blocks**. Neural steps stayed at 0.5 ms and body feedback at 2 ms. It recorded **419,301 neural spikes**. Final neural/native clock discrepancy was 1.3869794202037156e-11 ms.

The result was **time_limit; success=false**. Upright time was 108 ms, support 148 ms, maximum angular speed 278.97 rad/s, and food transfer and qualified flight were zero. The native preview shows the falling/tumbling body honestly. This is real integration evidence, not learned behavior or a performance comparison.

Paused evaluation consumed zero blocks; cancellation returned the actual zero duration, dynamic preview-off/work-budget messages applied, and Stop disposed the environment. Thirteen separate mathematical/lifecycle unit tests pass; those tests use synthetic fixtures and are not neural evidence. No root forces, teacher actions or direct motor-neuron current were introduced.

The executable/data manifest contains 39 assets; all source hashes still match after the run. Graph binaries are checked against the pinned BANC manifest. Fingerprint: `8bfd755893d540ed11e249c36ad7ae73d6a955c78bc57a3899ad18cc8c2bee13`. Config SHA256: `1bad7d5c80d5129dcbe4f94b66fc725ec343053fb50bf70769c4a8e75510d00e`. The only browser console 404 was independently identified as the missing favicon.

Stage observation rules use native cm, cm/s, rad/s and seconds. Sequence order is localization → approach → qualified landing → probing/feeding → takeoff → qualified flight. Feeding requires actual crop transfer plus mouth contact/extension. Landing requires the existing native monitor's prior qualified powered flight and 100 ms stable food contact; an initial ballistic drop cannot pass. Horizontal food-center/radius distance is an explicit approach proxy, not exact distance to the fruit collision surface. Thresholds are modeling criteria, not measured biological limits.

Trainable leak cells are disjoint from external sensory inputs: wing 65, leg 297, probing 44, grip 48, descending 1,316. Zero log-gains preserve original parameter/edge bits; chemical gains retain topology, neuromodulator edges and electrical edges. Sensory gains act through the existing calibrated rate-to-current mapper (bounded at its existing 200 Hz table); motor-family gains scale measured MN output before the existing 80 Hz muscle activation prior. Other physiology and rigid-wing/strain-feedback limitations remain unchanged.

Files: [raw results](result.json), [native preview](native-preview.png), [source/score audit](source-audit.json), [exact config](config.json). Reproduce with `PLAYWRIGHT_MODULE=/tmp/fruit-fly-browser-tests/node_modules/playwright/index.mjs node scripts/smoke-training-worker.mjs`.
