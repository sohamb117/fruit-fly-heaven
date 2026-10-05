# Live GCP browser verification

The public page at <https://35.209.223.157/train.html> passed the deployment browser check on 2026-09-13. Chrome used normal certificate validation (TLS 1.3); no certificate bypass was enabled.

- Opening the page and connecting loaded no worker, model assets, WebGL context, or simulation. Sharing was unchecked, and the coordinator URL defaulted to the same HTTPS origin.
- Explicit Connect and Start ran one actual BANC → VNC → motor neuron → muscle → native FlyBody episode using WebGPU.
- The browser submitted exactly one accepted result. The coordinator's accepted count increased from 8 to 9; generation remained 1.
- The episode simulated 1 second of the posture task, with reward 2.4295438457926406 and `success: false` (`time_limit`). This is deployment evidence, not successful training.
- Training work took 34.44 seconds, including model startup; the run and capture sequence took 36.31 seconds from Start.
- The preview contained 66 native frames, six articulated legs and two wings, with progressing body/neural clocks and nonzero neural activity. The final native position differed from the initial position.
- The preview rendered at 320 × 180, up to 3 fps. Desktop (1440 px) and mobile (390 px) screenshots were visually inspected: the fly, controls, reward, and contribution state are visible, with no horizontal overflow.
- Resuming leased the next real job; the test paused before its first physics step, then Stop released that unscored lease. No worker, heartbeat, or outstanding lease remained. Server state was one completed job and seven pending jobs.
- No JavaScript, console, network, or API errors were observed.

Configuration SHA256: `1bad7d5c80d5129dcbe4f94b66fc725ec343053fb50bf70769c4a8e75510d00e`.

Model fingerprint: `8bfd755893d540ed11e249c36ad7ae73d6a955c78bc57a3899ad18cc8c2bee13`.

Artifacts: [structured report](browser-verification.json), [desktop screenshot](live-training-1440.png), [mobile screenshot](live-training-390.png). The report includes native frames, before/after coordinator status, API responses with lease tokens removed, and source identity.

Reproduce with `node scripts/verify-gcp-training.mjs`. This performs one real production contribution and releases one subsequent unscored lease.
