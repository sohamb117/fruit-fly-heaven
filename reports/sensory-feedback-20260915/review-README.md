Local recorded-evaluation review, using the existing native 3D preview and training theme.

Start from the repository root:

```sh
node reports/sensory-feedback-20260915/review-server.mjs
```

Open `http://127.0.0.1:7884/` in Safari. Select a recording and a saved frame; drag the 3D view to orbit. The slider renders when released. Previous/Next select adjacent saved poses. Refresh checks for newly completed results. There is no playback, interpolation, neural loading, simulation or coordinator traffic.

The page exposes only `off-v1`, `vision-airflow-v1` and `vision-airflow-v2` saved frame files. Missing files display “Waiting”; results are available once the evaluation saves them. The current recordings save only their first 200 poses, so the slider does **not** span the complete trial. A coverage notice gives the native and scored timestamp ranges, full evaluated duration, and any missing final interval. Trial score/result refer to the whole evaluation, not the selected pose. Large gaps between saved poses are also shown explicitly.

The eye canvases are **reconstructed at 256×128 per eye from the selected native pose using the frozen v2 antialiased renderer**. They are not saved sensory samples from that evaluation. In particular, off-v1 did not receive visual input. The renderer and Three modules are read from the v2 bundle and checked against its pinned hashes. Only the native preview and CSS come from the checkout.

Validation without starting a server:

```sh
node reports/sensory-feedback-20260915/review-server.mjs --check
node --test reports/sensory-feedback-20260915/review-server.test.mjs
```

The server binds only `127.0.0.1:7884`, accepts GET/HEAD, and serves an explicit allowlist. It neither changes result files nor exposes general filesystem paths. It was left stopped for the root agent to launch and inspect in Safari.
