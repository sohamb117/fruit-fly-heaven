# Vision declaration contract review

This is an unapplied candidate. No runtime, active bundle, canonical config, coordinator, or deployment was modified by this review. Existing evaluations may continue against their frozen source hashes.

Apply after the active evaluations finish:

```sh
git apply --check reports/sensory-feedback-20260915/vision-contract-candidate/vision-contract.patch
git apply reports/sensory-feedback-20260915/vision-contract-candidate/vision-contract.patch
```

The patch changes seven files:

- `web/training/config-schema.js`: a pure `validateTrainingVisionConfig` contract. Explicit `vision:false` with no `visionFeedback` keeps its old semantics. Explicit `vision:true` requires the version-1 compact-retinal-motion declaration, valid resolution and a finite cadence aligned to the body block. Known optional camera/motion objects remain supported; their detailed numerical validation still belongs to the transducer constructors. Generic optimizer-only fixtures without either sensory field remain accepted by the generic schema validator, while episode execution requires an explicit boolean.
- `web/training/episode.js`: preserves the exact 0.5 ms neural / 2 ms body timing requirements, accepts the validated opt-in vision profile, and leaves both 27- and 672-parameter interpretations unchanged.
- `web/training/sensory-feedback.js`: uses that same declaration validator, preventing actual injection from disagreeing with the configuration flag. Projection identity checks remain unchanged.
- `scripts/prepare-training-manifest.mjs`: accepts an optional config argument and adds the computed visual-projections URL when vision is declared; the standalone CLI passes its loaded config. Static source imports already discover all three sensory modules and `banc-antenna.js`. Existing `extraUrls` and config-free builder calls remain valid.
- `scripts/prepare-flight-feedback-experiment.mjs`: writes a truthful boolean per generated variant. It already explicitly pins the projection for all four comparison variants; that remains unchanged so model-asset sets stay matched.
- `web/test/training-sensory-feedback.test.mjs`: gives all sensory fixtures explicit truthful flags.
- `web/test/training-vision-contract.test.mjs`: covers unchanged parameter interpretation, mismatch/malformed declaration rejection, and computed plus static asset discovery.

The existing environment checks for a pinned visual-projections asset before using it. No server or client code found assumes a false vision flag. `scripts/training_coordinator.py` accepts the preview `vision` field only when it is a boolean; it does not use it to compute or overwrite neural inputs. Config hashes already bind the declaration. No server changes are needed for this local contract fix.

Validation against candidate sources used a Node load hook rather than editing runtime files:

```sh
node --import ./reports/sensory-feedback-20260915/vision-contract-candidate/candidate-loader.mjs --test web/test/training-episode.test.mjs web/test/training-guard-schema.test.mjs web/test/training-sensory-feedback.test.mjs reports/sensory-feedback-20260915/vision-contract-candidate/candidate-tests.mjs
node --import ./reports/sensory-feedback-20260915/vision-contract-candidate/candidate-loader.mjs --test web/test/training-motor-decoder.test.mjs
```

Result: 22 tests passed; `git apply --check` passed. The asset-discovery test checks inclusion against the current files; after actual application, regenerate the bundle to pin the new source bytes. These tests do not constitute a new full flight evaluation.

Do not overwrite v1 evidence bundles when generating the next version. The running v1 variant has real visual injection even though its legacy config flag says false; label that provenance mismatch in the report. The final corrected runtime must have a new fingerprint. Existing generator notes also inherit obsolete upstream prose; replace that prose when preparing the final local experiment, without implying these local results were submitted to the cloud.
