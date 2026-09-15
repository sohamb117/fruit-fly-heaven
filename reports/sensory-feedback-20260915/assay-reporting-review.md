The active assay source, SHA256 `9e2d4730106a4e3ad1d9486409067f4f79b099cf3cebdebeafe2d91438368136`, already contains the two reported fixes. The independent review described an earlier draft. No live-source patch is required:

- Line99 sorts each0.5ms tick's events by exact timestamp, then neuron index. Mixed ionic0.1ms and generic0.5ms event times therefore give the correct first-event summary.
- Lines225/252 retain the input interval's start and report current onset from that start. Immediate input differences report0ms.
- Lines254–255 distinguish the earliest altered exact wing spike from the2ms packet detection boundary.

`flight-feedback-assay-reporting-fix.mjs` is an optional offline report auditor/corrector. It owns its output copy and changes only derived timing summaries. It neither launches a neural simulation nor changes raw records. The current frozen source should require zero corrections. The companion tests inspect the pinned frozen implementation and use synthetic saved records; they allocate no brain or body.

Interpretation limits remain material:

1. Signed probes replace captured pitch angular velocity with±5rad/s. They are symmetric about zero, not small perturbations around the captured pitch rate. The odd contrast is a valid signed response; the even term relative to native-rate sham also contains operating-point displacement and is not a pure nonlinearity estimate.
2. Sham recomputes the mechanical haltere prior under frozen body state. It is not the original live continuation, and caps/phase shifts change its inputs. Duplicate shams prove branch reproducibility within this protocol.
3. Current values near95%of a configured cap describe the transducer curve. Neural saturation needs the actual voltage/release/event/rate response. Graded afferents can transmit without spikes.
4. Cross-phase cosine values are descriptive alignment, not demonstrated generalization or controllability. The decoder receives exact restored history but samples an analytic continuation of the captured wing oscillator at1ms boundaries; no future native body trajectory is integrated.
5. These short frozen-context probes can establish signed sensory-to-motor sensitivity and timing. They cannot establish a stabilizing closed-loop controller, a correct physiological current scale, or successful flight.
