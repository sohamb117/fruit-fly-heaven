# /// script
# dependencies = ["matplotlib>=3.8,<4"]
# ///
"""Plot the gated mechanical replay; this is not a behavioral success assay."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

directory = Path("reports/flybody-onset-causal")
report = json.loads((directory / "replay.json").read_text())
capture = json.loads((directory / "capture.json").read_text())
assert report["passed"] and report["interpretationAllowed"]
initial_height = capture["initial"]["native"]["qpos"][2]
styles = {
    "baseline": ("Recorded motor input", "#c65a17", "-"),
    "dlm_dvm_zero": ("DLM/DVM input zero", "#187849", "-"),
    "dlm_dvm_zero_and_wing_controls_zero": ("DLM/DVM zero + wing controls zero", "#255fa1", "--"),
}
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
for case in report["cases"]:
    label, color, style = styles[case["name"]]
    samples = case["samples"]
    times = [s["time"] * 1000 for s in samples]
    axes[0].plot(times, [s["qpos"][2] - initial_height for s in samples],
                 label=label, color=color, linestyle=style, linewidth=1.6)
    axes[1].plot(times, [s["up"] for s in samples], color=color, linestyle=style, linewidth=1.6)
for ax in axes:
    ax.grid(alpha=.2)
    ax.axvline(104, color="#666666", linewidth=.7, linestyle=":")
    ax.axvline(112, color="#666666", linewidth=.7, linestyle=":")
axes[0].set_ylabel("Root height change (cm)")
axes[0].legend(loc="upper right", fontsize=8)
axes[1].set_ylabel("Body uprightness (up Z)")
axes[1].axhline(0, color="#444444", linewidth=.6)
axes[1].set_ylim(-1.05, 1.05)
axes[1].set_xlim(0, 400)
axes[1].set_xlabel("Simulated time (ms)")
fig.suptitle("Actual BANC onset: exact baseline replay, then isolated interventions\n"
             "Dotted lines: first airborne report (104 ms), first inverted 2 ms sample (112 ms)", fontsize=11)
fig.savefig(directory / "replay-timeline.png", dpi=160)
plt.close(fig)
print(directory / "replay-timeline.png")
