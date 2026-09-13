"""Plot the measured beginning of one native-body fall; no resimulation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
rows = json.loads((root/'reports/flybody-flight-live-audit.json').read_text())['flightTrace']
rows = [r for r in rows if r['t'] <= .24]
t = np.array([r['t'] for r in rows])*1000
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, layout='constrained')
axes[0, 0].plot(t, [r['velocity'][2]*10 for r in rows], color='#19647e')
axes[0, 0].axhline(0, color='#aaaaaa', lw=.7)
axes[0, 0].set(ylabel='Vertical speed (mm/s)', title='Falls first, then gets kicked upward')
axes[0, 1].plot(t, np.mean([r['wings'] for r in rows], axis=1), color='#19647e', label='Wing activation')
force = np.array([r['aerodynamicForceBodyWeights'][2] for r in rows])
average = np.array([force[max(0, i-19):i+1].mean() for i in range(len(force))])
axes[0, 1].plot(t, average, color='#de7b28', label='Upward air force / weight (20 ms mean)')
axes[0, 1].axhline(1, color='#999999', lw=.7, ls='--')
axes[0, 1].set(ylabel='Activation / force ratio', title='Active wings do not establish flight')
axes[0, 1].legend(fontsize=8)
axes[1, 0].plot(t, [r['constraintForceBodyWeights'][2] for r in rows], color='#de7b28')
axes[1, 0].set(yscale='symlog', ylabel='Vertical constraint force / weight', title='Large impulses at wing contact', xlabel='Simulated time (ms)')
axes[1, 1].plot(t, [r['angularSpeed'] for r in rows], color='#19647e')
axes[1, 1].set(ylabel='Angular speed (rad/s)', title='Rapid rotation follows the impacts', xlabel='Simulated time (ms)')
for ax in axes.flat:
    ax.axvline(136, color='#73836c', ls='--', lw=1)
    ax.axvline(167, color='#bd4935', ls='--', lw=1)
    ax.grid(alpha=.15)
    ax.set_xlim(0, 240)
fig.suptitle('One BANC-driven fly: slide → fall → wing strike → tumble\nDashed lines: first classified airborne (136 ms); first recorded wing strike (167 ms)', fontsize=13)
fig.savefig(root/'reports/flybody-flight-timeline.png', dpi=160)
