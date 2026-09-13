"""Plot recorded behavior; no simulation or control is performed here."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
before=json.loads((root/'reports/flybody-flight-live-audit.json').read_text())['flightTrace']
after_report=json.loads((root/'reports/flybody-flight-repair-final-live.json').read_text());after=after_report['flightTrace'];stop=after_report['disconnectedAtMs']/1000
fig,axes=plt.subplots(2,2,figsize=(12,6.5),layout='constrained')
for trace,name,color in [(before,'Before (0.61 s recorded)','#bc3e33'),(after,'After (7.01 s recorded)','#147c70')]:
 t=np.array([s['t'] for s in trace]);pos=np.array([s['position'] for s in trace]);omega=np.array([s['angularSpeed'] for s in trace])
 axes[0,0].plot(t,10*(pos[:,2]-pos[0,2]),label=name,color=color,lw=1.1)
 axes[0,1].plot(t,np.maximum(omega,.01),label=name,color=color,lw=.9)
 axes[1,0].plot(t,[s['tilt']*180/np.pi for s in trace],label=name,color=color,lw=1)
t=np.array([s['t'] for s in after]);axes[1,1].plot(t,[np.mean(s['wingDrive']) for s in after],color='#dc8c22',label='Power-muscle drive')
axes[1,1].plot(t,[np.mean(s['wings']) for s in after],color='#147c70',label='Transmitted beat power')
axes[0,0].set(ylabel='Vertical displacement from start (mm)',title='Body remains on the fruit')
axes[0,1].set(ylabel='Angular speed (rad/s)',yscale='log',title='No collision-driven tumbling')
axes[1,0].set(ylabel='Body tilt (degrees)',title='Upright throughout the repaired run')
axes[1,1].set(ylabel='Normalized drive / power',ylim=(-.04,1.04),title='Retraction now disengages the hinge')
for ax in axes.flat:
 ax.axvline(stop,color='#777777',ls='--',lw=.9);ax.set_xlabel('Simulated time (s)');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[0,0].legend(fontsize=8);axes[1,1].legend(fontsize=8)
fig.suptitle('One-fly BANC observation: falling-off sequence repaired',weight='bold',fontsize=15)
fig.text(.5,-.03,'Dashed line: motor connection disabled. This run does not demonstrate autonomous takeoff or sustained flight.',ha='center',fontsize=10)
fig.savefig(root/'reports/flybody-flight-repair.png',dpi=160,bbox_inches='tight')
