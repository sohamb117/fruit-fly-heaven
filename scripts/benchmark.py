from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from brain import Graph, Brain, ROOT

p = argparse.ArgumentParser()
p.add_argument('--flies', type=int, default=100)
p.add_argument('--ms', type=int, default=100)
p.add_argument('--workers', type=int, default=6)
args = p.parse_args()
g = Graph()
brains = [Brain(g, 1701+i) for i in range(args.flies)]
start = time.perf_counter()
def run(b):
    for _ in range(args.ms//20):
        b.step(200, 35, 45, 150)
    return b.stats()
with ThreadPoolExecutor(args.workers) as pool:
    states = list(pool.map(run, brains))
elapsed = time.perf_counter()-start
result = {'flies': args.flies, 'workers': args.workers, 'simulated_ms_per_fly': args.ms,
          'wall_seconds': elapsed, 'speed_vs_real_time': args.ms/1000/elapsed,
          'distinct_spike_counts': len(set(s['spikes'] for s in states)),
          'total_neural_spikes': sum(s['spikes'] for s in states),
          'min_neurons_fired': min(s['active_ever'] for s in states),
          'max_neurons_fired': max(s['active_ever'] for s in states),
          'brains_with_feeding_activity': sum(s['feed_hz']>0 for s in states),
          'brains_with_walking_activity': sum(s['walk_hz']>0 for s in states),
          'states': states}
(ROOT/'reports'/f'benchmark-{args.flies}.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='states'}, indent=2), flush=True)
for b in brains: b.close()
g.close()
