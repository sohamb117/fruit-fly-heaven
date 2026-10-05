from pathlib import Path
import json,subprocess
root=Path('/workspace/experiments/connectome_body/runs/learning-qualification-cuda-v2')
progress=root/'progress.json'
if progress.exists():
 p=json.loads(progress.read_text()); print(json.dumps({'completed':p['completed'],'planned':p['planned'],'passed':sum(r.get('passed',False) for r in p['cases']),'failures':[r for r in p['cases'] if not r.get('passed')]}))
logs=sorted(root.glob('case-*/worker.log'))
if logs:
 p=logs[-1]; print(str(p.relative_to(root)))
 for line in p.read_text().splitlines()[-4:]:
  try:
   d=json.loads(line)
   if 'hardware' in d: d={k:d[k] for k in ('passed','substrate','variant','plasticity','body','wall_seconds','cuda_peak_allocated_bytes') if k in d}
   print(json.dumps(d))
  except ValueError: print(line)
for name in ('launcher-exit-code','finished-at'):
 if (root/name).exists(): print(name, (root/name).read_text().strip())
print('archive_ready',Path(str(root)+'.tar.gz.sha256').exists())
subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'])
