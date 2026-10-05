from pathlib import Path
import json,time
from connectome_body.compatibility.learning_study import read_plan
from connectome_body.compatibility.bodies import BodySpec
from connectome_body.compatibility.imitation import build_demonstrations
from connectome_body.util import atomic_json
plan=read_plan('runs/learning-core-cuda-v2/plan.json')
records={}; start=time.monotonic()
for name,item in plan['datasets'].items():
 print(json.dumps({'task':name,'stage':'expert_dataset_start'}),flush=True)
 r=build_demonstrations(BodySpec(**item['body']),item['path'],**item['settings'])
 records[name]={k:r.get(k) for k in ('fingerprint','qualified','qualification_success_rate','unique_train_samples','teacher_environment_interactions')}
 atomic_json('validation/main-preparation-20260918/datasets.json',records)
 print(json.dumps({'task':name,'stage':'expert_dataset_complete',**records[name],'elapsed':time.monotonic()-start}),flush=True)
