import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import tyro
from scripts.train_sac import Args,validate_args
p=json.load(open('configs/act_initialization_execution_plan_002.json'));a=json.load(open(p['authorization']))
for name in p['smoke_jobs']+p['training_jobs']:
 args=tyro.cli(Args,args=a['jobs'][name]['command'][2:]);validate_args(args)
 for key,value in p['expected_configs'][name].items():assert getattr(args,key)==value,(name,key)
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in p['runtime_sources'].items())
assert sum(j['cap_seconds'] for j in a['jobs'].values())==11500
assert not json.load(open(p['test_gate']))['released']
print('Five exact registered training commands parsed/validated; source hashes match; final gate closed; zero simulator interactions.')

from scripts.run_initialization_controls import project_full_duration
f=Path("artifacts/act-r0-expert50-smoke-001");m=json.load(open(f/"manifest.json"));r=json.load(open(f/"result.json"));rows=[json.loads(x) for x in (f/"metrics.jsonl").read_text().splitlines()];last={x["metric"]:x["value"] for x in rows if x["step"]==8192};projection=project_full_duration(m["elapsed_seconds"],r["elapsed_seconds"],last["time/total_update_time"],last["time/total_rollout_time"]);projection["with_35_percent_plus60"]=projection["projected_full_seconds"]*1.35+60;assert projection["with_35_percent_plus60"]<=1900;Path("artifacts/act-initialization-preflight-003/resource_reconciliation.json").write_text(json.dumps(projection,indent=2)+"\n");print(projection)
