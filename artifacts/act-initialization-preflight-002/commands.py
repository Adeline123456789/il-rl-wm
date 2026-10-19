import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import tyro
from scripts.train_sac import Args,validate_args
p=json.load(open('configs/act_initialization_execution_plan.json'));a=json.load(open(p['authorization']))
for name in p['smoke_jobs']+p['training_jobs']:
 args=tyro.cli(Args,args=a['jobs'][name]['command'][2:]);validate_args(args)
 for key,value in p['expected_configs'][name].items():assert getattr(args,key)==value,(name,key)
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in p['runtime_sources'].items())
assert sum(j['cap_seconds'] for j in a['jobs'].values())==11500
assert not json.load(open(p['test_gate']))['released']
print('Five exact registered training commands parsed/validated; source hashes match; final gate closed; zero simulator interactions.')
