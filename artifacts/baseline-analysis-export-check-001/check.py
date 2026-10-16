import json,subprocess
from pathlib import Path
import torch
from scripts.evaluate_checkpoint import load_policies
from scripts import train_bc
source=Path('checkpoints/e3-sac-c-continue-full-001/final.pt')
out=Path('releases/c-continue-analysis-final-001')
subprocess.run(['.venv/bin/python','scripts/export_analysis_checkpoint.py','--checkpoint',str(source),'--output',str(out)],check=True)
torch.set_num_threads(2)
a=torch.load(source,map_location='cpu');b=torch.load(out/'weights.pt',map_location='cpu')
assert 'u0' not in b and not any(k in b for k in ['replay','rng','actor_optimizer','q_optimizer'])
for key in ['actor','qf1','qf2','qf1_target','qf2_target','frozen_il']:
 assert all(torch.equal(v,b[key][k]) for k,v in a[key].items())
def forbidden(*args,**kwargs):raise AssertionError('SAC evaluation constructed IL')
train_bc.Actor.__init__=forbidden
pa=load_policies(a,['sac'],'cpu')['sac'];pb=load_policies(b,['sac'],'cpu')['sac']
obs=a['replay']['obs'].flatten(0,1)[::4096][:64]
with torch.no_grad():
 aa=pa.get_eval_action(obs);bb=pb.get_eval_action(obs)
assert torch.equal(aa,bb)
Path('artifacts/baseline-analysis-export-check-001/result.json').write_text(json.dumps(dict(valid=True,states=len(obs),max_abs_action_error=float((aa-bb).abs().max()),all_learned_parameters_equal=True,no_learning_state=True,il_construction_forbidden=True),indent=2))
