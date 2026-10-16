import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import torch
from scripts.deployment import DeploymentPolicy
from scripts.evaluate_checkpoint import load_policies
from scripts import train_bc
out=Path('artifacts/u1-portable-no-il-check-001');torch.set_num_threads(2)
package_path=Path('releases/u1-final-001/policy.pt');analysis_path=Path('releases/u1-seed0-analysis-final-001/weights.pt')
def forbidden(*args,**kwargs):raise AssertionError('IL constructor called')
train_bc.Actor.__init__=forbidden
policy=DeploymentPolicy(torch.load(package_path,map_location='cpu'))
reference=load_policies(torch.load(analysis_path,map_location='cpu'),['sac_wm'],'cpu')['sac_wm']
with np.load('artifacts/u1-seed0-own-input-drift-001/fixed_batches.npz') as data:obs=torch.from_numpy(data['online_obs'][:64].copy())
with torch.no_grad():
 torch.manual_seed(719);expected=reference.get_eval_action(obs)
 torch.manual_seed(719);actual=policy.get_eval_action(obs)
assert torch.equal(actual,expected) and torch.isfinite(actual).all() and actual.abs().max()<=1
assert not any(p.requires_grad for p in policy.parameters())
np.savez_compressed(out/'actions.npz',observations=obs.numpy(),actual=actual.numpy(),expected=expected.numpy())
(out/'result.json').write_text(json.dumps(dict(valid=True,states=64,bitexact=True,il_constructor_forbidden=True,original_large_checkpoint_required=False,new_environment_interactions=0,source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [package_path,analysis_path]}),indent=2)+'\n')
