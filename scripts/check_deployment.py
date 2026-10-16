"""Check exported inference against original no-IL planner on saved real states."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.deployment import DeploymentPolicy
from scripts.load_components import sac_actor,world_model,task_value
from scripts.planning import generate_candidates,score_candidates
from scripts import train_bc


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2)
    original=torch.load(args.checkpoint,map_location='cpu');package=torch.load(args.package,map_location='cpu')
    actor=sac_actor(original,'cpu').eval();wm=world_model({'model':original['u0']['wm']},'cpu');value=task_value({'value':original['u0']['value']},'cpu')
    def forbidden(*args,**kwargs):raise AssertionError('Deployment attempted to construct IL')
    train_bc.Actor.__init__=forbidden
    policy=DeploymentPolicy(package)
    assert not any(p.requires_grad for p in policy.parameters())
    observations=original['replay']['obs'].flatten(0,1)[::4096][:64]
    with torch.no_grad():
        torch.manual_seed(719)
        candidates=generate_candidates(observations,actor,wm,il=None)
        reference=score_candidates(candidates,actor,value)['action']
        torch.manual_seed(719)
        actual=policy.get_eval_action(observations)
    assert torch.equal(actual,reference)
    assert candidates['actions'].shape[1:3]==(9,2)
    assert torch.isfinite(actual).all() and actual.abs().max()<=1
    assert not any('il' in key for key in policy.state_dict())
    args.output.write_text(json.dumps(dict(valid=True,states=len(observations),bit_exact_cpu_action_parity=True,
        no_il_weights=True,il_constructor_forbidden=True,no_trainable_parameters=True,new_environment_interactions=0,
        limitation='Inference parity on saved real states; does not improve or establish task performance.'),indent=2))


if __name__=='__main__':main()
