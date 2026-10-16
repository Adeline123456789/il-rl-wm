"""PickCube-only audit of initial grasp flags in a saved real SAC replay."""
import argparse
import json
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
torch.set_num_threads(2)
s=torch.load(args.checkpoint,map_location='cpu');assert s['config']['env_id']=='PickCube-v1'
r=s['replay'];mask=r['timesteps']==0;obs=r['obs'][mask];grasped=obs[:,18]>0.5;distance=obs[:,36:39].norm(dim=1)
result=dict(checkpoint=str(args.checkpoint),global_step=s['global_step'],initial_observations=len(obs),grasped_true=int(grasped.sum()),grasped_true_and_tcp_distance_over_0_1m=int((grasped&(distance>.1)).sum()),min_tcp_distance=float(distance.min()),note='Dim18 follows locked PickCube state dict agent qpos/qvel then extra is_grasped. This checks returned observations, not a learned reward or policy score.')
args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
