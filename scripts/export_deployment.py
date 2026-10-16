"""Export the fixed checkpoint's no-IL inference weights with hashes and protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    state=torch.load(args.checkpoint,map_location='cpu')
    protocol={k:state['config'][k] for k in ['env_id','control_mode','max_episode_steps','gamma','reconfiguration_freq']}
    protocol.update(obs_mode='state',reward_mode='sparse',sac_candidates=9,horizon=2,execute_actions=1,
                    objective='pure task return with deterministic student continuation',
                    observation_normalization='stored WM/value statistics; actor raw state',
                    action_bounds=[-1,1],candidate_sampling='stochastic SAC, externally seed Torch for repeatability',
                    calls_il=False)
    source=dict(checkpoint=str(args.checkpoint),sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                lineage_training_interactions=state['lineage_training_interactions'],
                warning='Exports fixed final checkpoint regardless of performance; export is not evidence of successful behavior.')
    package=dict(actor=state['actor'],wm=state['u0']['wm'],value=state['u0']['value'],protocol=protocol,source=source)
    path=args.output/'policy.pt';torch.save(package,path)
    manifest=dict(protocol=protocol,source=source,path=str(path),bytes=path.stat().st_size,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  includes=['SAC actor','world model','pure task value'],excludes=['IL','training optimizers','replay'],
                  intended_use='inference; original full training checkpoint remains separate')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
