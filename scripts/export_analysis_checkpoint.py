"""Export baseline/U0 analysis weights; this is NOT the no-IL deployment package."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def training_interactions(state):
    if 'lineage_training_interactions' in state:
        return state['lineage_training_interactions']
    # The original standalone SAC saver predates continuation metadata.
    # Never infer a continuation's inherited experience from its segment step.
    config = state['config']
    assert not config.get('resume_checkpoint') and not config.get('inherited_training_interactions', 0)
    return state['global_step']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    original=torch.load(args.checkpoint,map_location='cpu')
    keys=['actor','qf1','qf2','qf1_target','qf2_target','log_alpha','frozen_il','config',
          'global_step','global_update']
    state={key:original[key] for key in keys}
    state['lineage_training_interactions']=training_interactions(original)
    if 'u0' in original:
        state['u0']={key:original['u0'][key] for key in ['wm','value','value_target']}
    state['analysis_source']=dict(path=str(args.checkpoint),sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest())
    args.output.mkdir(parents=True,exist_ok=False)
    path=args.output/'weights.pt';torch.save(state,path)
    (args.output/'manifest.json').write_text(json.dumps(dict(path=str(path),bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source=state['analysis_source'],
        env_id=state['config']['env_id'],training_seed=state['config']['seed'],
        lineage_training_interactions=state['lineage_training_interactions'],
        purpose='Frozen diagnosis and independent evaluation. Use separate deployment package for IL-free planning.',
        includes_frozen_il=state['frozen_il'] is not None,
        omitted=['optimizers','replay','RNG'],training_resume_supported=False),indent=2)+'\n')


if __name__=='__main__':main()
