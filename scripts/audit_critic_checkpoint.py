"""Verify the real U1 C smoke: learned normalization, frozen IL and raw replay."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.train_sac import SoftQNetwork


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2)
    state=torch.load(args.checkpoint,map_location='cpu');config=state['config']
    assert config['critic_layer_norm'] and config['wm_checkpoint'] is None
    replay=state['replay'];assert replay['actions'].shape[:2].numel()==state['global_step']
    env=SimpleNamespace(single_observation_space=SimpleNamespace(shape=(replay['obs'].shape[-1],)),
        single_action_space=SimpleNamespace(shape=(replay['actions'].shape[-1],)))
    norms={}
    for name in ['qf1','qf2','qf1_target','qf2_target']:
        model=SoftQNetwork(env,layer_norm=True);model.load_state_dict(state[name])
        layers=[m for m in model.modules() if isinstance(m,torch.nn.LayerNorm)]
        assert len(layers)==3 and all(m.eps==1e-6 for m in layers)
        assert all(torch.isfinite(v).all() for v in model.parameters())
        norms[name]=[float((m.weight-1).abs().max()) for m in layers]
        assert all(delta>0 for delta in norms[name])
    assert all(int(s['step'])==state['global_update'] for s in state['q_optimizer']['state'].values())
    bc=torch.load(config['bc_checkpoint'],map_location='cpu')
    assert all(torch.equal(v,bc['actor'][k]) for k,v in state['frozen_il'].items())
    for name in ['obs','next_obs','actions','rewards']:
        assert torch.isfinite(replay[name]).all()
    assert replay['actions'].abs().max()<=1.00001
    assert torch.equal(replay['rewards'],replay['raw_terminated'].float())
    assert not replay['dones'].any()
    truncated=replay['raw_truncated'][:-1]
    assert truncated.any()
    different=(replay['next_obs'][:-1]-replay['obs'][1:]).abs().amax(-1)>1e-6
    assert different[truncated].all(), 'Physical final observations must not be reset observations'
    report=dict(valid=True,training_interactions=state['global_step'],critic_updates=state['global_update'],
        frozen_il_unchanged=True,normalization_weight_max_changes=norms,
        truncated_final_observations_distinct_from_reset=int(truncated.sum()),
        sparse_labels_equal=True,continuing_masks_zero=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
