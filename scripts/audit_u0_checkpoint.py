"""Recompute U0 execution/update accounting from saved real replay and weights."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.train_bc import Actor


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2)
    child=torch.load(args.checkpoint,map_location='cpu');parent=torch.load(child['config']['resume_checkpoint'],map_location='cpu')
    replay=child['replay'];first_id=int(parent['replay']['episode_ids'].max())+1
    fresh=replay['episode_ids']>=first_id;n=int(fresh.sum())
    assert n==child['global_step'], 'Audit scoped to new segment fitting inside replay capacity'
    sources=replay['action_sources'][fresh].long();counts=torch.bincount(sources,minlength=3)
    assert len(counts)==3 and torch.equal(counts,child['u0']['executed_source_counts'])
    assert torch.equal(replay['dones'][fresh],torch.zeros(n))
    frozen=child['frozen_il'];assert all(torch.equal(v,parent['frozen_il'][k]) for k,v in frozen.items())
    obs=replay['obs'][fresh];actions=replay['actions'][fresh]
    il=Actor(obs.shape[-1],actions.shape[-1]);il.load_state_dict(frozen)
    with torch.no_grad():
        error=(il(obs[sources==0])-actions[sources==0]).abs().max().item() if (sources==0).any() else None
    assert error is None or error<1e-4
    artifact=Path(child['config']['output']);rows=[json.loads(s) for s in (artifact/'planning.jsonl').read_text().splitlines()]
    for row in rows:
        index=(parent['replay']['pos']+row['step']//replay['num_envs']-1)%replay['capacity']
        assert torch.equal(replay['actions'][index],torch.tensor(row['executed_action'],dtype=torch.float32))
        assert torch.equal(replay['action_sources'][index].long(),torch.tensor(row['executed_source']))
        assert torch.equal(replay['planner_selected'][index].long(),torch.tensor(row['selected']))
    wm_parent=torch.load(child['config']['wm_checkpoint'],map_location='cpu')
    value_parent=torch.load(child['config']['task_value_checkpoint'],map_location='cpu')
    wm_changed=any(not torch.equal(v,wm_parent['model'][k]) for k,v in child['u0']['wm'].items())
    value_changed=any(not torch.equal(v,value_parent['value'][k]) for k,v in child['u0']['value'].items())
    assert wm_changed and value_changed
    assert all(int(s['step'])==wm_parent['updates']+child['u0']['wm_updates'] for s in child['u0']['wm_optimizer']['state'].values())
    assert all(int(s['step'])==value_parent['updates']+child['u0']['value_updates'] for s in child['u0']['value_optimizer']['state'].values())
    report=dict(valid=True,checkpoint=str(args.checkpoint),new_real_transitions=n,executed_il=int(counts[0]),executed_planned_sac=int(counts[1]),
        executed_random_sac=int(counts[2]),actual_exploration_fraction=float(counts[2]/n),logged_batches_exactly_match_replay=len(rows),
        il_action_max_abs_cpu_recompute_error=error,il_weights_unchanged=True,wm_weights_updated=wm_changed,value_weights_updated=value_changed,
        wm_updates=child['u0']['wm_updates'],value_updates=child['u0']['value_updates'],optimizer_steps_preserved=True,
        note='CPU recomputation tolerates floating-point kernel differences; logged executed actions and replay match exactly. This is an engineering audit, not a performance conclusion.')
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
