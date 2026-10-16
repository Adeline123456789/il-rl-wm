"""Check action identity and initial-state provenance across same-control replay."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
from scripts.replay_provenance import resolve_sources


def compare(source, target):
    sm=json.loads(source.with_suffix('.json').read_text())
    tm=json.loads(target.with_suffix('.json').read_text())
    matched=resolve_sources(sm['episodes'],tm['episodes'])
    rows=[]
    def state_arrays(group, prefix=''):
        out={}
        for key,item in group.items():
            if isinstance(item,h5py.Group): out.update(state_arrays(item,prefix+key+'/'))
            else: out[prefix+key]=item[0]
        return out
    with h5py.File(source) as src,h5py.File(target) as dst:
        for e,original in zip(tm['episodes'],matched):
            s,t=src[f"traj_{original['episode_id']}"],dst[f"traj_{e['episode_id']}"]
            a,b=state_arrays(s['env_states']),state_arrays(t['env_states'])
            state_match=a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a)
            rows.append(dict(target_episode_id=e['episode_id'],source_episode_id=original['episode_id'],seed=int(e['episode_seed']),source_steps=len(s['actions']),target_steps=len(t['actions']),actions_equal=np.array_equal(s['actions'][:],t['actions'][:]),first_state_equal=state_match))
    return dict(source=str(source),target=str(target),episodes=rows,valid=all(r['actions_equal'] and r['first_state_equal'] and r['source_steps']==r['target_steps'] for r in rows),mismatched_episodes=sum(not(r['actions_equal'] and r['first_state_equal'] and r['source_steps']==r['target_steps']) for r in rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=compare(args.source,args.target);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='episodes'}));raise SystemExit(0 if result['valid'] else 1)


if __name__=='__main__': main()
