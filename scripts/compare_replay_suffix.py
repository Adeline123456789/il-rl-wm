"""Compare old contaminated recordings' source-length suffix against repaired replay."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    old=json.loads(args.before.with_suffix('.json').read_text());new=json.loads(args.after.with_suffix('.json').read_text());by_seed={int(e['episode_seed']):e['episode_id'] for e in old['episodes']};rows=[]
    with h5py.File(args.before) as a,h5py.File(args.after) as b:
        for e in new['episodes']:
            x=a[f"traj_{by_seed[int(e['episode_seed'])]}" ];y=b[f"traj_{e['episode_id']}"];n=len(y['actions']);errors={};exact={}
            for key in ['obs','actions','rewards','terminated','truncated']:
                count=n+1 if key=='obs' else n
                left,right=x[key][-count:],y[key][:]
                assert left.shape==right.shape
                exact[key]=bool(np.array_equal(left,right));errors[key]=float(np.abs(left.astype(float)-right.astype(float)).max())
            rows.append(dict(seed=int(e['episode_seed']),removed_prefix_steps=len(x['actions'])-n,exact=exact,max_abs_error=errors))
    result=dict(before=str(args.before),after=str(args.after),episodes=rows,all_suffixes_bit_exact=all(all(r['exact'].values()) for r in rows),removed_prefix_steps=sum(r['removed_prefix_steps'] for r in rows),max_abs_error={k:max(r['max_abs_error'][k] for r in rows) for k in errors})
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='episodes'}))


if __name__=='__main__': main()
