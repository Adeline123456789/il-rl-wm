"""Six registered 10-environment validation checks; run inside the shared GPU runner."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.run_final_evaluation import ROOT, PYTHON, act_checkpoint_hash


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((ROOT/'configs/act_final_evaluation_plan.json').read_text())
    entries=[e for e in plan['entries'] if e['framework']=='act' and e['training_interactions']==0]
    assert len(entries)==6
    hashes={e['id']:act_checkpoint_hash(e,ROOT) for e in entries}
    state=dict(phase='running',completed=[],current=None,exit_code=None,
               seed=10000,num_envs=10,episodes_per_condition=10,training_interactions=0,final_test_accessed=False)
    def save():
        temp=args.output/'status.tmp';temp.write_text(json.dumps(state,indent=2)+'\n')
        temp.replace(args.output/'status.json')
    reference=None
    try:
        for entry in entries:
            folder=args.output/entry['id'];folder.mkdir(exist_ok=False)
            command=[PYTHON,'scripts/evaluate_act_checkpoint.py','--checkpoint',entry['checkpoint'],
                '--kind',entry['kind'],'--mode',entry['mode'],'--seed','10000',
                '--episodes','10','--num-envs','10','--output',str(folder)]
            state['current']=dict(id=entry['id'],command=command);save()
            with (folder/'command.log').open('w') as stream:
                process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                state['current']['pid']=process.pid;save()
                code=process.wait();state['current']['exit_code']=code;save()
                assert code==0, entry['id']
            result=json.loads((folder/'result.json').read_text())
            assert result['checkpoint_sha256']==hashes[entry['id']]
            assert result['purpose']=='validation' and result['seed']==10000 and result['episodes']==10
            assert result['interactions']==2000 and result['new_training_interactions']==0
            counter=json.loads((folder/'interactions.json').read_text())
            assert counter['total_attempted']==counter['total_completed']==2000
            assert set(counter['phases'])=={'validation'}
            rows=[json.loads(line) for line in (folder/'validation_episodes.jsonl').read_text().splitlines()]
            assert len(rows)==10 and {row['episode_index'] for row in rows}==set(range(10))
            assert all(row['eval_seed']==10000 and row['episode_len']==200 for row in rows)
            with np.load(folder/'final_validation_trace.npz' if entry['mode']!='ensemble'
                         else folder/'final_ensemble_trace.npz') as trace:
                initial=trace['obs'][0].copy()
                assert trace['obs'].shape==(200,10,42) and trace['action'].shape==(200,10,4)
                assert all(np.isfinite(trace[key]).all() for key in ['obs','action','reward'])
            if reference is None:reference=initial
            assert np.array_equal(initial,reference), 'Validation initial states differ'
            state['completed'].append(dict(id=entry['id'],exit_code=code,initial_state_max_error=0,
                validation_interactions=2000,checkpoint_sha256=hashes[entry['id']]))
            save()
        state.update(phase='completed',current=None,exit_code=0,validation_interactions=12000)
    except BaseException as error:
        state.update(phase='failed',error=repr(error),exit_code=1);raise
    finally:save()


if __name__=='__main__':main()
