"""Finite U1 comparison from the same BC as its seed's original U0."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_pickcube_repeat import plan as original_plan,ROOT,training_seconds_left


def plan(seed):
    jobs=[job for job in original_plan(seed) if any(job['id']==f'pick-seed{seed}-{name}-001'
          for name in ['c','wm','value','u0','c-continue'])]
    replacements={job['id']:'u1-'+job['id'] for job in jobs}
    if seed==0:replacements['pick-seed0-bc-001']='e1-bc-full-003'
    for job in jobs:
        job['id']=replacements[job['id']]
        for i,arg in enumerate(job['command']):
            for original,new in replacements.items():
                if arg==original or arg.endswith('/'+original) or '/'+original+'/' in arg:
                    arg=arg.replace(original,new)
            job['command'][i]=arg
        if job['command'][1]=='scripts/train_sac.py':job['command'].append('--critic-layer-norm')
    return jobs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[0,1,2,3,4],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args();jobs=plan(args.seed)
    if args.plan_only:args.output.write_text(json.dumps(jobs,indent=2)+'\n');return
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'plan.json').write_text(json.dumps(jobs,indent=2)+'\n')
    started=time.monotonic();record=dict(seed=args.seed,phase='running',completed=[],current=None,limit_seconds=5400)
    def save():
        temp=args.output/'status.tmp';temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(args.output/'status.json')
    try:
        for job in jobs:
            remaining=min(5400-(time.monotonic()-started),training_seconds_left())
            assert remaining>0,'U1 seed or training budget exhausted; preserve final evaluation reserve'
            record['current']=job['id'];save()
            subprocess.run([sys.executable,'scripts/run_experiment.py','--id',job['id'],
                '--seconds',str(min(job['seconds'],remaining)),'--',*job['command']],cwd=ROOT,check=True)
            record['completed'].append(job['id'])
        record.update(phase='completed',current=None)
    except BaseException as error:
        record.update(phase='failed',error=repr(error));raise
    finally:
        record['elapsed_seconds']=time.monotonic()-started;save()


if __name__=='__main__':main()
