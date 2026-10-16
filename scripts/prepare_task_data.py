"""Finite CPU conversion -> GPU action replay -> audit for the other two tasks.

Run after the current GPU queue. Each real replay goes through the same lock and
24-hour ledger. Stop on any failure; original demonstrations remain untouched.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PYTHON=str(ROOT/'.venv/bin/python')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=['StackCube-v1','PegInsertionSide-v1'],required=True)
    p.add_argument('--status-path',type=Path,required=True)
    p.add_argument('--revision',choices=['001','002'],default='001')
    args=p.parse_args()
    config=json.loads((ROOT/'configs'/('stackcube_protocol.json' if args.task=='StackCube-v1' else 'peg_protocol.json')).read_text())
    tag='stack' if args.task=='StackCube-v1' else 'peg'
    control=config['control_mode']
    args.status_path.parent.mkdir(parents=True,exist_ok=True)
    assert not args.status_path.exists(), 'Use a fresh pipeline status'
    def status(phase,**kwargs):
        temp=args.status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(phase=phase,task=args.task,**kwargs),indent=2))
        temp.replace(args.status_path)
    def run(command):
        subprocess.run(command,cwd=ROOT,check=True)
    try:
        smoke_count=128 if args.task=='PegInsertionSide-v1' else 64
        for count,label in [(smoke_count,'smoke'),(1000,'full')]:
            cpu_id=f'e0-{tag}-cpu-replay-{label}-{args.revision}'
            gpu_id=f'e0-{tag}-gpu-replay-{label}-{args.revision}'
            original=ROOT/'data/official'/args.task/'motionplanning'
            cpu=ROOT/'data/replay'/cpu_id;cpu.mkdir(exist_ok=False)
            for ext in ['h5','json']:
                shutil.copy2(original/f'trajectory.{ext}',cpu/f'trajectory.{ext}')
            common=['--use-first-env-state','--target-control-mode',control,'--obs-mode','state',
                    '--save-traj','--reward-mode','sparse','--record-rewards','--count',str(count),'--allow-failure']
            status('cpu_conversion',experiment=cpu_id,count=count)
            run([sys.executable,'scripts/run_experiment.py','--id',cpu_id,'--seconds','1800','--',
                 PYTHON,'scripts/replay_recorded.py',f'artifacts/{cpu_id}/interactions.json',
                 '--traj-path',str(cpu/'trajectory.h5'),'--num-envs','1','--sim-backend','physx_cpu',*common])
            source=cpu/f'trajectory.state.{control}.physx_cpu.h5'
            assert len(json.loads(source.with_suffix('.json').read_text())['episodes'])==count
            gpu=ROOT/'data/replay'/gpu_id;gpu.mkdir(exist_ok=False)
            shutil.copy2(source,gpu/'trajectory.h5')
            shutil.copy2(source.with_suffix('.json'),gpu/'trajectory.json')
            status('gpu_replay',experiment=gpu_id,count=count)
            run([sys.executable,'scripts/run_experiment.py','--id',gpu_id,'--seconds','900','--',
                 PYTHON,'scripts/replay_recorded.py',f'artifacts/{gpu_id}/interactions.json',
                 '--traj-path',str(gpu/'trajectory.h5'),'--num-envs','32','--sim-backend','physx_cuda',*common])
            audit_path=ROOT/'artifacts'/gpu_id/'dataset-audit.json'
            status('audit',experiment=gpu_id,count=count)
            run([PYTHON,'scripts/audit_demos.py','--path',str(gpu/f'trajectory.state.{control}.physx_cuda.h5'),
                 '--action-source-path',str(source),'--horizon',str(config['max_episode_steps']),
                 '--successful-only','--source-metadata',str(original/'trajectory.json'),'--output',str(audit_path)])
            audit=json.loads(audit_path.read_text())
            assert audit['alignment_validated'] and len(audit['episodes'])==count
            assert len(set(audit['source_episode_mapping'].values()))==count
        status('completed',audit=str(audit_path.relative_to(ROOT)),
               eligible=len(audit['train_episode_ids'])+len(audit['validation_episode_ids']),
               failed_retained=len(audit['excluded_episode_ids']))
    except BaseException as error:
        status('failed',error=repr(error))
        raise


if __name__=='__main__':main()
