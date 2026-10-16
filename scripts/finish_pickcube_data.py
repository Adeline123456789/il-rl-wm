"""Finite, fail-closed continuation of a completed or running CPU replay."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / '.venv/bin/python')


def main(args, status):
    manifest = ROOT / 'artifacts' / args.cpu_id / 'manifest.json'
    status('waiting_for_existing_cpu_replay')
    deadline = time.monotonic() + 1800
    while True:
        data = json.loads(manifest.read_text())
        if data['ended_at'] is not None:
            assert data['exit_code'] == 0, data
            break
        assert time.monotonic() < deadline, 'CPU replay did not finish within continuation wait budget'
        time.sleep(5)
    source = ROOT / 'data/replay' / args.cpu_id / 'trajectory.state.pd_ee_delta_pos.physx_cpu'
    assert len(json.loads(Path(str(source)+'.json').read_text())['episodes']) == 1000
    target = ROOT / 'data/replay' / args.gpu_id
    target.mkdir(exist_ok=False)
    for ext in ['h5','json']:
        shutil.copy2(str(source)+'.'+ext, target/f'trajectory.{ext}')
    status('gpu_replay')
    subprocess.run([PYTHON,'scripts/run_experiment.py','--id',args.gpu_id,'--seconds','900','--',PYTHON,'scripts/replay_recorded.py',f'artifacts/{args.gpu_id}/interactions.json','--traj-path',str(target/'trajectory.h5'),'--use-first-env-state','--target-control-mode','pd_ee_delta_pos','--obs-mode','state','--save-traj','--num-envs','32','--sim-backend','physx_cuda','--reward-mode','sparse','--record-rewards','--count','1000','--allow-failure'],cwd=ROOT,check=True)
    status('audit')
    subprocess.run([PYTHON,'scripts/audit_demos.py','--path',str(target/'trajectory.state.pd_ee_delta_pos.physx_cuda.h5'),'--action-source-path',str(source)+'.h5','--horizon','200','--successful-only','--source-metadata','data/official/PickCube-v1/motionplanning/trajectory.json','--output',f'artifacts/{args.gpu_id}/dataset-audit.json'],cwd=ROOT,check=True)
    audit=json.loads((ROOT/'artifacts'/args.gpu_id/'dataset-audit.json').read_text())
    assert len(audit['episodes']) == 1000
    assert len(set(audit['source_episode_mapping'].values())) == 1000
    assert audit['alignment_validated']
    status('completed', eligible_episodes=len(audit['train_episode_ids'])+len(audit['validation_episode_ids']), excluded_episodes=len(audit['excluded_episode_ids']))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu-id',default='e0-pickcube-replay-full-001')
    parser.add_argument('--gpu-id',required=True)
    parser.add_argument('--status-path',type=Path,required=True)
    args=parser.parse_args()
    args.status_path.parent.mkdir(exist_ok=True,parents=True)
    assert not args.status_path.exists(), 'Use a new continuation status path'
    def status(phase, **kwargs):
        temporary=args.status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(phase=phase,**kwargs),indent=2))
        temporary.replace(args.status_path)
    try:
        main(args,status)
    except BaseException as error:
        status('failed',error=repr(error))
        raise
