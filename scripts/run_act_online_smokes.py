"""Two bounded, serial online SAC throughput checks; no full-run auto-enlargement."""
import json
from pathlib import Path
import subprocess
import sys
import os
import time


def command(name,kind,path):
    return [sys.executable,'scripts/train_sac.py','--output','artifacts/'+name,'--exp-name',name,
        '--env-id','PickCube-v1','--seed','0','--control-mode','pd_ee_delta_pos','--max-episode-steps','200',
        '--total-timesteps','8192','--num-envs','16','--num-eval-envs','8','--num-eval-steps','1000',
        '--training-freq','64','--utd','0.25','--batch-size','1024','--buffer-size','262144',
        '--learning-starts','4096','--eval-freq','8192','--log-freq','4096','--gamma','0.99',
        '--no-capture-video','--no-track','--critic-layer-norm','--initial-log-std','-2',
        '--matched-eval-resets','--actor-init-kind',kind,'--actor-init-checkpoint',path]


def main():
    out=Path('artifacts/act-online-smokes-001');out.mkdir(exist_ok=False)
    state=dict(pid=os.getpid(),phase='running',completed=[],started_at=time.time())
    def save(): (out/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        assert json.loads(Path('artifacts/act-online-tests-001/exit.json').read_text())['exit_code']==0
        c=json.loads(Path('artifacts/act-c1-seed0-001/result.json').read_text())
        assert c['teacher_unchanged'] and c['methods']['transfer_mean']['success_once']>=8
        for name,kind,path in [('act-t5-online-smoke-001','act','releases/act-c1-init-001/weights.pt'),
                               ('act-t4-online-smoke-001','mlp','releases/act-d1-negative-001/weights.pt')]:
            budget=json.loads(Path('research_state.json').read_text())['budget']
            assert budget['gpu_hours_limit']*3600-budget['gpu_job_seconds_used']-budget['final_evaluation_reserve_seconds']>=600
            cmd=command(name,kind,path);state['current']=name;save()
            subprocess.run([sys.executable,'scripts/run_experiment.py','--id',name,'--seconds','600','--',*cmd],check=True)
            state['completed'].append(name)
        state.update(phase='completed',exit_code=0,next='Inspect throughput/skill loss before fixing equal full comparison budget; no full runs automatically scheduled')
    except BaseException as e:
        state.update(phase='failed',exit_code=1,error=repr(e));raise
    finally:
        state['ended_at']=time.time();save()

if __name__=='__main__':main()
