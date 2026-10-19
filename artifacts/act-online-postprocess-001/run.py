import hashlib,json,os,subprocess,time
from pathlib import Path
out=Path('artifacts/act-online-postprocess-001')
state=dict(pid=os.getpid(),phase='waiting',completed=[],current=None,exit_code=None,GPU_use=False)
def save():
    temp=out/'status.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(out/'status.json')
try:
    deadline=time.monotonic()+10800
    for number in [4,5,6]:
        name=f'act-t{number}-online-seed0-001';state['current']=name;state['phase']='waiting';save()
        manifest=Path('artifacts')/name/'manifest.json'
        while True:
            queue=json.loads(Path('artifacts/act-online-batch-002/status.json').read_text())
            assert queue['phase']!='failed', 'GPU queue failed; do not restart'
            if manifest.exists():
                run=json.loads(manifest.read_text())
                if run['ended_at'] is not None:
                    assert run['exit_code']==0 and run['stop_reason'] is None
                    break
            assert time.monotonic()<deadline, 'Postprocessing wait limit; no retries'
            time.sleep(15)
        state['phase']='processing';save()
        commands=[
            ['.venv/bin/python','scripts/summarize_act_online.py','--output',f'artifacts/act-online-summary-00{number-1}'],
            ['.venv/bin/python','scripts/export_act_online_actor.py','--checkpoint',f'checkpoints/{name}/final.pt','--output',f'releases/act-t{number}-online-final-001'],
            ['.venv/bin/python','scripts/diagnose_act_online_drift.py','--experiment',name,'--steps','16384','65536','131072','262144','--output',f'artifacts/act-t{number}-online-drift-001']]
        for command in commands:
            subprocess.run(command,check=True)
        state['completed'].append(name);save()
    state.update(phase='completed',current=None,exit_code=0)
except BaseException as error:
    state.update(phase='failed',exit_code=1,error=repr(error));raise
finally:save()
