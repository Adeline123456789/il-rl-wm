import json,subprocess,time,sys
from pathlib import Path
out=Path('artifacts/act-t3-postprocess-001')
try:
    deadline=time.monotonic()+240
    while True:
        manifest=json.loads(Path('artifacts/act-t3-online-seed0-001/manifest.json').read_text())
        if manifest['ended_at'] is not None:
            assert manifest['exit_code']==0 and manifest['stop_reason'] is None
            break
        assert time.monotonic()<deadline, 'Wait limit; do not restart training'
        time.sleep(5)
    for command in [
        ['.venv/bin/python','scripts/summarize_act_online.py','--output','artifacts/act-online-summary-002'],
        ['.venv/bin/python','scripts/export_act_online_actor.py','--checkpoint','checkpoints/act-t3-online-seed0-001/final.pt','--output','releases/act-t3-online-final-001']]:
        subprocess.run(command,check=True)
    code=0
except BaseException as error:
    print(repr(error),flush=True);code=1
(out/'exit.json').write_text(json.dumps(dict(exit_code=code)))
sys.exit(code)
