import json,subprocess,sys,os
from pathlib import Path
out=Path('artifacts/act-u2-smoke-launch-001')
p=json.loads(Path('configs/act_u2_smoke_plan.json').read_text())
used=sum(json.loads(f.read_text())['elapsed_seconds'] for f in Path('artifacts').glob('*/manifest.json'))
assert 86400-used-7200>=p['seconds']+p['full_cap_seconds']
cmd=[sys.executable,'scripts/run_experiment.py','--id',p['id'],'--seconds',str(p['seconds']),'--',*p['command']]
r=subprocess.run(cmd)
(out/'exit.json').write_text(json.dumps(dict(pid=os.getpid(),command=cmd,exit_code=r.returncode)))
raise SystemExit(r.returncode)
