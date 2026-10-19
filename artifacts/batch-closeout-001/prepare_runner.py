import subprocess,json,os,time
from pathlib import Path
p=Path('artifacts/batch-closeout-001');start=time.time();r=subprocess.run(['python','/tmp/act-stage22.py']);(p/'prepare_status.json').write_text(json.dumps(dict(pid=os.getpid(),exit_code=r.returncode,elapsed_seconds=time.time()-start)));raise SystemExit(r.returncode)
