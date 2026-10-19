import subprocess,json,os
from pathlib import Path
cmd=['.venv/bin/python','releases/source-history-act-20261019-001/restore.py','.cache/act-source-history-release-check-001']
r=subprocess.run(cmd)
Path('artifacts/act-source-history-001/release_restore_exit.json').write_text(json.dumps(dict(pid=os.getpid(),command=cmd,exit_code=r.returncode)))
raise SystemExit(r.returncode)
