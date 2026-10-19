import json,subprocess,sys
from pathlib import Path
out=Path('artifacts/act-online-behavior-audit-002')
cmd=['.venv/bin/python',str(out/'recompute.py'),'--input',str(out),'--output',str(out/'recomputed-counts.json')]
code=subprocess.call(cmd);(out/'recompute-status.json').write_text(json.dumps(dict(command=cmd,exit_code=code,GPU_use=False),indent=2)+'\n');sys.exit(code)
