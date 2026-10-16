import json,subprocess
from pathlib import Path
p=Path('artifacts/u1-seed0-behavior-launch-001')
for cmd in json.loads((p/'commands.json').read_text()):subprocess.run(cmd,check=True)
a=json.loads(Path('artifacts/u1-seed0-behavior-001/result.json').read_text());b=json.loads(Path('artifacts/u1-seed0-behavior-portable-001/result.json').read_text());assert a==b
(p/'result.json').write_text(json.dumps(dict(valid=True,portable_summary_bitexact=True,new_environment_interactions=0),indent=2)+'\n')
