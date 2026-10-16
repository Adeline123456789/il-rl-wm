import json,subprocess
from pathlib import Path
out=Path('artifacts/u1-no-il-export-001')
for command in json.loads((out/'commands.json').read_text()):subprocess.run(command,check=True)
