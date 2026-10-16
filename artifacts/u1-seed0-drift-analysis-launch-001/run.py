import json,subprocess
from pathlib import Path
import numpy as np
out=Path('artifacts/u1-seed0-drift-analysis-launch-001');protocol=json.loads((out/'protocol.json').read_text());completed=[]
try:
 for command in protocol['commands']:
  subprocess.run(command,check=True);completed.append(command)
 with np.load('artifacts/u1-seed0-own-input-drift-001/values.npz') as full,np.load('artifacts/u1-seed0-own-input-drift-portable-001/values.npz') as portable:
  assert set(full.files)==set(portable.files)
  assert all(np.array_equal(full[k],portable[k]) for k in full.files)
 (out/'result.json').write_text(json.dumps(dict(valid=True,all_arrays_bitexact=True,completed_commands=completed,new_environment_interactions=0),indent=2)+'\n')
finally:
 (out/'completed_commands.json').write_text(json.dumps(completed,indent=2)+'\n')
