import json,numpy as np
from pathlib import Path
from scripts.diagnose_behavior import summarize
for label in ['u0','control']:
 with np.load(f'artifacts/f-u0-behavior-{label}-001/transitions.npz') as f:data={key:f[key] for key in f.files}
 for size,target in [(65536,f'artifacts/f-u0-behavior-{label}-001/result.json'),(8192,f'artifacts/f-u0-behavior-{label}-fine-001/result.json')]:
  assert summarize(data,size)==json.loads(Path(target).read_text())
Path('artifacts/behavior-portable-regression-001/result.json').write_text(json.dumps(dict(valid=True,models=2,bin_sizes=[65536,8192],all_statistics_equal=True,new_environment_interactions=0),indent=2))
