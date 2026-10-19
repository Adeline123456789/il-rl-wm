import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from scripts.diagnose_behavior import extract,summarize
out=Path('artifacts/act-online-behavior-audit-002');state=dict(phase='running',completed=[],exit_code=None,GPU_use=False)
try:
 for name in ['act-t5-online-seed0-001']:
  folder=Path('artifacts')/name;manifest=json.loads((folder/'manifest.json').read_text())
  assert manifest['ended_at'] and manifest['exit_code']==0 and manifest['stop_reason'] is None
  records=[json.loads(line) for line in (folder/'checkpoints.jsonl').read_text().splitlines()]
  source=next(row for row in records if row['final'])
  path=Path(source['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
  data,provenance=extract(path);assert (data['source']==3).all()
  config=json.loads((folder/'sac_config.json').read_text());assert config['learning_starts']==4096 and not config['resume_checkpoint']
  destination=out/name;destination.mkdir();np.savez_compressed(destination/'transitions.npz',**data)
  (destination/'source.json').write_text(json.dumps(provenance,indent=2)+'\n')
  result=summarize(data,8192)
  with np.load(destination/'transitions.npz') as stored:
   rebuilt=summarize({key:stored[key] for key in stored.files},8192)
  assert result==rebuilt
  result['source_label_caveat']='Code3 is the bare-SAC collector route, including its first4096 uniform random warmup actions. It does not mean all actions came from the actor.'
  result['uniform_warmup_transitions']=4096;result['learned_policy_transitions']=262144-4096
  result['portable_recomputation_equal']=True
  (destination/'result.json').write_text(json.dumps(result,indent=2)+'\n')
  state['completed'].append(name);(out/'status.json').write_text(json.dumps(state,indent=2)+'\n')
 state.update(phase='completed',exit_code=0)
except BaseException as error:
 state.update(phase='failed',exit_code=1,error=repr(error));raise
finally:(out/'status.json').write_text(json.dumps(state,indent=2)+'\n')
