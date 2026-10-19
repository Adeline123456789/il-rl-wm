import sys,json,hashlib,tarfile,traceback,os
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch,numpy as np
from scripts.diagnose_behavior import extract,summarize
out=Path('artifacts/act-t6-evidence-audit-001'); root=Path('artifacts/act-t6-online-seed0-001')
try:
 torch.set_num_threads(2)
 m=json.loads((root/'manifest.json').read_text()); hashes=json.loads((root/'source_hashes.json').read_text())
 assert hashlib.sha256((root/'source_snapshot.tar.gz').read_bytes()).hexdigest()==m['source_snapshot_sha256']
 assert hashlib.sha256((root/'working-tree.patch').read_bytes()).hexdigest()==m['tracked_patch_sha256']
 with tarfile.open(root/'source_snapshot.tar.gz') as archive:
  assert set(archive.getnames())==set(hashes)
  for name,digest in hashes.items(): assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest
 data,source=extract(Path('checkpoints/act-t6-online-seed0-001/final.pt'))
 records=[json.loads(x) for x in (root/'checkpoints.jsonl').read_text().splitlines()]
 assert source['sha256']==next(x['sha256'] for x in records if x['final'])
 np.savez_compressed(out/'transitions.npz',**data)
 result=summarize(data,8192)
 with np.load(out/'transitions.npz') as z: rebuilt=summarize(dict(z),8192)
 assert result==rebuilt
 (out/'behavior.json').write_text(json.dumps(result,indent=2)+'\n')
 (out/'source.json').write_text(json.dumps(source,indent=2)+'\n')
 counts=dict(positive_reward_transitions=int((data['reward']>0).sum()),grasped_observation_transitions=int(data['grasped'].sum()),unique_grasped_episodes=len(np.unique(data['episode_id'][data['grasped']])),successful_episodes=len(np.unique(data['episode_id'][data['reward']>0])),episodes_seen=len(np.unique(data['episode_id'])))
 (out/'result.json').write_text(json.dumps(dict(exit_code=0,pid=os.getpid(),source_files_verified=len(hashes),actual_code_commit=m['code_commit'],dirty_source_captured=True,portable_bins_equal=True,new_environment_interactions=0,counts=counts),indent=2)+'\n')
except BaseException:
 (out/'exit.json').write_text(json.dumps(dict(exit_code=1,pid=os.getpid())))
 raise
else: (out/'exit.json').write_text(json.dumps(dict(exit_code=0,pid=os.getpid())))
