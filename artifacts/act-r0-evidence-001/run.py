import hashlib,json,os,tarfile,time
from pathlib import Path
OUT=Path('artifacts/act-r0-evidence-001')
state=dict(pid=os.getpid(),phase='running',started_at=time.time(),exit_code=None)
def read(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
try:
 (OUT/'status.json').write_text(json.dumps(state))
 name='act-r0-expert50-seed0-001'; folder=Path('artifacts')/name
 m=read(folder/'manifest.json'); assert m['exit_code']==0 and m['ended_at'] and m['stop_reason'] is None
 assert sha(folder/'source_snapshot.tar.gz')==m['source_snapshot_sha256']
 assert sha(folder/'working-tree.patch')==m['tracked_patch_sha256']
 hashes=read(folder/'source_hashes.json')
 with tarfile.open(folder/'source_snapshot.tar.gz') as archive:
  members={x.name:x for x in archive.getmembers() if x.isfile()}
  assert set(members)==set(hashes)
  for path,expected in hashes.items(): assert hashlib.sha256(archive.extractfile(members[path]).read()).hexdigest()==expected,path
 export=read('releases/act-r0-expert50-final-001/manifest.json')
 assert sha(export['source'])==export['source_sha256']
 assert sha(export['portable'])==export['portable_sha256']
 assert export['all_actor_tensors_equal'] and not export['includes_teacher']
 old=Path('artifacts/act-t2-online-seed0-002')
 a,b=read(folder/'actor_initialization.json'),read(old/'actor_initialization.json')
 for key in ['actor_hash','critic1_hash','critic2_hash']: assert a[key]==b[key]
 x,y=read(folder/'sac_config.json'),read(old/'sac_config.json')
 differences={k:[y.get(k),x.get(k)] for k in set(x)|set(y) if x.get(k)!=y.get(k)}
 assert set(differences)=={'exp_name','output','expert_audit','expert_dataset'}
 sampling=read(folder/'sampling.json'); assert sampling==dict(expert_samples=33038336,online_samples=33038336,expert_fraction=.5)
 plan=read('configs/act_initialization_budget.json'); assert m['command']==plan['jobs'][name]['command']
 assert m['budget_id']==plan['budget_id'] and sha('configs/act_initialization_budget.json')==m['budget_authorization_sha256']
 summary=read('artifacts/act-initialization-analysis-001/summary-001/summary.json')
 row=next(r for r in summary['conditions'] if r['id']==name)
 assert row['checkpoint_sha256']==export['source_sha256']
 result=dict(experiment=name,source_files_verified=len(hashes),manifest_sha256=sha(folder/'manifest.json'),
  source_checkpoint_sha256=export['source_sha256'],portable_sha256=export['portable_sha256'],
  initial_actor_and_critics_match_T2=True,config_differences_from_T2=differences,sampling=sampling,
  original_budget_seconds=read('research_state.json')['budget']['gpu_job_seconds_used'],
  result=row,scope='Seed0 development only; no new environment steps or optimization; T6 difference not statistical proof',
  queue_snapshot=read('artifacts/act-initialization-batch-002/status.json'))
 assert result['original_budget_seconds']==85238.75034914288
 (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 state.update(phase='completed',exit_code=0)
except BaseException as error:
 state.update(phase='failed',exit_code=1,error=repr(error));raise
finally:
 state['ended_at']=time.time();(OUT/'status.json').write_text(json.dumps(state,indent=2)+'\n')
