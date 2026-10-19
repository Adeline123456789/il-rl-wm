import hashlib,json,os,tarfile
from pathlib import Path
out=Path('artifacts/act-u2-comparison-audit-001')
def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
try:
 root=Path('artifacts/act-u2-retention-seed0-001');parent=Path('artifacts/act-t5-online-seed0-001')
 m=read(root/'manifest.json');assert m['ended_at'] and m['exit_code']==0 and m['stop_reason'] is None
 cfg=read(root/'sac_config.json');old=read(parent/'sac_config.json')
 assert all(cfg[k]==v for k,v in old.items() if k not in ['output','exp_name'])
 assert set(cfg)-set(old)=={'retention_config'} and cfg['retention_config']=='configs/act_u2_retention.json'
 assert read(root/'actor_initialization.json')==read(parent/'actor_initialization.json')
 counter=read(root/'retention_result.json')
 assert counter['updates']==15360 and counter['samples']==983040 and counter['final_coefficient']==0
 assert not counter['teacher_calls_in_sac_updates'] and not counter['teacher_required_at_deployment']
 labels=read(root/'retention_manifest.json');smoke=read('artifacts/act-u2-retention-smoke-001/retention_manifest.json')
 assert labels['states']==55092 and labels['teacher_frozen_verified'] and not labels['expert_transitions_in_critic']
 assert labels['labels_sha256']==smoke['labels_sha256']==digest(root/'retention_labels.npz')
 dataset=read(labels['config']['audit'])
 assert labels['train_episode_ids']==dataset['train_episode_ids']
 assert not set(labels['train_episode_ids']) & set(dataset['validation_episode_ids'])
 assert labels['dataset_sha256']==dataset['dataset_sha256']
 assert digest(labels['config']['teacher'])==labels['config']['teacher_sha256']
 assert read(root/'sampling.json')['expert_samples']==0
 metrics=rows(root/'metrics.jsonl');retention_rows=[x for x in metrics if x['metric'].startswith('retention/')]
 for x in retention_rows:
  step=x['step'];key=x['metric']
  if key=='retention/coefficient':assert x['value']==10*max(0.,1-step/65536)
  if key=='retention/samples':assert x['value']==max(0,(min(step,65472)-4096)//64+1)*16*64
  if key=='retention/mse' and step>=65536:assert x['value']==0
 assert len(retention_rows)==32*3
 a=[x for x in rows(root/'validation_episodes.jsonl') if x['step']==0]
 b=[x for x in rows(parent/'validation_episodes.jsonl') if x['step']==0]
 assert a==b and len(a)==40
 assert digest(root/'source_snapshot.tar.gz')==m['source_snapshot_sha256']
 assert digest(root/'working-tree.patch')==m['tracked_patch_sha256']
 hashes=read(root/'source_hashes.json')
 with tarfile.open(root/'source_snapshot.tar.gz') as archive:
  assert set(archive.getnames())==set(hashes)
  for name,value in hashes.items():assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==value
 summary=read('artifacts/act-online-summary-006/summary.json')
 assert len(summary['conditions'])==6 and all(x['status']=='completed' for x in summary['conditions'])
 curves=[x for x in summary['curves'] if x['id'] in [root.name,parent.name]]
 drift=read('artifacts/act-u2-online-drift-001/result.json')
 behavior=read('artifacts/act-u2-online-behavior-001/result.json')
 export=read('releases/act-u2-online-final-001/manifest.json')
 assert export['portable_sha256']==digest('releases/act-u2-online-final-001/weights.pt')
 assert export['all_actor_tensors_equal'] and not export['includes_teacher']
 (out/'result.json').write_text(json.dumps(dict(exit_code=0,source_files_verified=len(hashes),actual_code_commit=m['code_commit'],initial_episode_records_exact_match=True,retention_updates=15360,retention_samples=983040,retention_disabled_after_step=65536,source_config_single_mechanism_verified=True,teacher_training_split_only=True,portable=export,curves=curves,drift=drift['results'],behavior=behavior,new_environment_interactions=0,final_test_accessed=False,limitations='One development seed. Disabled retention/mse=0 is a zero auxiliary term, not measured action error. Fixed-state drift reports actual error separately. No causal isolation of other failure factors or ACT-WM evidence.'),indent=2)+'\n')
 (out/'exit.json').write_text(json.dumps(dict(pid=os.getpid(),exit_code=0)))
except BaseException as e:
 (out/'exit.json').write_text(json.dumps(dict(pid=os.getpid(),exit_code=1,error=repr(e))))
 raise
