import datetime as dt,hashlib,json,os,subprocess,sys,tarfile,time
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root))
from scripts.summarize_final_results import summarize
out=Path('artifacts/act-final-completion-postprocess-001');python=str(root/'.venv/bin/python')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=dict(pid=os.getpid(),phase='waiting',exit_code=None,started_at=time.time())
def save():(out/'status.json').write_text(json.dumps(s,indent=2)+'\n')
save()
try:
 deadline=time.monotonic()+360
 while read('artifacts/act-final-completion-001/status.json')['phase']=='running':
  assert time.monotonic()<deadline,'Completion did not settle within bounded wait'
  time.sleep(5)
 s['phase']='auditing';save()
 subprocess.run([python,'scripts/summarize_final_results.py','--plan','configs/act_final_evaluation_plan.json','--attempt-map','configs/act_final_completion_attempts.json','--output','artifacts/act-final-summary-004'],check=True)
 subprocess.run([python,'scripts/review_act_final_evidence.py','--summary','artifacts/act-final-summary-004/summary.json','--output','artifacts/act-final-review-cpu-002/analysis'],check=True)
 plan=read('configs/act_final_evaluation_plan.json');mapping=read('configs/act_final_completion_attempts.json');auth=read('configs/act_final_completion_authorization.json')
 summary=read('artifacts/act-final-summary-004/summary.json');old=read('artifacts/act-final-summary-003/summary.json')
 assert summary['conditions'][:-2]==old['conditions'][:-2]
 assert sha('configs/act_final_evaluation_plan.json')==auth['original_plan_sha256']
 assert sha('releases/act-u2-online-final-001/weights.pt')==auth['checkpoint_sha256']
 failed=read('artifacts/e7-test-act-u2-final-mean-001/manifest.json');assert failed['exit_code']==143 and failed['stop_reason']=='time_budget'
 assert len(Path('artifacts/e7-test-act-u2-final-mean-001/episodes.jsonl').read_text().splitlines())==70
 baseline=read('artifacts/e7-test-act-u2-final-mean-001/source_hashes.json');records=[]
 for key,attempt in mapping.items():
  folder=Path('artifacts')/attempt
  if not (folder/'manifest.json').exists():continue
  m=read(folder/'manifest.json');assert m['ended_at']
  assert m['limit_seconds']<=120 and sha(folder/'source_snapshot.tar.gz')==m['source_snapshot_sha256']
  assert sha(folder/'working-tree.patch')==m['tracked_patch_sha256']
  hashes=read(folder/'source_hashes.json')
  with tarfile.open(folder/'source_snapshot.tar.gz') as archive:
   assert set(archive.getnames())==set(hashes)
   for name,digest in hashes.items():assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest
  # No model/environment execution code changed from the original attempt.
  changed=[name for name,digest in baseline.items() if name.startswith(('scripts/','src/','vendor/')) and hashes.get(name)!=digest]
  assert set(changed)<=set(['scripts/summarize_final_results.py','scripts/review_act_final_evidence.py']),changed
  subprocess.run(['git','-C','.cache/act-source-history-release-check-001/history.git','cat-file','-e',m['code_commit']+'^{commit}'],check=True)
  counter=read(folder/'interactions.json');resources=[json.loads(x) for x in (folder/'resources.jsonl').read_text().splitlines()]
  assert all(x['index']==0 for x in resources)
  records.append(dict(condition=key,attempt=attempt,exit_code=m['exit_code'],stop_reason=m['stop_reason'],elapsed_seconds=m['elapsed_seconds'],source_files_verified=len(hashes),changed_non_execution_scripts=changed,interactions=counter,peak_gpu_mib=max(x['memory_mib'] for x in resources),manifest_sha256=sha(folder/'manifest.json')))
 manifests=[read(p) for p in Path('artifacts').glob('*/manifest.json')];assert all(m['ended_at'] for m in manifests)
 total=sum(m['elapsed_seconds'] for m in manifests);assert abs(total-read('research_state.json')['budget']['gpu_job_seconds_used'])<1e-6 and total<=86400
 intervals=sorted((dt.datetime.fromisoformat(m['started_at']),dt.datetime.fromisoformat(m['ended_at'])) for m in manifests)
 assert all(x[1]<=y[0] for x,y in zip(intervals,intervals[1:]))
 batch=read('artifacts/act-final-completion-001/status.json');assert batch['elapsed_seconds']<=300
 report=dict(phase='completed',completed_conditions=summary['completed_conditions'],registered_conditions=106,unchanged_other_conditions=104,old_failed_attempt_preserved=True,all_gpu_job_seconds=total,remaining_gpu_job_seconds=86400-total,gpu_job_count=len(manifests),completion_wall_seconds=batch['elapsed_seconds'],completion_gpu_job_seconds=sum(r['elapsed_seconds'] for r in records),original_final_wall_seconds=7200.778426306031,original_shutdown_overrun_seconds=0.778426306031,added_cap_seconds=300,manifest_intervals_disjoint=True,checkpoint_sha256=auth['checkpoint_sha256'],records=records,new_training_updates=0,audit_new_environment_interactions=0)
 (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
 s.update(phase='completed',exit_code=0)
except BaseException as error:
 s.update(phase='failed',exit_code=1,error=repr(error));raise
finally:
 s['ended_at']=time.time();save()
