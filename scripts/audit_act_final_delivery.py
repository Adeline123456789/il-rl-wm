"""Read-only terminal-batch provenance and accounting audit; no simulator/ML imports."""
import argparse,datetime as dt,hashlib,json,math,subprocess,tarfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.summarize_final_results import summarize

def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
 p.add_argument('--allow-running-snapshot',action='store_true',help='Provenance preflight only; never a terminal completion audit')
 a=p.parse_args()
 a.output.mkdir(exist_ok=False,parents=True)
 launch=read('artifacts/act-final-evaluation-launch-003/status.json')
 terminal=launch['phase'] in ['completed','failed'] and launch['exit_code'] is not None
 assert terminal or a.allow_running_snapshot,'Final batch is still active'
 plan_path=Path('configs/act_final_evaluation_plan.json');plan=read(plan_path)
 freeze=read('artifacts/act-comparison-freeze-001/result.json');gate=read('configs/act_final_test_gate.json')
 assert digest(plan_path)==freeze['plan_sha256'] and gate['released'] and freeze['development_closed']
 result=summarize(plan,Path('artifacts'))
 if terminal:
  published=read('artifacts/act-final-summary-003/summary.json')
  assert all(published[k]==v for k,v in result.items()),'Final summary differs from raw episodes'
 assert result['registered_conditions']==106
 for row in result['conditions']:
  if row['status']!='completed':assert row['success_rate'] is None and row['episodes'] is None
 frozen=dt.datetime.fromisoformat(freeze['frozen_at']);records=[];problems=[]
 manifests={str(p):read(p) for p in Path('artifacts').glob('*/manifest.json')}
 live=[m['experiment_id'] for m in manifests.values() if not m.get('ended_at')]
 if terminal:assert not live,'Unfinished GPU job requires reconciliation'
 settled=[m for m in manifests.values() if m.get('ended_at')]
 commits=sorted({m['code_commit'] for m in manifests.values()})
 history=Path('.cache/act-source-history-release-check-001/history.git')
 for commit in commits:
  subprocess.run(['git','-C',str(history),'cat-file','-e',commit+'^{commit}'],check=True)
 intervals=sorted([(dt.datetime.fromisoformat(m['started_at']),dt.datetime.fromisoformat(m['ended_at']),m['experiment_id']) for m in settled])
 overlaps=[dict(previous=x[2],next=y[2],seconds=(x[1]-y[0]).total_seconds()) for x,y in zip(intervals,intervals[1:]) if x[1]>y[0]]
 if overlaps:problems.append('GPU runner manifest intervals overlap; inspect before claiming serial execution')
 for entry in plan['entries']:
  root=Path('artifacts')/f"e7-test-{entry['id']}-001"
  if not (root/'manifest.json').exists():continue
  m=read(root/'manifest.json');assert dt.datetime.fromisoformat(m['started_at'])>=frozen
  if not m.get('ended_at'):continue
  assert digest(root/'source_snapshot.tar.gz')==m['source_snapshot_sha256']
  assert digest(root/'working-tree.patch')==m['tracked_patch_sha256']
  hashes=read(root/'source_hashes.json')
  with tarfile.open(root/'source_snapshot.tar.gz') as archive:
   assert set(archive.getnames())==set(hashes)
   for name,value in hashes.items():assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==value
  weight=None
  if m['exit_code']==0 and m['stop_reason'] is None:
   evidence=read(root/'result.json');path=evidence.get('checkpoint');expected=evidence.get('checkpoint_sha256',evidence.get('sha256'))
   if entry['framework']=='rlpd':
    path=str(Path(evidence['policy'])/'actor.msgpack');expected=evidence['actor_sha256']
   if path and expected and Path(path).is_file():
    assert digest(path)==expected;weight=dict(path=path,sha256=expected,verified=True)
   else:
    weight=dict(path=path,sha256=expected,verified=False,reason='Result format lacks a single file hash; inspect framework-specific portable manifest')
    problems.append('Weight provenance requires framework-specific check: '+entry['id'])
  resources=[json.loads(x) for x in (root/'resources.jsonl').read_text().splitlines()]
  assert resources and all(x['index']==0 for x in resources)
  peak=max(x['memory_mib'] for x in resources)
  if peak>m['memory_limit_mib']:problems.append('Memory limit exceeded: '+entry['id'])
  records.append(dict(id=entry['id'],manifest_sha256=digest(root/'manifest.json'),code_commit=m['code_commit'],source_files_verified=len(hashes),source_snapshot_sha256=m['source_snapshot_sha256'],tracked_patch_sha256=m['tracked_patch_sha256'],weight=weight,exit_code=m['exit_code'],stop_reason=m['stop_reason'],gpu_job_seconds=m['elapsed_seconds'],whole_gpu_peak_mib=peak))
 state=read('research_state.json');total=sum(m['elapsed_seconds'] for m in settled);ledger=state['budget']['gpu_job_seconds_used']
 assert math.isclose(total,ledger,rel_tol=0,abs_tol=1e-6)
 assert total<=86400 and state['budget']['gpu_hours_limit']==24 and state['budget']['local_changes_used']==2
 previous=read('artifacts/act-final-evaluation-batch-002/status.json')['elapsed_seconds']
 active=read('artifacts/act-final-evaluation-batch-003/status.json')
 final_elapsed=previous+active['elapsed_seconds']
 if final_elapsed>7200:problems.append('Final batch wall time includes '+str(final_elapsed-7200)+' seconds beyond cap; report shutdown/cleanup overhead explicitly')
 report=dict(terminal_batch=terminal,running_gpu_manifests=live,completed_conditions=result['completed_conditions'],registered_conditions=106,condition_statuses={status:sum(r['status']==status for r in result['conditions']) for status in sorted({r['status'] for r in result['conditions']})},missing_conditions=[r['id']+':'+r['method'] for r in result['conditions'] if r['status']!='completed'],
  final_plan_sha256=digest(plan_path),summary_recomputed_from_raw_episodes=True,gate_before_test_verified=True,
  all_gpu_manifest_count=len(manifests),source_commits_verified_in_offline_restored_history=commits,
  final_job_records=records,all_gpu_job_seconds=total,ledger_equal=True,gpu_limit_seconds=86400,local_hypotheses_used=2,
  final_batch_wall_seconds=final_elapsed,original_final_limit_seconds=7200,manifest_interval_overlaps=overlaps,
  audit_findings=problems,new_environment_interactions=0,new_training_updates=0,
  completion_scope='Provenance/accounting only. Missing final conditions, ACT-WM, independent ACT seeds/tasks, and unresolved scientific claims remain incomplete; this audit does not declare research goal achieved.')
 (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ['final_job_records','source_commits_verified_in_offline_restored_history']},indent=2))
if __name__=='__main__':main()
