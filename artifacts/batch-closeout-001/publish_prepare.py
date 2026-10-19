import json,os,subprocess,hashlib,time
from pathlib import Path
out=Path('.cache/github-act-stage-022');out.mkdir(exist_ok=False)
base='5d0bd5304c83d1a39fd05069cf491a0ef2880164'
env=dict(os.environ,GIT_INDEX_FILE=str((out/'index').resolve()))
def git(*args):return subprocess.check_output(['git',*args],env=env,text=True).strip()
git('read-tree',base)
paths=['FINAL_BATCH_REPORT.md','configs/act_final_completion_proposal.json','scripts/review_act_final_evidence.py','ACT_MILESTONE_REPORT.md','ACT_U2_FINAL_REVIEW.md','ACT_TRANSFER_REPORT.md','ACT_TRANSFER_CAUSAL_GAPS.md','ACT_SAC_PLAN.md','PROGRESS_SNAPSHOT.md','README.md','NEXT_STEPS.md','NEXT_RESEARCH_BATCH.md','BATCH_REQUIREMENTS_AUDIT.md','AGENTS.md','ACT_FAILURES.md','RESEARCH_LOG.md','ACT_CHANGE_LOG.md','REPRODUCING.md','releases/ACT_REPRODUCTION.md','research_state.json','artifacts/batch-closeout-001','artifacts/act-final-review-cpu-001','artifacts/batch-final-accounting-001','artifacts/batch-final-accounting-002','artifacts/act-final-evaluation-launch-003','artifacts/act-final-evaluation-batch-003','artifacts/act-final-summary-003','artifacts/act-final-delivery-audit-001','artifacts/github-act-stage-upload-021','artifacts/act-final-increment-002']
plan=json.loads(Path('configs/act_final_evaluation_plan.json').read_text());act=[x for x in plan['entries'] if x.get('framework')=='act'];assert len(act)==18
inventory=[]
for entry in act:
 d=Path('artifacts')/('e7-test-'+entry['id']+'-001')
 if not d.exists():
  assert entry['id']=='act-u2-final-stochastic';inventory.append(dict(id=entry['id'],status='not_started'));continue
 m=json.loads((d/'manifest.json').read_text());assert m['ended_at']
 assert m['exit_code']==0 or (entry['id']=='act-u2-final-mean' and m['exit_code']==143 and m['stop_reason']=='time_budget')
 paths.append(str(d));files=[]
 for p in sorted(d.iterdir()):
  assert p.is_file();files.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 inventory.append(dict(id=entry['id'],status='completed' if m['exit_code']==0 else 'failed',files=files))
close=Path('artifacts/batch-closeout-001');(close/'act_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n');(close/'publish_prepare.py').write_text(Path(__file__).read_text())
git('add','--',*paths);tree=git('write-tree');files=[]
for line in git('ls-tree','-rl',tree).splitlines():
 meta,path=line.split('\t',1);mode,kind,sha,size=meta.split();files.append(dict(path=path,mode=mode,type=kind,sha=sha,size=int(size)))
changed=set(git('diff','--name-only',base,tree).splitlines());delta=[x for x in files if x['path'] in changed];assert len(delta)==len(changed)
assert all(x['size']<10_000_000 for x in delta)
by_path={x['path']:x for x in files}
for entry in act:
 path=entry['checkpoint'];assert path in by_path and git('hash-object',path)==by_path[path]['sha']
result=dict(local_source_head=git('rev-parse','HEAD'),capture='temporary index final evidence snapshot; local HEAD unchanged',tree=tree,base_tree=base,remote_parent='8e91f23044bca9dfcde55f926743027c6dc2c49c',files=files,delta=delta)
(out/'plan.json').write_text(json.dumps(result));print(json.dumps({k:v for k,v in result.items() if k not in ['files','delta']}));print(len(delta),sum(x['size'] for x in delta))
