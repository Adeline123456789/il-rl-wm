import json,os,subprocess
from pathlib import Path
out=Path('.cache/github-act-stage-024');out.mkdir(exist_ok=False)
env=dict(os.environ,GIT_INDEX_FILE=str((out/'index').resolve()))
def git(*args):return subprocess.check_output(['git',*args],env=env,text=True).strip()
base='2723aa23694e4dd32809d4201034193c7a8a684a';git('read-tree',base)
paths=['scripts/complete_u2_final_evaluation.py','scripts/summarize_final_results.py','scripts/review_act_final_evidence.py','configs/act_final_completion_authorization.json','configs/act_final_completion_attempts.json','AGENTS.md','RESEARCH_LOG.md','ACT_CHANGE_LOG.md','ACT_FAILURES.md','research_state.json','FINAL_BATCH_REPORT.md','ACT_MILESTONE_REPORT.md','ACT_TRANSFER_REPORT.md','ACT_U2_FINAL_REVIEW.md','ACT_TRANSFER_CAUSAL_GAPS.md','ACT_SAC_PLAN.md','PROGRESS_SNAPSHOT.md','README.md','NEXT_STEPS.md','NEXT_RESEARCH_BATCH.md','BATCH_REQUIREMENTS_AUDIT.md','REPRODUCING.md','releases/ACT_REPRODUCTION.md','artifacts/act-final-completion-preflight-001','artifacts/act-final-completion-launch-001','artifacts/act-final-completion-001','artifacts/act-final-completion-postprocess-001','artifacts/act-final-completion-delivery-001','artifacts/act-final-summary-004','artifacts/act-final-review-cpu-002','artifacts/e7-test-act-u2-final-mean-002','artifacts/e7-test-act-u2-final-stochastic-002','artifacts/github-act-stage-upload-023']
git('add','--',*paths);tree=git('write-tree');files=[]
for line in git('ls-tree','-rl',tree).splitlines():
 meta,path=line.split('\t',1);mode,kind,sha,size=meta.split();files.append(dict(path=path,mode=mode,type=kind,sha=sha,size=int(size)))
changed=set(git('diff','--name-only',base,tree).splitlines());delta=[x for x in files if x['path'] in changed];assert len(delta)==len(changed)
assert max(x['size'] for x in delta)<10*1024*1024
assert not any('/__pycache__/' in x['path'] or '/videos/' in x['path'] for x in delta)
p=dict(local_source_head=git('rev-parse','HEAD'),capture='temporary index tree; local HEAD and normal index unchanged',tree=tree,base_tree=base,remote_parent='5adb2c8155c9c94095f2567824ac4e860338d7d0',files=files,delta=delta)
(out/'plan.json').write_text(json.dumps(p));print(json.dumps({k:v for k,v in p.items() if k not in ['files','delta']}));print(len(delta),sum(x['size'] for x in delta))
