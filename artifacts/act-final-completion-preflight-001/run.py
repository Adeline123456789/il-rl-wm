import ast,json,hashlib,sys,time,os
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from scripts.summarize_final_results import summarize
out=Path('artifacts/act-final-completion-preflight-001')
s=dict(pid=os.getpid(),started_at=time.time(),phase='running',exit_code=None)
try:
 for name in ['scripts/complete_u2_final_evaluation.py','scripts/summarize_final_results.py']:ast.parse(Path(name).read_text())
 plan=json.load(open('configs/act_final_evaluation_plan.json'));old=json.load(open('artifacts/act-final-summary-003/summary.json'))
 actual=summarize(plan,Path('artifacts'))
 assert all(old[k]==v for k,v in actual.items())
 mapping=json.load(open('configs/act_final_completion_attempts.json'));new=summarize(plan,Path('artifacts'),mapping)
 assert new['completed_conditions']==104
 assert all(x==y for x,y in zip(actual['conditions'][:-2],new['conditions'][:-2]))
 assert all(x['status']=='not_started' for x in new['conditions'][-2:])
 s.update(phase='completed',exit_code=0,original_summary_unchanged=True,completed_conditions=104,non_u2_rows_unchanged=True,new_attempts_not_started=True)
except BaseException as e:
 s.update(phase='failed',exit_code=1,error=repr(e));raise
finally:
 s['ended_at']=time.time();(out/'status.json').write_text(json.dumps(s,indent=2)+'\n')
