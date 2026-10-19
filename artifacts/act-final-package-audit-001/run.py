import json,py_compile,sys
from pathlib import Path
from scripts.run_final_evaluation import act_checkpoint_hash
out=Path('artifacts/act-final-package-audit-001');result=dict(exit_code=None,verified=[],pending=[],GPU_use=False,final_test_accessed=False)
try:
 for path in ['scripts/check_act_evaluation_entry.py','scripts/run_final_evaluation.py','scripts/evaluate_act_checkpoint.py','scripts/summarize_final_results.py']:
  py_compile.compile(path,doraise=True)
 plan=json.loads(Path('configs/act_final_evaluation_plan.json').read_text())
 for entry in plan['entries']:
  if entry['framework']!='act':continue
  if not Path(entry['checkpoint']).exists():
   result['pending'].append(entry['id']);continue
  digest=act_checkpoint_hash(entry,Path('.'))
  result['verified'].append(dict(id=entry['id'],sha256=digest))
 result['exit_code']=0
except BaseException as error:
 result.update(exit_code=1,error=repr(error))
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));sys.exit(result['exit_code'])
