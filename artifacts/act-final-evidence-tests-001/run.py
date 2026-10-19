import json,subprocess,sys
from pathlib import Path
out=Path('artifacts/act-final-evidence-tests-001')
commands=[['.venv/bin/python','-m','pytest','-q','tests/test_act_final_evidence.py','tests/test_final_results.py','tests/test_act_inference.py'],['.venv/bin/python','scripts/summarize_final_results.py','--plan','configs/act_final_evaluation_plan.json','--output',str(out/'pending-summary')]]
code=0
for command in commands:
 code=subprocess.call(command)
 if code:break
(out/'status.json').write_text(json.dumps(dict(exit_code=code,GPU_use=False,final_test_accessed=False),indent=2)+'\n')
sys.exit(code)
