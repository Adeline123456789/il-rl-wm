import subprocess,json,pathlib
commands=[['.venv/bin/python', 'scripts/diagnose_behavior.py', '--checkpoint', 'checkpoints/e6-u0-full-001/final.pt', '--output', 'artifacts/f-u0-behavior-u0-001'], ['.venv/bin/python', 'scripts/diagnose_behavior.py', '--input-arrays', 'artifacts/f-u0-behavior-u0-001/transitions.npz', '--output', 'artifacts/f-u0-behavior-u0-001-portable'], ['.venv/bin/python', 'scripts/diagnose_behavior.py', '--checkpoint', 'checkpoints/e3-sac-c-continue-full-001/final.pt', '--output', 'artifacts/f-u0-behavior-control-001'], ['.venv/bin/python', 'scripts/diagnose_behavior.py', '--input-arrays', 'artifacts/f-u0-behavior-control-001/transitions.npz', '--output', 'artifacts/f-u0-behavior-control-001-portable']]
results=[]
for command in commands:
 r=subprocess.run(command);results.append(dict(command=command,exit_code=r.returncode))
 if r.returncode:break
pathlib.Path('artifacts/f-u0-behavior-launch-001/result.json').write_text(json.dumps(results,indent=2))
