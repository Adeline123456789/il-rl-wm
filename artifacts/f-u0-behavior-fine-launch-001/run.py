import subprocess,json,pathlib
commands=[['.venv/bin/python', 'scripts/diagnose_behavior.py', '--input-arrays', 'artifacts/f-u0-behavior-u0-001/transitions.npz', '--bin-size', '8192', '--output', 'artifacts/f-u0-behavior-u0-fine-001'], ['.venv/bin/python', 'scripts/diagnose_behavior.py', '--input-arrays', 'artifacts/f-u0-behavior-control-001/transitions.npz', '--bin-size', '8192', '--output', 'artifacts/f-u0-behavior-control-fine-001']]
results=[]
for command in commands:
 r=subprocess.run(command);results.append(dict(command=command,exit_code=r.returncode))
 if r.returncode:break
pathlib.Path('artifacts/f-u0-behavior-fine-launch-001/result.json').write_text(json.dumps(results,indent=2))
