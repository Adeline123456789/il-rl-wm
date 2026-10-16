import json,subprocess,py_compile
from pathlib import Path
from scripts.run_pickcube_repeat import plan
from scripts.run_u1_repeat import plan as u1_plan
for path in ['scripts/run_final_evaluation.py','scripts/diagnose_behavior.py','scripts/export_analysis_checkpoint.py']:
 py_compile.compile(path,doraise=True)
lock=Path('external/rlpd/requirements.lock').read_text()
actual=subprocess.check_output(['uv','pip','freeze','--python','.venv-rlpd/bin/python'],text=True)
for name,sha in [('ManiSkill','62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3'),('rfcl','7bd660f05dbda1ec8881bcfb250a8d1881532dbb')]:
 path=Path('.cache/upstreams')/name
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()==sha
 actual=actual.replace('-e '+path.resolve().as_uri(),'-e ./'+str(path))
assert actual==lock
registry=json.loads(Path('configs/final_evaluation_plan.json').read_text())
jobs={}
for task,seed in [('PickCube-v1',1),('PickCube-v1',2),('StackCube-v1',0),('PegInsertionSide-v1',0)]:
 jobs.update({j['id']:j for j in plan(seed,task)})
for seed in [0,1,2]:jobs.update({j['id']:j for j in u1_plan(seed)})
for e in registry['entries']:
 if e['framework']=='main':
  assert e['checkpoint']==f"checkpoints/{e['training_experiment']}/final.pt"
  if e['training_experiment'] in jobs:
   cmd=jobs[e['training_experiment']]['command']
   assert cmd[cmd.index('--exp-name')+1]==e['training_experiment']
  else:
   rows=[json.loads(line) for line in Path('artifacts',e['training_experiment'],'checkpoints.jsonl').read_text().splitlines()]
   assert len([r for r in rows if r['path']==e['checkpoint']])==1
assert sum(len(e['methods']) for e in registry['entries'])==60
Path('artifacts/reproduction-preflight-001/result.json').write_text(json.dumps(dict(valid=True,registered_models=len(registry['entries']),method_conditions=60,final_test_accessed=False,rlpd_lock_matches_installed_versions=True,upstream_commits_match=True,syntax_validated=True,limitation='Final evaluation orchestrator not yet run end-to-end; pending training and GPU RLPD evaluator smoke.'),indent=2))
