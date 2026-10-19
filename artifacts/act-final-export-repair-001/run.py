import os,json,hashlib,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from scripts.export_analysis_checkpoint import training_interactions
out=Path('artifacts/act-final-export-repair-001')
def read(p):return json.loads(Path(p).read_text())
status=dict(pid=os.getpid(),started_at=time.time(),exit_code=None,GPU_use=False)
try:
 subprocess.run(['.venv/bin/python','-m','pytest','-q','tests/test_analysis_export.py','tests/test_final_results.py','tests/test_act_final_evidence.py'],check=True)
 rows=[]
 for e in read('configs/act_final_evaluation_plan.json')['entries']:
  if e['framework']!='main':continue
  s=torch.load(e['checkpoint'],map_location='cpu')
  total=training_interactions(s)
  assert (total,s['config']['env_id'],s['config']['seed'])==(e['training_interactions'],e['task'],e['training_seed'])
  assert all(k in s for k in ['actor','qf1','qf2','qf1_target','qf2_target','log_alpha','frozen_il','config','global_step','global_update'])
  if 'sac_wm' in e['methods']:assert 'lineage_training_interactions' in s and all(k in s['u0'] for k in ['wm','value','value_target'])
  rows.append(dict(id=e['id'],checkpoint=e['checkpoint'],legacy_counter='lineage_training_interactions' not in s,global_step=s['global_step'],lineage_training_interactions=total))
  del s
 (out/'checkpoint_metadata.json').write_text(json.dumps(rows,indent=2)+'\n')
 empty=Path('releases/final-pick-seed0-c-001')
 assert empty.is_dir() and not list(empty.iterdir())
 empty.rename(out/'failed_empty_package')
 subprocess.run(['.venv/bin/python','scripts/export_analysis_checkpoint.py','--checkpoint','checkpoints/e3-sac-c-full-001/final.pt','--output','releases/final-pick-seed0-c-001'],check=True)
 original=torch.load('checkpoints/e3-sac-c-full-001/final.pt',map_location='cpu');export=torch.load('releases/final-pick-seed0-c-001/weights.pt',map_location='cpu')
 for key in ['actor','qf1','qf2','qf1_target','qf2_target','frozen_il']:
  assert set(original[key])==set(export[key])
  for name in original[key]:assert torch.equal(original[key][name],export[key][name])
 assert export['lineage_training_interactions']==262144
 # Preserve verification export separately; the registered runner creates its own fresh package.
 Path('releases/final-pick-seed0-c-001').rename(out/'verified_legacy_package')
 status.update(exit_code=0,phase='completed',checked_main_entries=len(rows),legacy_ids=[r['id'] for r in rows if r['legacy_counter']],all_model_tensors_equal=True,new_environment_interactions=0)
except BaseException as error:
 status.update(exit_code=1,phase='failed',error=repr(error));raise
finally:
 status['ended_at']=time.time();(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
