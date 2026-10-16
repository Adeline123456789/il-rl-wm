import json,subprocess
from pathlib import Path
output=Path('.cache/checks/final-export-metadata-001')
subprocess.run(['.venv/bin/python','scripts/export_analysis_checkpoint.py','--checkpoint','releases/c-continue-analysis-final-001/weights.pt','--output',str(output)],check=True)
m=json.loads((output/'manifest.json').read_text());r=next(e for e in json.loads(Path('configs/final_evaluation_plan.json').read_text())['entries'] if e['id']=='pick-seed0-c-continue')
assert (m['env_id'],m['training_seed'],m['lineage_training_interactions'])==(r['task'],r['training_seed'],r['training_interactions'])
Path('artifacts/final-export-metadata-check-001/export_manifest.json').write_text(json.dumps(m,indent=2))
Path('artifacts/final-export-metadata-check-001/result.json').write_text(json.dumps(dict(valid=True,task=m['env_id'],seed=m['training_seed'],training_interactions=m['lineage_training_interactions'],test_episodes_executed=0,source_is_portable_baseline_for_metadata_check_only=True),indent=2))
