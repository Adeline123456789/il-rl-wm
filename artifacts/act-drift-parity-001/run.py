import hashlib,json,numpy as np
from pathlib import Path
out=Path('artifacts/act-drift-parity-001')
try:
    old=np.load('artifacts/act-t5-smoke-diagnosis-001/arrays.npz');new=np.load('artifacts/act-t5-smoke-drift-002/arrays.npz')
    pairs={'observations':'observations','initial':'step0_action','step8192':'step8192_action','initial_log_std':'step0_log_std','step8192_log_std':'step8192_log_std'}
    checks={a:bool(np.array_equal(old[a],new[b])) for a,b in pairs.items()};assert all(checks.values())
    files=['scripts/diagnose_act_online_drift.py','tests/test_act_online_drift.py']
    result=dict(exit_code=0,exact_existing_probe_parity=checks,source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},new_environment_interactions=0)
except BaseException as error:result=dict(exit_code=1,error=repr(error))
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
