"""CPU-only terminal audit for the explicitly authorized initialization batch."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time
from datetime import datetime
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = Path('artifacts/act-initialization-delivery-audit-001')
def read(path): return json.loads(Path(path).read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines()]
def epoch(value): return datetime.fromisoformat(value).timestamp()
state = dict(pid=os.getpid(), started_at=time.time(), phase='waiting', exit_code=None,
             new_environment_interactions=0, new_training_updates=0, gpu_jobs_started=0)
def save(): (OUT/'status.json').write_text(json.dumps(state, indent=2)+'\n')
try:
    save()
    deadline = time.monotonic() + 14400
    while True:
        queue = read('artifacts/act-initialization-batch-002/status.json')
        if queue['exit_code'] is not None: break
        assert time.monotonic() < deadline, 'Bounded terminal wait expired'
        time.sleep(30)
    assert queue['phase']=='completed' and queue['exit_code']==0, queue
    state['phase']='auditing'; save()
    authorization_path = Path('configs/act_initialization_budget.json')
    auth = read(authorization_path)
    plan = read('configs/act_initialization_execution_plan_002.json')
    evaluation_path = Path(plan['evaluation_plan'])
    evaluation = read(evaluation_path)
    protocol_path = Path(evaluation['final_protocol'])
    protocol = read(protocol_path)
    gate = read(plan['test_gate'])
    summary = read('artifacts/act-initialization-summary-001/summary.json')
    assert gate['released'] and gate['evaluation_plan_sha256']==digest(evaluation_path)
    assert len(summary['conditions'])==len(evaluation['entries'])==8
    original=[]; additional=[]
    for path in Path('artifacts').glob('*/manifest.json'):
        m=read(path)
        if 'experiment_id' not in m or 'elapsed_seconds' not in m: continue
        if not m.get('budget_id'): original.append(m)
        elif m['budget_id']==auth['budget_id']: additional.append(m)
    assert set(m['experiment_id'] for m in additional)==set(auth['jobs'])
    original_seconds=sum(m['elapsed_seconds'] for m in original)
    assert abs(original_seconds-auth['original_24h_gpu_seconds'])<1e-6
    totals={key:0.0 for key in auth['category_caps']}
    source_counts={}
    for m in additional:
        name=m['experiment_id']; job=auth['jobs'][name]; folder=Path('artifacts')/name
        assert m['ended_at'] and m['exit_code']==0 and m['stop_reason'] is None
        assert m['command']==job['command'] and m['budget_authorization_sha256']==digest(authorization_path)
        assert m['elapsed_seconds']<=job['cap_seconds']
        totals[job['category']]+=m['elapsed_seconds']
        assert digest(folder/'source_snapshot.tar.gz')==m['source_snapshot_sha256']
        assert digest(folder/'working-tree.patch')==m['tracked_patch_sha256']
        source_hashes=read(folder/'source_hashes.json')
        with tarfile.open(folder/'source_snapshot.tar.gz') as archive:
            files={x.name:x for x in archive.getmembers() if x.isfile()}
            assert set(files)==set(source_hashes)
            for path,sha in source_hashes.items():
                assert hashlib.sha256(archive.extractfile(files[path]).read()).hexdigest()==sha,(name,path)
        source_counts[name]=len(source_hashes)
    assert all(totals[k]<=auth['category_caps'][k] for k in totals)
    assert sum(totals.values())<=auth['total_seconds']
    ledger=read('research_state.json')
    assert abs(ledger['budget']['gpu_job_seconds_used']-original_seconds)<1e-6
    assert abs(ledger['additional_budgets'][auth['budget_id']]['gpu_job_seconds_used']-sum(totals.values()))<1e-6
    intervals=sorted((epoch(m['started_at']),epoch(m['ended_at']),m['experiment_id']) for m in original+additional)
    assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
    training_end=max(epoch(read(Path('artifacts')/name/'manifest.json')['ended_at']) for name in plan['training_jobs'])
    first_start=min(epoch(m['started_at']) for m in additional)
    assert epoch(protocol['registered_at'])<=first_start
    assert gate['released_at']>=training_end
    reference=None; actions={}; conditions=[]; sources={str(evaluation_path):digest(evaluation_path),str(protocol_path):digest(protocol_path)}
    for entry in evaluation['entries']:
        folder=Path('artifacts')/('e7-test-'+entry['id']+'-001')
        m=read(folder/'manifest.json'); result=read(folder/'result.json'); records=rows(folder/'episodes.jsonl')
        assert epoch(m['started_at'])>=gate['released_at']
        assert result['purpose']=='final_test' and result['seed']==protocol['seed']==30000
        assert result['checkpoint_sha256']==digest(entry['checkpoint'])==gate['checkpoint_hashes'][entry['checkpoint']]
        assert result['final_protocol_sha256']==digest(protocol_path)
        assert result['new_training_interactions']==0 and result['interactions']==20000
        assert len(records)==100 and [r['episode_index'] for r in records]==list(range(100))
        assert [r['eval_seed'] for r in records]==[30000+10*(i//10) for i in range(100)]
        assert all(r['episode_len']==200 for r in records)
        with np.load(folder/'final_test_trace.npz') as trace:
            assert trace['obs'].shape==(2000,10,42) and trace['action'].shape==(2000,10,4)
            initial=trace['obs'][::200].reshape(100,42).copy()
            actions[entry['id']]=trace['action'][::200].reshape(100,4).copy()
        if reference is None: reference=initial
        assert np.array_equal(initial,reference),entry['id']
        c=next(x for x in summary['conditions'] if x['id']==entry['id'])
        assert c['status']=='completed'
        for key in ['success_once','success_at_end']:
            assert c[key]==result[key]==sum(r[key] for r in records)
        assert abs(c['mean_return']-np.mean([r['return'] for r in records]))<1e-6
        stage={key:sum(bool(r[key]) for r in records) for key in ['approached_once','is_grasped_once','lifted_once','is_obj_placed_once']}
        grasp=[r['first_grasp_step'] for r in records if r['first_grasp_step'] is not None]
        conditions.append(dict(c,stage_counts=stage,first_grasp_step_conditional_median=float(np.median(grasp)) if grasp else None,
                               first_grasp_denominator=len(grasp),initial_observation_max_difference=0.0))
        for name in ['manifest.json','result.json','episodes.jsonl','final_test_trace.npz']:
            sources[str(folder/name)]=digest(folder/name)
    result=dict(conditions=conditions,matched_initial_conditions=8,category_gpu_job_seconds=totals,
        total_gpu_job_seconds=sum(totals.values()),original_gpu_job_seconds_unchanged=original_seconds,
        total_seconds_authorized=auth['total_seconds'],source_files_verified=source_counts,job_intervals_disjoint=True,
        protocol_registered_before_first_new_job=True,final_gate_after_all_training=True,
        new_environment_interactions=0,new_training_updates=0,
        scope='Single training seed, held-out final evaluation after fixed training. No checkpoint selection, extra tuning, WM, or multi-seed claim.')
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    sources[str(Path(__file__).relative_to(ROOT))]=digest(Path(__file__))
    (OUT/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    np.savez_compressed(OUT/'matched_initial_actions.npz',initial_observations=reference,**actions)
    state.update(phase='completed',exit_code=0)
except BaseException as error:
    state.update(phase='failed',exit_code=1,error=repr(error)); raise
finally:
    state['ended_at']=time.time();save()
