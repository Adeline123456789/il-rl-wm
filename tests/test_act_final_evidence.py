"""Synthetic final-evidence contracts; never launch a simulator or access test states."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_final_evaluation import act_checkpoint_hash
from scripts.summarize_final_results import summarize


def test_extension_preserves_history_and_shared_budget():
    root=Path(__file__).resolve().parents[1]
    original=root/'configs/final_evaluation_plan.json'
    old=json.loads(original.read_text())
    new=json.loads((root/'configs/act_final_evaluation_plan.json').read_text())
    assert new['entries'][:len(old['entries'])]==old['entries']
    assert new['historical_plan_sha256']==hashlib.sha256(original.read_bytes()).hexdigest()
    assert new['total_evaluation_limit_seconds']==old['total_evaluation_limit_seconds']==7200
    assert sum(len(e['methods']) for e in new['entries'])==106
    assert len({e['id'] for e in new['entries']})==len(new['entries'])


def test_online_export_must_be_registered_final(tmp_path):
    package=tmp_path/'release';package.mkdir();weights=package/'weights.pt';weights.write_bytes(b'synthetic')
    folder=tmp_path/'artifacts/run';folder.mkdir(parents=True)
    digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    exported=dict(portable_sha256=digest,all_actor_tensors_equal=True,
                  source='checkpoints/run/final.pt',source_sha256='registered')
    (package/'manifest.json').write_text(json.dumps(exported))
    row=dict(final=True,path=exported['source'],sha256='registered',step=262144)
    records=folder/'checkpoints.jsonl';records.write_text(json.dumps(row)+'\n')
    entry=dict(id='synthetic',checkpoint='release/weights.pt',training_experiment='run',
               training_interactions=262144,kind='act',mode='mean',methods=['act_mean'])
    assert act_checkpoint_hash(entry,tmp_path)==digest
    row['final']=False;records.write_text(json.dumps(row)+'\n')
    with pytest.raises(AssertionError):act_checkpoint_hash(entry,tmp_path)
    entry.update(training_interactions=0,checkpoint_sha256=digest)
    assert act_checkpoint_hash(entry,tmp_path)==digest
    weights.write_bytes(b'changed')
    with pytest.raises(AssertionError):act_checkpoint_hash(entry,tmp_path)


def test_act_rows_recompute_and_reject_wrong_identity(tmp_path):
    entry=dict(id='act-synthetic',framework='act',task='PickCube-v1',training_seed=0,
        training_interactions=262144,training_role='development',variant='synthetic',
        kind='act',mode='mean',methods=['act_mean'],checkpoint='synthetic.pt')
    plan=dict(seed=20000,episodes_per_model_method=100,num_envs=10,entries=[entry])
    folder=tmp_path/'e7-test-act-synthetic-001';folder.mkdir()
    (folder/'manifest.json').write_text(json.dumps(dict(ended_at='synthetic',exit_code=0,stop_reason=None)))
    rows=[dict(method='act_mean',episode_index=i,iteration=262144,episode_len=200,
        eval_seed=20000+i//10*10,success_once=int(i<20),success_at_end=0,**{'return':float(i<20)}) for i in range(100)]
    (folder/'episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    result=dict(purpose='final_test',seed=20000,num_envs=10,new_training_interactions=0,
        training_seed=0,training_interactions=262144,kind='act',mode='mean',method='act_mean',
        checkpoint='synthetic.pt',success_once=20,success_at_end=0,mean_return=.2,interactions=20000,
        protocol=dict(env_id='PickCube-v1',reset_protocol='explicit_batch_seed',max_episode_steps=200))
    (folder/'result.json').write_text(json.dumps(result))
    (folder/'interactions.json').write_text(json.dumps(dict(total_attempted=20000,total_completed=20000,
        phases=dict(final_test=dict(attempted_transitions=20000,completed_transitions=20000)))))
    checked=summarize(plan,tmp_path)
    assert checked['completed_conditions']==1 and checked['conditions'][0]['success_rate']==.2
    bad=copy.deepcopy(plan);bad['entries'][0]['mode']='stochastic'
    with pytest.raises(AssertionError):summarize(bad,tmp_path)
    rows[-1]['eval_seed']=20000
    (folder/'episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(AssertionError):summarize(plan,tmp_path)
