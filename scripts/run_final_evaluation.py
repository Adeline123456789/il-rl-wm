"""One finite registered evaluation batch; never choose checkpoints by scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON=str(ROOT/'.venv/bin/python')
RLPD_PYTHON=str(ROOT/'.venv-rlpd/bin/python')


def act_checkpoint_hash(entry, root):
    """Verify a registered offline package or fixed-final online export before test access."""
    path=root/entry['checkpoint']
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if entry['training_interactions']==0:
        assert digest==entry['checkpoint_sha256'], entry['id']
    else:
        exported=json.loads((path.parent/'manifest.json').read_text())
        assert digest==exported['portable_sha256'] and exported['all_actor_tensors_equal']
        assert exported['source']==f"checkpoints/{entry['training_experiment']}/final.pt"
        records=[json.loads(line) for line in
            (root/'artifacts'/entry['training_experiment']/'checkpoints.jsonl').read_text().splitlines()]
        assert any(r['final'] and r['path']==exported['source'] and
                   r['sha256']==exported['source_sha256'] and r['step']==entry['training_interactions']
                   for r in records)
    assert entry['methods']==[entry['kind']+'_'+entry['mode']]
    return digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ids',nargs='+',help='Explicit registered subset; unselected entries remain pending')
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    # ACT task supersedes the earlier evaluation start authorization: protect the
    # same test seeds until the new comparison conditions are explicitly frozen.
    gate=ROOT/'configs/act_final_test_gate.json'
    if gate.exists():
        deadline=time.monotonic()+86400
        while not json.loads(gate.read_text())['released']:
            if time.monotonic()>=deadline:
                raise RuntimeError('ACT comparison not frozen; final-test gate remains closed')
            time.sleep(30)
    assert (plan['seed'],plan['episodes_per_model_method'],plan['num_envs'])==(20000,100,10)
    entries=plan['entries'];ids=[e['id'] for e in entries]
    assert len(set(ids))==len(ids)
    if args.ids:
        assert len(set(args.ids))==len(args.ids) and set(args.ids)<=set(ids)
        entries=[e for e in entries if e['id'] in args.ids]
    # Verify all selected training runs and loader smokes before any test access.
    required=['e7-independent-eval-smoke-001']
    if any(e['framework']=='rlpd' for e in entries):required.append('e7-rlpd-independent-eval-smoke-001')
    if any(e['framework']=='act' for e in entries):
        smoke=json.loads((ROOT/'artifacts/act-independent-eval-smoke-001/status.json').read_text())
        assert smoke['phase']=='completed' and smoke['exit_code']==0
        required.append('act-independent-eval-smoke-001')
    for name in required+[e['training_experiment'] for e in entries]:
        m=json.loads((ROOT/'artifacts'/name/'manifest.json').read_text())
        assert m['ended_at'] and m['exit_code']==0 and m['stop_reason'] is None, name
        assert not (ROOT/'artifacts'/name/'INVALIDATED.json').exists()
    expected={}
    for e in entries:
        if e['framework']=='act':
            expected[e['id']]=act_checkpoint_hash(e,ROOT)
            continue
        assert (ROOT/e['checkpoint']).exists(), e['id']
        records=[json.loads(line) for line in (ROOT/'artifacts'/e['training_experiment']/'checkpoints.jsonl').read_text().splitlines()]
        key='path' if e['framework']=='main' else 'directory'
        matched=[r for r in records if r[key]==e['checkpoint']];assert len(matched)==1
        expected[e['id']]=matched[0]['sha256'] if key=='path' else matched[0]['files']['learner.msgpack']['sha256']
    args.output.mkdir(parents=True,exist_ok=False)
    record=dict(phase='running',plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        selected=[e['id'] for e in entries],unselected_pending=[i for i in ids if i not in {e['id'] for e in entries}],
        completed=[],current=None)
    (args.output/'registered_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    started=time.monotonic();cpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    def save():
        record['elapsed_seconds']=time.monotonic()-started
        tmp=args.output/'status.tmp';tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(args.output/'status.json')
    def cpu(command):subprocess.run(command,cwd=ROOT,env=cpu_env,check=True)
    try:
        for e in entries:
            record['current']=e['id'];save()
            package=Path('releases')/f"final-{e['id']}-001"
            if e['framework']=='main':
                cpu([PYTHON,'scripts/export_analysis_checkpoint.py','--checkpoint',e['checkpoint'],'--output',str(package)])
                manifest=json.loads((ROOT/package/'manifest.json').read_text())
                assert manifest['source']['sha256']==expected[e['id']]
                assert (manifest['env_id'],manifest['training_seed'],manifest['lineage_training_interactions'])==(
                    e['task'],e['training_seed'],e['training_interactions'])
                command=[PYTHON,'scripts/evaluate_checkpoint.py','--checkpoint',str(package/'weights.pt'),'--methods',*e['methods']]
                if 'sac_wm' in e['methods']:
                    deployment=Path('releases')/f"no-il-{e['id']}-001"
                    cpu([PYTHON,'scripts/export_deployment.py','--checkpoint',e['checkpoint'],'--output',str(deployment)])
                    cpu([PYTHON,'scripts/check_deployment.py','--checkpoint',e['checkpoint'],'--package',str(deployment/'policy.pt'),
                         '--output',str(deployment/'action-parity.json')])
            elif e['framework']=='rlpd':
                cpu([RLPD_PYTHON,'external/rlpd/policy.py','--checkpoint',e['checkpoint'],'--output',str(package)])
                manifest=json.loads((ROOT/package/'manifest.json').read_text())
                assert manifest['source_sha256']==expected[e['id']]
                assert (manifest['config']['env_id'],manifest['config']['seed'],manifest['step'])==(
                    e['task'],e['training_seed'],e['training_interactions'])
                command=[RLPD_PYTHON,'external/rlpd/evaluate.py','--policy',str(package)]
            else:
                assert e['framework']=='act'
                package=Path(e['checkpoint']).parent
                assert act_checkpoint_hash(e,ROOT)==expected[e['id']]
                command=[PYTHON,'scripts/evaluate_act_checkpoint.py','--checkpoint',e['checkpoint'],
                    '--kind',e['kind'],'--mode',e['mode'],'--training-seed',str(e['training_seed']),
                    '--training-interactions',str(e['training_interactions'])]
            experiment=f"e7-test-{e['id']}-001"
            command+=['--output',f'artifacts/{experiment}','--seed','20000','--episodes','100','--num-envs','10']
            remaining=plan['total_evaluation_limit_seconds']-(time.monotonic()-started)
            assert remaining>0,'Registered evaluation budget exhausted'
            subprocess.run([PYTHON,'scripts/run_experiment.py','--id',experiment,'--seconds',str(min(remaining,plan['per_job_limit_seconds'])),
                            '--',*command],cwd=ROOT,check=True)
            record['completed'].append(dict(id=e['id'],experiment=experiment,package=str(package)));save()
        record.update(phase='completed',current=None)
    except BaseException as error:
        record.update(phase='failed',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
