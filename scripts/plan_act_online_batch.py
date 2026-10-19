"""Freeze equal-interaction T2--T6 commands using completed throughput evidence."""
import json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_act_online_smokes import command


def main():
    out=Path('artifacts/act-online-budget-001');out.mkdir(exist_ok=False)
    estimate={};hashes=[]
    for tag in ['t5','t4']:
        p=Path(f'artifacts/act-{tag}-online-smoke-001')
        m=json.loads((p/'manifest.json').read_text());r=json.loads((p/'result.json').read_text())
        assert m['exit_code']==0 and r['training_interactions']==8192 and r['gradient_updates']==1040
        rows=[json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines()]
        final={x['metric']:x['value'] for x in rows if x['step']==8192}
        updates=final['time/total_update_time'];rollout=final['time/total_rollout_time']
        other=max(0,m['elapsed_seconds']-updates-rollout)
        projection=updates/1040*64528+rollout/8192*262144+other/2*17
        limit=math.ceil(projection*1.35+60)
        estimate[tag]=dict(measured_gpu_job_seconds=m['elapsed_seconds'],update_seconds=updates,rollout_seconds=rollout,
            other_seconds=other,projected_full_seconds=projection,limit_seconds=limit,
            rule='64528 updates + 262144 rollout transitions + 17 evaluations; all remaining overhead scaled as evaluation; 35 percent margin +60s')
        h=json.loads((p/'actor_initialization.json').read_text());hashes.append((h['critic1_hash'],h['critic2_hash']))
    assert hashes[0]==hashes[1]
    jobs=[]
    for tag in ['t2','t3','t4','t5','t6']:
        name=f'act-{tag}-online-seed0-001';kind='act' if tag=='t5' else 'mlp'
        path='releases/act-c1-init-001/weights.pt' if tag=='t5' else 'releases/act-d1-negative-001/weights.pt'
        cmd=command(name,kind,path)
        # Planning may run in system Python; training uses the verified project environment.
        cmd[0]=str(Path(__file__).resolve().parents[1]/'.venv/bin/python')
        for flag,value in [('--total-timesteps','262144'),('--eval-freq','16384'),('--log-freq','8192')]:cmd[cmd.index(flag)+1]=value
        if tag in ['t2','t3']:
            i=cmd.index('--actor-init-checkpoint');del cmd[i:i+2]
        if tag=='t3':cmd+=['--bc-checkpoint','checkpoints/e1-bc-full-003/final.pt']
        if tag=='t6':cmd+=['--expert-dataset','data/replay/e0-pickcube-gpu-replay-full-003/trajectory.state.pd_ee_delta_pos.physx_cuda.h5','--expert-audit','artifacts/e0-pickcube-gpu-replay-full-003/dataset-audit.json']
        jobs.append(dict(id=name,seconds=estimate['t5' if tag=='t5' else 't4']['limit_seconds'],command=cmd))
    budget=json.loads(Path('research_state.json').read_text())['budget']
    available=budget['gpu_hours_limit']*3600-budget['gpu_job_seconds_used']-budget['final_evaluation_reserve_seconds']
    required=sum(j['seconds'] for j in jobs)
    record=dict(training_interactions_each=262144,updates_each=64528,validation_episodes_per_checkpoint=40,
        validation_checkpoints=17,selection='fixed_final',training_seed=0,final_test_accessed=False,
        fixed_critic_hashes=hashes[0],estimate=estimate,available_seconds=available,required_max_seconds=required,
        fits=available>=required,reserve_seconds=7200,jobs=jobs)
    (out/'plan.json').write_text(json.dumps(record,indent=2)+'\n')
    assert record['fits'],'Preserve plan and request bounded redesign; do not silently change one method budget'
    print(json.dumps({k:v for k,v in record.items() if k!='jobs'},indent=2))

if __name__=='__main__':main()
