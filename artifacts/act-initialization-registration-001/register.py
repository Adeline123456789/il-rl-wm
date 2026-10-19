import datetime,hashlib,json,subprocess,os,time
from pathlib import Path
root=Path.cwd();python=str(root/'.venv/bin/python')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
proposal=json.load(open('configs/act_next_initialization_controls.json'));now=datetime.datetime.now(datetime.timezone.utc).isoformat()
assert not Path('configs/act_initialization_budget.json').exists()
previous=[json.loads(p.read_text()) for p in Path('artifacts').glob('*/manifest.json')];assert len(previous)==216 and all(m['ended_at'] for m in previous)
original=sum(m['elapsed_seconds'] for m in previous);assert abs(original-85238.75034914288)<1e-6
configs={c['id']:c['config'] for c in proposal['conditions']};jobs={};expected={}
def training(tag,name,cap,smoke=False):
 c=dict(configs[tag]);c.update(output='artifacts/'+name,exp_name=name)
 if smoke:c.update(total_timesteps=8192,eval_freq=8192,log_freq=4096)
 command=[python,'scripts/train_sac.py']
 for key,value in c.items():
  if value is None:continue
  flag=key.replace('_','-')
  if isinstance(value,bool):command+=['--'+('' if value else 'no-')+flag]
  else:command+=['--'+flag,str(value)]
 jobs[name]=dict(category='smoke' if smoke else 'training',cap_seconds=cap,command=command)
 expected[name]=c
training('R0','act-r0-expert50-smoke-001',90,True)
training('R3','act-r3-expert50-smoke-001',210,True)
for tag,cap in [('R0',1900),('R1',1900),('R3',6200)]:training(tag,f'act-{tag.lower()}-expert50-seed0-001',cap)
protocol='configs/act_initialization_final_protocol.json';gate='configs/act_initialization_test_gate.json'
write(protocol,dict(status='registered_before_training',registered_at=now,seed=30000,episodes=100,num_envs=10,gate=gate,selection='fixed final for all four conditions; R2 reuses completed T6 without training',no_test_driven_tuning=True))
write(gate,dict(released=False,reason='Wait for all three fixed training controls; no final access during resource smoke/development'))
entries=[];evaljobs=[];exports={f'act-{t.lower()}-expert50-seed0-001':f'releases/act-{t.lower()}-expert50-final-001' for t in ['R0','R1','R3']}
for tag in ['R0','R1','R2','R3']:
 experiment='act-t6-online-seed0-001' if tag=='R2' else f'act-{tag.lower()}-expert50-seed0-001'
 checkpoint='releases/act-t6-online-final-001/weights.pt' if tag=='R2' else exports[experiment]+'/weights.pt'
 kind=configs[tag]['actor_init_kind']
 for mode in ['mean','stochastic']:
  condition=f'act-{tag.lower()}-controls-{mode}';name=f'e7-test-{condition}-001';evaljobs.append(name)
  command=[python,'scripts/evaluate_act_checkpoint.py','--checkpoint',checkpoint,'--kind',kind,'--mode',mode,'--seed','30000','--episodes','100','--num-envs','10','--training-seed','0','--training-interactions','262144','--output',f'artifacts/{name}','--final-protocol',protocol]
  jobs[name]=dict(category='evaluation',cap_seconds=150,command=command)
  entries.append(dict(id=condition,framework='act',task='PickCube-v1',training_seed=0,training_interactions=262144,training_experiment=experiment,training_role='development',variant=tag+' expert50 initialization control',kind=kind,mode=mode,methods=[kind+'_'+mode],checkpoint=checkpoint))
auth=dict(status='authorized',budget_id='act-initialization-controls-20261019',user_message='批准下一批 11,500 秒',authorized_at=now,total_seconds=11500,category_caps=dict(smoke=300,training=10000,evaluation=1200),original_24h_gpu_seconds=original,original_24h_balance_not_borrowed=True,new_local_mechanisms=0,scope='Only fixed50% expert R0/R1/R3 training and resource/final evaluation; R2 training reused. No tuning/WM/warmup/automation.',jobs=jobs)
write('configs/act_initialization_budget.json',auth)
write('configs/act_initialization_evaluation_plan.json',dict(seed=30000,episodes_per_model_method=100,num_envs=10,final_protocol=protocol,selection='fixed_final',entries=entries))
critic=json.load(open('artifacts/act-t6-online-seed0-001/actor_initialization.json'))
plan=dict(authorization='configs/act_initialization_budget.json',smoke_jobs=['act-r0-expert50-smoke-001','act-r3-expert50-smoke-001'],smoke_full_cap={'act-r0-expert50-smoke-001':1900,'act-r3-expert50-smoke-001':6200},training_jobs=[f'act-{t.lower()}-expert50-seed0-001' for t in ['R0','R1','R3']],exports=exports,evaluation_jobs=evaljobs,evaluation_plan='configs/act_initialization_evaluation_plan.json',test_gate=gate,expected_configs=expected,fixed_critic_hashes=[critic['critic1_hash'],critic['critic2_hash']],runtime_sources={str(p):digest(p) for p in sorted(Path('scripts').glob('*.py'))})
write('configs/act_initialization_execution_plan.json',plan)
s=json.load(open('research_state.json'));s['additional_budgets']={auth['budget_id']:dict(authorization='configs/act_initialization_budget.json',limit_seconds=11500,gpu_job_seconds_used=0)};s['phase']='Explicitly authorized separate11500s R0/R1/R3 controls; preregistered before GPU smoke';s['blockers']=[];s['next_experiment']=plan['smoke_jobs'][0];s['act_research']['next_gpu_training_authorized']=True;s['act_research']['next_stage']='Resource-gated serial R0/R1/R3; new final seed30000 preregistered, no WM';write('research_state.json',s)
with open('AGENTS.md','a') as f:f.write('\n2026-10-19 新授权：用户明确批准下一批11500 GPU作业秒，仅R0/R1/R3固定50%专家对照＋资源短检/评测；R2复用不重训。configs/act_initialization_budget.json独立账，不借旧余额；smoke300/train10000/eval1200，测试seed30000预登记。无额外调参/WM/预热或恢复定时任务。新队列按实际status/PID核实，不重复启动。\n')
with open('RESEARCH_CHARTER.md','a') as f:f.write('\n## 2026-10-19 新批算力授权（原批次边界不变）\n用户明确回复“批准下一批 11,500 秒”。新批仅固定50%专家回放的初始化因果对照R0/R1/R3及短检/评测；原24小时实际85238.750349秒关闭保存，其余额不并入本批。独立cap11500秒，单3090串行，无第三机制、无WM/预热调参、无定时任务。授权及精确命令见configs/act_initialization_budget.json。\n')
with open('RESEARCH_LOG.md','a') as f:f.write('\n## '+now+' 新批11500秒明确授权及预登记\n原216作业全部结束，原总账85238.750349秒；GPU查询为空。新批分账smoke300/train10000/eval1200；两短检及R0→R1→R3串行，R2不重训。各262144交互/64528更新、LN Critic哈希固定、50%专家。最终seed30000在训练前固定，R0/R1/R2/R3各均值/随机100回合；关闭独立gate直到三完整运行结束。原seed20000不用于本轮调参。无新增算法机制。\n')
print(json.dumps(dict(jobs=len(jobs),authorized=11500,old_gpu_seconds=original,seed=30000)))
