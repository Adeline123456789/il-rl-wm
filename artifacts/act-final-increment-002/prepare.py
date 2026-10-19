import json,os,subprocess,hashlib,time,tarfile
from pathlib import Path
out=Path('.cache/github-act-stage-021');out.mkdir(exist_ok=False)
receipt=Path('artifacts/act-final-increment-002');receipt.mkdir(exist_ok=False)
(receipt/'status.json').write_text(json.dumps(dict(pid=os.getpid(),phase='running',started_at=time.time())))
try:
 selected=json.loads(Path('artifacts/act-final-evaluation-batch-003/status.json').read_text())['completed'][25:50]
 assert len(selected)==25
 paths=[];entries=[]
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 for x in selected:
  d=Path('artifacts')/x['experiment']; package=Path(x['package'])
  m=json.loads((d/'manifest.json').read_text());p=json.loads((package/'manifest.json').read_text());r=json.loads((d/'result.json').read_text())
  assert m['exit_code']==0 and m['ended_at']
  if x['id'].startswith('rlpd-'):
   assert sha(package/'actor.msgpack')==p['actor_sha256']==r['actor_sha256']
   conditions=1; episode_count=r['episodes']
  else:
   assert sha(package/'weights.pt')==p['sha256']==r['sha256']
   conditions=len(r['methods']); episode_count=r['episodes_per_method']
  assert sha(d/'source_snapshot.tar.gz')==m['source_snapshot_sha256']
  assert sha(d/'working-tree.patch')==m['tracked_patch_sha256']
  assert r['purpose']=='final_test' and r['seed']==20000 and episode_count==100
  episodes=[json.loads(l) for l in (d/'episodes.jsonl').read_text().splitlines()]
  assert len(episodes)==100*conditions
  files=[dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)) for folder in [d,package] for f in sorted(folder.rglob('*')) if f.is_file()]
  assert all(f['bytes']<10_000_000 for f in files)
  entries.append(dict(**x,conditions=conditions,episodes=len(episodes),files=files))
  paths.extend([str(d),str(package)])
 (receipt/'inventory.json').write_text(json.dumps(dict(captured_at=time.time(),source_queue='act-final-evaluation-batch-003',complete_batch=False,entries=entries),indent=2)+'\n')
 (receipt/'README.md').write_text('# 正式评测增量证据 002\n\n仅收录003队列第26至50个已正常结束的登记项；此前25项及002的A/B已交付，合计覆盖全部52个历史模型项、88条件。ACT正式评测继续按原106条件计划运行，此包不表示整批完成。每项保留原始逐回合、资源/交互记录、执行源码tar/patch和便携权重，逐文件SHA256见inventory.json。没有训练、测试选模或新增GPU预算。\n\n用户已取消每小时定时续接；未终止已授权后台评测和终态CPU审计。\n')
 (receipt/'status.json').write_text(json.dumps(dict(pid=os.getpid(),phase='completed',exit_code=0,entries=len(entries),conditions=sum(x['conditions'] for x in entries),ended_at=time.time()),indent=2)+'\n')
 env=dict(os.environ,GIT_INDEX_FILE=str((out/'index').resolve()))
 def git(*args):return subprocess.check_output(['git',*args],env=env,text=True).strip()
 base='fb413e488c64bf302f867142516dbe34a20c8709';git('read-tree',base)
 (receipt/'prepare.py').write_text(Path(__file__).read_text())
 paths += [str(receipt),'artifacts/github-act-stage-upload-020','artifacts/act-final-increment-001','RESEARCH_LOG.md']
 git('add','--',*paths);tree=git('write-tree');files=[]
 for line in git('ls-tree','-rl',tree).splitlines():
  meta,path=line.split('\t',1);mode,kind,blob,size=meta.split();files.append(dict(path=path,mode=mode,type=kind,sha=blob,size=int(size)))
 changed=set(git('diff','--name-only',base,tree).splitlines());delta=[x for x in files if x['path'] in changed];assert len(delta)==len(changed)
 plan=dict(local_source_head=git('rev-parse','HEAD'),capture='temporary index tree; local HEAD unchanged',tree=tree,base_tree=base,remote_parent='3841964cbb26a38d26a55eea0611bd058f189ce9',files=files,delta=delta)
 (out/'plan.json').write_text(json.dumps(plan));print(json.dumps({k:v for k,v in plan.items() if k not in ['files','delta']}));print(len(delta),sum(x['size'] for x in delta))
except BaseException as e:
 (receipt/'status.json').write_text(json.dumps(dict(pid=os.getpid(),phase='failed',exit_code=1,error=repr(e))))
 raise
