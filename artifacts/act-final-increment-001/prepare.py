import json,os,subprocess,hashlib,time,tarfile
from pathlib import Path
out=Path('.cache/github-act-stage-020');out.mkdir(exist_ok=False)
receipt=Path('artifacts/act-final-increment-001');receipt.mkdir(exist_ok=False)
(receipt/'status.json').write_text(json.dumps(dict(pid=os.getpid(),phase='running',started_at=time.time())))
try:
 selected=json.loads(Path('artifacts/act-final-evaluation-batch-003/status.json').read_text())['completed'][:25]
 assert len(selected)==25
 paths=[];entries=[]
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 for x in selected:
  d=Path('artifacts')/x['experiment']; package=Path(x['package'])
  m=json.loads((d/'manifest.json').read_text());p=json.loads((package/'manifest.json').read_text());r=json.loads((d/'result.json').read_text())
  assert m['exit_code']==0 and m['ended_at']
  assert sha(package/'weights.pt')==p['sha256']==r['sha256']
  assert sha(d/'source_snapshot.tar.gz')==m['source_snapshot_sha256']
  assert sha(d/'working-tree.patch')==m['tracked_patch_sha256']
  assert r['purpose']=='final_test' and r['seed']==20000 and r['episodes_per_method']==100
  episodes=[json.loads(l) for l in (d/'episodes.jsonl').read_text().splitlines()]
  assert len(episodes)==100*len(r['methods'])
  files=[dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)) for folder in [d,package] for f in sorted(folder.rglob('*')) if f.is_file()]
  assert all(f['bytes']<10_000_000 for f in files)
  entries.append(dict(**x,conditions=len(r['methods']),episodes=len(episodes),files=files))
  paths.extend([str(d),str(package)])
 (receipt/'inventory.json').write_text(json.dumps(dict(captured_at=time.time(),source_queue='act-final-evaluation-batch-003',complete_batch=False,entries=entries),indent=2)+'\n')
 (receipt/'README.md').write_text('# 正式评测增量证据 001\n\n仅收录003队列前25个已正常结束的登记项；002的A/B已在前次交付。其余正式评测仍按原106条件计划运行，此包不表示整批完成。每项保留原始逐回合、资源/交互记录、执行源码tar/patch和便携权重，逐文件SHA256见inventory.json。没有训练、测试选模或新增GPU预算。\n\n用户已取消每小时定时续接；未终止已授权后台评测和终态CPU审计。\n')
 (receipt/'status.json').write_text(json.dumps(dict(pid=os.getpid(),phase='completed',exit_code=0,entries=len(entries),conditions=sum(x['conditions'] for x in entries),ended_at=time.time()),indent=2)+'\n')
 env=dict(os.environ,GIT_INDEX_FILE=str((out/'index').resolve()))
 def git(*args):return subprocess.check_output(['git',*args],env=env,text=True).strip()
 base='2f96b05c0376a255bc73d09d821d56d3ffaa9a85';git('read-tree',base)
 paths += [str(receipt),'artifacts/github-act-stage-upload-019']
 git('add','--',*paths);tree=git('write-tree');files=[]
 for line in git('ls-tree','-rl',tree).splitlines():
  meta,path=line.split('\t',1);mode,kind,blob,size=meta.split();files.append(dict(path=path,mode=mode,type=kind,sha=blob,size=int(size)))
 changed=set(git('diff','--name-only',base,tree).splitlines());delta=[x for x in files if x['path'] in changed];assert len(delta)==len(changed)
 plan=dict(local_source_head=git('rev-parse','HEAD'),capture='temporary index tree; local HEAD unchanged',tree=tree,base_tree=base,remote_parent='c7d82c4ef636389b6f3b492ea37fd65b5fca20a4',files=files,delta=delta)
 (out/'plan.json').write_text(json.dumps(plan));print(json.dumps({k:v for k,v in plan.items() if k not in ['files','delta']}));print(len(delta),sum(x['size'] for x in delta))
except BaseException as e:
 (receipt/'status.json').write_text(json.dumps(dict(pid=os.getpid(),phase='failed',exit_code=1,error=repr(e))))
 raise
