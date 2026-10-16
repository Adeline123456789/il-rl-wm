import hashlib,json,subprocess,tempfile
from pathlib import Path
out=Path('artifacts/source-history-preflight-001');report=json.loads((out/'result.json').read_text())
recovered=[];unresolved=[]
for sha,item in report['not_in_committed_blobs'].items():
 exp=item['experiments'][0];m=json.loads((Path('artifacts')/exp/'manifest.json').read_text());path=Path(item['path']);patch=(Path('artifacts')/exp/'working-tree.patch').resolve()
 base=subprocess.run(['git','show',f"{m['code_commit']}:{path}"],capture_output=True)
 with tempfile.TemporaryDirectory(prefix='ilrl-source-recovery-') as directory:
  folder=Path(directory);target=folder/path;target.parent.mkdir(parents=True)
  if base.returncode==0:target.write_bytes(base.stdout)
  applied=subprocess.run(['git','apply',f'--include={path}',str(patch)],cwd=folder,capture_output=True)
  if applied.returncode==0 and target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==sha:
   dest=out/'recovered'/sha;dest.parent.mkdir(exist_ok=True);dest.write_bytes(target.read_bytes())
   recovered.append(dict(path=str(path),sha256=sha,experiment=exp,base_commit=m['code_commit'],patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest()))
  else:unresolved.append(dict(sha256=sha,**item,reason='Not recovered from committed source blobs or recorded launch patch; do not substitute current bytes.'))
(out/'patch_recovery.json').write_text(json.dumps(recovered,indent=2)+'\n')
(out/'unresolved.json').write_text(json.dumps(unresolved,indent=2)+'\n')
assert len(recovered)==2 and len(unresolved)==1
