"""Offline restoration from the two declared manifests; no GPU or network."""
import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
new=json.loads((Path(__file__).parent/'manifest.json').read_text())
old=json.loads((root/new['base_manifest']).read_text())
for name,manifest in [('base',old),('increment',new)]:
 bundle=out/(name+'.bundle')
 with bundle.open('wb') as stream:
  for item in manifest['parts']:
   data=(root/item['path']).read_bytes()
   assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
   stream.write(data)
 assert bundle.stat().st_size==manifest['bundle_bytes']
 assert hashlib.sha256(bundle.read_bytes()).hexdigest()==manifest['bundle_sha256']
repo=out/'history.git'
subprocess.run(['git','init','--bare',str(repo)],check=True)
for name in ['base','increment']:
 subprocess.run(['git','-C',str(repo),'bundle','verify',str(out/(name+'.bundle'))],check=True)
 subprocess.run(['git','-C',str(repo),'fetch',str(out/(name+'.bundle')),'HEAD:refs/heads/'+name],check=True)
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','refs/heads/increment'],text=True).strip()==new['source_head']
for commit in new['experiment_code_commits_verified_in_fresh_bare_repository']:
 subprocess.run(['git','-C',str(repo),'cat-file','-e',commit+'^{commit}'],check=True)
subprocess.run(['git','-C',str(repo),'fsck','--full','--no-reflogs'],check=True)
print('Verified original source history:',repo)
