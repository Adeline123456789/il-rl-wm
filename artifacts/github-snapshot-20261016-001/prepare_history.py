import hashlib,json,subprocess,time
from pathlib import Path
out=Path('artifacts/github-snapshot-20261016-001');target=Path('releases/source-history-20261016-001')
target.mkdir(exist_ok=False);head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
bundle=Path('.cache/source-history-20261016-001.bundle')
with (out/'bundle.log').open('w') as log:
 subprocess.run(['git','-c','pack.threads=2','bundle','create',str(bundle),head],stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run(['git','bundle','verify',str(bundle)],stdout=log,stderr=subprocess.STDOUT,check=True)
 restore=Path('.cache/source-history-upload-restore-001.git')
 subprocess.run(['git','init','--bare',str(restore)],stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run(['git','-C',str(restore),'fetch',str(bundle.resolve()),head],stdout=log,stderr=subprocess.STDOUT,check=True)
 commits=sorted({json.loads(p.read_text()).get('code_commit') for p in Path('artifacts').glob('*/manifest.json') if json.loads(p.read_text()).get('code_commit')})
 for sha in commits:subprocess.run(['git','-C',str(restore),'cat-file','-e',sha+'^{commit}'],check=True,stdout=log,stderr=subprocess.STDOUT)
 parts=[];h=hashlib.sha256()
 with bundle.open('rb') as stream:
  for i,block in enumerate(iter(lambda:stream.read(4*1024*1024),b'')):
   p=target/f'source-history.bundle.part{i:03d}';p.write_bytes(block);h.update(block)
   parts.append({'path':str(p),'bytes':len(block),'sha256':hashlib.sha256(block).hexdigest()})
 record={'source_head':head,'bundle_bytes':bundle.stat().st_size,'bundle_sha256':h.hexdigest(),'parts':parts,'experiment_code_commits_verified_in_fresh_bare_repository':commits,'full_offline_restore_verified':True}
 (target/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
 (out/'bundle-status.json').write_text(json.dumps({'exit_code':0,**record},indent=2)+'\n')
