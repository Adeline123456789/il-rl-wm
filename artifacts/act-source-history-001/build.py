import hashlib,json,os,subprocess,tarfile
from pathlib import Path
out=Path('artifacts/act-source-history-001');cache=Path('.cache/act-source-history-001').resolve();cache.mkdir()
release=Path('releases/source-history-act-20261019-001');release.mkdir()
status=dict(pid=os.getpid(),phase='running',exit_code=None,GPU_use=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd):return subprocess.run(cmd,check=True)
try:
 old=json.loads(Path('releases/source-history-20261016-001/manifest.json').read_text())
 base=cache/'base.bundle'
 with base.open('wb') as f:
  for part in old['parts']:
   p=Path(part['path']);assert sha(p)==part['sha256'];f.write(p.read_bytes())
 assert base.stat().st_size==old['bundle_bytes'] and sha(base)==old['bundle_sha256']
 head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 bundle=cache/'increment.bundle'
 run(['git','-c','pack.threads=2','bundle','create',str(bundle),'HEAD','^'+old['source_head']])
 restored=cache/'restored.git'
 run(['git','init','--bare',str(restored)])
 run(['git','-C',str(restored),'fetch',str(base),'HEAD:refs/heads/base'])
 run(['git','-C',str(restored),'bundle','verify',str(bundle)])
 run(['git','-C',str(restored),'fetch',str(bundle),'HEAD:refs/heads/snapshot'])
 assert subprocess.check_output(['git','-C',str(restored),'rev-parse','refs/heads/snapshot'],text=True).strip()==head
 manifests={str(p):json.loads(p.read_text()) for p in Path('artifacts').glob('*/manifest.json')}
 commits=sorted({m['code_commit'] for m in manifests.values()})
 for commit in commits:run(['git','-C',str(restored),'cat-file','-e',commit+'^{commit}'])
 run(['git','-C',str(restored),'fsck','--full','--no-reflogs'])
 parts=[]
 with bundle.open('rb') as f:
  i=0
  while data:=f.read(4194304):
   p=release/f'source-history-increment.bundle.part{i:03d}';p.write_bytes(data)
   parts.append(dict(path=str(p),bytes=len(data),sha256=sha(p)));i+=1
 assert sum(p['bytes'] for p in parts)==bundle.stat().st_size
 report=dict(source_head=head,base_source_head=old['source_head'],base_manifest='releases/source-history-20261016-001/manifest.json',base_bundle_sha256=old['bundle_sha256'],bundle_bytes=bundle.stat().st_size,bundle_sha256=sha(bundle),parts=parts,experiment_manifest_count=len(manifests),experiment_code_commits_verified_in_fresh_bare_repository=commits,full_offline_restore_verified=True,network_used=False,scope='Committed local history through captured head, including the running U2 code commit. Dirty execution sources remain in individual source_snapshot.tar.gz/working-tree.patch; old unresolved plotting bytes remain explicitly missing.')
 (release/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 (out/'result.json').write_text(json.dumps({k:v for k,v in report.items() if k!='parts'},indent=2)+'\n')
 status.update(phase='completed',exit_code=0)
except BaseException as e:
 status.update(phase='failed',exit_code=1,error=repr(e));raise
finally:(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
