import hashlib,json,subprocess
from pathlib import Path
out=Path('artifacts/source-history-preflight-001');bundle=Path('.cache/source-history-preflight-001.bundle').resolve()
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
subprocess.run(['git','-c','pack.threads=2','bundle','create',str(bundle),'HEAD'],check=True)
subprocess.run(['git','bundle','verify',str(bundle)],check=True)
restored=Path('.cache/source-history-restored-001.git').resolve()
subprocess.run(['git','init','--bare',str(restored)],check=True)
subprocess.run(['git','-C',str(restored),'fetch',str(bundle),'HEAD:refs/heads/snapshot'],check=True)
report=json.loads((out/'result.json').read_text())
for commit in report['commits']:
 subprocess.run(['git','-C',str(restored),'cat-file','-e',commit+'^{commit}'],check=True)
assert subprocess.check_output(['git','-C',str(restored),'rev-parse','refs/heads/snapshot'],text=True).strip()==head
(out/'bundle_result.json').write_text(json.dumps(dict(valid=True,head=head,bundle=str(bundle),bytes=bundle.stat().st_size,sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),restored_original_commits=report['commits'],network_used=False,note='Preflight bundle is ignored local cache; regenerate final bundle after code freezes and deliver it. This test does not upload or include later uncommitted changes.'),indent=2)+'\n')
