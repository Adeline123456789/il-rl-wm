import hashlib,json,subprocess
from pathlib import Path
out=Path('artifacts/source-history-preflight-001')
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
lines=subprocess.check_output(['git','rev-list','--objects',head,'--','scripts','configs','tests','src','vendor','external'],text=True).splitlines()
ids=[line.split()[0] for line in lines]
types=subprocess.check_output(['git','cat-file','--batch-check=%(objectname) %(objecttype)'],input='\n'.join(ids)+'\n',text=True).splitlines()
index={}
for line in types:
 oid,kind=line.split()
 if kind=='blob':
  data=subprocess.check_output(['git','cat-file','blob',oid]);index[hashlib.sha256(data).hexdigest()]=oid
missing={};total=0;experiments=[];commits=set()
for file in sorted(Path('artifacts').glob('*/source_hashes.json')):
 hashes=json.loads(file.read_text());experiments.append(file.parent.name)
 manifest=json.loads((file.parent/'manifest.json').read_text());commits.add(manifest['code_commit'])
 for name,sha in hashes.items():
  total+=1
  if sha not in index:missing.setdefault(sha,dict(path=name,experiments=[]))['experiments'].append(file.parent.name)
for commit in commits:subprocess.run(['git','merge-base','--is-ancestor',commit,head],check=True)
(out/'historical_blob_index.json').write_text(json.dumps(index,indent=2)+'\n')
(out/'result.json').write_text(json.dumps(dict(head=head,experiments=experiments,commits=sorted(commits),all_commits_reachable=True,recorded_source_entries=total,historical_source_blobs=len(index),not_in_committed_blobs=missing,scope='Compare recorded SHA256 with all committed source versions; missing entries may still be recoverable from saved tracked patches or current untracked files. No training execution.'),indent=2)+'\n')
print(json.dumps(dict(experiments=len(experiments),source_entries=total,unique_missing_hashes=len(missing))))
