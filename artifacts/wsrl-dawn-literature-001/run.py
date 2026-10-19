import urllib.request,subprocess,json,hashlib,time,os
from pathlib import Path
p=Path('artifacts/wsrl-dawn-literature-001');t=time.time()
try:
 url='https://raw.githubusercontent.com/mlresearch/v306/main/assets/ma26s/ma26s.pdf'
 data=urllib.request.urlopen(url,timeout=40).read();assert data.startswith(b'%PDF')
 target=Path('.cache/dawn-paper.pdf');target.write_bytes(data)
 subprocess.run(['pdftotext','-layout',str(target),'.cache/dawn-paper.txt'],check=True)
 result=dict(phase='completed',exit_code=0,pid=os.getpid(),url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),elapsed_seconds=time.time()-t)
except Exception as e: result=dict(phase='failed',exit_code=1,pid=os.getpid(),error=repr(e),elapsed_seconds=time.time()-t)
(p/'status.json').write_text(json.dumps(result,indent=2));print(result)
