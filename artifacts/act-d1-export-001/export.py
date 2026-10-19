import torch,json,hashlib
from pathlib import Path
from scripts.distill_act_sac import make_student
p=Path('checkpoints/act-d1-seed0-001/final.pt');full=torch.load(p,map_location='cpu',weights_only=False)
a=make_student(-2).eval();a.load_state_dict(full['actor'])
out=Path('releases/act-d1-negative-001');out.mkdir()
w=out/'weights.pt';torch.save(dict(actor=full['actor'],config=full['config'],step=full['step'],teacher_sha256=full['teacher_sha256']),w)
r=torch.load(w,map_location='cpu',weights_only=False)
assert all(torch.equal(v,r['actor'][k]) for k,v in full['actor'].items())
(out/'manifest.json').write_text(json.dumps(dict(source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),portable_sha256=hashlib.sha256(w.read_bytes()).hexdigest(),actor='scripts.train_sac.Actor',state_dim=42,action_dim=4,teacher_required_at_deployment=False,negative_result='mean 1/40, stochastic 1/40; teacher 8/40',resume_training=False),indent=2)+'\n')
