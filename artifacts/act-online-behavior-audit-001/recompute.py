"""Recompute portable real-replay counts and check all recorded behavior bins (CPU)."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.diagnose_behavior import summarize

parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
rows=[]
for folder in sorted(args.input.glob('act-*')):
    with np.load(folder/'transitions.npz') as stored:
        data={key:stored[key] for key in stored.files}
    rebuilt=summarize(data,8192)
    recorded=json.loads((folder/'result.json').read_text())
    assert all(recorded[key]==value for key,value in rebuilt.items())
    rows.append(dict(experiment=folder.name,transitions=len(data['source']),
        positive_reward_transitions=int((data['reward']>0).sum()),
        grasped_observation_transitions=int(data['grasped'].sum()),
        unique_grasped_episodes=len(np.unique(data['episode_id'][data['grasped']])),
        episodes_seen=len(np.unique(data['episode_id'])),
        successful_episodes=len(np.unique(data['episode_id'][data['reward']>0]))))
assert rows==json.loads((args.input/'counts.json').read_text())
with args.output.open('x') as stream:stream.write(json.dumps(rows,indent=2)+'\n')
