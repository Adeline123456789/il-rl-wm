"""Audit recorded real transitions and split whole episodes without rewriting data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_replay_alignment import compare
from scripts.replay_provenance import resolve_sources

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def split_episodes(episodes, seed):
    # Peg's duplicate reset seeds have identical original actions/first states
    # (F-E0-005); keep their copies together to prevent train/validation leakage.
    groups = {}
    for e in episodes:
        groups.setdefault(int(e['episode_seed']), []).append(e['episode_id'])
    assert len(groups) >= 2, 'Need at least two independent source groups'
    order = np.random.default_rng(seed).permutation(list(groups)).tolist()
    cut = min(len(order)-1, max(1, int(0.8*len(order))))
    return ([i for key in order[:cut] for i in groups[key]],
            [i for key in order[cut:] for i in groups[key]])


def audit(path, seed, horizon, successful_only=False, source_metadata=None, action_source_path=None):
    metadata_path = path.with_suffix('.json')
    metadata = json.loads(metadata_path.read_text())
    env_info = metadata['env_info']
    if env_info['env_kwargs'].get('reward_mode') != 'sparse':
        raise ValueError('Require an explicit sparse reward replay; no implicit relabeling')
    if env_info['env_id'] not in {'PickCube-v1','StackCube-v1','PegInsertionSide-v1'}:
        raise ValueError('Reward semantics only verified for the initial three tasks')
    alignment = None
    if env_info['env_kwargs'].get('sim_backend') in {'gpu','physx_cuda'}:
        if action_source_path is None:
            raise ValueError('GPU replay requires --action-source-path to verify lengths, exact actions and first states')
        alignment = compare(action_source_path, path)
        if not alignment['valid']:
            raise ValueError(f"Replay alignment failed for {alignment['mismatched_episodes']} episodes")
    episodes = metadata['episodes']
    ids = [e['episode_id'] for e in episodes]
    assert len(set(ids)) == len(ids) and len(ids) >= 2
    rows = []
    initial_contacts = []
    with h5py.File(path, 'r') as dataset:
        for episode in episodes:
            key = f'traj_{episode["episode_id"]}'
            data = dataset[key]
            for required in ['obs','actions','rewards','terminated','truncated','success']:
                if required not in data:
                    raise ValueError(f'{key} missing {required}; replay with --record-rewards')
            obs, action, reward = (data[k][:] for k in ['obs','actions','rewards'])
            n = len(action)
            assert obs.ndim == 2 and action.ndim == 2
            assert len(obs) == n+1 and n > 0
            for name in ['rewards','terminated','truncated','success']:
                assert data[name].shape == (n,), (key,name,data[name].shape,n)
            for array in [obs, action, reward]:
                assert np.isfinite(array).all(), key
            assert np.max(np.abs(action)) <= 1.00001, (key,'expected normalized actions')
            if env_info['env_id'] == 'PickCube-v1':
                # Locked state order: qpos(9), qvel(9), is_grasped, ..., tcp_to_obj(3).
                initial_contacts.append(dict(episode_id=episode['episode_id'],
                    grasped=bool(obs[0,18] > 0.5), distance=float(np.linalg.norm(obs[0,36:39]))))
            # Locked task evaluate methods have success and no fail: reward == success.
            assert 'fail' not in data, 'Re-audit reward semantics if a failure label is present'
            assert np.array_equal(reward, data['success'][:].astype(float)), key
            assert episode['elapsed_steps'] == n
            rows.append(dict(episode_id=episode['episode_id'], episode_seed=episode.get('episode_seed'),
                             transitions=n, success_once=bool(data['success'][:].any()),
                             success_at_end=bool(data['success'][-1]),
                             nonzero_rewards=int(np.count_nonzero(reward)), return_=float(reward.sum()),
                             terminated_count=int(data['terminated'][:].sum()),
                             truncated_count=int(data['truncated'][:].sum()),
                             beyond_planned_horizon=max(n-horizon,0)))
    assert not any(r['grasped'] and r['distance'] > .1 for r in initial_contacts), 'F-E0-004: stale initial contact observation'
    eligible = [r['episode_id'] for r in rows if not successful_only or r['success_at_end']]
    if len(eligible) < 2:
        raise ValueError('Need at least two eligible episodes for an independent split')
    train_ids, validation_ids = split_episodes(
        [r for r in rows if r['episode_id'] in eligible], seed)
    source_mapping = None
    if source_metadata is not None:
        original = json.loads(source_metadata.read_text())
        assert original['env_info']['env_id'] == env_info['env_id']
        matched = resolve_sources(original['episodes'], episodes, 'original_episode_id')
        source_mapping = {str(e['episode_id']):s['episode_id'] for e,s in zip(episodes,matched)}
    return dict(initial_contact_audit=initial_contacts, alignment_validated=alignment is not None and alignment['valid'], action_source_sha256=digest(action_source_path) if action_source_path else None, dataset=str(path), dataset_sha256=digest(path), metadata_sha256=digest(metadata_path),
                env_info=env_info, split_seed=seed, train_episode_ids=sorted(train_ids),
                validation_episode_ids=sorted(validation_ids), episodes=rows,
                split_unit='whole episode groups sharing original reset seed; duplicate originals retained within group',
                total_transitions=sum(r['transitions'] for r in rows), planned_horizon=horizon,
                successful_only=successful_only, excluded_episode_ids=sorted(set(ids)-set(eligible)),
                source_episode_mapping=source_mapping,
                source_metadata_sha256=digest(source_metadata) if source_metadata else None,
                caution='Raw terminated flags are preserved. Decide and document continuing-vs-episodic TD masks separately; never concatenate episode boundaries for multi-step WM training.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--successful-only', action='store_true')
    parser.add_argument('--source-metadata', type=Path)
    parser.add_argument('--action-source-path', type=Path)
    args = parser.parse_args()
    result = audit(args.path, args.seed, args.horizon, args.successful_only, args.source_metadata, args.action_source_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['episodes','env_info']}, indent=2))


if __name__ == '__main__':
    main()
