import json

import h5py
import numpy as np
import pytest

from scripts.replay_provenance import annotate, resolve_sources
from scripts.audit_replay_alignment import compare
from scripts.audit_demos import split_episodes


def test_duplicate_seeds_survive_two_reordered_replays(tmp_path):
    original = [dict(episode_id=i, episode_seed=100) for i in [76, 100]]
    cpu = [dict(episode_id=i, episode_seed=100) for i in [0, 1]]
    annotate(cpu, original)
    gpu = [dict(episode_id=i, episode_seed=100) for i in [0, 1]]
    annotate(gpu, cpu[::-1])
    assert [e['episode_id'] for e in resolve_sources(original, gpu, 'original_episode_id')] == [100, 76]
    for name, episodes in [('cpu', cpu), ('gpu', gpu)]:
        path = tmp_path / f'{name}.h5'
        path.with_suffix('.json').write_text(json.dumps(dict(episodes=episodes)))
        with h5py.File(path, 'w') as f:
            for e in episodes:
                g = f.create_group(f"traj_{e['episode_id']}")
                value = e['original_episode_id']
                g['actions'] = np.full((2, 4), value)
                g.create_group('env_states')['position'] = np.full((3, 3), value)
    assert compare(tmp_path/'cpu.h5', tmp_path/'gpu.h5')['valid']
    # A wrong map with the same seed must still fail the physical/action audit.
    gpu[0]['source_episode_id'], gpu[1]['source_episode_id'] = 0, 1
    (tmp_path/'gpu.json').write_text(json.dumps(dict(episodes=gpu)))
    assert not compare(tmp_path/'cpu.h5', tmp_path/'gpu.h5')['valid']


def test_legacy_unique_seeds_only():
    source = [dict(episode_id=i, episode_seed=i) for i in [4, 7]]
    target = [dict(episode_id=0, episode_seed=7)]
    assert resolve_sources(source, target)[0]['episode_id'] == 7
    source[0]['episode_seed'] = 7
    with pytest.raises(AssertionError, match='Ambiguous'):
        resolve_sources(source, target)


def test_duplicate_sources_cannot_cross_split_and_unique_split_unchanged():
    unique = [dict(episode_id=i+50, episode_seed=i+200) for i in range(10)]
    train, validation = split_episodes(unique, 42)
    old = np.random.default_rng(42).permutation([e['episode_id'] for e in unique]).tolist()
    assert train == old[:8] and validation == old[8:]
    duplicate = [dict(episode_id=i+100, episode_seed=i+200) for i in range(10)]
    train, validation = split_episodes(unique+duplicate, 42)
    by_id = {e['episode_id']: e['episode_seed'] for e in unique+duplicate}
    assert not ({by_id[i] for i in train} & {by_id[i] for i in validation})
    assert len(train)+len(validation) == 20
