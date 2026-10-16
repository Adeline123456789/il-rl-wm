"""Explicit episode identity across CPU conversion and reordered GPU replay."""


def annotate(saved, sources):
    assert len(saved) == len(sources), 'Every flushed episode must have one source'
    for target, source in zip(saved, sources):
        target['source_episode_id'] = int(source['episode_id'])
        target['original_episode_id'] = int(source.get('original_episode_id', source['episode_id']))
        assert int(target['episode_seed']) == int(source['episode_seed'])


def resolve_sources(sources, targets, field='source_episode_id'):
    by_id = {int(e['episode_id']): e for e in sources}
    assert len(by_id) == len(sources)
    if all(field in e for e in targets):
        matched = [by_id[int(e[field])] for e in targets]
    else:
        # Existing validated Pick/Stack recordings predate explicit provenance.
        # Accept that recorded format only when seeds uniquely identify sources.
        assert not any(field in e for e in targets), 'Mixed provenance metadata'
        by_seed = {int(e['episode_seed']): e for e in sources}
        assert len(by_seed) == len(sources), 'Ambiguous source seeds; explicit episode provenance required'
        matched = [by_seed[int(e['episode_seed'])] for e in targets]
    assert len({e['episode_id'] for e in matched}) == len(targets), 'Repeated source episode'
    assert all(int(s['episode_seed']) == int(t['episode_seed']) for s, t in zip(matched, targets))
    return matched
