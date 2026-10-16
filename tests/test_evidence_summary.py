import json

from scripts.summarize_evidence import summarize


def test_four_policy_evaluation_is_not_averaged_into_one_method(tmp_path):
    run=tmp_path/'u0';run.mkdir()
    (run/'manifest.json').write_text(json.dumps(dict(experiment_id='u0',code_commit='test',exit_code=0,stop_reason=None,elapsed_seconds=1,ended_at='done')))
    episodes=[dict(step=16,method=method,success_once=success,success_at_end=success,**{'return':success*200,'episode_len':200}) for method,success in [('il',0),('sac',1),('il_sac_wm',0),('sac_wm',1)]]
    (run/'validation_episodes.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in episodes))
    result=summarize(tmp_path)['experiments'][0]['validation_by_method_and_checkpoint']
    assert len(result)==4
    assert result['il']['16']['success_once']==0
    assert result['sac_wm']['16']['success_once']==1
    assert all(x['16']['episodes']==1 for x in result.values())
