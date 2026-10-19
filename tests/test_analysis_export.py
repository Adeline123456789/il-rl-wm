import pytest
from scripts.export_analysis_checkpoint import training_interactions


def test_legacy_standalone_counter_without_mutation():
    state = dict(global_step=262144, config=dict(env_id='PickCube-v1'))
    assert training_interactions(state) == 262144
    assert 'lineage_training_interactions' not in state


def test_continuation_keeps_inherited_total():
    state = dict(global_step=262144, lineage_training_interactions=524288,
                 config=dict(resume_checkpoint='parent.pt', inherited_training_interactions=262144))
    assert training_interactions(state) == 524288


@pytest.mark.parametrize('config', [dict(resume_checkpoint='parent.pt'), dict(inherited_training_interactions=262144)])
def test_missing_continuation_total_is_not_guessed(config):
    with pytest.raises(AssertionError):
        training_interactions(dict(global_step=262144, config=config))
