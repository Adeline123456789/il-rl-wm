import pytest
from scripts.run_experiment import job_budget


def authorization():
    return dict(status='authorized', budget_id='new', total_seconds=11500,
                category_caps=dict(train=10000, evaluation=1200, smoke=300),
                jobs={'train': dict(cap_seconds=1900, category='train'),
                      'test': dict(cap_seconds=150, category='evaluation'),
                      'smoke': dict(cap_seconds=90, category='smoke')})


def test_original_and_new_budgets_do_not_borrow_each_other():
    state = dict(budget=dict(gpu_hours_limit=24))
    previous = [dict(elapsed_seconds=85238.75),
                dict(budget_id='new', experiment_id='train', elapsed_seconds=1000)]
    assert job_budget(state, previous, 'old', 2000) == (85238.75, 1161.25)
    assert job_budget(state, previous, 'train', 1900, authorization()) == (1000, 1900)


def test_category_reserve_and_cleanup_margin():
    previous = [dict(budget_id='new', experiment_id='train', elapsed_seconds=9990)]
    assert job_budget({}, previous, 'train', 1900, authorization())[1] < 0
    assert job_budget({}, previous, 'test', 150, authorization()) == (9990, 150)


def test_unregistered_or_oversized_job_rejected():
    with pytest.raises(AssertionError):
        job_budget({}, [], 'other', 1, authorization())
    with pytest.raises(AssertionError):
        job_budget({}, [], 'train', 1901, authorization())
    a = authorization(); a['status'] = 'proposal'
    with pytest.raises(AssertionError):
        job_budget({}, [], 'train', 1900, a)
