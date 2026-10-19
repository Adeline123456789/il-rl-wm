import pytest
from scripts.run_initialization_controls import project_full_duration


def test_setup_is_charged_once_while_repeated_validation_is_scaled():
    p = project_full_duration(80, 64, 8, 12)
    assert p['one_time_setup_cleanup_seconds'] == 16
    assert p['in_loop_other_seconds'] == 44
    assert p['projected_full_seconds'] == pytest.approx(8 / 1040 * 64528 + 12 * 32 + 44 * 8.5 + 16)
    assert p['previous_all_overhead_scaled_seconds'] - p['projected_full_seconds'] == 120


def test_impossible_timer_order_rejected():
    with pytest.raises(AssertionError):
        project_full_duration(60, 64, 8, 12)
    with pytest.raises(AssertionError):
        project_full_duration(80, 64, 60, 12)
