"""Small hand-calculated checks for order-specific eligibility and loss weights."""
import numpy as np
import pytest

from gx1.scripts.audit_entry_direction_order_geometry_v1 import compare_orders, MINUTE_NS


def fixture():
    n = 5
    parents = np.arange(n, dtype=np.int64)
    decision = 1_700_000_000_000_000_000 + parents * 5 * MINUTE_NS
    terminal = decision[:, None] + np.asarray([1, 5, 15], np.int64) * MINUTE_NS
    signs = np.asarray([[1, 1, 1], [-1, -1, -1], [0, 0, 0],
                        [1, 1, 1], [1, 1, 1]], dtype=np.int8)
    observed = np.asarray([[1, 1, 1], [1, 1, 1], [0, 0, 0],
                           [1, 1, 1], [1, 1, 1]], dtype=np.bool_)
    return parents, decision, terminal, observed, signs


def test_chronological_batch_counts_and_mean_coefficients_are_hand_calculated():
    a = fixture()
    got = compare_orders(*a, cutoff_ns=int(a[1][-1] + 16 * MINUTE_NS), batch_size=2, seed=17)
    q = got['chronological']
    assert got['eligible_by_horizon'] == [4, 4, 4]
    assert q['batches'] == q['nonempty_update_batches'] == 3
    assert q['five_minute_batch_classes'] == {'up_only': 2, 'down_only': 0, 'both': 1, 'no_eligible_five_minute': 0}
    assert q['last_batch']['parents'] == 1 and q['last_batch']['eligible_cells'] == 3
    assert q['mean_batch_loss_coefficient_mass_by_horizon'] == pytest.approx([1/3] * 3)
    assert q['mean_batch_loss_coefficient_mass_up_by_horizon'] == pytest.approx([5/18] * 3)


def test_empty_batch_skips_update_and_strict_cutoff_masks_immature_labels():
    a = fixture()
    a[3][:2] = False
    a[4][:2] = 0
    got = compare_orders(*a, cutoff_ns=int(a[1][-1] + MINUTE_NS), batch_size=2, seed=17)
    assert got['chronological']['empty_update_batches'] == 2
    assert got['chronological']['nonempty_update_batches'] == 1
    assert got['chronological']['last_batch']['eligible_cells'] == 0
    assert got['immature_by_horizon'][0] == 1


def test_seeded_mixed_order_is_reproducible_and_rejects_bad_target_storage():
    a = fixture()
    opts = {'cutoff_ns': int(a[1][-1] + 16 * MINUTE_NS), 'batch_size': 2, 'seed': 17}
    state = np.random.get_state()
    assert compare_orders(*a, **opts) == compare_orders(*a, **opts)
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    a[4][2, 0] = 1
    with pytest.raises(RuntimeError, match='TARGET_CLOCK_DOMAIN'):
        compare_orders(*a, **opts)
