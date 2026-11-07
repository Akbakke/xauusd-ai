"""Small synthetic checks for exact clocks, cutoff censoring and mask semantics."""
import io

import numpy as np
import pytest

from gx1.contracts import market_state_five_endpoint_target_binding_v1 as binding


def _archives(*, missing_join=False, bad_bar=False, bad_side=False, bad_source=False):
    start = binding.TRAIN_START_NS
    cutoff = binding.FOLD_BOUNDARIES_NS[0]
    model_issues = np.array([start + binding.M5_NS, cutoff - binding.M5_NS,
                             cutoff + binding.M5_NS], dtype=np.int64)
    target_issues = np.array([model_issues[0], model_issues[0] + binding.M5_NS,
                              model_issues[1], model_issues[2]], dtype=np.int64)
    if missing_join:
        model_issues[2] += binding.M5_NS
    starts = model_issues - binding.M5_NS
    if bad_bar:
        starts[0] += 1
    model = {key: np.zeros(3) for key in binding.MODEL_KEYS}
    model.update(parent_rows=np.array([10, 20, 30], dtype=np.int64),
                 bar_start_ns=starts, decision_ns=model_issues)
    allowed = np.array([[1, 2, 4, 5, 7], [7, 7, 7, 7, 7],
                        [5, 1, 2, 4, 7], [7, 5, 7, 1, 4]], dtype=np.uint8)
    source = np.arange(20, dtype=np.int64).reshape(4, 5) + 100
    source[1, :] = -1
    if bad_source:
        source[0, 0] = -1
    side = np.isin(allowed, (1, 4))
    if bad_side:
        side[0, 3] = True
    mask = {
        "issue_ns": target_issues,
        "m5_source_row": np.array([987341, 987342, 987343, 987344], dtype=np.int64),
        "lead_minutes": np.array(binding.LEAD_MINUTES, dtype=np.int16),
        "m1_source_row": source,
        "allowed_set": allowed,
        "activity_observed": np.isin(allowed, (1, 2, 4, 5)),
        "activity_active": np.isin(allowed, (1, 4, 5)),
        "side_observed": side,
        "side_up": allowed == 4,
        "singleton_observed": np.isin(allowed, (1, 2, 4)),
    }
    model_buf, mask_buf = io.BytesIO(), io.BytesIO()
    np.savez(model_buf, **model)
    np.savez(mask_buf, **mask)
    model_buf.seek(0)
    mask_buf.seek(0)
    return np.load(model_buf, allow_pickle=False), np.load(mask_buf, allow_pickle=False)


def _bind(**changes):
    model, mask = _archives(**changes)
    with model, mask:
        return binding._bind_arrays(model, mask)


def test_exact_time_join_preserves_parent_order_and_censors_at_fold_boundary():
    bound = _bind()
    assert bound.unjoined_target_rows == 1
    assert bound.parent_rows.tolist() == [10, 20, 30]
    assert bound.m5_source_row.tolist() == [987341, 987343, 987344]
    issues = bound.decision_ns[[1, 0]]
    target = bound.gather(np.array([20, 10], np.int64), issues,
                          fold_id=1, partition="fit")
    assert target["allowed"].tolist() == [[5, 1, 2, 4, 7], [1, 2, 4, 5, 7]]
    assert target["eligible"].tolist() == [[True] * 4 + [False]] * 2
    assert target["mature"].tolist() == [[False] * 5, [True] * 5]
    assert target["validity_ns"][0, 0] == binding.FOLD_BOUNDARIES_NS[0]
    score = bound.gather(np.array([30], np.int64), bound.decision_ns[[2]],
                         fold_id=1, partition="score")
    assert score["eligible"].tolist() == [[False, True, False, True, True]]
    assert score["mature"].all()


@pytest.mark.parametrize("change,reason", [
    ({"missing_join": True}, "EXACT_ISSUE_JOIN"),
    ({"bad_bar": True}, "AXES_AND_BITS"),
    ({"bad_side": True}, "SIDE_OBSERVED_PARITY"),
    ({"bad_source": True}, "INFORMATIVE_SOURCE"),
])
def test_invalid_axis_or_semantics_fails_closed(change, reason):
    with pytest.raises(ValueError, match=reason):
        _bind(**change)


def test_wrong_query_clock_or_partition_fails_closed():
    bound = _bind()
    with pytest.raises(ValueError, match="QUERY_EXACT_CLOCK"):
        bound.gather(np.array([10], np.int64), bound.decision_ns[[1]],
                     fold_id=1, partition="fit")
    with pytest.raises(ValueError, match="QUERY_PARTITION"):
        bound.gather(np.array([30], np.int64), bound.decision_ns[[2]],
                     fold_id=1, partition="fit")
    with pytest.raises(ValueError, match="FOLD"):
        bound.gather(np.array([10], np.int64), bound.decision_ns[[0]],
                     fold_id=6, partition="fit")


def test_pinned_file_hash_is_checked(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    path.write_bytes(b"{}")
    monkeypatch.setitem(binding.REFS, "mask_result", (str(path), "0" * 64))
    with pytest.raises(ValueError, match="MASK_RESULT_HASH"):
        binding._checked_bytes("mask_result", 100)
