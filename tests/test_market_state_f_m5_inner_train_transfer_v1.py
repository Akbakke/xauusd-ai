import numpy as np
import pytest

from gx1.audit import run_market_state_f_m5_inner_train_transfer_v1 as work


def test_inner_split_purges_unmatured_five_minute_label():
    issues = np.arange(1001, dtype=np.int64) * work.LEAD_NS + 1_000_000_000_000
    validity = issues + work.LEAD_NS
    selected = np.ones(len(issues), dtype=np.bool_)
    train, validation, boundary = work.inner_split(issues, validity, selected)
    assert boundary == issues[800]
    assert np.array_equal(train, np.arange(799))
    assert np.array_equal(validation, np.arange(800, 1001))
    assert validity[train[-1]] < boundary
    assert validity[799] == boundary
    with pytest.raises(RuntimeError, match='SPLIT_INPUT'):
        work.inner_split(issues, validity + 1, selected)


def test_inner_checkpoint_choice_uses_only_strict_brier_gain_and_earliest_tie():
    records = [{'rounds': r, 'brier': 0.25} for r in work.CHECKPOINTS]
    assert work.choose_round(records, 0.25) == 0
    records[2]['brier'] = 0.249
    records[3]['brier'] = 0.249
    assert work.choose_round(records, 0.25) == 4
    records[0]['brier'] = 0.248
    assert work.choose_round(records, 0.25) == 1


def test_terminal_payload_separates_success_and_failure_receipts():
    plan = {'path': 'plan', 'sha256': 'a' * 64}
    claim = {'path': 'claim', 'sha256': 'b' * 64}
    receipt = {'path': 'receipt', 'sha256': 'c' * 64}
    success = work.terminal_payload('COMPLETE', plan, claim, receipt, 5)
    failure = work.terminal_payload('FAILED', plan, claim, receipt, 2)
    assert success['exit_code'] == 0 and success['result'] == receipt
    assert 'failure' not in success and success['test_data_used'] is False
    assert failure['exit_code'] == 1 and failure['failure'] == receipt
    assert 'result' not in failure and failure['no_relaunch_under_this_plan'] is True
