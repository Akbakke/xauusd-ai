import pytest

from gx1.audit.fit_market_state_fold2_fixed_25_cell_prior_v1 import (
    _check_week_partition, terminal_payload,
)


def test_terminal_success_and_consumed_failure_are_distinct():
    claim = {'path': '/tmp/claim', 'sha256': 'a' * 64}
    result = {'path': '/tmp/result', 'sha256': 'b' * 64}
    success = terminal_payload(
        status='COMPLETE', exit_code=0, claim=claim,
        result=result, head='c' * 40)
    failure = terminal_payload(
        status='FAILED_CONSUMED', exit_code=1, claim=claim,
        result=None, head='c' * 40, reason='RuntimeError:example')
    assert success['result'] == result and success['exit_code'] == 0
    assert failure['result'] is None and failure['exit_code'] == 1
    assert failure['reason'] == 'RuntimeError:example'
    with pytest.raises(RuntimeError, match='TERMINAL_ARGS'):
        terminal_payload(
            status='COMPLETE', exit_code=1, claim=claim,
            result=result, head='c' * 40)


def test_week_partition_rejects_missing_score_mass():
    def cells(count, brier, log):
        return {'per_tf': {
            tf: {m: {'count': count, 'brier_sum': brier,
                     **({} if m == 'activity' else {'log_sum': log})}
                 for m in ('activity', 'conditional_side', 'three_class')}
            for tf in ('M5', 'M15', 'H1', 'H4', 'D1')}}
    overall = [cells(3, 1.5, 2.0) for _ in range(5)]
    weekly = {'w1': [cells(1, 0.5, 0.75) for _ in range(5)],
              'w2': [cells(2, 1.0, 1.25) for _ in range(5)]}
    _check_week_partition(overall, weekly)
    weekly['w2'][0]['per_tf']['M5']['conditional_side']['count'] = 1
    with pytest.raises(RuntimeError, match='WEEK_PARTITION'):
        _check_week_partition(overall, weekly)
