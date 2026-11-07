"""Only the plan-bound runner drift can be accepted by later read-only diagnostics."""
from pathlib import Path

import pytest

from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero


def test_receipt_reviewed_runner_is_exact_and_default_remains_strict():
    if not zero.FIXED['fold2_normalization'][0].exists():
        pytest.skip('Local immutable fold-2 receipt is unavailable')
    runner = zero.REPO / zero.RUNNER_SOURCE
    ref = {'path': str(runner), 'sha256': zero._sha(runner)}
    with pytest.raises(RuntimeError, match='REVIEWED_RUNTIME_DRIFT'):
        zero._inspect_receipts()
    with pytest.raises(RuntimeError, match='REVIEWED_RUNNER_REF'):
        zero._inspect_receipts(reviewed_runner_ref={**ref, 'sha256': '0' * 64})
    with pytest.raises(RuntimeError, match='REVIEWED_RUNNER_REF'):
        zero._inspect_receipts(reviewed_runner_ref={**ref, 'path': str(Path('/tmp/other.sh'))})
    receipts = zero._inspect_receipts(reviewed_runner_ref=ref)
    assert set(receipts['historical_lineage']['reviewed_runtime_drift']) == (
        set(zero.HISTORICAL_RUNTIME_BRIDGE) | {zero.RUNNER_SOURCE})
    assert receipts['fold1']['lineage']['current_source_owner_count'] == 25
