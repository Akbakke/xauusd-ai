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
    assert zero._reviewed_runner_sha(ref) == ref['sha256']
    # New TA code is a separate design change, not covered by the old runner-only bridge.
    with pytest.raises(RuntimeError, match='REVIEWED_RUNTIME_DRIFT'):
        zero._inspect_receipts(reviewed_runner_ref=ref)


def test_ta_loader_preserves_historical_fit_and_binds_current_sources(monkeypatch):
    if not zero.FIXED['fold1_normalization'][0].exists():
        pytest.skip('Local immutable fold-1 receipt is unavailable')
    original = zero._read_ref
    def no_fold2(ref, **kwargs):
        assert ref not in (zero._fixed_ref('fold2_normalization'), zero._fixed_ref('fold2_qualification'))
        return original(ref, **kwargs)
    monkeypatch.setattr(zero, '_read_ref', no_fold2)
    receipts = zero.load_ta_input_receipts()
    assert receipts['fold_ids'] == [1]
    assert receipts['fold1']['cohort']['joined_fit_rows'] == 29574
    assert receipts['fold1']['profile']['contract_sha256'] == zero.fold1.PROFILE_SHA256
    lineage = receipts['fold1']['lineage']
    assert len(lineage['historical_source_owners']) == 25
    assert set(lineage['changed_runtime_owners']) <= zero.TA_PREFIX_RUNTIME_SOURCE_CHANGES
    projected = receipts['ta_normalization']
    assert projected['ta_projection']['source_contract'] == receipts['fold1']['normal']['base_normalization']
    assert projected['ta_projection']['unobserved_units_profile'] == receipts['fold1']['profile']
    mtf = projected['surfaces']['mtf_m5']
    assert mtf['scale_source'][mtf['field_names'].index('geomchan_apex_proximity')] == 'source_bounded_unit_scale'
    view = zero._bind_ta_view(receipts)
    assert view['physical_widths'] == {'signal': 315, 'ctx_cont': 132, 'ctx_cat': 1, 'mtf': 253}
    assert view['prefix_normalization']['ta_base_normalization_sha256'] == projected['contract_sha256']
    assert view['prefix_normalization']['source_owner_profile_sha256'] == (
        lineage['historical_source_owners']['gx1/features/entry_specialist_feature_groups_v1.py'])
    bad = dict(receipts['ta_source_owners'])
    bad['gx1/features/htf_features.py'] = '0' * 64
    with pytest.raises(RuntimeError, match='TA_CURRENT_SOURCE'):
        zero.load_ta_input_receipts(current_source_owners=bad)
