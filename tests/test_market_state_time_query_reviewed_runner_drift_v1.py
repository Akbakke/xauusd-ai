"""Reviewed runner and exact Counter drift retain separate source gates."""
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
    assert set(lineage['changed_runtime_owners']) - zero.TA_PREFIX_RUNTIME_SOURCE_CHANGES == {
        zero.TA_SIGNAL_COUNTER_SOURCE}
    assert (lineage['historical_source_owners'][zero.TA_SIGNAL_COUNTER_SOURCE],
            lineage['current_source_owners'][zero.TA_SIGNAL_COUNTER_SOURCE]) == (
                zero.TA_SIGNAL_COUNTER_SOURCE_SHA256)
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

def test_ta_counter_bridge_requires_exact_source_pair_and_bound_parity(monkeypatch):
    source = zero.TA_SIGNAL_COUNTER_SOURCE
    old_sha, new_sha = zero.TA_SIGNAL_COUNTER_SOURCE_SHA256
    semantics = '4402d8dc6be0fdae3e8fadf322403f2fe40e41a1c4787ba080d7e49da68b216d'
    before_ref = {'path': '/before.json', 'sha256': '1' * 64}
    after_ref = {'path': '/after.json', 'sha256': '2' * 64}
    parity = {
        'schema_version': 'gx1_ta_duplicate_check_performance_result_v1',
        'status': 'PASS_METADATA_AND_FOCUSED_TA_TESTS',
        'valid_outputs_and_error_semantics_identical': True,
        'test_data_used': False, 'before': before_ref, 'after': after_ref,
        'semantics_sha256': semantics,
    }
    before = {'owner_sha256': old_sha, 'semantics_sha256': semantics}
    after = {'owner_sha256': new_sha, 'semantics_sha256': semantics,
             'identical_valid_outputs_contracts_and_errors': True}
    reads = []
    def read_ref(ref):
        reads.append(ref)
        if ref == zero.TA_SIGNAL_COUNTER_PARITY_REF:
            return parity
        if ref == before_ref:
            return before
        assert ref == after_ref
        return after
    monkeypatch.setattr(zero, '_read_ref', read_ref)
    assert source not in zero.TA_PREFIX_RUNTIME_SOURCE_CHANGES
    assert zero._require_ta_prefix_source_changes({source: old_sha}, {source: old_sha}) == set()
    assert reads == []
    assert zero._require_ta_prefix_source_changes({source: old_sha}, {source: new_sha}) == {source}
    assert reads == [zero.TA_SIGNAL_COUNTER_PARITY_REF, before_ref, after_ref]
    reads.clear()
    for old, new in (('0' * 64, new_sha), (old_sha, '0' * 64), (new_sha, old_sha)):
        with pytest.raises(RuntimeError, match='TA_UNREVIEWED_NUMERIC_SOURCE_DRIFT'):
            zero._require_ta_prefix_source_changes({source: old}, {source: new})
    other = 'gx1/features/htf_features.py'
    with pytest.raises(RuntimeError, match='TA_UNREVIEWED_NUMERIC_SOURCE_DRIFT'):
        zero._require_ta_prefix_source_changes(
            {source: old_sha, other: '1' * 64}, {source: new_sha, other: '2' * 64})
    assert reads == []
    after['semantics_sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match='TA_COUNTER_PARITY'):
        zero._require_ta_prefix_source_changes({source: old_sha}, {source: new_sha})
