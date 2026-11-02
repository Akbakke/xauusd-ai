"""Metadata gate checks only: no model, model forward, optimizer or market data."""
import hashlib
import json

import pytest

from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as owner


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _case(tmp, fault=None):
    tfs = ('M5', 'M15', 'H1', 'H4', 'D1')
    owner_key = 'gx1/contracts/entry_dynamic_mtf_state_targets_v1.py'
    owner_ref = _file(tmp / owner_key, 'synthetic target owner')
    dependencies = {name:_file(tmp / name, 'synthetic ' + name) for name in
                    ('gx1/features/htf_features.py', 'gx1/features/technical_indicators_v1.py',
                     'gx1/time/session_detector.py')}
    worker_ref = _file(tmp / 'review.py', 'synthetic review')
    target_ref = _file(tmp / 'target.json', 'synthetic target manifest')
    sources = {'m1': _file(tmp / 'm1', 'synthetic m1'), 'm5_atr': _file(tmp / 'm5', 'synthetic m5')}
    fold = {'prefix_rows': 4, 'train_cutoff_ns': 100, 'oof_cutoff_ns': 200}
    counts = {tf: {'parents': 4, 'activity_label_eligible': 3, 'activity_label_ineligible': 1,
                   'flat_known': 1, 'nonflat_direction_known': 1, 'nonflat_direction_unknown': 1} for tf in tfs}
    if fault == 'partition': counts['M5']['activity_label_eligible'] = 4
    if fault == 'no_flat': counts['M5']['flat_known'] = 0; counts['M5']['nonflat_direction_known'] = 2
    published = _write(tmp/'published.json', {'schema_version': 'synthetic'})
    source = _write(tmp/'source.json', {'m1_source': sources['m1'], 'calendar': published, 'test_data_used': False})
    schedule_sha, authority_sha, target_sha = '1'*64, '2'*64, '3'*64
    schedule = _write(tmp/'schedule.json', {'source_method': 'externally_sourced_exact_xau_utc_closure_intervals_v1',
        'instrument':'XAU_USD', 'timeframe':'M1', 'test_data_used':False,
        'source_reference_sha256': source['sha256'], 'schedule_sha256': '9'*64 if fault == 'calendar' else schedule_sha})
    authority = _write(tmp/'authority.json', {'artifact_sha256':authority_sha})
    calendar = {'source_reference':source, 'exact_schedule':schedule, 'closure_authority':authority,
                'published_calendar':published}
    target_result = _write(tmp/'target_result.json', {'target_manifest':target_ref, 'target_sources':sources,
                                                       'test_data_used':False, 'calendar_binding':calendar})
    previous = _write(tmp/'previous.json', {'schema_version':'gx1_first_flat_activity_train_audit_v1',
        'status':'TRAIN_LABEL_FACTS_NOT_AN_ADMITTED_OBJECTIVE', 'new_scoring_or_test_used':False,
        'references':{'target_manifest':target_ref}, 'per_timeframe':counts})
    tf_results = {tf: {'counts':counts[tf], 'mature_first_parent_rows_compared':4,
                       'binary_mask_or_label_disagreements':0, 'atr_threshold_bitwise_differences':0,
                       'atr_threshold_max_abs_difference':0., 'local_range_max_abs_difference_on_eligible':0.}
                  for tf in tfs}
    if fault == 'mask': tf_results['M5']['binary_mask_or_label_disagreements'] = 1
    if fault == 'atr': tf_results['M5']['atr_threshold_bitwise_differences'] = 1
    raw = {'schema_version':'gx1_first_flat_activity_raw_train_review_v1',
        'status':'PASS_RAW_SOURCE_TRAIN_FIRST_LABEL_PARITY_NO_LAUNCH_AUTHORITY',
        'worker':worker_ref, 'source_head':'a'*40, 'target_manifest':target_ref,
        'target_sources':sources, 'fold':fold, 'target_result':target_result, 'calendar_refs':calendar,
        'original_train_facts':previous, 'tf_results':tf_results, 'model_calls':0, 'optimizer_steps':0,
        'backward_calls':0, 'new_scoring':0, 'test_data_used':fault == 'test_data', 'native_launch_authorized':False}
    if fault == 'raw_fold': raw['fold'] = {**fold, 'prefix_rows':5}
    raw_ref = _write(tmp/'raw_review.json', raw)
    proof = {'schema_version':'gx1_first_flat_activity_train_source_support_v1',
        'status':'PASS_INDEPENDENT_RAW_REVIEW_NO_LAUNCH_AUTHORITY', 'raw_review':raw_ref, 'raw_worker':worker_ref,
        'reviewed_source_head':'a'*40, 'target_owner':owner_ref, 'target_dependencies':dependencies,
        'target_contract_sha256':target_sha,
        'target_manifest':target_ref, 'target_sources':sources, 'fold':fold, 'per_timeframe':counts,
        'model_calls':0, 'optimizer_steps':0, 'test_data_used':False, 'native_launch_authorized':False}
    if fault == 'owner': proof['target_owner'] = {**owner_ref, 'sha256':'f'*64}
    if fault == 'dependency': proof['target_dependencies']['gx1/time/session_detector.py']['sha256'] = 'e'*64
    support = _write(tmp/'support.json', proof)
    if fault == 'worker': (tmp/'review.py').write_text('changed after hashing')
    audit = {} if fault == 'missing' else {'first_flat_activity_source_support':support}
    recipe = {'source_repo':str(tmp), 'source_owners':{owner_key:owner_ref, **dependencies}}
    if fault == 'owner_missing_dependency': del recipe['source_owners']['gx1/time/session_detector.py']
    meta = {'definition':{'contract_sha256':target_sha},
            'closure_binding':{'schedule_sha256':schedule_sha,'authority_sha256':authority_sha}}
    return audit, recipe, dict(target_ref=target_ref, sources=sources, fold=fold, target_metadata=meta)


def test_independent_support_is_bound_to_target_calendar_sources_and_train_population(tmp_path):
    audit, recipe, args = _case(tmp_path)
    proof = owner._require_first_flat_activity_support(audit, recipe, **args)
    assert proof['per_timeframe']['M5']['nonflat_direction_unknown'] == 1
    assert proof['native_launch_authorized'] is False


@pytest.mark.parametrize('fault', ['missing', 'owner', 'raw_fold', 'test_data', 'worker', 'calendar',
                                    'partition', 'no_flat', 'mask', 'atr', 'dependency', 'owner_missing_dependency'])
def test_support_refuses_missing_or_unproven_evidence(tmp_path, fault):
    audit, recipe, args = _case(tmp_path, fault)
    with pytest.raises(RuntimeError):
        owner._require_first_flat_activity_support(audit, recipe, **args)
