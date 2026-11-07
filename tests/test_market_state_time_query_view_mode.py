"""Synthetic time-query view identity and constructor routing; no native fit."""
from copy import deepcopy
import hashlib

import pytest

from gx1.contracts import entry_native_direction_feature_view_v1 as view
from gx1.contracts.market_state_time_query_model_v1 import time_query_model_contract
from tests.test_entry_native_direction_feature_view_v1 import _ref, _seal, bound_case
from tests.test_entry_native_direction_prefix_units_routing import binding_case


@pytest.fixture
def time_query_case(bound_case, monkeypatch):
    args = deepcopy(bound_case['arguments'])
    args['full_feature_profile'] = True
    args['market_state_time_query'] = True
    sources = args['source_bindings']
    prefix_root = view.time_query_targets.RUNS / 'MARKET_STATE_MATURE_PREFIX_NORMALIZATION_20261107_001'
    args['prefix_normalization_ref'] = {'path': str(prefix_root / 'PREFIX_NORMALIZATION_FOLD_2.json'),
                                        'sha256': '1' * 64}
    sources['input_qualification'] = {'path': str(prefix_root / 'QUALIFICATION_FOLD_2.json'),
                                      'sha256': '1' * 64}
    root = view.time_query_targets.ROOT
    owner_paths = {
        'mtf_window_source': 'gx1/features/htf_features.py',
        'gc_gather_source': 'gx1/contracts/entry_gc_mtf_history_v1.py',
        'owner_profile_source': 'gx1/features/entry_specialist_feature_groups_v1.py',
    }
    for key, path in owner_paths.items():
        sources[key] = {'path': str(root / path), 'sha256': '1' * 64}
    for key in view.TIME_QUERY_SOURCE_REF_KEYS - {'time_query_candidate_tape', 'time_query_mature_prefix_plan'}:
        sources[key] = _ref(key + '.py')
    model_path = root / 'gx1/contracts/market_state_time_query_model_v1.py'
    sources['time_query_model_source'] = {'path': str(model_path),
                                          'sha256': hashlib.sha256(model_path.read_bytes()).hexdigest()}
    path, sha = view.time_query_targets.CANDIDATE_REFS['tape']
    sources['time_query_candidate_tape'] = {'path': str(path), 'sha256': sha}
    sources['time_query_mature_prefix_plan'] = view.TIME_QUERY_MATURE_PLAN_REF.copy()

    fitted = args['prefix_normalization']
    report = fitted['prefix_qualification']
    proof = report['proof']
    proof['source_bindings'] = {
        'boolean_raw_domain_transition': sources['raw_gc_transition'],
        'boolean_raw_gc_layout': sources['raw_gc_layout'],
        'boolean_raw_result': _ref('boolean_raw_result.json'),
        'cohort_rule': 'decision_ns_strictly_below_official_cutoff_minus_1440_minutes',
        'current_plan': view.TIME_QUERY_MATURE_PLAN_REF.copy(),
        'current_policy': _ref('prior_policy.json'),
        'current_source_head': 'a' * 40,
        'current_source_owners': {path: sources[key]['sha256'] for key, path in owner_paths.items()},
        'model_axes': proof['full_axes']['source'],
        'provider_plan': sources['raw_input_provider_plan'],
        'provider_worker': _ref('provider_worker.py'),
    }
    fitted['prefix_qualification'] = _seal(report)
    args['prefix_normalization'] = _seal(fitted)
    cutoff = proof['prefix']['cutoff_ns']
    monkeypatch.setattr(view.five_endpoint_targets, 'FOLD_BOUNDARIES_NS',
                        (0, cutoff + 288 * view.BAR_NS, 2, 3, 4, 5))
    monkeypatch.setattr(view, 'TIME_QUERY_MATURE_FIT_ROWS',
                        (proof['prefix']['rows'], 2, 3, 4))
    monkeypatch.setattr(view, 'TIME_QUERY_MODEL_AXIS_ROWS', proof['full_axes']['rows'])
    return args


def test_time_query_view_uses_distinct_objective_and_rebinds(time_query_case, bound_case):
    bound = view.bind_native_direction_feature_view(**time_query_case)
    assert bound['market_state_time_query'] is True
    assert bound['full_feature_profile'] is True
    assert bound['research_objective'] == time_query_model_contract()
    assert bound['contract_sha256'] != bound_case['view']['contract_sha256']
    assert bound['source_bindings']['time_query_candidate_tape'] == time_query_case['source_bindings']['time_query_candidate_tape']
    assert view.bind_native_direction_feature_view(**time_query_case, expected=bound) == bound
    assert view.bind_native_direction_feature_view(
        **bound_case['arguments'], expected=bound_case['view']) == bound_case['view']


@pytest.mark.parametrize('change,reason', [
    ('old_mode', 'OBJECTIVE_MODE'),
    ('old_provenance', 'TIME_QUERY_MATURE_PROVENANCE'),
    ('wrong_plan', 'TIME_QUERY_MATURE_PROVENANCE_JOINS'),
    ('wrong_tape', 'TIME_QUERY_CANDIDATE_TAPE'),
    ('wrong_cohort', 'TIME_QUERY_MATURE_PROVENANCE_JOINS'),
    ('missing_target_source', 'SOURCE_REF_KEYS'),
])
def test_time_query_rejects_wrong_mode_source_or_prefix(time_query_case, monkeypatch, change, reason):
    args = deepcopy(time_query_case)
    if change == 'old_mode':
        args['market_state_five_endpoint'] = True
        for key in view.FIVE_ENDPOINT_SOURCE_REF_KEYS:
            args['source_bindings'][key] = _ref(key + '.py')
    elif change == 'old_provenance':
        normal = args['prefix_normalization']
        normal['prefix_qualification']['proof']['source_bindings'] = {
            'ancestral264_plan': args['source_bindings']['raw_input_provider_plan']}
        normal['prefix_qualification'] = _seal(normal['prefix_qualification'])
        args['prefix_normalization'] = _seal(normal)
    elif change == 'wrong_plan':
        args['source_bindings']['time_query_mature_prefix_plan'] = _ref('other_plan.json')
    elif change == 'wrong_tape':
        args['source_bindings']['time_query_candidate_tape'] = _ref('other_tape.npz')
    elif change == 'wrong_cohort':
        monkeypatch.setattr(view, 'TIME_QUERY_MATURE_FIT_ROWS', (999, 2, 3, 4))
    elif change == 'missing_target_source':
        del args['source_bindings']['time_query_target_source']
    with pytest.raises(RuntimeError, match=reason):
        view.bind_native_direction_feature_view(**args)


def test_time_query_builder_selects_only_new_wrapper(time_query_case, monkeypatch):
    bound = view.bind_native_direction_feature_view(**time_query_case)
    calls = []

    class BackboneShell:
        def __init__(self, **kwargs):
            calls.append(('backbone', kwargs))
            self.entry_execution_profile = kwargs['entry_execution_profile']
            self.gc_mtf_history = kwargs['gc_mtf_history']

        def to(self, device):
            return self

        def require_input_normalization_state(self):
            calls.append(('normalization', None))

    class QueryShell:
        def __init__(self, backbone):
            self.backbone = backbone
            calls.append(('query_head', None))

    def forbidden(*args, **kwargs):
        raise AssertionError('old state or endpoint wrapper constructed')

    monkeypatch.setattr(view, 'EntryV10CtxHybridTransformer', BackboneShell)
    monkeypatch.setattr(view, 'TimeQueryMarketStateModel', QueryShell)
    monkeypatch.setattr(view, 'FiveEndpointMarketStateModel', forbidden)
    monkeypatch.setattr(view, 'NativeDynamicMTFStateResearch', forbidden)
    model = view.build_native_direction_research_model(
        bound, prefix_normalization=time_query_case['prefix_normalization'],
        constructor_metadata=time_query_case['constructor_metadata'], device='synthetic')
    assert [kind for kind, _ in calls] == ['backbone', 'normalization', 'query_head']
    assert calls[0][1]['gc_mtf_history'] == time_query_case['prefix_normalization']['gc_mtf_history']
    assert model.feature_view_identity['research_objective_sha256'] == time_query_model_contract()['contract_sha256']


def test_fold1_view_accepts_only_exact_mature_units_profile(time_query_case):
    from gx1.contracts import entry_native_direction_prefix_units_v1 as units

    fold2 = view.bind_native_direction_feature_view(**time_query_case)
    profile = units.mature_fold1_prefix_units_profile()
    assert profile['fit_prefix']['rows'] == 29574
    assert profile['fit_prefix']['cutoff_ns'] == 1727654400000000000
    assert len(profile['fields']) * len(profile['surfaces']) == 153
    fold1 = deepcopy(fold2)
    fold1['unobserved_units_profile'] = profile
    fold1['prefix'].update(profile['fit_prefix'])
    fold1['prefix_normalization']['unobserved_units_profile_sha256'] = profile['contract_sha256']
    fold1['source_bindings']['time_query_mature_prefix_plan'] = {
        'path': view.TIME_QUERY_MATURE_FOLD1_PLAN_PATH, 'sha256': '1' * 64}
    fold1 = _seal(fold1)
    assert view._require_view(fold1, unobserved_units_profile=profile) == fold1
    with pytest.raises(RuntimeError, match='EXPLICIT_UNOBSERVED_UNITS_PROFILE'):
        view._require_view(fold1)
    with pytest.raises(RuntimeError, match='PROFILE_IDENTITY'):
        view._require_view(fold1, unobserved_units_profile=units.october_prefix_units_profile())
    assert view._require_view(fold2) == fold2



def test_october_mode_still_rejects_mature_fold1_profile(binding_case):
    from gx1.contracts import entry_native_direction_prefix_units_v1 as units

    with pytest.raises(RuntimeError, match='PROFILE_IDENTITY'):
        view._require_view(binding_case['view'],
                           unobserved_units_profile=units.mature_fold1_prefix_units_profile())
    assert view._require_view(binding_case['view'],
        unobserved_units_profile=binding_case['profile']) == binding_case['view']
