"""Small synthetic V3 review controls; no model, physical archive or admission."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gx1.audit import native_all_tf_provider_hlc_review_v3 as m


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setattr(m, 'ROWS', 8)
    target = m.hlc.all_tf_provider_hlc_state_contract()
    tick = 300_000_000_000
    issues = np.arange(8, dtype=np.int64) * tick + int(np.datetime64('2021-01-04T00:00', 'ns').astype(np.int64))
    duration = np.array([target['tf_duration_ns'][tf] for tf in m.TF], np.int64)
    origin = np.array([target['tf_origin_ns'][tf] for tf in m.TF], np.int64)
    starts = origin + ((issues[:, None] - origin + duration - 1) // duration) * duration
    yes = np.ones((8, 5), np.bool_); complete = np.zeros_like(yes); complete[:, 0] = True
    active = yes.copy(); active[-1] = False
    side = yes.copy(); side[-2:] = False
    up = np.broadcast_to(np.array([0, 1, 0, 1, 0, 1, 0, 0], np.bool_)[:, None], (8, 5)).copy()
    provider = yes.copy(); provider[:, 0] = False
    cert = {'path': '/synthetic/certificate.json', 'sha256': 'a' * 64,
            'target_contract_sha256': target['contract_sha256'],
            'source_binding_sha256': target['provider_hlc_source_binding_sha256']}
    truth = {'issue_ns': issues, 'target_start_ns': starts, 'validity_ns': starts + duration,
             'target_present': yes.copy(), 'mature': yes.copy(), 'nominal_complete': complete.copy(),
             'state_window_complete': complete, 'partial_range_active': active & ~complete,
             'activity_known': yes.copy(), 'activity': active, 'side_known': side, 'side_up': up,
             'provider_hlc_observed': provider, 'state_hlc_observed': yes.copy(),
             'target_observation_certificate_ref': cert}
    upper = int(np.datetime64('2026-01-01', 'ns').astype(np.int64))
    prior = m.objective.fit_native_future_state_train_prior(truth=truth, fold_upper_ns=upper, target_contract=target)
    initial_outputs = {k: v.numpy() for k, v in m.objective.native_future_state_train_prior_logits(prior, rows=8, target_contract=target).items()}
    canonical = m.objective.native_future_state_score_sums(model_output={k: torch.from_numpy(v) for k, v in initial_outputs.items()}, truth=truth, fold_upper_ns=upper, target_contract=target)
    refs = {k: {'path': '/synthetic/' + k + '.json', 'sha256': 'b' * 64} for k in m.REFERENCE_KEYS}
    model = {'contract_sha256': 'c' * 64, 'target_contract_sha256': target['contract_sha256'],
             'train_prior_sha256': prior['contract_sha256'], 'objective_contract_sha256': prior['objective_contract_sha256'],
             'timeframes': list(m.TF), 'input_binding_sha256': 'd' * 64, 'input_normalization_sha256': 'e' * 64}
    metadata = {'target_contract': target, 'truth_sha256': prior['fit_population']['truth_sha256'],
                'dataset': refs['dataset'], 'prior': refs['prior'], 'normalization': refs['normalization'],
                'input_binding_ref': refs['input_binding']}
    initial = {'status': 'COMPLETE_NATIVE_ALL_TF_PROVIDER_HLC_V3_SEGMENTED_WHOLE_TRAIN_INITIAL', 'test_data_used': False,
               'report': {'whole_TRAIN_initialbaseline_qualified': True, 'initial_state_sha256': 'f' * 64,
                          'initial_rng_sha256': '1' * 64, 'initial': {'rows': 8, 'initial_prior_exact': True,
                          'truth_sha256': metadata['truth_sha256'], 'scores': canonical,
                          'target_observation_certificate_ref': cert}}}
    receipt = {'initial_only': True, 'optimizer_updates': 0, 'model_contract_sha256': model['contract_sha256'],
               'target_contract_sha256': target['contract_sha256'], 'model_state_sha256': 'f' * 64, 'rng_sha256': '1' * 64}
    args = dict(metadata=metadata, model_contract=model, prior=prior, initial_result=initial, initial_receipt=receipt, references=refs)
    spec = m.build_spec(**args)
    context = {'truth': truth, 'target_contract': target, 'prior': prior, 'truth_sha256': metadata['truth_sha256'],
               'input_binding': {'contract_sha256': 'd' * 64}, 'normalization': {'contract_sha256': 'e' * 64}}
    return args, spec, context, initial_outputs


def review(case, **overrides):
    args, spec, context, outputs = case
    return m.review_outputs(**dict(context=context, outputs=outputs, spec=spec, references=args['references'], **overrides))


def test_builder_has_no_io_rng_model_or_fit(case, monkeypatch):
    args, spec, _, _ = case
    def denied(*a, **k): raise AssertionError('No IO or fit in metadata builder')
    monkeypatch.setattr('builtins.open', denied)
    monkeypatch.setattr(np, 'load', denied)
    monkeypatch.setattr(np.random, 'default_rng', denied)
    monkeypatch.setattr(m.objective, 'fit_native_future_state_train_prior', denied)
    monkeypatch.setattr(torch.cuda, 'init', denied)
    assert m.build_spec(**args) == spec


def test_initial_review_all_five_tf_and_all_fixed_years(case):
    result = review(case, expect_initial=True)
    assert set(result['all_TF']) == set(m.TF)
    assert all(set(result['by_UTC_target_year'][tf]) == set(map(str, m.YEARS)) for tf in m.TF)
    assert result['primary_proper_score_verdict'] == 'FAIL'  # strict equality is not learning
    assert result['candidate_TRAIN_comparison_verdict'] == 'FAIL'
    assert result['timeblock_stability_verdict'] == 'UNKNOWN'  # empty years stay visible
    assert all(result[k] == 0 for k in ['new_NN_forwards', 'optimizer_updates', 'prior_fits'])
    assert not result['learning_qualified'] and not result['training_authorized']


def test_repeated_targets_keep_mixed_issue_labels_and_score_row_losses(case):
    _, _, context, outputs = case
    outputs = {k: v.copy() for k, v in outputs.items()}
    outputs['side_logits'][:, 4] = np.array([-4, 4, -1, 1, -.5, .5, 0, 0], np.float32)
    result = m.review_outputs(context=context, outputs=outputs, spec=case[1], references=case[0]['references'])
    panel = result['all_TF']['D1']['side']; eq = panel['equal_target_weight']
    assert panel['count'] == 6 and panel['unique_target_starts'] == 1
    assert eq['targets_with_mixed_known_labels'] == 1
    assert panel['UP_DOWN_target_support']['targets_in_both'] == 1
    assert eq['model']['NLL'] == pytest.approx(panel['model']['NLL'])
    assert panel['model']['NLL'] < np.log(2)  # scoring mean logit would give log(2)
    assert result['support']['D1']['ACTIVE_unknown_side'] == 1
    assert result['support']['D1']['FLAT'] == 1


def test_calibration_bins_and_weights_include_edges_and_empty_support():
    z = np.array([-1000, 0, 1000, 1], np.float32); y = np.array([0, 1, 1, 0], np.bool_)
    result = m._calibration(z, y, np.ones(4, np.bool_), np.array([1, 1, 2, 3], np.int64))
    assert result['row']['bins'][0]['issue_rows'] == 1
    assert result['row']['bins'][9]['issue_rows'] == 1
    for weighting in ['row', 'equal_target']:
        assert sum(b['weight_mass'] for b in result[weighting]['bins']) == pytest.approx(1)
    assert m._calibration(z, y, np.zeros(4, np.bool_), np.arange(4))['status'] == 'UNKNOWN'


@pytest.mark.parametrize('field', ['dataset', 'prior', 'initial_receipt', 'membership_manifest'])
def test_changed_frozen_reference_rejected(case, field):
    refs = copy.deepcopy(case[0]['references']); refs[field]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='AUTHENTICATED_REFERENCES'):
        m.review_outputs(context=case[2], outputs=case[3], spec=case[1], references=refs)


@pytest.mark.parametrize('kind', ['v2', 'truth', 'normalization', 'mask', 'logits', 'initial_logits'])
def test_wrong_generation_population_and_outputs_fail_closed(case, kind):
    args, spec, context, outputs = copy.deepcopy(case)
    expected = 'CONTEXT_IDENTITY'
    if kind == 'v2': context['target_contract'] = m.hlc.provider_hlc_state_contract()
    elif kind == 'truth': context['truth_sha256'] = '0' * 64
    elif kind == 'normalization': context['normalization']['contract_sha256'] = '0' * 64; expected = 'FEATURE_IDENTITY'
    elif kind == 'mask': context['truth']['side_known'][0, 0] = False; expected = 'TRUTH_SHA'
    elif kind == 'logits': outputs['side_logits'][0, 0] = np.nan; expected = 'LOGIT_HEADER'
    else: outputs['side_logits'][0, 0] += .1; expected = 'INITIAL_EXACT_PRIOR'
    with pytest.raises(ValueError, match=expected):
        m.review_outputs(context=context, outputs=outputs, spec=spec, references=args['references'], expect_initial=kind == 'initial_logits')


def test_scope_change_rejected_even_if_resealed(case):
    spec = copy.deepcopy(case[1]); spec['training_authorized'] = True
    spec = m._seal({k: v for k, v in spec.items() if k != 'contract_sha256'})
    with pytest.raises(ValueError, match='FROZEN_SCOPE'): m.validate_spec(spec)


def test_spec_json_is_versioned_and_closed():
    spec = json.loads(Path('configs/research/NATIVE_ALL_TF_PROVIDER_HLC_V3_COMPARISON_SPEC_20261125_001.json').read_text())
    assert m.validate_spec(spec) == spec
    assert spec['populations']['side']['D1']['unique_target_starts'] == 875
    assert spec['populations']['side']['H4']['unique_target_starts'] == 4478


def test_candidate_comparison_keeps_empty_timeblocks_unknown_despite_pooled_gain(case):
    args, spec, context, outputs = copy.deepcopy(case)
    outputs['side_logits'] = np.where(context['truth']['side_up'], 2., -2.).astype(np.float32)
    result = m.review_outputs(context=context, outputs=outputs, spec=spec, references=args['references'])
    assert result['primary_proper_score_verdict'] == 'PASS'
    assert set(result['side_rank_guard_by_TF'].values()) == {'PASS'}
    assert result['timeblock_stability_verdict'] == 'UNKNOWN'
    assert result['candidate_TRAIN_comparison_verdict'] == 'UNKNOWN'
    assert not result['learning_qualified']
    assert all(result['support'][tf]['mature_unknown_activity'] == 0 for tf in m.TF)


def test_rank_guard_reports_unknown_or_fail_without_argmax_quota():
    assert m._side_rank_guard({'rank_and_variation': {'positive': 0, 'negative': 2}}) == 'UNKNOWN'
    assert m._side_rank_guard({'rank_and_variation': {'positive': 1, 'negative': 1, 'nonconstant': False, 'AUC': .5}}) == 'FAIL'
    # All positive logits can still rank; strict argmax support is reported separately.
    panel = m.binary_panel(np.array([1., 2.]), np.zeros(2), np.array([False, True]),
                           np.ones(2, np.bool_), np.arange(2))
    assert not panel['model']['both_strict_argmax_sides_observed']
    assert m._side_rank_guard(panel) == 'PASS'
