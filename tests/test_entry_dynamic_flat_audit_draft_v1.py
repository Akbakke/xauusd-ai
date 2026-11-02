"""TRAIN draft provenance and fail-closed admission; synthetic only, no model."""
from copy import deepcopy

import pytest

from gx1.contracts import entry_dynamic_mtf_state_report_v1 as report
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
from gx1.contracts import entry_model_native_train_recipe_v1 as recipes
from tests.test_entry_dynamic_state_train_counts_v1 import _draft_fixture


def _fixture(monkeypatch, *, extended=False):
    owner, series, rule, old = _draft_fixture(monkeypatch)
    leads = list(target.default_trial_lead_minutes())
    counts = report.dense_train_query_counts(series, prefix_rows=4, cutoff_ns=old['fold']['train_cutoff_ns'],
                                             lead_minutes=leads)
    prior = report.dense_train_prior_from_counts(counts, lead_minutes=leads)
    train_ref = {'path':'/synthetic/TRAIN.json','sha256':'c'*64}
    prior_ref = {'path':'/synthetic/PRIOR.json','sha256':'d'*64}
    support_ref = {'path':'/synthetic/SUPPORT.json','sha256':'e'*64}
    train = {'train_prior':prior_ref, 'series':old['target_manifest'], 'train_cutoff_ns':old['fold']['train_cutoff_ns'],
             'train_parents':4, 'later_aggregates_computed':False, 'target_contract':series.metadata['definition'],
             'class_counts':deepcopy(prior['counts']), 'transition_counts':deepcopy(prior['transition_counts'])}
    rule.update(lead_minutes=leads, prior=deepcopy(prior), train_audit=train_ref)
    objects = {old['prior_rule_review']['path']:rule, train_ref['path']:train, prior_ref['path']:prior}
    monkeypatch.setattr(owner.campaign, 'read_bound_json', lambda path, digest: objects[str(path)])
    monkeypatch.setattr(target, 'read_dynamic_mtf_state_series_metadata', lambda *a, **kw: {'metadata':series.metadata})
    checked=[]
    def check(audit, recipe, **kw):
        checked.append((dict(audit), dict(recipe), {'target_ref':kw['target_ref'], 'fold':kw['fold']}))
        assert audit['first_flat_activity_source_support'] == support_ref
        assert kw['target_ref'] == old['target_manifest'] and kw['fold'] == old['fold']
    monkeypatch.setattr(owner, '_require_first_flat_activity_support', check)
    options = {'extended_forecast_minutes':1441, 'extended_storage_reserve_bytes':5*1024**3} if extended else {}
    numeric = recipes.native_direction_research_recipe(batch_size=16, max_updates=2, dynamic_state=True,
        current_state_first_loss_weight=1.0, first_flat_activity_weight=0.25, **options)
    args = {'numerical_recipe':numeric, 'target_manifest':old['target_manifest'], 'target_sources':old['target_sources'],
            'fold':old['fold'], 'prior_rule_review':old['prior_rule_review'],
            'first_flat_activity_source_support':support_ref, 'source_repo':'/synthetic', 'source_owners':{}}
    return owner, series, rule, train, prior, args, checked


def test_default_grid_draft_reuses_bound_train_prior_without_admitting_launch(monkeypatch):
    owner, series, rule, train, prior, args, checked = _fixture(monkeypatch)
    draft = owner.build_first_flat_activity_train_audit_draft(**args)
    assert len(checked) == 1 and draft['prior'] == prior == rule['prior']
    assert draft['objective_sha256'] == args['numerical_recipe']['research_objective_sha256']
    assert draft['numerical_recipe_sha256'] == args['numerical_recipe']['contract_sha256']
    assert draft['first_flat_activity_source_support'] == args['first_flat_activity_source_support']
    assert draft['reused_train_audit'] == rule['train_audit'] and draft['reused_train_prior'] == train['train_prior']
    assert draft['lead_minutes'] == list(target.default_trial_lead_minutes())
    assert draft['support_status'] == 'DRAFT_NOT_ADMITTED' and draft['native_launch_authorized'] is False
    assert draft['model_forwards'] == draft['optimizer_steps'] == 0
    assert draft['test_data_used'] is draft['later_metrics_inspected'] is False
    monkeypatch.setattr(owner.campaign, 'read_bound_json', lambda *a: draft)
    launch = {'fold':args['fold'],'target_sources':args['target_sources'],'target_manifest':args['target_manifest'],
              'label_audit':{'path':'/synthetic/draft.json','sha256':'f'*64},'numerical_recipe':args['numerical_recipe']}
    with pytest.raises(RuntimeError, match='STATE_TRAIN_LABEL_AUDIT'):
        owner.require_dynamic_state_target_binding(launch)


def test_extended_grid_draft_reuses_existing_train_counter_and_binds_activity(monkeypatch):
    owner, _, _, _, _, args, checked = _fixture(monkeypatch, extended=True)
    draft = owner.build_first_flat_activity_train_audit_draft(**args)
    assert len(checked) == 1 and draft['lead_minutes'][-1] == 1441
    assert draft['first_flat_activity_source_support'] == args['first_flat_activity_source_support']
    assert draft['objective_sha256'] == args['numerical_recipe']['research_objective_sha256']
    assert draft['train_query_counts']['test_data_used'] is False
    assert draft['support_status'] == 'DRAFT_NOT_ADMITTED' and draft['native_launch_authorized'] is False
    assert 'independent_first_flat_activity_objective_review_required' in draft['blockers']


def test_missing_activity_mode_is_rejected_before_metadata_or_tape(monkeypatch):
    owner, _, _, _, _, args, _ = _fixture(monkeypatch)
    args['numerical_recipe'] = recipes.native_direction_research_recipe(batch_size=16, max_updates=2,
        dynamic_state=True, current_state_first_loss_weight=1.0)
    monkeypatch.setattr(target, 'read_dynamic_mtf_state_series_metadata', lambda *a, **kw: pytest.fail('no metadata read'))
    monkeypatch.setattr(target, 'load_dynamic_mtf_state_series', lambda *a, **kw: pytest.fail('no tape read'))
    with pytest.raises(RuntimeError, match='STATE_ACTIVITY_DRAFT_RECIPE'):
        owner.build_first_flat_activity_train_audit_draft(**args)


@pytest.mark.parametrize('fault', ['unreviewed_rule','later_data','wrong_counts','wrong_cutoff','different_prior'])
def test_draft_rejects_mismatched_or_later_selected_evidence_before_tape(monkeypatch, fault):
    owner, _, rule, train, prior, args, _ = _fixture(monkeypatch)
    if fault == 'unreviewed_rule': rule['independent_rule_review']['decision'] = 'DRAFT'
    if fault == 'later_data': rule['later_metrics_inspected'] = True
    if fault == 'wrong_counts': train['class_counts'][0][0] += 1
    if fault == 'wrong_cutoff': train['train_cutoff_ns'] += target.MINUTE_NS
    if fault == 'different_prior': rule['prior']['counts'][0][0] += 1
    monkeypatch.setattr(target, 'load_dynamic_mtf_state_series', lambda *a, **kw: pytest.fail('no tape read'))
    with pytest.raises(RuntimeError, match='STATE_ACTIVITY_DRAFT_(RULE_REVIEW|PRIOR_BINDING)'):
        owner.build_first_flat_activity_train_audit_draft(**args)


def test_draft_refuses_unverified_flat_support_before_tape(monkeypatch):
    owner, _, _, _, _, args, _ = _fixture(monkeypatch)
    def reject(*a, **kw): raise RuntimeError('STATE_FIRST_ACTIVITY_SUPPORT_IDENTITY')
    monkeypatch.setattr(owner, '_require_first_flat_activity_support', reject)
    monkeypatch.setattr(target, 'load_dynamic_mtf_state_series', lambda *a, **kw: pytest.fail('no tape read'))
    with pytest.raises(RuntimeError, match='STATE_FIRST_ACTIVITY_SUPPORT_IDENTITY'):
        owner.build_first_flat_activity_train_audit_draft(**args)
