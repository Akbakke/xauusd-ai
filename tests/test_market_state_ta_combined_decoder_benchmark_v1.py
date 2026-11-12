"""Exact one-arm metadata boundaries on synthetic dictionaries; no market data."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from gx1.contracts import market_state_ta_combined_decoder_benchmark_v1 as scope


def ref(path, sha='a' * 64):
    return dict(path=str(path), sha256=sha)


def seal(value, field):
    value[field] = scope.campaign.canonical_sha256({k: v for k, v in value.items() if k != field})
    return value


def metadata_documents():
    old = dict(contract_sha256='1' * 64, source_bindings={
        'time_query_v2_model_source': ref(scope.REPO / 'gx1/contracts/market_state_time_query_model_v2.py')},
        dedicated_tf_activity=True, research_objective={'contract_sha256': '2' * 64},
        full_axes={'rows': 118249}, prefix={'rows': 29574}, prefix_normalization={'binding': ref('/synthetic/norm.json')},
        constructor_metadata={'binding': ref('/synthetic/constructor.json')},
        physical_field_names_sha256={'signal': '3' * 64}, anchor_span_input_extension={'same': True})
    source = deepcopy(old); source['contract_sha256'] = '4' * 64
    source['source_bindings']['time_query_v2_model_source']['sha256'] = 'b' * 64
    target = dict(deepcopy(source), contract_sha256='5' * 64, dedicated_non_m5_side=True,
                  research_objective={'contract_sha256': '6' * 64})
    numerical = dict(batch_size=16, future_weight=1., optimizer={'lr': .0001},
        research_objective_sha256='7' * 64, native_F_loss_contract_sha256='8' * 64, contract_sha256='9' * 64)
    target_numerical = dict(deepcopy(numerical), research_objective_sha256='b' * 64,
        native_F_loss_contract_sha256='c' * 64, contract_sha256='d' * 64)
    owners = {n: ref(scope.REPO / n) for n in scope.QUALIFIED_NUMERICAL_OWNERS | scope.ORCHESTRATION_OWNERS
              | {'gx1/contracts/unchanged_numerics.py'}}
    batches, excluded = scope.coverage._selection()
    control = dict(source_owners=owners, numerical_recipe=numerical,
        input_refs={'feature_view': scope.base.VIEW_REF, 'short_initial_checkpoint': ref('/synthetic/INITIAL.pt')},
        batch_indices_by_pass=batches, excluded_complement_batch_indices=excluded,
        selected_fit_rows=12288, unique_training_rows=12288, selected_score_rows=4096,
        expected_source_model_state_sha256=scope.SOURCE_MODEL_SHA256,
        common_pass1_model_state_sha256=scope.coverage.COMMON_PASS1_MODEL_SHA256,
        common_pass1_rng_state_sha256=scope.coverage.COMMON_PASS1_RNG_SHA256,
        mechanical_source_commit=scope.BASE_COMMIT, recipe_sha256='e' * 64)
    qualified = dict(status='SOURCE_QUALIFIED_NATIVE_BASELINE_AND_MATCHED_COMPARISON_PENDING',
        metadata=scope.METADATA_QUALIFICATION_REF, production_files_changed=5,
        source_hashes={n: 'b' * 64 for n in scope.QUALIFIED_NUMERICAL_OWNERS},
        forward_and_loss_arithmetic_changed=False, training_authorized=False, test_data_used=False,
        initialization=dict(new_tensors=16, new_parameters=1032))
    metadata = dict(status='PASS_CURRENT_REAL_METADATA_BINDINGS', original_recipe=scope.CONTROL_RECIPE_REF,
        original_view=scope.base.VIEW_REF, old_view_rebuilt_exact=True, source_numerical_rebuilt_exact=True,
        input_normalization_and_constructor_unchanged=True, changed_source_bindings=['time_query_v2_model_source'],
        test_data_used=False, source_model_contract='2' * 64, target_model_contract='6' * 64,
        target_dual_contract='b' * 64, target_loss_contract='c' * 64, target_numerical_contract='d' * 64,
        **{'SOURCE_FEATURE_VIEW.json': scope.SOURCE_VIEW_REF, 'TARGET_FEATURE_VIEW.json': scope.TARGET_VIEW_REF,
           'NUMERICAL_RECIPE.json': scope.NUMERICAL_REF})
    return {r['path']: v for r, v in ((scope.CONTROL_RECIPE_REF, control),
        (scope.SOURCE_QUALIFICATION_REF, qualified), (scope.METADATA_QUALIFICATION_REF, metadata),
        (scope.base.VIEW_REF, old), (scope.SOURCE_VIEW_REF, source), (scope.TARGET_VIEW_REF, target),
        (scope.NUMERICAL_REF, target_numerical))}


def bind_documents(monkeypatch, docs):
    monkeypatch.setattr(scope, '_read', lambda r: docs[r['path']])
    monkeypatch.setattr(scope.campaign, 'require_binding', lambda r, **kw: r)


def test_fixed_exposure_is_reused_without_changing_the_existing_scopes():
    assert scope.ARMS == ('COMBINED_DECODER',)
    assert scope.coverage.ARMS == ('UNIQUE_COVERAGE',)
    assert scope.base.ARMS == ('FULL_OBJECTIVE', 'M5_DIRECTION_ONLY')
    for p in (1, 2, 3):
        assert scope.training_batch_indices(p) == scope.coverage.training_batch_indices(p)
    assert len(set(sum([scope.training_batch_indices(p) for p in (1, 2, 3)], []))) == 768
    assert scope.TRAIN_BATCH_INDICES == scope.coverage.TRAIN_BATCH_INDICES
    assert scope.LATER_BATCH_INDICES == scope.coverage.LATER_BATCH_INDICES
    assert scope.BUDGETS['max_forwards'] == 2304 and scope.BUDGETS['max_private_updates'] == 768
    assert scope.BUDGETS['max_model_constructions'] == scope.BUDGETS['max_control_capture_archive_decodes'] == 2
    assert scope.coverage.BUDGETS['max_model_constructions'] == 1


@pytest.mark.parametrize('fault', ['none', 'extra_owner', 'changed_loss', 'input_drift', 'side_false', 'numeric_drift', 'bad_reference'])
def test_qualified_metadata_allows_only_the_reviewed_decoder_difference(monkeypatch, fault):
    docs = metadata_documents()
    if fault == 'extra_owner': docs[scope.SOURCE_QUALIFICATION_REF['path']]['source_hashes']['gx1/extra.py'] = 'a' * 64
    elif fault == 'changed_loss': docs[scope.SOURCE_QUALIFICATION_REF['path']]['forward_and_loss_arithmetic_changed'] = True
    elif fault == 'input_drift': docs[scope.TARGET_VIEW_REF['path']]['prefix']['rows'] -= 1
    elif fault == 'side_false': docs[scope.TARGET_VIEW_REF['path']]['dedicated_non_m5_side'] = False
    elif fault == 'numeric_drift': docs[scope.NUMERICAL_REF['path']]['optimizer']['lr'] *= 2
    elif fault == 'bad_reference': docs[scope.METADATA_QUALIFICATION_REF['path']]['original_recipe'] = ref(scope.coverage.ROOT / 'other.json')
    bind_documents(monkeypatch, docs)
    if fault == 'none': assert scope._metadata()[0] is docs[scope.CONTROL_RECIPE_REF['path']]
    else:
        with pytest.raises(RuntimeError): scope._metadata()


def test_recipe_binds_measured_target_initial_without_relabelling_source(monkeypatch):
    docs = metadata_documents(); bind_documents(monkeypatch, docs)
    before = deepcopy(docs); value, control = scope._expected()
    assert docs == before
    assert value['expected_initial_model_state_sha256'] is None
    assert value['expected_source_initial_model_state_sha256'] == scope.SOURCE_INITIAL_SHA256
    assert value['expected_source_model_state_sha256'] == scope.SOURCE_MODEL_SHA256
    assert value['source_numerical_recipe'] == control['numerical_recipe']
    assert value['objective_recipes'] == {'COMBINED_DECODER': docs[scope.NUMERICAL_REF['path']]}
    assert value['input_refs']['feature_view'] == scope.TARGET_VIEW_REF
    assert all(value['input_refs'][k] == v for k, v in scope.CONTROL_REFS.items())
    assert 'common_pass1_model_state_sha256' not in value
    assert value['new_side_parameter_tensors'] == 16 and value['new_side_parameters'] == 1032


@pytest.mark.parametrize('fault', ['none', 'qualified_hash', 'unreviewed_numeric', 'wrong_path'])
def test_source_bridge_is_exact_except_the_declared_owners(monkeypatch, fault):
    docs = metadata_documents(); bind_documents(monkeypatch, docs)
    control, qualified, *_ = scope._metadata()
    owners = deepcopy(control['source_owners'])
    for n in scope.QUALIFIED_NUMERICAL_OWNERS: owners[n]['sha256'] = 'b' * 64
    for n in scope.ORCHESTRATION_OWNERS: owners[n]['sha256'] = 'c' * 64
    if fault == 'qualified_hash': owners[sorted(scope.QUALIFIED_NUMERICAL_OWNERS)[0]]['sha256'] = 'd' * 64
    elif fault == 'unreviewed_numeric': owners['gx1/contracts/unchanged_numerics.py']['sha256'] = 'd' * 64
    elif fault == 'wrong_path': owners[scope.WORKER]['path'] = '/wrong/worker.py'
    if fault == 'none': scope._require_source_bridge({'source_owners': owners}, control, qualified)
    else:
        with pytest.raises(RuntimeError): scope._require_source_bridge({'source_owners': owners}, control, qualified)


def test_later_binding_preserves_immutable_coverage_and_all_physical_fields():
    docs = metadata_documents()
    old, source, target = [docs[r['path']] for r in (scope.base.VIEW_REF, scope.SOURCE_VIEW_REF, scope.TARGET_VIEW_REF)]
    original = dict(view_sha256=old['contract_sha256'], full_axes=deepcopy(old['full_axes']),
        normalization_prefix=deepcopy(old['prefix']), physical_populations={'M5': {'rows': [10, 20]}})
    before = deepcopy(original)
    rebound, proof = scope._later_binding(original, old, source, target)
    assert original == before and scope._differences(rebound, original) == {'view_sha256'}
    assert rebound['view_sha256'] == target['contract_sha256']
    assert proof['original_coverage'] == scope.base.LATER_REFS['later_coverage']
    assert proof['changed_fields'] == ['view_sha256'] and proof['inputs_unchanged'] is True
    rebound['physical_populations']['M5']['rows'][0] = -1
    assert original == before
    target['prefix']['rows'] -= 1
    with pytest.raises(RuntimeError, match='LATER_UNCHANGED_INPUTS'):
        scope._later_binding(original, old, source, target)


def test_recipe_rejects_resealed_scope_budget_and_target_digest_changes(monkeypatch):
    docs = metadata_documents(); bind_documents(monkeypatch, docs)
    expected, control = scope._expected()
    monkeypatch.setattr(scope, 'read_origin', lambda value: {})
    monkeypatch.setattr(scope.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0))
    owners = {n: ref(scope.REPO / n) for n in set(control['source_owners']) | {scope.OWNER}}
    value = dict(expected, source_owners=owners, trace_worker=owners[scope.WORKER], mechanical_source_commit=scope.BASE_COMMIT)
    seal(value, 'recipe_sha256')
    assert scope.require_recipe(value, source_repo=scope.REPO, source_commit=scope.BASE_COMMIT) == value
    for key, wrong in [('expected_initial_model_state_sha256', scope.SOURCE_INITIAL_SHA256),
        ('max_model_constructions', 1), ('max_private_updates', 769), ('run_scope', scope.coverage.SCOPE),
        ('initial_parity', dict(scope.INITIAL_PARITY, atol=3e-6))]:
        bad = deepcopy(value); bad[key] = wrong; seal(bad, 'recipe_sha256')
        with pytest.raises(RuntimeError): scope.require_recipe(bad, source_repo=scope.REPO, source_commit=scope.BASE_COMMIT)
    with pytest.raises(RuntimeError): scope.require_recipe(value, source_repo=scope.REPO, source_commit=scope.BASE_COMMIT, verify_files=False)


def test_only_new_exact_gate_can_admit_the_declared_scope():
    recipe = dict(run_scope=scope.SCOPE, admission_stage=scope.STAGE)
    scope.require_gate(scope.gate_fields(), recipe)
    for change in ({'ta_combined_decoder_benchmark_authorized': False}, {'model_constructions_authorized': 1},
                   {'optimizer_steps_authorized': True}, {'canonical_resume_authority': True}):
        with pytest.raises(RuntimeError): scope.require_gate(dict(scope.gate_fields(), **change), recipe)
    with pytest.raises(RuntimeError): scope.require_gate(scope.coverage.gate_fields(), recipe)


def cursor(outcome='COMPLETE', visits=768, updates=768):
    return scope.build_cursor(recipe_ref=ref(scope.RECIPE_PATH), outcome=outcome, optimizer_updates=updates,
        parent_cursor=16 * visits, parent_order_sha256=None, full_state=None, result=ref(scope.ROOT / 'RESULT.json'))


@pytest.mark.parametrize('outcome,visits,updates,valid', [
    ('COMPLETE', 768, 768, True), ('FAILED', 0, 0, True), ('FAILED', 17, 16, True),
    ('COMPLETE', 768, 767, False), ('FAILED', 769, 768, False), ('FAILED', 0, 1, False)])
def test_failure_cursor_retains_actual_progress_without_canonical_resume(outcome, visits, updates, valid):
    if valid:
        value = cursor(outcome, visits, updates)
        assert value['full_state'] is None and value['canonical_resume_authority'] is False
    else:
        with pytest.raises(RuntimeError): cursor(outcome, visits, updates)


def panel_proof(panel):
    return dict(rows=4096, axes_targets_exact=True, m5_side_exact=True, activity_exact=True,
        m5_representations_exact=True if panel == 'LATER' else None,
        m5_representation_arrays_checked=2 if panel == 'LATER' else 0,
        non_m5_side_close=True, canonical_loss_close=True,
        non_m5_side_max_abs={'state': 3e-6, 'future': 0.},
        non_m5_side_max_scaled_error={'state': .5, 'future': 0.}, loss_max_abs=0., **scope.INITIAL_PARITY)


@pytest.mark.parametrize('fault', ['none', 'false_train_repr', 'missing_later_repr', 'scaled_error', 'nan', 'loss', 'm5', 'loose_tolerance'])
def test_initial_parity_is_panel_specific_and_uses_relative_plus_absolute_tolerance(fault):
    proof = dict(initial={p: panel_proof(p) for p in ('TRAIN', 'LATER')}, rng_after_each_pass_exact=True)
    if fault == 'false_train_repr': proof['initial']['TRAIN']['m5_representations_exact'] = True
    elif fault == 'missing_later_repr': proof['initial']['LATER']['m5_representation_arrays_checked'] = 0
    elif fault == 'scaled_error': proof['initial']['TRAIN']['non_m5_side_max_scaled_error']['state'] = 1.01
    elif fault == 'nan': proof['initial']['TRAIN']['non_m5_side_max_abs']['future'] = float('nan')
    elif fault == 'loss': proof['initial']['LATER']['loss_max_abs'] = 5.1e-6
    elif fault == 'm5': proof['initial']['TRAIN']['m5_side_exact'] = False
    elif fault == 'loose_tolerance': proof['initial']['LATER']['atol'] = 3e-6
    if fault == 'none': scope._require_control_parity(proof)
    else:
        with pytest.raises(RuntimeError): scope._require_control_parity(proof)


def completed_documents():
    docs = metadata_documents(); value = cursor(); initial = 'e' * 64
    rng = {str(p): str(p) * 64 for p in (1, 2, 3)}
    control = dict(arms={'UNIQUE_COVERAGE': {'training_rng_by_pass': deepcopy(rng)}})
    docs[scope.CONTROL_RESULT_REF['path']] = control
    old, source, target = [docs[r['path']] for r in (scope.base.VIEW_REF, scope.SOURCE_VIEW_REF, scope.TARGET_VIEW_REF)]
    original = dict(view_sha256=old['contract_sha256'])
    docs[scope.base.LATER_REFS['later_coverage']['path']] = original
    _, later_proof = scope._later_binding(original, old, source, target)
    side = dict(source_model_state_sha256=scope.SOURCE_INITIAL_SHA256, model_state_sha256=initial,
        added_parameters=1032, cursor=0, optimizer_updates=0,
        optimizer_policy='fresh_AdamW_empty_history_no_inherited_updates', copied_side_rows={
            prefix + '.' + tf + '.' + kind: shared + '.' + kind
            for prefix, shared in (('state_model.non_m5_side_heads', 'state_model.query_head'),
                ('future_non_m5_side_heads', 'future_head'))
            for tf in ('M15', 'H1', 'H4', 'D1') for kind in ('weight', 'bias')})
    import_proof = dict(source_checkpoint=docs[scope.CONTROL_RECIPE_REF['path']]['input_refs']['short_initial_checkpoint'],
        source_initial_model_state_sha256=scope.SOURCE_INITIAL_SHA256, original_rng_restored=True,
        additional_checkpoint_decode=False, side_migration=side, later_coverage_binding_proof=later_proof, migration=dict(
            model_state_sha256=scope.SOURCE_INITIAL_SHA256, expanded_parameter_tensors=14,
            added_parameters=2314, buffers_preserved=True))
    arm = dict(batch_visits=768, optimizer_updates=768, skipped_batches=0, passes_completed=3,
        initial_model_state_sha256=initial, initial_rng_state_sha256=scope.INITIAL_RNG_SHA256,
        fixed_pass0_query_policy=True, training_rng_by_pass=deepcopy(rng), final_model_state_sha256='f' * 64,
        evidence_capsule=ref(scope.ROOT / 'COMBINED_DECODER_EVIDENCE.pt'))
    counts = dict(model_constructions=2, model_construction_returns=2, optimizer_constructions=1,
        checkpoint_loads=1, resume_state_load_calls=1, resume_state_load_returns=1,
        strict_full_state_restore_calls=1, strict_full_state_restore_returns=1, initial_rng_imports=1,
        training_forwards=768, evaluation_forwards=1536, forwards=2304, forward_returns=2304,
        candidate_evidence_writes=1, checkpoint_writes=0, control_capture_archive_decodes=2,
        input_sidecar_archive_decodes=12, skipped_batches=0,
        **{k: 768 for k in ('backwards', 'backward_returns', 'optimizer_step_calls',
            'optimizer_step_returns', 'optimizer_updates', 'batch_visits')})
    refs = {k: ref(scope.ROOT / (k + '.json')) for k in scope.MEASUREMENT_LABELS}
    captures = {k: ref(scope.ROOT / (k + '_READOUT.npz')) for k in scope.MEASUREMENT_LABELS}
    for label in scope.MEASUREMENT_LABELS:
        panel = label.rsplit('_', 1)[1]; p = 0 if label.startswith('INITIAL_') else 1 if '_PASS1_' in label else 3
        docs[refs[label]['path']] = dict(capture=captures[label], evaluation_objective='FULL_OBJECTIVE',
            measurement=dict(rows=4096, batches=256, observation_chain_sha256=('1' if panel == 'TRAIN' else '2') * 64),
            optimizer_updates=256 * p, evaluated_model_state_sha256={0: initial, 1: 'a' * 64, 3: 'f' * 64}[p],
            panel_batch_indices=scope.TRAIN_BATCH_INDICES if panel == 'TRAIN' else scope.LATER_BATCH_INDICES,
            evaluation_state_unchanged=True, test_data_used=False)
    report = dict(schema_version=scope.RESULT_SCHEMA, outcome='COMPLETE', recipe=value['recipe'],
        parent_cursor=12288, optimizer_updates=768, batch_visits=768, skipped_batches=0,
        canonical_resume_authority=False, test_data_used=False, counts=counts, arms={'COMBINED_DECODER': arm},
        measurement_refs=refs, panel_captures=captures, initial_restore_exact=True, evaluation_state_unchanged=True,
        source_unchanged=True, same_training_queries=True, rng_stream_parity=True,
        optimizer_initialization='fresh_adamw_empty_state', initial_model_state_sha256=initial, import_proof=import_proof,
        combined_control_parity=dict(initial={p: panel_proof(p) for p in ('TRAIN', 'LATER')}, rng_after_each_pass_exact=True),
        training_records={'COMBINED_DECODER': [dict(pass_number=p, original_batch_index=b, skip_update=False,
            loss=.7, gradient_norm=.9) for p in scope.PASSES for b in scope.training_batch_indices(p)]})
    docs[value['result']['path']] = report
    return value, report, docs


@pytest.mark.parametrize('fault', ['none', 'old_initial_digest', 'side_map', 'constructs', 'decodes', 'skip',
    'rng', 'repeated_batch', 'pass1_panel_state', 'wrong_chain', 'wrong_evidence', 'later_provenance', 'source'])
def test_complete_requires_actual_migration_parity_counts_and_six_matched_panels(monkeypatch, fault):
    value, report, docs = completed_documents()
    if fault == 'old_initial_digest': report['initial_model_state_sha256'] = scope.SOURCE_INITIAL_SHA256
    elif fault == 'side_map': report['import_proof']['side_migration']['copied_side_rows'].pop('future_non_m5_side_heads.D1.bias')
    elif fault == 'constructs': report['counts']['model_constructions'] = 1
    elif fault == 'decodes': report['counts']['control_capture_archive_decodes'] = 4
    elif fault == 'skip': report['skipped_batches'] = 1
    elif fault == 'rng': report['arms']['COMBINED_DECODER']['training_rng_by_pass']['3'] = '0' * 64
    elif fault == 'repeated_batch': report['training_records']['COMBINED_DECODER'][256]['original_batch_index'] = 0
    elif fault == 'pass1_panel_state': docs[report['measurement_refs']['COMBINED_DECODER_PASS1_LATER']['path']]['evaluated_model_state_sha256'] = 'b' * 64
    elif fault == 'wrong_chain': docs[report['measurement_refs']['COMBINED_DECODER_PASS3_LATER']['path']]['measurement']['observation_chain_sha256'] = '0' * 64
    elif fault == 'wrong_evidence': report['arms']['COMBINED_DECODER']['evidence_capsule'] = ref(scope.coverage.ROOT / 'UNIQUE_COVERAGE_EVIDENCE.pt')
    elif fault == 'later_provenance': report['import_proof']['later_coverage_binding_proof']['changed_fields'].append('full_axes')
    elif fault == 'source': report['source_unchanged'] = False
    bind_documents(monkeypatch, docs)
    if fault == 'none': assert scope.require_cursor(value, expected_recipe=value['recipe']) == value
    else:
        with pytest.raises(RuntimeError): scope.require_cursor(value, expected_recipe=value['recipe'])


def test_preflight_delegates_to_the_same_metadata_gates_without_runtime_work(monkeypatch):
    calls = []
    monkeypatch.setattr(scope, 'require_recipe', lambda value, **kw: calls.append(kw))
    result = scope.preflight_read_only({}, source_commit=scope.BASE_COMMIT)
    assert len(calls) == 1
    assert result['model_constructions'] == result['checkpoint_loads'] == result['data_payload_reads'] == 0
    assert result['optimizer_updates'] == 0 and result['actual_initial_parity_still_required_before_updates'] is True
