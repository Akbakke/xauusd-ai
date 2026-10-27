"""Exact frozen276 head-input extraction authority; no fitting or updates.

One separately admitted state/cohort per canonical invocation. Four fixed units
are required by the paired readout study. This module imports no numerics.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PurePosixPath
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import entry_native_direction_checkpoint_diagnostic_v1 as previous

SCHEMA = 'gx1_native_direction_frozen_extraction_plan_v1'
OPERATION = 'F_saved276_frozen_readout_extraction_v1'
STAGE = 'dynamic_m5_direction_frozen_extraction_f'
PHASE = 'native_direction_frozen_extraction'
KIND = 'native_direction_frozen_extraction_window'
WINDOW_SCHEMA = 'gx1_native_direction_frozen_extraction_window_policy_v1'
POINTER_SCHEMA = 'gx1_native_direction_frozen_extraction_pointer_v1'
MODULE = previous.MODULE
REPO, PROGRAM, BASE = previous.REPO, previous.PROGRAM, previous.BASE
PROPOSAL, INITIAL_REPORT, INITIAL_PREDICTIONS = previous.PROPOSAL, previous.INITIAL_REPORT, previous.INITIAL_PREDICTIONS
FOLD = dict(previous.FOLD)
STATE_SPEC = (previous.STATE_SPEC[0], previous.STATE_SPEC[-1])
TERMINAL_JSON_RESERVE_BYTES = previous.TERMINAL_JSON_RESERVE_BYTES
RESOURCE_LIMITS = dict(previous.RESOURCE_LIMITS)
DESIGN = {'path': PROGRAM + '/F_FROZEN_READOUT_DESIGN_001.json',
    'sha256': 'f09158bc72287bfd9b154273812355c3634e39ddce018d72720f110856d14713'}
UNITS_DESIGN = {'path': PROGRAM + '/F_FROZEN_READOUT_EXTRACTION_UNITS_001.json',
    'sha256': 'd2933f31c243f67c874a6203848bf1940ba26579ccef8ad67c749f34f08fca79'}
TERMINAL_REPORT = {'path': BASE + '/RUNTIME/prefix-fit/window-0002/OOF.json',
    'sha256': '76cf81ac015abe30e9c1d27d3e837ee477a83703d6775cc518c6100662ca395f'}
TERMINAL_PREDICTIONS = {'path': BASE + '/RUNTIME/prefix-fit/window-0002/OOF.npz',
    'sha256': 'fd7d3177f1a4d588f235156a2dbc02d670137ab91862e779606bb932e0e4ef57'}
TIMING_UNIT = 'own0U_prefix_timing256'
TIMING_DESIGN = {'path': PROGRAM + '/F_FROZEN_EXTRACTION_THROUGHPUT_MEASUREMENT_DESIGN_001.json',
    'sha256': '572bcbe54cb67e7d477c40272c3bde3ff179d9b349499643849d09493c1f4684'}
# Exact metadata transition; full observations use only the separately bound partitions.
GC_LAYOUT_METADATA_PARITY = {'path': PROGRAM + '/DIRECTION_GC_LAYOUT_REUSE_QUALIFICATION_ROOT_REVIEW_003.json',
    'sha256': '1d3347ee52f02af7afcac5d55c02532c0e2823e1ce20397633785a2512d39263'}
GC_METADATA_SOURCE_TRANSITIONS = {
    'gx1/contracts/entry_gc_mtf_history_v1.py': (
        '82ead3da60383ac816f7b75963cbe5118520e2d058ec120edb1dff169e0362cd',
        'de3a254f605c8e84543855b09f9ab98e6401225f7c212af5926c6b58ae250d12'),
    'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py': (
        'db7bd2823b5fd1c36a42da2433bef22ac106ed682e700b476234d6b7f9b19bba',
        '407725139b0df536e30713e560621cb056739753ff401242657d1e836f46982f')}
# These four full cohorts remain the complete readout study; timing is separate.
UNITS = {
    'own0U_prefix': ('own0U', 'prefix'),
    'own0U_later': ('own0U', 'later'),
    'terminal_prefix': ('terminal_u2976', 'prefix'),
    'terminal_later': ('terminal_u2976', 'later'),
}
PARTITION_DESIGN = {'path': PROGRAM + '/F_FROZEN_READOUT_PARTITION_DESIGN_001.json',
    'sha256': 'e9f205865da12b8237f0729518d716394c34efea044cb799eda1b1e77981c7fc'}
PARTITION_PARITY = {'path': PROGRAM + '/F_FROZEN_EXTRACTION_own0U_prefix_timing256_CAMPAIGN_20261027_004/ACTUAL_ROOT_REVIEW.json',
    'sha256': '97296dc91e9ffb843499eb89f463358407fd6203b457f46d0ef8cfed57e6f7cb'}
# Presentation offsets, never canonical-row or outcome-selected subsets.
PARTITION_UNITS = {
    'own0U_prefix_part1': ('own0U_prefix', 0, 15872),
    'own0U_prefix_part2': ('own0U_prefix', 15872, 31744),
    'own0U_prefix_part3': ('own0U_prefix', 31744, 47602),
    'own0U_later_full': ('own0U_later', 0, 17288),
    'terminal_prefix_part1': ('terminal_prefix', 0, 15872),
    'terminal_prefix_part2': ('terminal_prefix', 15872, 31744),
    'terminal_prefix_part3': ('terminal_prefix', 31744, 47602),
    'terminal_later_full': ('terminal_later', 0, 17288),
}
NEW_FILES = ('gx1/contracts/entry_native_direction_frozen_extraction_v1.py',
    'gx1/scripts/extract_entry_native_direction_frozen_readout_v1.py',
    'tests/test_entry_native_direction_frozen_extraction_v1.py')
TRANSPORT_ORIGINAL_SHA256 = {**previous.TRANSPORT_ORIGINAL_SHA256,
    'gx1/contracts/entry_direction_probability_reference_v1.py':
        '89ece5378ac777701aada7a8a80e14f4a3f2f7fd8c5a95bae7fc3c36d06edf94'}
# Exact transport and metric pins, including the qualified F cap-route repair.
PINNED_PRE_F = {
    'scripts/gx1_capped_run.sh': '42be63429d4b5c12377da5ac7a6631252fa4b083ad4a5b6595e76eb80e971e55',
    'gx1/contracts/entry_direction_probability_reference_v1.py':
        '59d57110cf7a51da8158f9535088147d50673dea10728a2ebe3c9e469f0cee76'}
LIMITS = (
    'Frozen EVAL representation; no native optimizer steps or reverse passes.',
    'All four fixed units required before separately admitted readout fitting.',
    'Prefix and later cohorts are reused development TRAIN, never sealed confirmation.',
    'No predictive information, convergence, economics or checkpoint-selection claim.')
plain_ref, _identity, _seal, _path = previous.plain_ref, previous._identity, previous._seal, previous._path
PARTITION_LIMITS = (LIMITS[0],
    'All eight fixed partitions and exact full-cohort assembly required before separately admitted readout fitting.',
    *LIMITS[2:])
require_proposal = previous.require_proposal
INDEPENDENT_ROOT = '/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63'
INDEPENDENT_DESIGN = {'path': INDEPENDENT_ROOT + '/NATIVE_0U_COMPATIBILITY_DESIGN_001.json',
    'sha256': '3c1b5694ecd97c7c6ac8d4b1315f86d9b24c45d7b3d61c3a9e0533564047b5ed'}
INDEPENDENT_UNITS = {
    TIMING_UNIT: ('independent_direction_0u_timing256_20261027', 'NATIVE_0U_TIMING256_001'),
    'own0U_prefix_part3': ('independent_direction_0u_part3_20261027', 'NATIVE_0U_PREFIX_PART3_001')}
INDEPENDENT_CODE_FILES = (NEW_FILES[0],
    'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py', NEW_FILES[1])
INDEPENDENT_REVIEW_SCHEMA = 'gx1_independent_default_sign_0u_timing_review_v1'


def read_independent_design(completion, read=None):
    require(type(completion) is dict and set(completion) == {'design', 'initial_parity'}
        and completion['design'] == INDEPENDENT_DESIGN, 'INDEPENDENT_DESIGN_REF')
    if read is None:
        read = lambda ref: campaign.read_bound_json(Path(ref['path']), ref['sha256'])
    design = read(INDEPENDENT_DESIGN)
    require(design.get('schema_version') == 'gx1_independent_default_sign_0u_completion_design_v1'
        and design.get('canonical_repo') == REPO and design.get('canonical_branch') == 'work/gx1-current'
        and design['authority']['execution_authorized'] is False, 'INDEPENDENT_SOURCE_DESIGN')
    return design


def _independent_scope(value):
    require(value.get('unit') in INDEPENDENT_UNITS, 'INDEPENDENT_FIXED_UNIT')
    stage, name = INDEPENDENT_UNITS[value['unit']]
    root = INDEPENDENT_ROOT + '/' + name
    require(value.get('admission_stage') == stage and value.get('run_root') == root
        and value.get('runtime_root') == root + '_RUNTIME'
        and value.get('gc_layout_metadata_parity') == GC_LAYOUT_METADATA_PARITY,
        'INDEPENDENT_FIXED_NAMESPACE')
    completion = value['independent_completion']
    require(type(completion) is dict and set(completion) == {'design', 'initial_parity'}
        and completion['design'] == INDEPENDENT_DESIGN, 'INDEPENDENT_DESIGN_REF')
    parity = completion['initial_parity']
    if value['unit'] == TIMING_UNIT:
        require(parity is None, 'INDEPENDENT_TIMING_NO_PRESUMED_PARITY')
    else:
        require(plain_ref(parity)['path'] == INDEPENDENT_ROOT
            + '/NATIVE_0U_TIMING256_001/ACTUAL_ROOT_REVIEW.json', 'INDEPENDENT_FRESH_PARITY_REF')


def require_independent_completion(recipe, read=None):
    """Validate the two fixed metadata-only transitions; grant no execution."""
    if 'independent_completion' not in recipe:
        return None
    _independent_scope(recipe)
    if read is None:
        read = lambda ref: campaign.read_bound_json(Path(ref['path']), ref['sha256'])
    design = read_independent_design(recipe['independent_completion'], read)
    closure = design['source_closure']
    baseline = closure['baseline_current57_references']
    require(len(baseline) == 57 and set(recipe['source_owners']) == set(baseline)
        and set(closure['only_runtime_code_edit_exceptions']) == set(INDEPENDENT_CODE_FILES),
        'INDEPENDENT_SOURCE_CLOSURE')
    exceptions = {*INDEPENDENT_CODE_FILES, NEW_FILES[-1]}
    require(NEW_FILES[-1] in closure['qualification_test_edit_allowlist'], 'INDEPENDENT_TEST_PIN')
    for name, ref in baseline.items():
        if name not in exceptions:
            require(recipe['source_owners'][name] == ref, 'INDEPENDENT_UNCHANGED_SOURCE:' + name)
    if recipe['unit'] != TIMING_UNIT:
        _require_independent_timing(recipe, design, read)
    return design


def _require_independent_timing(recipe, design, read):
    review = read(recipe['independent_completion']['initial_parity'])
    require(review.get('schema_version') == INDEPENDENT_REVIEW_SCHEMA
        and review.get('status') == 'COMPLETE_AND_QUALIFIED_CURRENT_SIGN_ONLY_OWN0U256'
        and review.get('design') == INDEPENDENT_DESIGN and review.get('test_data_used') is False
        and review.get('state_preserved') is True and review.get('all_original_fields_bit_identical') is True,
        'INDEPENDENT_TIMING_REVIEW')
    timing_ref = plain_ref(review['recipe'])
    timing = require_plan(read(timing_ref))
    require(timing['unit'] == TIMING_UNIT and PurePosixPath(timing_ref['path']).parent == PurePosixPath(INDEPENDENT_ROOT),
        'INDEPENDENT_TIMING_RECIPE')
    changing = {'unit', 'admission_stage', 'units_design', 'run_root', 'runtime_root',
        'execution_binding_path', 'budget', 'operation_limits', 'independent_completion'}
    require({k: v for k, v in recipe.items() if k not in changing}
        == {k: v for k, v in timing.items() if k not in changing}, 'INDEPENDENT_TIMING_SAME_BINDINGS')
    root = timing['run_root']
    for key, path in (('result', root + '/RESULT.json'), ('native_report', root + '/NATIVE_REPORT.json'),
            ('pointer', root + '/DIAGNOSTIC_POINTER.json'),
            ('canonical_terminal_receipt', timing['runtime_root'] + '/receipts/invocation-0001.json')):
        require(plain_ref(review[key])['path'] == path, 'INDEPENDENT_TIMING_ARTIFACT:' + key)
    result, report = read(review['result']), read(review['native_report'])
    original = design['original_bindings']
    parity = {'reference': INITIAL_PREDICTIONS, 'rows': 256, 'bit_identical': True}
    limits = call_limits(TIMING_UNIT)
    require(result.get('decision') == 'COMPLETE_FROZEN_READOUT_EXTRACTION' and result.get('error') is None
        and result.get('recipe') == timing_ref and result.get('unit') == TIMING_UNIT
        and result.get('diagnostic_only') is True and result.get('state') == original['own0U_state']['ref']
        and result.get('state_role') == 'own0U' and result.get('cohort') == 'prefix'
        and result.get('representation_rows') == 256 and result.get('hidden_width') == 128
        and result.get('native_report') == review['native_report']
        and result.get('fresh_identity_sha256') == original['fresh_identity_sha256']
        and result.get('feature_view_identity') == original['feature_view_identity']
        and result.get('original_prediction_parity') == parity
        and result.get('observed_calls') == result.get('observed_returns') == limits
        and result.get('calls') == {'calls': limits, 'returns': limits, 'optimizer_steps': 0, 'reverse_calls': 0}
        and all(result.get(k) is True for k in ('source_unchanged', 'originals_stat_preserved'))
        and all(type(result.get(k)) is int and result[k] == 0 for k in ('optimizer_steps', 'native_optimizer_steps'))
        and result.get('test_data_used') is False
        and result.get('denied_phase_opens') == result.get('delegated_denied_opens') == [],
        'INDEPENDENT_TIMING_RESULT')
    require(report.get('unit') == TIMING_UNIT and report.get('cutoff_ns') == FOLD['train_cutoff_ns']
        and report.get('original_prediction_parity') == parity
        and report.get('own_prefix_prior') == result.get('own_prefix_prior')
        and type(report.get('state_preservation')) is dict
        and set(report['state_preservation']) == {'model', 'all_buffers', 'rng', 'modes'},
        'INDEPENDENT_TIMING_REPORT')
    pointer = require_frozen_extraction_pointer(read(review['pointer']), expected_recipe=timing_ref,
        expected_unit=TIMING_UNIT, verify_files=False)
    require(pointer['outcome'] == 'COMPLETE' and pointer['result'] == review['result'], 'INDEPENDENT_TIMING_POINTER')
    receipt = read(review['canonical_terminal_receipt'])
    require(receipt.get('receipt_sha256') == campaign.canonical_sha256({k: v for k, v in receipt.items() if k != 'receipt_sha256'})
        and receipt.get('outcome') == 'COMPLETE' and receipt.get('kind') == KIND
        and receipt.get('guard_decision') == 'PASS' and receipt.get('trainer_guard_exit_code') == 0
        and receipt.get('progress_observer_exit_code') == 0 and receipt.get('invocation_number') == 1
        and receipt.get('test_data_used') is False and receipt.get('checkpoint_pointer_after') == review['pointer'],
        'INDEPENDENT_TIMING_RECEIPT')
    campaign_ref = plain_ref(result['campaign_plan'])
    require(PurePosixPath(campaign_ref['path']).parent == PurePosixPath(INDEPENDENT_ROOT),
        'INDEPENDENT_TIMING_CAMPAIGN_REF')
    bound_campaign = read(campaign_ref)
    require(bound_campaign.get('plan_sha256') == campaign.canonical_sha256({k: v for k, v in bound_campaign.items() if k != 'plan_sha256'})
        and bound_campaign.get('recipe') == timing_ref and bound_campaign.get('runtime_root') == timing['runtime_root']
        and bound_campaign.get('phase') == PHASE and bound_campaign.get('test_data_used') is False
        and receipt.get('plan_sha256') == bound_campaign['plan_sha256']
        and type(bound_campaign.get('invocations')) is list and len(bound_campaign['invocations']) == 1,
        'INDEPENDENT_TIMING_CAMPAIGN')
    invocation = read(plain_ref(bound_campaign['invocations'][0]))
    require(invocation.get('invocation_sha256') == campaign.canonical_sha256({k: v for k, v in invocation.items() if k != 'invocation_sha256'})
        and receipt.get('invocation_sha256') == invocation['invocation_sha256']
        and invocation.get('kind') == KIND and invocation.get('invocation_number') == 1
        and invocation.get('optimizer_step_budget') == 0 and invocation.get('test_data_used') is False
        and invocation.get('source_commit') == bound_campaign.get('source_commit'), 'INDEPENDENT_TIMING_INVOCATION')
    execution = read(plain_ref(invocation['execution_manifest']))
    require(execution.get('prelaunch_manifest') == timing_ref and execution.get('invocation_kind') == KIND
        and execution.get('source_commit') == invocation['source_commit'], 'INDEPENDENT_TIMING_EXECUTION')



def require(ok, reason):
    if not ok:
        raise RuntimeError('NATIVE_DIRECTION_FROZEN_EXTRACTION_' + reason)


def unit_spec(unit):
    require(type(unit) is str and (unit in UNITS or unit in PARTITION_UNITS or unit == TIMING_UNIT), 'FIXED_UNIT')
    if unit in PARTITION_UNITS:
        parent, start, end = PARTITION_UNITS[unit]
        base = unit_spec(parent)
        require(0 <= start < end <= base['rows'] and start % 16 == 0
            and (end % 16 == 0 or end == base['rows']), 'ORIGINAL_BATCH_BOUNDARIES')
        return {**base, 'unit': unit, 'rows': end - start, 'forwards': (end - start + 15) // 16,
            'parity': base['parity'] if start == 0 else None,
            'partition': {'parent_unit': parent, 'presentation_start': start,
                'presentation_end': end, 'full_rows': base['rows']}}
    diagnostic = unit == TIMING_UNIT
    role, cohort = ('own0U', 'prefix') if diagnostic else UNITS[unit]
    state = next(s for s in STATE_SPEC if s[0] == role)
    rows = 256 if diagnostic else (FOLD['prefix_rows'] if cohort == 'prefix'
        else FOLD['oof_end_row'] - FOLD['oof_start_row'])
    parity = INITIAL_PREDICTIONS if diagnostic or unit == 'own0U_prefix' else (
        TERMINAL_PREDICTIONS if unit == 'terminal_later' else None)
    return {'unit': unit, 'state': state, 'cohort': cohort, 'rows': rows,
        'forwards': (rows + 15) // 16, 'parity': deepcopy(parity), 'diagnostic_only': diagnostic,
        'cutoff_ns': FOLD['train_cutoff_ns'] if cohort == 'prefix' else FOLD['oof_cutoff_ns']}


def unit_design(unit):
    spec = unit_spec(unit)
    return deepcopy(PARTITION_DESIGN if unit in PARTITION_UNITS else (
        TIMING_DESIGN if spec['diagnostic_only'] else UNITS_DESIGN))


def resource_limits(unit):
    limits = dict(RESOURCE_LIMITS)
    if unit_spec(unit)['diagnostic_only']:
        limits['max_wall_seconds'] = 600
    return limits


def call_limits(unit):
    spec = unit_spec(unit)
    result = dict(previous.CALL_LIMITS)
    result.update(checkpoint_load=1, panel_forward=spec['forwards'], gradient_forward=0,
        reverse=0, batch=spec['forwards'], torch_load=1,
        # C preparation consumed32 loads; one roundtrip of our new output,
        # plus a saved-prediction load only in the two exact parity units.
        numpy_load=33 + int(spec['parity'] is not None),
        native_forward=spec['forwards'], research_forward=spec['forwards'],
        trainer_forward=spec['forwards'], autograd_grad=0, engine_reverse=0,
        gather=spec['forwards'])
    return result


def historical_gc_view_source(ref, metadata_parity):
    """Keep historical manifest identity while checking one qualified current file."""
    require(type(metadata_parity) is dict and metadata_parity == GC_LAYOUT_METADATA_PARITY,
        'GC_METADATA_PARITY_REF')
    source = plain_ref(ref)
    name = 'gx1/contracts/entry_gc_mtf_history_v1.py'
    if source['path'] == REPO + '/' + name:
        old, current = GC_METADATA_SOURCE_TRANSITIONS[name]
        require(source['sha256'] == old, 'HISTORICAL_GC_SOURCE')
        return {'path': source['path'], 'sha256': current}
    return source


def require_plan(value):
    fields = {'schema_version', 'operation', 'admission_stage', 'proposal', 'design', 'units_design', 'unit',
        'source_owners', 'transport_source_transitions', 'run_root', 'runtime_root',
        'execution_binding_path', 'budget', 'operation_limits', 'initial_report',
        'initial_predictions', 'terminal_report', 'terminal_predictions', 'max_invocations',
        'execution_authorized', 'data_reads_authorized', 'checkpoint_reads_authorized',
        'genuine_forwards_authorized', 'gradient_reverse_authorized', 'optimizer_steps_authorized', 'test_data_used'}
    require(type(value) is dict, 'PLAN_TYPE')
    independent = 'independent_completion' in value
    if independent:
        fields.add('independent_completion')
        _independent_scope(value)
    metadata_transition = 'gc_layout_metadata_parity' in value
    if metadata_transition:
        fields.add('gc_layout_metadata_parity')
    require(set(value) == fields and value['schema_version'] == SCHEMA
        and value['operation'] == OPERATION
        and (independent or value['admission_stage'] == STAGE), 'PLAN_SCHEMA')
    unit_spec(value['unit'])
    require(value['unit'] not in PARTITION_UNITS or metadata_transition, 'PARTITION_METADATA_PROOF_REQUIRED')
    require(value['proposal'] == PROPOSAL and value['design'] == DESIGN and value['units_design'] == unit_design(value['unit'])
        and value['initial_report'] == INITIAL_REPORT and value['initial_predictions'] == INITIAL_PREDICTIONS
        and value['terminal_report'] == TERMINAL_REPORT and value['terminal_predictions'] == TERMINAL_PREDICTIONS,
        'FIXED_INPUT_IDENTITIES')
    owners = value['source_owners']
    require(type(owners) is dict and all(plain_ref(ref)['path'] == REPO + '/' + name for name, ref in owners.items())
        and all(name in owners for name in (*NEW_FILES, *previous.NEW_FILES)), 'SOURCE_OWNERS')
    transitions = value['transport_source_transitions']
    require(type(transitions) is dict and set(transitions) == set(TRANSPORT_ORIGINAL_SHA256), 'EXACT_TRANSITIONS')
    for name, digest in TRANSPORT_ORIGINAL_SHA256.items():
        item = transitions[name]
        require(type(item) is dict and set(item) == {'old276', 'current'}
            and item['old276'] == {'path': REPO + '/' + name, 'sha256': digest}
            and plain_ref(item['current']) == owners.get(name)
            and item['current']['sha256'] != digest, 'SOURCE_TRANSITION:' + name)
    require(all(owners[name]['sha256'] == digest for name, digest in PINNED_PRE_F.items()), 'PRE_F_PINNED_SOURCES')
    if metadata_transition:
        require((value['unit'] == TIMING_UNIT or value['unit'] in PARTITION_UNITS)
            and value['gc_layout_metadata_parity'] == GC_LAYOUT_METADATA_PARITY
            and type(value['gc_layout_metadata_parity']) is dict, 'GC_METADATA_TIMING_SCOPE')
        for name, (_old, current) in GC_METADATA_SOURCE_TRANSITIONS.items():
            if independent and name in INDEPENDENT_CODE_FILES:
                continue
            require(owners.get(name) == {'path': REPO + '/' + name, 'sha256': current},
                'GC_METADATA_CURRENT_SOURCE:' + name)
    root, runtime = (PurePosixPath(value[k]) for k in ('run_root', 'runtime_root'))
    require(independent or all(str(p) == value[k] and p.parent == PurePosixPath(PROGRAM)
        and p.name.startswith('F_FROZEN_EXTRACTION_' + value['unit'] + '_') for k, p in
        (('run_root', root), ('runtime_root', runtime))) and root != runtime, 'SEPARATE_UNIT_ROOTS')
    require(value['execution_binding_path'] == str(runtime / 'EXECUTION_BINDING.json'), 'EXTERNAL_BINDING_PATH')
    require(type(value['max_invocations']) is int and value['max_invocations'] == 1, 'SINGLE_INVOCATION')
    require(type(value['budget']) is type(value['operation_limits']) is dict
        and value['budget'] == resource_limits(value['unit']) and value['operation_limits'] == call_limits(value['unit'])
        and all(type(n) is int for n in (*value['budget'].values(), *value['operation_limits'].values())), 'FIXED_CEILINGS')
    for key in ('execution_authorized', 'data_reads_authorized', 'checkpoint_reads_authorized', 'genuine_forwards_authorized'):
        require(value[key] is True, 'ADMISSION_' + key)
    for key in ('gradient_reverse_authorized', 'optimizer_steps_authorized', 'test_data_used'):
        require(value[key] is False, 'CLOSED_' + key)
    if independent:
        require_independent_completion(value)
    return deepcopy(value)


def require_admission(admission, recipe_ref, unit):
    unit_spec(unit)
    require(type(admission) is dict and admission.get('operation') == OPERATION
        and admission.get('unit') == unit and admission.get('execution_authorized') is True
        and admission.get('recipe') == plain_ref(recipe_ref)
        and type(admission.get('max_invocations')) is int and admission['max_invocations'] == 1
        and admission.get('genuine_payload_reads_authorized') is True
        and admission.get('genuine_forwards_authorized') is True
        and all(admission.get(k) is False for k in ('gradient_reverse_authorized',
            'optimizer_steps_authorized', 'genuine_predictor_fit_authorized', 'test_data_used')), 'ROOT_EXECUTION_ADMISSION')
    return deepcopy(admission)


def require_source_closure(recipe, original, proposal):
    previous_owners, reused = original['source_owners'], proposal['existing_sources_reused_without_change']
    overlap = set(previous_owners) & set(reused)
    require(len(previous_owners) == 50 and len(reused) == 7 and len(overlap) == 6
        and all(previous_owners[n] == reused[n] for n in overlap), 'ORIGINAL_SOURCE_INVENTORY')
    required = {**previous_owners, **reused}
    require(set(recipe['source_owners']) == set(required) | set(previous.NEW_FILES) | set(NEW_FILES),
        'EXACT_SOURCE_CLOSURE')
    if 'independent_completion' in recipe:
        require_independent_completion(recipe)
        for name in TRANSPORT_ORIGINAL_SHA256:
            require(recipe['transport_source_transitions'][name]['old276'] == required[name], 'ORIGINAL_SOURCE:' + name)
        return deepcopy(recipe['source_owners'])
    for name, ref in required.items():
        expected = recipe['transport_source_transitions'][name]['current'] if name in TRANSPORT_ORIGINAL_SHA256 else ref
        if 'gc_layout_metadata_parity' in recipe and name in GC_METADATA_SOURCE_TRANSITIONS:
            require((recipe['unit'] == TIMING_UNIT or (
                recipe['unit'] in PARTITION_UNITS and recipe['units_design'] == PARTITION_DESIGN))
                and recipe['gc_layout_metadata_parity'] == GC_LAYOUT_METADATA_PARITY,
                'GC_METADATA_TIMING_SCOPE')
            old, current = GC_METADATA_SOURCE_TRANSITIONS[name]
            require(ref == {'path': REPO + '/' + name, 'sha256': old}, 'GC_METADATA_ORIGINAL_SOURCE:' + name)
            expected = {'path': REPO + '/' + name, 'sha256': current}
        require(recipe['source_owners'][name] == expected, 'UNCHANGED_SOURCE_OWNER:' + name)
        if name in TRANSPORT_ORIGINAL_SHA256:
            require(recipe['transport_source_transitions'][name]['old276'] == ref, 'ORIGINAL_SOURCE:' + name)
    return deepcopy(recipe['source_owners'])


class CallBudget:
    def __init__(self, unit):
        self.limits = call_limits(unit)
        self.calls = {k: 0 for k in self.limits}
        self.returns = dict(self.calls)

    def start(self, key):
        require(key in self.limits and self.calls[key] < self.limits[key], 'CALL_CEILING_' + key)
        self.calls[key] += 1

    def returned(self, key):
        require(key in self.limits and self.returns[key] < self.calls[key], 'RETURN_' + key)
        self.returns[key] += 1

    def complete(self):
        require(self.calls == self.returns == self.limits, 'EXACT_COMPLETED_CALLS')
        return {'calls': dict(self.calls), 'returns': dict(self.returns), 'optimizer_steps': 0, 'reverse_calls': 0}


def require_frozen_extraction_recipe(binding, *, source_repo, source_commit, verify_files=True):
    require(verify_files is True and Path(source_repo) == Path(REPO)
        and campaign._COMMIT.fullmatch(source_commit) is not None, 'RECIPE_SOURCE')
    ref = campaign.require_binding(binding, label='F static recipe')
    recipe = require_plan(campaign.read_bound_json(Path(ref['path']), ref['sha256']))
    proposal = require_proposal(campaign.read_bound_json(Path(PROPOSAL['path']), PROPOSAL['sha256']))
    original_ref = proposal['existing_input_refs']['recipe']
    original = campaign.read_bound_json(Path(original_ref['path']), original_ref['sha256'])
    require_source_closure(recipe, original, proposal)
    for name, owner in recipe['source_owners'].items():
        require(campaign.require_binding(owner, label='F source owner')['path'] == REPO + '/' + name, 'CURRENT_SOURCE:' + name)
    for key in ('run_root', 'runtime_root', 'execution_binding_path'):
        _path(recipe[key], key)
    return recipe


def require_frozen_extraction_window_policy(value, *, verify_files=True):
    fields = {'schema_version', 'recipe', 'invocation_number', 'max_invocation_seconds',
        'progress_path', 'campaign_cursor_path', 'output_directory', 'unit', 'test_data_used'}
    window = _identity(value, fields, WINDOW_SCHEMA, 'policy_sha256')
    unit_spec(window['unit'])
    require(type(window['invocation_number']) is int and window['invocation_number'] == 1
        and type(window['max_invocation_seconds']) is int and window['max_invocation_seconds'] == 12000
        and window['test_data_used'] is False, 'WINDOW_SCOPE')
    campaign.require_binding(window['recipe'], label='F window recipe', verify_file=verify_files)
    output = _path(window['output_directory'], 'F output')
    require(_path(window['progress_path'], 'F progress') == output / 'PROGRESS.json'
        and _path(window['campaign_cursor_path'], 'F pointer') == output / 'DIAGNOSTIC_POINTER.json', 'WINDOW_LAYOUT')
    return window


def require_frozen_extraction_invocation_execution(result, execution, *, source_repo, source_commit):
    recipe_ref = campaign.require_binding(execution['prelaunch_manifest'], label='F recipe')
    recipe = require_frozen_extraction_recipe(recipe_ref, source_repo=source_repo, source_commit=source_commit)
    window_ref = campaign.require_binding(execution['train_session_manifest'], label='F window')
    window = require_frozen_extraction_window_policy(campaign.read_bound_json(Path(window_ref['path']), window_ref['sha256']))
    separator = result['launcher_argv'].index('--')
    require(result['launcher_argv'][separator + 1:] == [str(Path(source_repo) / '.venv/bin/python'), '-m', MODULE,
        '--window-policy', window_ref['path'], '--window-policy-file-sha256', window_ref['sha256'],
        '--progress-path', window['progress_path']], 'INVOCATION_ARGV')
    require(result['kind'] == KIND and type(result['batch_size']) is int and result['batch_size'] == 16
        and result['epoch_index'] is None and type(result['invocation_number']) is int and result['invocation_number'] == 1
        and type(result['window_index']) is int and result['window_index'] == 0
        and type(result['window_count']) is int and result['window_count'] == 1
        and type(result['optimizer_step_budget']) is int and result['optimizer_step_budget'] == 0
        and result['expected_success_outcome'] == 'COMPLETE' and result['maximum_wall_seconds'] == 13800
        and result['checkpoint'] == {'pointer_path': window['campaign_cursor_path'], 'before_mode': 'GENESIS',
            'predecessor_invocation_number': None, 'write_mode': 'ATOMIC_UPDATE'}
        and result['progress_path'] == window['progress_path'] and window['recipe'] == recipe_ref
        and window['output_directory'] == recipe['run_root'] and window['unit'] == recipe['unit']
        and execution['prelaunch_manifest_sha256'] == campaign.canonical_sha256(recipe)
        and execution['train_session_manifest_sha256'] == window['policy_sha256'], 'INVOCATION_BINDING')
    result['frozen_extraction_recipe'] = recipe_ref
    result['frozen_extraction_window'] = window


def require_frozen_extraction_campaign_plan(value, *, verify_files=True):
    fields = {'schema_version', 'decision', 'campaign_id', 'phase', 'created_utc', 'source_repo', 'source_commit',
        'runtime_root', 'gpu_uuid', 'prepared_windows_boot', 'recipe', 'invocations', 'signed_guard_sources',
        'controller_sources', 'policy', 'authority', 'test_data_used'}
    plan = _identity(value, fields, campaign.PLAN_SCHEMA, 'plan_sha256')
    require(plan['phase'] == PHASE and plan['decision'] == 'PASS_PREPARED'
        and type(plan['campaign_id']) is str and bool(plan['campaign_id']) and plan['source_repo'] == REPO
        and type(plan['source_commit']) is str and campaign._COMMIT.fullmatch(plan['source_commit']) is not None
        and type(plan['gpu_uuid']) is str and plan['gpu_uuid'].startswith('GPU-')
        and plan['test_data_used'] is False, 'CAMPAIGN_IDENTITY')
    campaign._utc(plan['created_utc'], 'F campaign creation')
    runtime = _path(plan['runtime_root'], 'F runtime')
    plan['prepared_windows_boot'] = campaign.require_boot_identity(plan['prepared_windows_boot'])
    require(plan['policy'] == campaign.campaign_hardware_policy(PHASE)
        and plan['authority'] == {'test': False, 'promotion': False, 'paper': False, 'live': False, 'cloud_spend': False}, 'CAMPAIGN_AUTHORITY')
    for key, names in (('signed_guard_sources', {'runner', 'guard', 'query', 'certificate'}),
            ('controller_sources', {'controller', 'observer', 'campaign_cli'})):
        require(type(plan[key]) is dict and set(plan[key]) == names, 'CAMPAIGN_SOURCES:' + key)
        plan[key] = {name: campaign.require_binding(ref, label=key, verify_file=verify_files) for name, ref in plan[key].items()}
    plan['recipe'] = campaign.require_binding(plan['recipe'], label='F campaign recipe', verify_file=verify_files)
    require(type(plan['invocations']) is list and len(plan['invocations']) == 1, 'ONE_CAMPAIGN_INVOCATION')
    plan['invocations'] = [campaign.require_binding(plan['invocations'][0], label='F invocation', verify_file=verify_files)]
    plan['selection_artifact_sha256'] = None
    if verify_files:
        recipe = require_frozen_extraction_recipe(plan['recipe'], source_repo=Path(REPO), source_commit=plan['source_commit'])
        invocation_ref = plan['invocations'][0]
        invocation = campaign.require_invocation(campaign.read_bound_json(Path(invocation_ref['path']), invocation_ref['sha256']),
            source_repo=Path(REPO), source_commit=plan['source_commit'], verify_files=True)
        window = invocation.get('frozen_extraction_window', {})
        require(invocation['kind'] == KIND and invocation.get('frozen_extraction_recipe') == plan['recipe']
            and recipe['runtime_root'] == str(runtime) and window['output_directory'] == recipe['run_root'] and window['unit'] == recipe['unit']
            and invocation['guard_log_path'] == str(runtime / 'logs/invocation-0001.signed-guard.log'), 'CAMPAIGN_LAYOUT')
        plan['checked_invocations'] = [invocation]
    return plan


def require_native_direction_frozen_extraction_readiness(repo, *, window_path, window_file_sha256):
    import subprocess
    repo = Path(repo)
    window = require_frozen_extraction_window_policy(campaign.read_bound_json(Path(window_path), window_file_sha256))
    policy_ref = {'path': str(repo / 'NEXT_RUN_POLICY.json'), 'sha256': campaign.file_sha256(repo / 'NEXT_RUN_POLICY.json')}
    policy = campaign.read_bound_json(Path(policy_ref['path']), policy_ref['sha256'])
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    branch = subprocess.check_output(['git', '-C', str(repo), 'branch', '--show-current'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain=v1', '--untracked-files=all'], text=True)
    recipe = require_frozen_extraction_recipe(window['recipe'], source_repo=repo, source_commit=head)
    require(repo == Path(REPO) and branch == 'work/gx1-current' and not dirty
        and policy.get('schema_version') == 'gx1_next_native_run_policy_v1'
        and policy.get('canonical_source_repo') == REPO and policy.get('canonical_branch') == branch
        and policy.get('training_module') == MODULE and all(policy.get(key) is False for key in
            ('training_enabled', 'native_training_enabled', 'full_epoch_training_allowed', 'full_val_allowed')), 'READINESS_GLOBAL')
    require(window['output_directory'] == recipe['run_root'] and window['unit'] == recipe['unit'], 'READINESS_OUTPUT')
    gate = policy.get(recipe['admission_stage'])
    require(type(gate) is dict and gate.get('recipe') == window['recipe']
        and gate.get('unit') == window['unit'] and gate.get('run_root') == recipe['run_root'] and gate.get('runtime_root') == recipe['runtime_root']
        and type(gate.get('max_invocations')) is int and gate['max_invocations'] == 1
        and type(gate.get('claimed_invocations')) is int and gate['claimed_invocations'] == 0
        and type(gate.get('completed_invocations')) is int and gate['completed_invocations'] == 0
        and gate.get('relaunch_allowed') is False, 'READINESS_ONCE')
    for key in ('execution_authorized', 'data_reads_authorized', 'checkpoint_reads_authorized',
            'genuine_constructor_authorized', 'genuine_forwards_authorized'):
        require(gate.get(key) is True, 'READINESS_GATE:' + key)
    for key in ('gradient_reverse_authorized', 'optimizer_steps_authorized', 'training_enabled', 'native_training_enabled',
            'full_epoch_training_allowed', 'full_val_allowed', 'genuine_predictor_fit_authorized',
            'normalization_fit_authorized', 'synthetic_test_execution_authorized', 'native_launch_authorized',
            'test_data_used', 'exit_evaluation', 'broker_access_allowed'):
        require(gate.get(key) is False, 'READINESS_CLOSED_GATE:' + key)
    admission_ref = campaign.require_binding(gate.get('admission'), label='F runtime admission')
    require_admission(campaign.read_bound_json(Path(admission_ref['path']), admission_ref['sha256']), window['recipe'], window['unit'])
    campaign.require_binding(policy.get('gpu_clock_launcher'), label='existing GPU clock launcher')
    return {'decision': 'READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES', 'blocked_reasons': [],
        'canonical_source_repo': REPO, 'source_commit': head, 'policy_sha256': policy_ref['sha256'],
        'training_started': False, 'native_invocation_seconds': 12000, 'outer_guard_seconds': 13800,
        'native_window_scope': {'frozen_extraction_only': True, 'invocation_number': 1,
            'optimizer_step_ceiling': 0, 'forward_ceiling': unit_spec(window['unit'])['forwards'], 'reverse_ceiling': 0,
            'checkpoint_load_ceiling': 1, 'full_epoch_training_allowed': False,
            'window_policy_file_sha256': window_file_sha256}}


def require_native_direction_frozen_extraction_context(policy_path, policy_file_sha256, progress_path):
    import os
    with campaign.file_verification_scope():
        window = require_frozen_extraction_window_policy(campaign.read_bound_json(Path(policy_path), policy_file_sha256))
        names = ('GX1_CAMPAIGN_PLAN_PATH', 'GX1_CAMPAIGN_PLAN_FILE_SHA256', 'GX1_CAMPAIGN_PLAN_SHA256', 'GX1_CAMPAIGN_INVOCATION_SHA256')
        require(all(os.environ.get(name) for name in names), 'CAMPAIGN_ENV')
        plan_ref = {'path': os.environ[names[0]], 'sha256': os.environ[names[1]]}
        plan = campaign.require_plan(campaign.read_bound_json(Path(plan_ref['path']), plan_ref['sha256']))
        require(plan['phase'] == PHASE and len(plan['checked_invocations']) == 1, 'CAMPAIGN_PHASE')
        invocation = plan['checked_invocations'][0]
        recipe = require_frozen_extraction_recipe(window['recipe'], source_repo=Path(REPO), source_commit=plan['source_commit'])
        marker_path = Path(plan['runtime_root']) / 'ACTIVE_INVOCATION.json'
        marker_ref = {'path': str(marker_path), 'sha256': campaign.file_sha256(marker_path)}
        marker = campaign.read_bound_json(marker_path, marker_ref['sha256'])
        require(plan['plan_sha256'] == os.environ[names[2]] and invocation['invocation_sha256'] == os.environ[names[3]]
            and invocation['frozen_extraction_window'] == window and plan['recipe'] == window['recipe']
            and str(progress_path) == window['progress_path'], 'WINDOW_CONTEXT')
        require(marker.get('schema_version') == campaign.ACTIVE_SCHEMA
            and type(marker.get('invocation_number')) is int
            and marker.get('marker_sha256') == campaign.canonical_sha256({k: v for k, v in marker.items() if k != 'marker_sha256'})
            and all(marker.get(key) == value for key, value in {'plan_sha256': plan['plan_sha256'],
                'invocation_sha256': invocation['invocation_sha256'], 'invocation_number': 1,
                'invocation_id': 'invocation-0001', 'kind': KIND, 'launcher_argv_sha256': invocation['launcher_argv_sha256'],
                'pointer_before_sha256': 'GENESIS'}.items()), 'ACTIVE_MARKER')
        require(campaign.fresh_boot(marker.get('boot'), plan['prepared_windows_boot']), 'ACTIVE_FRESH_BOOT')
        campaign._utc(marker.get('started_utc'), 'F active start')
        require(os.environ.get('GX1_CAMPAIGN_GUARD_LOG_PATH') == invocation['guard_log_path'], 'ACTIVE_GUARD_LOG')
        binding_ref = {'path': recipe['execution_binding_path'], 'sha256': campaign.file_sha256(recipe['execution_binding_path'])}
        binding = campaign.read_bound_json(Path(binding_ref['path']), binding_ref['sha256'])
        policy_ref = {'path': REPO + '/NEXT_RUN_POLICY.json', 'sha256': campaign.file_sha256(REPO + '/NEXT_RUN_POLICY.json')}
        require(set(binding) == {'plan', 'source_head', 'policy'} and binding['plan'] == plan_ref
            and binding['source_head'] == plan['source_commit'] and binding['policy'] == policy_ref, 'FROZEN_EXECUTION_BINDING')
        campaign.require_clean_source(plan)
        require_native_direction_frozen_extraction_readiness(Path(REPO), window_path=Path(policy_path), window_file_sha256=policy_file_sha256)
        policy = campaign.read_bound_json(Path(policy_ref['path']), policy_ref['sha256'])
        admission_ref = campaign.require_binding(policy[recipe['admission_stage']]['admission'], label='F execution admission')
        output = Path(recipe['run_root'])
        require(not output.exists() and not output.is_symlink(), 'GENESIS_NOT_FRESH')
        return {'window': window, 'plan': plan, 'invocation': invocation, 'marker': marker,
            'recipe': recipe, 'recipe_ref': window['recipe'], 'admission_ref': admission_ref, 'binding': binding,
            'metadata_refs': [plan_ref, {'path': str(Path(policy_path)), 'sha256': policy_file_sha256},
                marker_ref, binding_ref, policy_ref, admission_ref]}


def _operation_counts(calls, returns, outcome, unit):
    limits = call_limits(unit)
    require(type(calls) is type(returns) is dict and set(calls) == set(returns) == set(limits), 'POINTER_COUNTER_KEYS')
    require(all(type(calls[k]) is type(returns[k]) is int and 0 <= returns[k] <= calls[k] <= limit
        for k, limit in limits.items()), 'POINTER_COUNTER_BOUNDS')
    require(outcome in ('COMPLETE', 'FAILED') and (outcome != 'COMPLETE' or calls == returns == limits), 'POINTER_COMPLETE')
    return {'completed_units': returns['panel_forward'], 'forward_calls': calls['panel_forward'],
        'forward_returns': returns['panel_forward'], 'reverse_calls': calls['reverse'],
        'reverse_returns': returns['reverse'], 'checkpoint_loads': calls['checkpoint_load'],
        'checkpoint_load_returns': returns['checkpoint_load'], 'optimizer_steps': 0}


def build_frozen_extraction_pointer(*, recipe_ref, result_ref, outcome, calls, returns, unit):
    return _seal({'schema_version': POINTER_SCHEMA, 'recipe': plain_ref(recipe_ref), 'result': plain_ref(result_ref),
        'unit': unit, 'outcome': outcome, 'operation_calls': dict(calls), 'operation_returns': dict(returns),
        **_operation_counts(calls, returns, outcome, unit), 'test_data_used': False}, 'pointer_sha256')


def require_frozen_extraction_pointer(value, *, expected_recipe, expected_unit, verify_files=True):
    fields = {'schema_version', 'recipe', 'result', 'unit', 'outcome', 'operation_calls', 'operation_returns',
        'completed_units', 'forward_calls', 'forward_returns', 'reverse_calls', 'reverse_returns',
        'checkpoint_loads', 'checkpoint_load_returns', 'optimizer_steps', 'test_data_used'}
    pointer = _identity(value, fields, POINTER_SCHEMA, 'pointer_sha256')
    counters = _operation_counts(pointer['operation_calls'], pointer['operation_returns'], pointer['outcome'], pointer['unit'])
    require(pointer['recipe'] == expected_recipe and pointer['unit'] == expected_unit and pointer['test_data_used'] is False
        and all(type(pointer[k]) is int and pointer[k] == v for k, v in counters.items()), 'POINTER_ROLE')
    ref = campaign.require_binding(pointer['result'], label='F terminal result', verify_file=verify_files)
    if verify_files:
        result = campaign.read_bound_json(Path(ref['path']), ref['sha256'])
        require(result['recipe'] == expected_recipe and result['unit'] == expected_unit
            and type(result['optimizer_steps']) is int and result['optimizer_steps'] == 0
            and result['test_data_used'] is False and result['observed_calls'] == pointer['operation_calls']
            and result['observed_returns'] == pointer['operation_returns']
            and (result['decision'] == 'COMPLETE_FROZEN_READOUT_EXTRACTION') == (pointer['outcome'] == 'COMPLETE'), 'POINTER_RESULT')
        if pointer['outcome'] == 'COMPLETE':
            require(result['representation_rows'] == unit_spec(expected_unit)['rows']
                and result['hidden_width'] == 128 and result['originals_stat_preserved'] is True
                and result['source_unchanged'] is True and result['native_optimizer_steps'] == 0, 'RESULT_COMPLETE_DOMAIN')
            spec = unit_spec(expected_unit)
            diagnostic = spec['diagnostic_only']
            require(result.get('diagnostic_only', False) is diagnostic, 'RESULT_DIAGNOSTIC_ROLE')
            if expected_unit in PARTITION_UNITS:
                part = result.get('partition')
                require(type(part) is dict and set(part) == set(spec['partition']) | {'full_presentation_parent_rows_sha256'}
                    and all(type(part[k]) is type(v) and part[k] == v for k, v in spec['partition'].items())
                    and type(part['full_presentation_parent_rows_sha256']) is str
                    and len(part['full_presentation_parent_rows_sha256']) == 64
                    and all(c in '0123456789abcdef' for c in part['full_presentation_parent_rows_sha256']),
                    'RESULT_PARTITION_IDENTITY')
            else:
                require('partition' not in result, 'RESULT_UNEXPECTED_PARTITION')
            if diagnostic:
                timing = result.get('phase_timings', {})
                require(timing.get('pending_frames') == 0 and timing.get('scope') == 'inclusive_wall_and_process_cpu'
                    and all(timing.get('phases', {}).get(name, {}).get('calls') == count
                        and timing['phases'][name].get('returns') == count
                        for name, count in {'prepare': 1, 'evaluate': 1, 'batch': 16, 'context_tags': 16,
                            'device_batch': 16, 'forward': 16, 'hidden_capture': 16}.items()),
                    'RESULT_TIMING_COMPLETE')
            campaign.require_binding(result['representations'], label='F hidden output')
    return pointer


def _progress(*, window, plan, invocation, calls, returns, outcome, checkpoint_pointer):
    from datetime import datetime, timezone
    fields = _operation_counts(calls, returns, outcome if outcome != 'RUNNING' else 'FAILED', window['unit'])
    return _seal({'schema_version': campaign.PROGRESS_SCHEMA, 'plan_sha256': plan['plan_sha256'],
        'invocation_sha256': invocation['invocation_sha256'], 'phase': KIND, 'recipe': window['recipe'],
        'unit': window['unit'], **fields, 'total_units': unit_spec(window['unit'])['forwards'],
        'checkpoint_pointer': checkpoint_pointer, 'terminal': outcome != 'RUNNING', 'outcome': outcome,
        'observed_utc': datetime.now(timezone.utc).isoformat()}, 'progress_sha256')


def build_frozen_extraction_progress(*, window, plan, invocation, checkpoint_pointer, pointer):
    return _progress(window=window, plan=plan, invocation=invocation, calls=pointer['operation_calls'],
        returns=pointer['operation_returns'], outcome=pointer['outcome'], checkpoint_pointer=plain_ref(checkpoint_pointer))


def build_frozen_extraction_running_progress(*, window, plan, invocation, calls, returns):
    return _progress(window=window, plan=plan, invocation=invocation, calls=calls,
        returns=returns, outcome='RUNNING', checkpoint_pointer=None)


def require_frozen_extraction_progress(value, *, plan_sha256, invocation, expected_selection_receipt_sha256, verify_file=True):
    fields = {'schema_version', 'plan_sha256', 'invocation_sha256', 'phase', 'recipe', 'unit', 'completed_units',
        'forward_calls', 'forward_returns', 'reverse_calls', 'reverse_returns', 'checkpoint_loads',
        'checkpoint_load_returns', 'optimizer_steps', 'total_units', 'checkpoint_pointer', 'terminal', 'outcome', 'observed_utc'}
    progress = _identity(value, fields, campaign.PROGRESS_SCHEMA, 'progress_sha256')
    unit = invocation['frozen_extraction_window']['unit']
    require(expected_selection_receipt_sha256 is None and invocation['kind'] == KIND
        and progress['plan_sha256'] == plan_sha256 and progress['invocation_sha256'] == invocation['invocation_sha256']
        and progress['phase'] == KIND and progress['recipe'] == invocation['frozen_extraction_recipe']
        and progress['unit'] == unit and progress['terminal'] is True and progress['outcome'] in ('COMPLETE', 'FAILED'),
        'PROGRESS_IDENTITY')
    campaign._utc(progress['observed_utc'], 'F progress')
    n = unit_spec(unit)['forwards']
    require(type(progress['total_units']) is int and progress['total_units'] == n
        and type(progress['optimizer_steps']) is int and progress['optimizer_steps'] == 0, 'PROGRESS_ZERO_STEPS')
    for calls, returns, limit in (('forward_calls', 'forward_returns', n), ('reverse_calls', 'reverse_returns', 0),
            ('checkpoint_loads', 'checkpoint_load_returns', 1)):
        require(type(progress[calls]) is type(progress[returns]) is int and 0 <= progress[returns] <= progress[calls] <= limit,
            'PROGRESS_COUNTER:' + calls)
        require(progress['outcome'] != 'COMPLETE' or progress[calls] == progress[returns] == limit, 'PROGRESS_COMPLETE:' + calls)
    require(type(progress['completed_units']) is int and progress['completed_units'] == progress['forward_returns'], 'PROGRESS_UNITS')
    ref = campaign.require_binding(progress['checkpoint_pointer'], label='F extraction pointer', verify_file=verify_file)
    require(ref['path'] == invocation['checkpoint']['pointer_path'], 'PROGRESS_POINTER_PATH')
    if verify_file:
        pointer = require_frozen_extraction_pointer(campaign.read_bound_json(Path(ref['path']), ref['sha256']),
            expected_recipe=progress['recipe'], expected_unit=unit, verify_files=True)
        require(all(progress[k] == pointer[k] for k in ('unit', 'outcome', 'completed_units', 'forward_calls', 'forward_returns',
            'reverse_calls', 'reverse_returns', 'checkpoint_loads', 'checkpoint_load_returns', 'optimizer_steps')),
            'PROGRESS_DURABLE_POINTER')
    return progress
