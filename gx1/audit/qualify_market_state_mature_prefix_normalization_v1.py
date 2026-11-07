"""One-use, input-only qualification and conditional normalization of mature TRAIN prefixes.

Target fold 1 (October 2024) is report-only. Target folds 2..5 may fit
statistics after all five raw qualifications pass their source/axis gates.
No target values, model, TEST, or optimizer are read or constructed here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import signal
import time

import numpy as np

from gx1.contracts import market_state_five_endpoint_target_binding_v1 as target
from gx1.contracts.entry_native_direction_prefix_normalization_v1 import (
    fit_native_direction_prefix_normalization,
    qualify_native_direction_prefix_population,
)
from gx1.contracts.immutable_event_authority_v1 import _fsync_directory, _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNS = Path('/home/andre2/GX1_RUNS')
STAGE = 'market_state_mature_prefix_normalization_20261107_001'
SCHEMA = 'gx1_market_state_mature_prefix_normalization_plan_v1'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = RUNS / STAGE.upper()
POLICY = REPO / 'NEXT_RUN_POLICY.json'
FOLD_IDS = [1, 2, 3, 4, 5]
FIT_FOLD_IDS = [2, 3, 4, 5]
MATURE_ROWS = [29574, 47338, 64615, 82152, 100224]
OLD_ROWS = [29849, 47602, 64890, 82427, 100499]
SOURCE_ROWS = 118249
DAY_NS = 86_400_000_000_000
LIMITS = {'memory_max': 4 * 1024**3, 'memory_high': 4 * 1024**3,
          'swap': 512 * 1024**2, 'pids': 64, 'numerical_threads': 1,
          'internal_wall_seconds': 2100}
EXECUTION = {'class': 'audit', 'runner': 'scripts/gx1_capped_run.sh',
             'memory': '4G', 'swap': '512M', 'timeout_seconds': 2400,
             'kill_after_seconds': 10,
             'module': 'gx1.audit.qualify_market_state_mature_prefix_normalization_v1',
             'model_forwards': 0, 'optimizer_steps': 0}
SOURCE_PATHS = frozenset({
    'gx1/audit/qualify_market_state_mature_prefix_normalization_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/market_state_five_endpoint_contract_v1.py',
    'gx1/contracts/entry_native_direction_prefix_normalization_v1.py',
    'gx1/contracts/entry_native_direction_prefix_units_v1.py',
    'gx1/contracts/entry_model_native_signal_v1.py',
    'gx1/contracts/entry_model_native_training_objective_v1.py',
    'gx1/contracts/entry_model_native_input_normalization_v1.py',
    'gx1/contracts/entry_gc_mtf_history_v1.py',
    'gx1/contracts/entry_exit_feature_surface_v1.py',
    'gx1/contracts/entry_exit_production_architecture_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/models/entry_v10/entry_v10_input_normalization.py',
    'gx1/features/htf_features.py',
    'gx1/features/volatility_squeeze_state_v1.py',
    'gx1/features/entry_specialist_feature_groups_v1.py',
    'gx1/features/entry_model_native_feature_layers_v1.py',
    'gx1/features/gc_native_context_v1.py',
    'gx1/features/micro_structure_v1.py',
    'gx1/features/basic_v1.py',
    'gx1/contracts/registry_hyperparameter_fit_v1.py',
    'gx1/utils/artifact_primitives_v1.py',
    'scripts/gx1_capped_run.sh',
})
_PROVIDER_ROOT = RUNS / 'NATIVE_M5_DIRECTION_NATIVE_PREFIX_INPUT_BINDING_REPAIR_20261024_264'
_BOOLEAN_ROOT = RUNS / 'NATIVE_M5_DIRECTION_GC_BOOLEAN_PHYSICAL_PREFIX_NORMALIZATION_20261024_267'
INPUT_REFS = {
    'provider_plan': {'path': str(_PROVIDER_ROOT / 'PLAN_SUPPORT.json'),
                      'sha256': '96ff82533042ab069292505b3a44315c13223a9b9d9e9ee669cd757e3f84c086'},
    'provider_worker': {'path': str(_PROVIDER_ROOT / 'qualify_physical_prefix_support_guarded.py'),
                        'sha256': 'a007b01d5fb41785ffae3de11be71b50b7b3959875a69966c830e5a0c7b2d5e9'},
    'boolean_result': {'path': str(_BOOLEAN_ROOT / 'RESULT.json'),
                       'sha256': 'e4359717d818e6c7c55382d36a941765cc7c177b19b3eb89a50e11ec11684fe9'},
    'boolean_terminal': {'path': str(_BOOLEAN_ROOT / 'TERMINAL.json'),
                         'sha256': 'ae53d3274fb84487c247969177dc13506352db509d9ea17b414f290a5df7020a'},
    'gc_layout': {'path': str(_BOOLEAN_ROOT / 'CURRENT_GC_LAYOUT.json'),
                  'sha256': 'c0cdc60447ceb303ebdcdc790716789585fb32c0c30918f5d58d030ad6baffde'},
    'raw_domain_transition': {'path': str(_BOOLEAN_ROOT / 'RAW_DOMAIN_TRANSITION.json'),
                              'sha256': '98bc67279431199725ff64630dcce0687e6b0908bef0595724b8e861a7306a00'},
    'model_axes': {'path': target.REFS['model_axes'][0], 'sha256': target.REFS['model_axes'][1]},
}
MAX_OUTPUT_BYTES = 48 * 1024**2
PUBLISHED_BYTES = 0


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('MATURE_PREFIX_NORMALIZATION_' + reason)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _axis_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype='<i8').tobytes()).hexdigest()


def _checked_ref(item: dict, *, limit: int = 16 * 1024**2) -> dict:
    path = Path(item['path'])
    _need(path.is_absolute() and not path.is_symlink() and path.is_file()
          and 0 < path.stat().st_size <= limit and _sha(path) == item['sha256'],
          'REF:' + str(path))
    return json.loads(path.read_text())


def _write(root: Path, name: str, value: dict) -> dict:
    global PUBLISHED_BYTES
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _need(name in ('FAILED.json', 'TERMINAL.json') or
          PUBLISHED_BYTES + len(data) <= MAX_OUTPUT_BYTES, 'OUTPUT_BUDGET')
    _publish_bytes(root / name, data)
    PUBLISHED_BYTES += len(data)
    return {'path': str(root / name), 'sha256': hashlib.sha256(data).hexdigest()}


def _clean_head() -> str:
    _need(Path.cwd().resolve() == REPO
          and subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'work/gx1-current'
          and subprocess.check_output(['git', 'status', '--porcelain'], text=True) == '',
          'CLEAN_SOURCE')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def _source_bookend(plan: dict, plan_sha256: str, policy_sha256: str, head: str) -> None:
    _need(_clean_head() == head and _sha(PLAN_PATH) == plan_sha256
          and _sha(POLICY) == policy_sha256
          and all(_sha(REPO / path) == digest for path, digest in plan['source_owners'].items()),
          'SOURCE_BOOKEND')
    for item in INPUT_REFS.values():
        _need(_sha(Path(item['path'])) == item['sha256'], 'INPUT_BOOKEND')


def _receipt_chain() -> dict:
    result = _checked_ref(INPUT_REFS['boolean_result'])
    terminal = _checked_ref(INPUT_REFS['boolean_terminal'])
    layout = _checked_ref(INPUT_REFS['gc_layout'])
    transition = _checked_ref(INPUT_REFS['raw_domain_transition'])
    provider_plan = _checked_ref(INPUT_REFS['provider_plan'])
    _need(result.get('status') == 'COMPLETE_FIVE_RAW_QUALIFICATIONS_FOUR_BOOLEAN_PREFIX_NORMALIZATIONS_NO_MODEL_ADMISSION'
          and terminal.get('status') == 'COMPLETE' and terminal.get('exit_code') == 0
          and terminal.get('result') == INPUT_REFS['boolean_result']
          and result.get('input_provider_plan') == INPUT_REFS['provider_plan']
          and result.get('input_provider_worker') == INPUT_REFS['provider_worker']
          and result.get('current_gc_layout') == INPUT_REFS['gc_layout']
          and result.get('raw_domain_transition') == INPUT_REFS['raw_domain_transition']
          and result.get('test_data_used') is False
          and result.get('model_constructions') == result.get('optimizer_updates') == 0
          and transition.get('normalization_applied') is False
          and transition.get('legacy_statistics_admitted_as_current') is False
          and provider_plan.get('axes', {}).get('source') == INPUT_REFS['model_axes']
          and provider_plan.get('axes', {}).get('rows') == SOURCE_ROWS,
          'HISTORICAL_RECEIPTS')
    _need(_sha(Path(INPUT_REFS['provider_worker']['path'])) == INPUT_REFS['provider_worker']['sha256'],
          'PROVIDER_WORKER')
    return {'provider_plan': provider_plan, 'gc_layout': layout}


def _validate_cohort_contract(cohorts: list[dict]) -> None:
    keys = {'fold_id', 'official_fit_cutoff_ns', 'effective_fit_cutoff_ns',
            'joined_fit_rows', 'old_joined_fit_rows', 'fit_parent_rows_sha256',
            'fit_decision_ns_sha256'}
    _need(isinstance(cohorts, list) and len(cohorts) == 5, 'COHORT_COUNT')
    for index, item in enumerate(cohorts):
        _need(isinstance(item, dict) and set(item) == keys
              and item['fold_id'] == FOLD_IDS[index]
              and item['official_fit_cutoff_ns'] == target.FOLD_BOUNDARIES_NS[index]
              and item['effective_fit_cutoff_ns'] == target.FOLD_BOUNDARIES_NS[index] - DAY_NS
              and item['joined_fit_rows'] == MATURE_ROWS[index]
              and item['old_joined_fit_rows'] == OLD_ROWS[index]
              and all(isinstance(item[key], str) and len(item[key]) == 64
                      for key in ('fit_parent_rows_sha256', 'fit_decision_ns_sha256')),
              'COHORT_DECLARATION:' + str(index + 1))


def _validate_axes(cohorts: list[dict], axes: dict[str, np.ndarray]) -> None:
    _need(len(axes['parent_rows']) == len(axes['bar_start_ns']) == len(axes['decision_ns']) == SOURCE_ROWS,
          'AXIS_ROWS')
    for item in cohorts:
        stop = int(np.searchsorted(axes['decision_ns'], item['effective_fit_cutoff_ns'], side='left'))
        old = int(np.searchsorted(axes['decision_ns'], item['official_fit_cutoff_ns'], side='left'))
        _need(stop == item['joined_fit_rows'] and old == item['old_joined_fit_rows']
              and _axis_sha(axes['parent_rows'][:stop]) == item['fit_parent_rows_sha256']
              and _axis_sha(axes['decision_ns'][:stop]) == item['fit_decision_ns_sha256'],
              'COHORT_AXIS:' + str(item['fold_id']))


def _policy_gate(policy: dict, plan_ref: dict, out_dir: Path) -> None:
    gate = policy.get(STAGE, {})
    _need(policy.get('training_enabled') is False
          and policy.get('native_training_enabled') is False
          and policy.get('native_launch_authorized') is False
          and policy.get('full_val_allowed') is False
          and gate.get('status') == gate.get('decision') == 'ADMITTED'
          and gate.get('plan') == plan_ref
          and gate.get('execution_authorized') is True
          and gate.get('data_reads_authorized') is True
          and gate.get('normalization_fit_authorized') is True
          and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
          and gate.get('max_invocations') == 1
          and gate.get('relaunch_allowed') is False
          and gate.get('memory_max') == EXECUTION['memory']
          and gate.get('swap_max') == EXECUTION['swap']
          and gate.get('wall_seconds') == EXECUTION['timeout_seconds']
          and gate.get('test_data_used') is False
          and gate.get('checkpoint_reads_authorized') is False
          and gate.get('broker_access_allowed') is False
          and gate.get('model_forwards_authorized') == 0
          and gate.get('optimizer_steps_authorized') == 0
          and gate.get('training_authorized') is False
          and gate.get('test_authorized') is False
          and gate.get('output_root') == str(out_dir), 'POLICY_GATE')


def preflight(plan_path: Path, plan_sha256: str, out_dir: Path, *, admitted: bool = False) -> tuple[dict, dict, str, dict | None]:
    head = _clean_head()
    _need(plan_path == PLAN_PATH and out_dir == OUT_ROOT and not out_dir.exists()
          and len(plan_sha256) == 64 and _sha(plan_path) == plan_sha256, 'PLAN_IDENTITY')
    plan = json.loads(plan_path.read_text())
    _validate_cohort_contract(plan.get('cohorts'))
    _need(plan.get('schema_version') == SCHEMA and plan.get('stage') == STAGE
          and plan.get('status') == 'PREPARED_NOT_ADMITTED'
          and plan.get('output_root') == str(OUT_ROOT)
          and plan.get('max_invocations') == 1
          and plan.get('fold_ids') == FOLD_IDS and plan.get('fit_fold_ids') == FIT_FOLD_IDS
          and plan.get('input_refs') == INPUT_REFS
          and plan.get('execution') == EXECUTION and plan.get('limits') == LIMITS
          and plan.get('normalization_fit_authorized') is True
          and plan.get('training_authorized') is False
          and plan.get('test_data_used') is False
          and plan.get('model_forwards_authorized') == 0
          and plan.get('optimizer_steps_authorized') == 0
          and isinstance(plan.get('source_base_head'), str)
          and len(plan['source_base_head']) == 40
          and subprocess.run(['git', 'merge-base', '--is-ancestor', plan['source_base_head'], head],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0,
          'PLAN_SCOPE')
    owners = plan.get('source_owners')
    _need(isinstance(owners, dict) and set(owners) == SOURCE_PATHS
          and all(_sha(REPO / path) == digest for path, digest in owners.items()),
          'SOURCE_OWNERS')
    _receipt_chain()
    plan_ref = {'path': str(plan_path), 'sha256': plan_sha256}
    policy_ref = None
    if admitted:
        from gx1.contracts.gx1_capped_execution_v1 import (
            require_capped_cpu_audit_execution, require_capped_lock_ancestry)
        cap = require_capped_cpu_audit_execution()
        require_capped_lock_ancestry()
        _need(all(cap.get(key) == LIMITS[key] for key in ('memory_max', 'memory_high', 'swap',
                                                        'pids', 'numerical_threads')),
              'CAP_LIMITS')
        policy_ref = {'path': str(POLICY), 'sha256': _sha(POLICY)}
        policy = json.loads(POLICY.read_text())
        _need(_sha(POLICY) == policy_ref['sha256'], 'POLICY_STABLE_READ')
        _policy_gate(policy, plan_ref, out_dir)
    return plan, plan_ref, head, policy_ref


def _provider_module():
    item = INPUT_REFS['provider_worker']
    _need(_sha(Path(item['path'])) == item['sha256'], 'PROVIDER_IMPORT_SHA')
    spec = importlib.util.spec_from_file_location('gx1_mature_prefix_input_provider264', item['path'])
    _need(spec is not None and spec.loader is not None, 'PROVIDER_IMPORT_SPEC')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _need(_sha(Path(item['path'])) == item['sha256'], 'PROVIDER_IMPORT_BOOKEND')
    return module


def _prepare_inputs(provider, provider_plan: dict, info: dict, saved_layout: dict) -> dict:
    from gx1.contracts.entry_exit_feature_surface_v1 import load_m1_feature_surface_times
    from gx1.contracts.entry_gc_mtf_history_v1 import load_gc_mtf_history_raw_artifact
    from gx1.features import htf_features as htf

    axes, expected_axes = provider.read_axes(provider_plan, info, np)
    source_times = load_m1_feature_surface_times(Path(info['source']['output_parquet']),
        context='MARKET_STATE_MATURE_PREFIX_CLOCK_ONLY', expected_bar_seconds=300)
    source_ns = source_times.asi8
    _need(source_ns.shape == (provider.SOURCE_ROWS,), 'SOURCE_CLOCK_ROWS')
    groups = provider.map_reuse(info['reuse'], np)
    source, current = groups['source050'], groups['current051']
    positions = np.searchsorted(source_ns, axes['bar_start_ns']).astype(np.int64, copy=False)
    _need(np.all(positions >= 95) and np.all(positions < provider.SOURCE_ROWS)
          and np.all(positions[1:] > positions[:-1])
          and np.array_equal(source_ns[positions], axes['bar_start_ns']), 'SOURCE_POINTERS')
    cache = htf.load_multi_tf_v4_cache(provider_plan['native_cache_dir'])
    names_sha = hashlib.sha256(json.dumps(list(htf.MULTI_TF_PER_BAR_FEATURES_V4),
        sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    base = provider_plan['base_lineage']
    _need(base['mtf_builder_version'] == htf.HTF_V4_CACHE_BUILDER_VERSION
          and base['mtf_feature_names_sha256'] == names_sha
          and cache.manifest_sha256 == base['mtf_cache_manifest_sha256']
          and cache.m5_prebuilt_source == base['m5_prebuilt_path']
          and cache.m5_prebuilt_source_sha256 == base['m5_prebuilt_sha256']
          and cache.v29_registry_constants['contract_sha256'] == info['v29']['expected_registry_contract_sha256'],
          'MTF_CACHE_IDENTITY')
    gc = load_gc_mtf_history_raw_artifact(provider_plan['gc_surfaces'])
    _need(gc['layout'] == saved_layout, 'CURRENT_BOOLEAN_GC_LAYOUT')
    return {'axes': axes, 'expected_axes': expected_axes, 'source': source,
            'current': current, 'cache': cache, 'gc': gc,
            'entry_sequence_source': {'signal': source['signal'], 'times_ns': source_ns,
                'sample_positions': positions, 'audit': info['audit']}}


def _release_pages(provider, prepared: dict) -> None:
    from gx1.features.htf_features import discard_multi_tf_v4_cache_pages
    for value in (*prepared['source'].values(), *prepared['current'].values()):
        provider.flush_pages(value)
    discard_multi_tf_v4_cache_pages(prepared['cache'])
    for surface in prepared['gc']['surfaces'].values():
        for key in ('native_row_indices', 'available_ns', 'features'):
            provider.flush_pages(surface[key])


def _time_guard(deadline: float) -> None:
    # A new costly unit is never begun inside the final five-minute reserve.
    _need(time.monotonic() + 300 < deadline, 'DEADLINE_RESERVE')


def _source_bindings(plan: dict, plan_ref: dict, policy_ref: dict, head: str) -> dict:
    return {'provider_plan': INPUT_REFS['provider_plan'],
            'provider_worker': INPUT_REFS['provider_worker'],
            'boolean_raw_result': INPUT_REFS['boolean_result'],
            'boolean_raw_gc_layout': INPUT_REFS['gc_layout'],
            'boolean_raw_domain_transition': INPUT_REFS['raw_domain_transition'],
            'model_axes': INPUT_REFS['model_axes'],
            'current_plan': plan_ref, 'current_policy': policy_ref,
            'current_source_head': head, 'current_source_owners': plan['source_owners'],
            'cohort_rule': 'decision_ns_strictly_below_official_cutoff_minus_1440_minutes'}


def _process(plan: dict, provider_plan: dict, info: dict, provider, prepared: dict,
             plan_ref: dict, policy_ref: dict, head: str, out_dir: Path,
             deadline: float) -> tuple[list[dict], list[dict]]:
    axes = prepared['axes']
    _validate_axes(plan['cohorts'], axes)
    bindings = _source_bindings(plan, plan_ref, policy_ref, head)
    qualifications, retained = [], []
    for cohort in plan['cohorts']:
        _time_guard(deadline)
        fold_id = cohort['fold_id']
        count = cohort['joined_fit_rows']
        qualified = qualify_native_direction_prefix_population(
            parent_rows=axes['parent_rows'], bar_start_ns=axes['bar_start_ns'],
            decision_ns=axes['decision_ns'], expected_axes=prepared['expected_axes'],
            cutoff_ns=cohort['effective_fit_cutoff_ns'],
            expected_prefix_parent_rows=axes['parent_rows'][:count],
            entry_sequence_source=prepared['entry_sequence_source'],
            snap_x=prepared['current']['snap'], ctx_cont=prepared['current']['ctx_cont'],
            ctx_cat=prepared['current']['ctx_cat'],
            ordered_signal_names=info['source']['feature_field_order'],
            multi_tf_sources=prepared['cache'], gc_surfaces=prepared['gc']['surfaces'],
            gc_layout=prepared['gc']['layout'], base_lineage=provider_plan['base_lineage'],
            source_bindings=bindings)
        report = qualified['report']
        prefix = report['proof']['prefix']
        _need(report['statistics_fits'] == 0
              and prefix['rows'] == count
              and prefix['cutoff_ns'] == cohort['effective_fit_cutoff_ns']
              and prefix['parent_rows_sha256'] == cohort['fit_parent_rows_sha256']
              and prefix['bar_start_ns_sha256'] == _axis_sha(axes['bar_start_ns'][:count])
              and prefix['decision_ns_sha256'] == cohort['fit_decision_ns_sha256']
              and ('unobserved_units_profile' not in report)
              and (qualified['population'] is not None) is report['fit_allowed'],
              'RAW_QUALIFICATION:' + str(fold_id))
        report_ref = _write(out_dir, f'QUALIFICATION_FOLD_{fold_id}.json', report)
        qualifications.append({'fold_id': fold_id, 'official_fit_cutoff_ns': cohort['official_fit_cutoff_ns'],
            'effective_fit_cutoff_ns': cohort['effective_fit_cutoff_ns'],
            'joined_fit_rows': count, 'fit_allowed': report['fit_allowed'],
            'blockers': report['blockers'], 'report': report_ref,
            'action': 'REPORT_ONLY' if fold_id == 1 else
                      ('FIT_ELIGIBLE' if report['fit_allowed'] else 'FIT_BLOCKED')})
        if fold_id in FIT_FOLD_IDS and report['fit_allowed']:
            retained.append((fold_id, report, qualified['population']['inputs']))
        del qualified, report
        _release_pages(provider, prepared)
    _need([item['fold_id'] for item in qualifications] == FOLD_IDS, 'ALL_FIVE_BEFORE_FIT')
    fits = []
    for fold_id, report, inputs in retained:
        _time_guard(deadline)
        fitted = fit_native_direction_prefix_normalization(
            {'report': report, 'population': {'inputs': inputs}})
        expected = report['support']
        actual = dict(fitted['base_normalization']['surfaces'])
        actual.update(fitted['gc_mtf_history']['normalization']['surfaces'])
        _need(fitted['prefix_qualification'] == report
              and fitted['old055_or223_normalization_identity_reused'] is False
              and fitted['native_launch_authorized'] is False
              and fitted['genuine_model_fit_authorized'] is False
              and set(actual) == set(expected) and len(actual) == 12
              and all(actual[key]['field_names'] == expected[key]['field_names'] for key in expected),
              'FIT_SCOPE:' + str(fold_id))
        output = _write(out_dir, f'PREFIX_NORMALIZATION_FOLD_{fold_id}.json', fitted)
        fits.append({'fold_id': fold_id, 'output': output,
                     'qualification': next(item['report'] for item in qualifications
                                           if item['fold_id'] == fold_id),
                     'contract_sha256': fitted['contract_sha256']})
        del fitted
        _release_pages(provider, prepared)
    _need(all(item['fold_id'] in FIT_FOLD_IDS for item in fits), 'FIT_FOLD_SCOPE')
    return qualifications, fits


def _wall_limit(_signum, _frame) -> None:
    raise TimeoutError('MATURE_PREFIX_NORMALIZATION_INTERNAL_WALL_LIMIT')


def run(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    global PUBLISHED_BYTES
    started = time.monotonic()
    plan, plan_ref, head, policy_ref = preflight(plan_path, plan_sha256, out_dir, admitted=True)
    _need(policy_ref is not None, 'POLICY_REQUIRED')
    deadline = started + LIMITS['internal_wall_seconds']
    remaining = deadline - time.monotonic()
    _need(remaining > 0, 'WALL_BUDGET_PRECLAIM')
    _need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'AMBIENT_ALARM')
    out_dir.mkdir(parents=True, exist_ok=False)
    PUBLISHED_BYTES = 0
    claim = None
    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _wall_limit)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        _fsync_directory(out_dir.parent)
        claim = _write(out_dir, 'CLAIM.json', {
            'schema_version': 'gx1_market_state_mature_prefix_normalization_claim_v1',
            'plan': plan_ref, 'policy': policy_ref, 'source_head': head,
            'invocation': 1, 'fold_ids': FOLD_IDS, 'fit_fold_ids': FIT_FOLD_IDS,
            'normalization_only': True, 'model_forwards': 0,
            'optimizer_steps': 0, 'test_data_used': False})
        chain = _receipt_chain()
        provider_plan = chain['provider_plan']
        provider = _provider_module()
        info = provider.validate_metadata(provider_plan)
        prepared = _prepare_inputs(provider, provider_plan, info, chain['gc_layout'])
        qualifications, fits = _process(plan, provider_plan, info, provider, prepared,
                                        plan_ref, policy_ref, head, out_dir, deadline)
        _time_guard(deadline)
        provider.validate_metadata(provider_plan)
        _source_bookend(plan, plan_sha256, policy_ref['sha256'], head)
        result = _write(out_dir, 'RESULT.json', {
            'schema_version': 'gx1_market_state_mature_prefix_normalization_result_v1',
            'status': 'COMPLETE_FIVE_MATURE_INPUT_QUALIFICATIONS_CONDITIONAL_FITS_NO_MODEL',
            'plan': plan_ref, 'claim': claim, 'policy': policy_ref,
            'source_head': head, 'elapsed_seconds': time.monotonic() - started,
            'input_refs': INPUT_REFS, 'cohorts': plan['cohorts'],
            'qualifications': qualifications, 'fits': fits,
            'qualifications_before_first_fit': True,
            'target_fold_1_report_only': True,
            'decoded_target_arrays': 0, 'decoded_label_arrays': 0,
            'model_forwards': 0, 'model_constructions': 0,
            'optimizer_steps': 0, 'test_data_used': False,
            'model_training_authorized': False, 'native_launch_authorized': False})
        signal.setitimer(signal.ITIMER_REAL, 0)
        terminal = _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_mature_prefix_normalization_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
            'claim': claim, 'result': result, 'qualification_count': 5,
            'fit_count': len(fits), 'model_forwards': 0,
            'optimizer_steps': 0, 'test_data_used': False,
            'no_relaunch_under_this_plan': True})
        return {'result': result, 'terminal': terminal}
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        failed = _write(out_dir, 'FAILED.json', {
            'schema_version': 'gx1_market_state_mature_prefix_normalization_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'elapsed_seconds': time.monotonic() - started,
            'no_relaunch_under_this_plan': True})
        _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_mature_prefix_normalization_terminal_v1',
            'status': 'FAILED', 'exit_code': 1, 'plan': plan_ref,
            'claim': claim, 'failure': failed, 'model_forwards': 0,
            'optimizer_steps': 0, 'test_data_used': False,
            'no_relaunch_under_this_plan': True})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        plan, ref, head, _ = preflight(args.plan, args.plan_sha256, args.out_dir)
        print(json.dumps({'status': 'READY_NO_CLAIM_NO_PAYLOAD', 'plan': ref,
                          'source_head': head, 'cohorts_checked': len(plan['cohorts']),
                          'claim_created': False, 'test_data_used': False}, sort_keys=True))
        return 0
    print(json.dumps(run(args.plan, args.plan_sha256, args.out_dir), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
