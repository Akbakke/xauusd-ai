"""One-use TRAIN fold-1 normalization using the sealed mature-prefix units.

The original five-fold raw qualification remains consumed. This owner reuses
its exact physical input axis, qualifies the 153 declared unobserved fields,
and fits input statistics once. It reads no labels, model, TEST or broker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.contracts.entry_native_direction_prefix_normalization_v1 import (
    fit_native_direction_prefix_normalization,
    qualify_native_direction_prefix_population,
)
from gx1.contracts.immutable_event_authority_v1 import (
    _fsync_directory, _publish_bytes,
)

REPO = base.REPO
RUNS = base.RUNS
STAGE = 'market_state_mature_fold1_normalization_20261107_001'
SCHEMA = 'gx1_market_state_mature_fold1_normalization_plan_v1'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = RUNS / STAGE.upper()
POLICY = REPO / 'NEXT_RUN_POLICY.json'
FOLD_ID = 1
FIT_ROWS = 29574
EFFECTIVE_CUTOFF_NS = 1727654400000000000
PROFILE_SHA256 = '8fe7982aaf6ef8de947997cc945669cb58e94c6729282a38f57e7a967d3013bc'
LIMITS = dict(base.LIMITS)
EXECUTION = {**base.EXECUTION,
             'module': 'gx1.audit.fit_market_state_mature_fold1_units_v1'}
SOURCE_PATHS = base.SOURCE_PATHS | frozenset({
    'gx1/audit/fit_market_state_mature_fold1_units_v1.py',
})
_OLD_ROOT = RUNS / 'MARKET_STATE_MATURE_PREFIX_NORMALIZATION_20261107_001'
INPUT_REFS = {
    'original_plan': {'path': str(REPO / 'configs/research/MARKET_STATE_MATURE_PREFIX_NORMALIZATION_20261107_001.json'),
                      'sha256': 'b77795e606b35424d6189b7866337ca74c9f7dfc22371168452f5a6f6d665029'},
    'original_result': {'path': str(_OLD_ROOT / 'RESULT.json'),
                        'sha256': '4b2772e893cdca98af91b04daa817d5b289c234f9dcd5cab5e2bac0f680c6833'},
    'original_terminal': {'path': str(_OLD_ROOT / 'TERMINAL.json'),
                          'sha256': 'c88cca2d44189ec0284ebfea7f55e431759a43ecce49c801f59a880dff88f1bc'},
    'original_fold1_report': {'path': str(_OLD_ROOT / 'QUALIFICATION_FOLD_1.json'),
                              'sha256': '6df90574ef2e414a0b544f69c25de35b488147d91ad413916691b973b525ff3f'},
    'units_plan': {'path': str(REPO / units.MATURE_FOLD1_PLAN_PATH),
                   'sha256': units.MATURE_FOLD1_PLAN_SHA256},
}
MAX_OUTPUT_BYTES = 16 * 1024**2
_published_bytes = 0


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('MATURE_FOLD1_UNITS_' + reason)


def _sha(path: Path) -> str:
    return base._sha(path)


def _ref(path: Path) -> dict:
    return {'path': str(path), 'sha256': _sha(path)}


def _checked_json(item: dict) -> dict:
    return base._checked_ref(item)


def _write(root: Path, name: str, value: dict) -> dict:
    global _published_bytes
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _need(name in ('FAILED.json', 'TERMINAL.json')
          or _published_bytes + len(raw) <= MAX_OUTPUT_BYTES, 'OUTPUT_BUDGET')
    _publish_bytes(root / name, raw)
    _published_bytes += len(raw)
    return {'path': str(root / name), 'sha256': hashlib.sha256(raw).hexdigest()}


def _original_receipts() -> tuple[dict, dict, dict]:
    old_plan = _checked_json(INPUT_REFS['original_plan'])
    old_result = _checked_json(INPUT_REFS['original_result'])
    old_terminal = _checked_json(INPUT_REFS['original_terminal'])
    old_report = _checked_json(INPUT_REFS['original_fold1_report'])
    units_plan = _checked_json(INPUT_REFS['units_plan'])
    profile = units.mature_fold1_prefix_units_profile()
    base._validate_cohort_contract(old_result.get('cohorts'))
    cohort = old_result['cohorts'][0]
    prior = old_result['qualifications'][0]
    _need(old_plan.get('stage') == base.STAGE
          and old_result.get('status') ==
              'COMPLETE_FIVE_MATURE_INPUT_QUALIFICATIONS_CONDITIONAL_FITS_NO_MODEL'
          and old_result.get('plan') == INPUT_REFS['original_plan']
          and old_result.get('source_head') == '39b4ac93af9938a3452e457d3d14d5a3c8a5846a'
          and old_result.get('target_fold_1_report_only') is True
          and old_result.get('qualifications_before_first_fit') is True
          and old_result.get('input_refs') == base.INPUT_REFS
          and old_result.get('decoded_target_arrays') == 0
          and old_result.get('decoded_label_arrays') == 0
          and old_result.get('model_forwards') == old_result.get('optimizer_steps') == 0
          and old_result.get('test_data_used') is False
          and old_terminal.get('status') == 'COMPLETE'
          and old_terminal.get('exit_code') == 0
          and old_terminal.get('plan') == INPUT_REFS['original_plan']
          and old_terminal.get('claim') == old_result.get('claim')
          and old_terminal.get('result') == INPUT_REFS['original_result']
          and old_terminal.get('fit_count') == 4
          and prior.get('report') == INPUT_REFS['original_fold1_report']
          and prior.get('fold_id') == FOLD_ID
          and prior.get('action') == 'REPORT_ONLY'
          and prior.get('fit_allowed') is False
          and prior.get('joined_fit_rows') == FIT_ROWS
          and prior.get('effective_fit_cutoff_ns') == EFFECTIVE_CUTOFF_NS
          and all(item.get('fold_id') != FOLD_ID for item in old_result.get('fits', []))
          and old_report.get('fit_allowed') is False
          and old_report.get('statistics_fits') == 0
          and old_report.get('blockers') == prior.get('blockers')
          and len(old_report['blockers']) == 153
          and cohort['fit_parent_rows_sha256'] == profile['fit_prefix']['parent_rows_sha256']
          and cohort['fit_decision_ns_sha256'] == profile['fit_prefix']['decision_ns_sha256']
          and cohort['joined_fit_rows'] == FIT_ROWS
          and cohort['effective_fit_cutoff_ns'] == EFFECTIVE_CUTOFF_NS
          and all(old_report['proof']['prefix'][key] == value
                  for key, value in profile['fit_prefix'].items())
          and profile['contract_sha256'] == PROFILE_SHA256
          and units_plan.get('status') == 'PREPARATION_ONLY_NOT_NATIVE_ADMISSION'
          and profile['support_evidence'] == INPUT_REFS['original_fold1_report'],
          'ORIGINAL_CHAIN')
    declared = {(surface, entry['field'])
                for surface in old_report['support']
                for entry in units.prefix_units_surface_columns(
                    profile, surface=surface,
                    field_names=old_report['support'][surface]['field_names']).values()}
    blockers = old_report['blockers']
    _need(len(declared) == 153
          and len({(item['surface'], item['field']) for item in blockers}) == 153
          and all(item['reason'] == 'no_observed_prefix_values'
                  and (item['surface'], item['field']) in declared
                  for item in blockers),
          'EXACT_153_DECLARED_FIELDS')
    return old_result, old_report, profile


def _policy_gate(policy: dict, plan_ref: dict) -> None:
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
          and gate.get('output_root') == str(OUT_ROOT),
          'POLICY_GATE')


def preflight_read_only(plan_path: Path, plan_sha256: str,
                        out_dir: Path) -> dict:
    """Verify small receipts and exact source identity without CLAIM or data payload."""
    return _preflight(plan_path, plan_sha256, out_dir, admitted=False)[0]


def _preflight(plan_path: Path, plan_sha256: str, out_dir: Path,
               *, admitted: bool) -> tuple[dict, dict, str, dict, dict | None]:
    head = base._clean_head()
    _need(plan_path == PLAN_PATH and out_dir == OUT_ROOT and not out_dir.exists()
          and len(plan_sha256) == 64 and _sha(plan_path) == plan_sha256,
          'PLAN_IDENTITY')
    plan = json.loads(plan_path.read_text())
    old_result, old_report, profile = _original_receipts()
    cohort = old_result['cohorts'][0]
    _need(plan.get('schema_version') == SCHEMA
          and plan.get('stage') == STAGE
          and plan.get('status') == 'PREPARED_NOT_ADMITTED'
          and plan.get('output_root') == str(OUT_ROOT)
          and plan.get('max_invocations') == 1
          and plan.get('fold_id') == FOLD_ID
          and plan.get('cohort') == cohort
          and plan.get('input_refs') == INPUT_REFS
          and plan.get('units_profile_sha256') == PROFILE_SHA256
          and plan.get('execution') == EXECUTION
          and plan.get('limits') == LIMITS
          and plan.get('normalization_fit_authorized') is True
          and plan.get('training_authorized') is False
          and plan.get('test_data_used') is False
          and plan.get('model_forwards_authorized') == 0
          and plan.get('optimizer_steps_authorized') == 0
          and isinstance(plan.get('source_base_head'), str)
          and len(plan['source_base_head']) == 40
          and subprocess.run(['git', 'merge-base', '--is-ancestor',
                              plan['source_base_head'], head],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode == 0,
          'PLAN_SCOPE')
    owners = plan.get('source_owners')
    _need(isinstance(owners, dict) and set(owners) == SOURCE_PATHS
          and all(_sha(REPO / path) == digest for path, digest in owners.items()),
          'SOURCE_OWNERS')
    base._receipt_chain()
    plan_ref = {'path': str(plan_path), 'sha256': plan_sha256}
    policy_ref = None
    if admitted:
        from gx1.contracts.gx1_capped_execution_v1 import (
            require_capped_cpu_audit_execution, require_capped_lock_ancestry,
        )
        cap = require_capped_cpu_audit_execution()
        require_capped_lock_ancestry()
        _need(all(cap.get(key) == LIMITS[key]
                  for key in ('memory_max', 'memory_high', 'swap',
                              'pids', 'numerical_threads')), 'CAP_LIMITS')
        policy_ref = _ref(POLICY)
        policy = _checked_json(policy_ref)
        _policy_gate(policy, plan_ref)
    return ({'status': 'READY_NO_CLAIM_NO_PAYLOAD', 'plan': plan_ref,
             'source_head': head, 'fold_id': FOLD_ID,
             'old_raw_report': INPUT_REFS['original_fold1_report'],
             'profile_sha256': profile['contract_sha256'],
             'claim_created': False, 'test_data_used': False},
            plan, head, {'old_report': old_report, 'profile': profile,
                         'cohort': cohort}, policy_ref)


def _source_bindings(plan: dict, plan_ref: dict,
                     policy_ref: dict, head: str) -> dict:
    return {**{key: base.INPUT_REFS[source] for key, source in (
                ('provider_plan', 'provider_plan'),
                ('provider_worker', 'provider_worker'),
                ('boolean_raw_result', 'boolean_result'),
                ('boolean_raw_gc_layout', 'gc_layout'),
                ('boolean_raw_domain_transition', 'raw_domain_transition'),
                ('model_axes', 'model_axes'))},
            'current_plan': plan_ref, 'current_policy': policy_ref,
            'current_source_head': head,
            'current_source_owners': plan['source_owners'],
            'cohort_rule': 'decision_ns_strictly_below_official_cutoff_minus_1440_minutes'}


def _validate_new_report(report: dict, original: dict,
                         profile: dict, sources: dict) -> None:
    _need(report.get('fit_allowed') is True
          and report.get('blockers') == []
          and report.get('statistics_fits') == 0
          and report.get('unobserved_units_profile') == profile
          and report.get('declared_unit_support') == original['blockers']
          and report.get('support') == original['support']
          and report.get('proof', {}).get('source_bindings') == sources
          and {key: value for key, value in report['proof'].items()
               if key != 'source_bindings'}
              == {key: value for key, value in original['proof'].items()
                  if key != 'source_bindings'},
          'EXACT_REQUALIFICATION')


def _source_bookend(plan: dict, plan_sha256: str, policy_sha256: str,
                    head: str) -> None:
    _need(base._clean_head() == head
          and _sha(PLAN_PATH) == plan_sha256
          and _sha(POLICY) == policy_sha256
          and all(_sha(REPO / path) == digest
                  for path, digest in plan['source_owners'].items())
          and all(_sha(Path(item['path'])) == item['sha256']
                  for item in INPUT_REFS.values()),
          'SOURCE_BOOKEND')
    base._receipt_chain()


def _timeout(_signum, _frame) -> None:
    raise TimeoutError('MATURE_FOLD1_UNITS_INTERNAL_WALL_LIMIT')


def run(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    """One capped, one-use physical TRAIN input normalization, no model."""
    global _published_bytes
    started = time.monotonic()
    _, plan, head, bundle, policy_ref = _preflight(
        plan_path, plan_sha256, out_dir, admitted=True)
    _need(policy_ref is not None, 'POLICY_REQUIRED')
    plan_ref = {'path': str(plan_path), 'sha256': plan_sha256}
    deadline = started + LIMITS['internal_wall_seconds']
    remaining = deadline - time.monotonic()
    _need(remaining > 0, 'WALL_BUDGET_PRECLAIM')
    _need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0),
          'AMBIENT_ALARM')
    out_dir.mkdir(parents=True, exist_ok=False)
    _published_bytes = 0
    old_handler = signal.getsignal(signal.SIGALRM)
    claim = None
    try:
        signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        _fsync_directory(out_dir.parent)
        claim = _write(out_dir, 'CLAIM.json', {
            'schema_version': 'gx1_market_state_mature_fold1_normalization_claim_v1',
            'plan': plan_ref, 'policy': policy_ref, 'source_head': head,
            'invocation': 1, 'fold_id': FOLD_ID, 'normalization_only': True,
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False})
        original_result, original_report, profile = _original_receipts()
        chain = base._receipt_chain()
        provider_plan = chain['provider_plan']
        provider = base._provider_module()
        info = provider.validate_metadata(provider_plan)
        prepared = base._prepare_inputs(provider, provider_plan, info,
                                        chain['gc_layout'])
        try:
            base._validate_axes(original_result['cohorts'], prepared['axes'])
            axes = prepared['axes']
            cohort = bundle['cohort']
            rows = cohort['joined_fit_rows']
            sources = _source_bindings(plan, plan_ref, policy_ref, head)
            qualified = qualify_native_direction_prefix_population(
                parent_rows=axes['parent_rows'],
                bar_start_ns=axes['bar_start_ns'],
                decision_ns=axes['decision_ns'],
                expected_axes=prepared['expected_axes'],
                cutoff_ns=cohort['effective_fit_cutoff_ns'],
                expected_prefix_parent_rows=axes['parent_rows'][:rows],
                entry_sequence_source=prepared['entry_sequence_source'],
                snap_x=prepared['current']['snap'],
                ctx_cont=prepared['current']['ctx_cont'],
                ctx_cat=prepared['current']['ctx_cat'],
                ordered_signal_names=info['source']['feature_field_order'],
                multi_tf_sources=prepared['cache'],
                gc_surfaces=prepared['gc']['surfaces'],
                gc_layout=prepared['gc']['layout'],
                base_lineage=provider_plan['base_lineage'],
                source_bindings=sources,
                unobserved_units_profile=profile)
            report = qualified['report']
            _validate_new_report(report, original_report, profile, sources)
            _need(qualified['population'] is not None,
                  'FIT_POPULATION')
            report_ref = _write(out_dir, 'QUALIFICATION_FOLD_1.json', report)
            fitted = fit_native_direction_prefix_normalization(qualified)
            _need(fitted['prefix_qualification'] == report
                  and fitted.get('unobserved_units_profile') == profile
                  and fitted['old055_or223_normalization_identity_reused'] is False
                  and fitted['native_launch_authorized'] is False
                  and fitted['genuine_model_fit_authorized'] is False
                  and set(fitted['base_normalization']['surfaces'])
                      | set(fitted['gc_mtf_history']['normalization']['surfaces'])
                      == set(original_report['support'])
                  and len(original_report['support']) == 12,
                  'FIT_SCOPE')
            output_ref = _write(out_dir,
                                'PREFIX_NORMALIZATION_FOLD_1.json', fitted)
        finally:
            base._release_pages(provider, prepared)
        provider.validate_metadata(provider_plan)
        _source_bookend(plan, plan_sha256, policy_ref['sha256'], head)
        result_ref = _write(out_dir, 'RESULT.json', {
            'schema_version': 'gx1_market_state_mature_fold1_normalization_result_v1',
            'status': 'COMPLETE_FOLD1_MATURE_UNITS_PREFIX_NORMALIZATION_NO_MODEL',
            'plan': plan_ref, 'claim': claim, 'policy': policy_ref,
            'source_head': head, 'elapsed_seconds': time.monotonic() - started,
            'input_refs': INPUT_REFS, 'cohort': bundle['cohort'],
            'old_raw_qualification': INPUT_REFS['original_fold1_report'],
            'unit_profile_sha256': profile['contract_sha256'],
            'new_qualification': report_ref,
            'normalization': output_ref,
            'normalization_contract_sha256': fitted['contract_sha256'],
            'declared_unit_support_rows': len(report['declared_unit_support']),
            'fit_rows': FIT_ROWS, 'statistics_fits': 1,
            'decoded_target_arrays': 0, 'decoded_label_arrays': 0,
            'model_forwards': 0, 'model_constructions': 0,
            'optimizer_steps': 0, 'test_data_used': False,
            'model_training_authorized': False,
            'native_launch_authorized': False})
        signal.setitimer(signal.ITIMER_REAL, 0)
        terminal_ref = _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_mature_fold1_normalization_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
            'claim': claim, 'result': result_ref,
            'qualification_count': 1, 'fit_count': 1,
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'no_relaunch_under_this_plan': True})
        return {'result': result_ref, 'terminal': terminal_ref}
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        failed_ref = _write(out_dir, 'FAILED.json', {
            'schema_version': 'gx1_market_state_mature_fold1_normalization_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'elapsed_seconds': time.monotonic() - started,
            'no_relaunch_under_this_plan': True})
        _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_mature_fold1_normalization_terminal_v1',
            'status': 'FAILED', 'exit_code': 1, 'plan': plan_ref,
            'claim': claim, 'failure': failed_ref,
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'no_relaunch_under_this_plan': True})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    supplied = __import__('sys').argv[1:]
    for option in ('--plan', '--plan-sha256', '--out-dir', '--preflight'):
        if sum(value.partition('=')[0] == option for value in supplied) > 1:
            parser.error('duplicate flag: ' + option)
    args = parser.parse_args(supplied)
    if not args.plan.is_absolute() or not args.out_dir.is_absolute():
        parser.error('plan and out-dir must be absolute paths')
    report = (preflight_read_only(args.plan, args.plan_sha256, args.out_dir)
              if args.preflight else
              run(args.plan, args.plan_sha256, args.out_dir))
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
