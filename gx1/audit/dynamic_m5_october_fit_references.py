"""Freeze October OOF references using only the earlier TRAIN prefix."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.audit import dynamic_mtf_december_candle as candle
from gx1.audit import dynamic_mtf_december_normalization as raw_owner
from gx1.audit import dynamic_mtf_october_feature_delivery as delivery
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as state
from gx1.contracts import entry_dynamic_mtf_state_report_v1 as report
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
from gx1.features import htf_features as htf


REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / 'configs/research/DYNAMIC_M5_OCTOBER_FIT_REFERENCES_20261106_001.json'
POLICY = REPO / 'NEXT_RUN_POLICY.json'
POLICY_KEY = 'dynamic_m5_october_fit_references_20261106_001'


def _require(ok, reason):
    if not ok:
        raise RuntimeError('OCTOBER_FIT_REFERENCES_' + reason)


def _read(ref):
    return delivery._read(ref)


def _sha(path):
    return delivery._sha(path)


def _ref(path):
    return delivery._ref(path)


def _twice_wins_independent(raw, binary, eligible):
    """Independent sorted-pair oracle for the frozen candle orientation."""
    up = raw[eligible & (binary == 1)]
    down = np.sort(raw[eligible & (binary == 0)])
    pairs = len(up) * len(down)
    if not pairs:
        return None
    lower = np.searchsorted(down, up, side='left')
    upper = np.searchsorted(down, up, side='right')
    return int((lower + upper).sum(dtype=np.int64))


def run():
    require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    _require(Path.cwd().resolve() == REPO
             and subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'work/gx1-current'
             and subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == '', 'CLEAN_REPO')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    plan_ref = _ref(PLAN)
    plan = _read(plan_ref)
    policy_bytes = POLICY.read_bytes()
    policy = json.loads(policy_bytes)
    admitted = policy.get(POLICY_KEY)
    _require(plan['schema_version'] == 'gx1_dynamic_m5_october_fit_references_plan_v1'
             and plan['status'] == 'PREPARED_NOT_ADMITTED' and plan['max_invocations'] == 1
             and plan['output_root'] == '/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_FIT_REFERENCES_20261106_001'
             and plan['execution']['command'] ==
                 'bash scripts/gx1_capped_run.sh --class audit --mem 4G --swap 512M -- timeout --signal=TERM --kill-after=10s 1800s .venv/bin/python -m gx1.audit.dynamic_m5_october_fit_references'
             and plan['restrictions'] == {'new_prior_fit': False, 'october_outcomes': False,
                 'model_construction': False, 'model_forward': False, 'optimizer_updates': False,
                 'test_split': False, 'broker_or_network': False, 'spending_usd': 0}
             and isinstance(admitted, dict) and admitted.get('status') == 'ADMITTED_ONCE'
             and admitted.get('plan') == plan_ref and admitted.get('operator_approved') is True
             and admitted.get('remaining_invocations') == 1
             and policy['training_enabled'] is False and policy['full_val_allowed'] is False
             and policy['full_epoch_training_allowed'] is False
             and policy['direction_edge_research_goal_20261105']['native_launch_authorized'] is False,
             'ADMISSION')
    _require(plan['standing_authorization'] == policy['no_cost_local_standing_authorization_20261105']['operator_instruction']
             and all(_sha(REPO / path) == digest for path, digest in plan['source_files_sha256'].items()),
             'SOURCE_IDENTITY')
    prior_result = _read(plan['prior_result'])
    target_only = _read(plan['october_target_only_result'])
    oct_result = _read(plan['october_series_result'])
    counts = _read(plan['fit_counts'])
    prior = _read(plan['fit_prior'])
    control = _read(plan['control_view'])
    qualification = _read(plan['input_qualification'])
    old_manifest = _read(plan['fit_series'])
    new_manifest = _read(plan['october_series'])
    target_only_plan = _read(plan['october_target_only_plan'])
    oct_plan = _read(plan['october_series_plan'])
    _require(prior_result['counts'] == plan['fit_counts'] and prior_result['prior'] == plan['fit_prior']
             and prior_result['series'] == plan['fit_series']
             and target_only['fit_series'] == plan['fit_series']
             and target_only['original_series'] == oct_plan['original_series']
             and target_only_plan['original_series'] == oct_plan['original_series']
             and oct_result['series'] == plan['october_series']
             and oct_plan['fit_series'] == plan['fit_series']
             and oct_plan['prior_target_result'] == plan['october_target_only_result']
             and target_only_plan['source_files_sha256']['gx1/audit/dynamic_mtf_train_prefix_targets.py']
                 == oct_plan['source_files_sha256']['gx1/audit/dynamic_mtf_train_prefix_targets.py']
             and old_manifest['metadata']['definition'] == new_manifest['metadata']['definition']
             and old_manifest['metadata']['source_refs'] == new_manifest['metadata']['source_refs']
             and old_manifest['metadata']['source_row_offset'] == new_manifest['metadata']['source_row_offset']
             and old_manifest['metadata']['parents'] == counts['prefix_rows'] == prior['parents'] == 29849
             and old_manifest['metadata']['cutoff_ns'] == counts['cutoff_ns'] == plan['fit_cutoff_ns']
             and new_manifest['metadata']['parents'] == 36172
             and new_manifest['metadata']['cutoff_ns'] == plan['development_cutoff_ns']
             and qualification['fit_allowed'] is True
             and control['october_observer_axes']['fit']['cutoff_ns'] == plan['fit_cutoff_ns'],
             'EXACT_REUSED_PREFIX')
    leads = np.arange(1, 1441, dtype=np.int64)
    _require(report.dense_train_prior_from_counts(counts, lead_minutes=leads) == prior,
             'FROZEN_PRIOR_REPLAY')
    root = Path(plan['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    try:
        with campaign.file_verification_scope():
            fit_series = state.load_dynamic_mtf_state_series(plan['fit_series'],
                expected_source_refs=plan['target_source_refs'],
                expected_cutoff_ns=plan['fit_cutoff_ns'])
            parents, issues = fit_series.arrays['parent_rows'], fit_series.arrays['issue_ns']
            _require(len(parents) == 29849 and np.all(issues < plan['fit_cutoff_ns'])
                     and raw_owner._axis_digest(parents) == plan['fit_parent_rows_sha256']
                     and raw_owner._axis_digest(issues) == plan['fit_issue_ns_sha256'],
                     'EXACT_FIT_AXES')
            native_leads = np.array([int(htf.MULTI_TF_SHIFT[tf].value) // state.MINUTE_NS
                                     for tf in state.TIMEFRAMES], dtype=np.int64)
            _require(native_leads.tolist() == [5, 15, 60, 240, 1440], 'NATIVE_LEADS')
            states, reasons, observed, native_digest = candle._native_labels(
                fit_series, plan['fit_cutoff_ns'], native_leads)
            lineage = control['base_lineage']
            cache_ref = {'path': lineage['mtf_cache_manifest_path'],
                         'sha256': lineage['mtf_cache_manifest_sha256']}
            cache = _read(cache_ref)
            names = cache['feature_names']
            _require(names == list(htf.MULTI_TF_PER_BAR_FEATURES_V4)
                     and names.count(candle.FIELD) == 1
                     and cache['cache_identity_sha256'] == htf.compute_htf_v4_cache_identity(cache),
                     'CANDLE_CACHE')
            column = names.index(candle.FIELD)
            panels = {}
            for j, tf in enumerate(state.TIMEFRAMES):
                info = cache['tfs'][tf]
                evidence = qualification['proof']['tf'][tf]
                support = qualification['support']['mtf_' + tf.lower()]
                _require(support['field_names'] == names
                         and support['observed_counts'][column] == support['population_rows']
                         and column not in [i for i, _ in support['masking_pairs']]
                         and evidence['seq_len'] == control['window_rows'][tf]
                         and info['causal_warmup_rows'] == evidence['source_warmup_rows'],
                         'CANDLE_SUPPORT:' + tf)
                cache_root = Path(cache_ref['path']).parent
                clock_ref = {'path': str(cache_root / info['ts_npy']),
                             'sha256': info['ts_npy_sha256']}
                feat_ref = {'path': str(cache_root / info['feats_npy']),
                            'sha256': info['feats_npy_sha256']}
                opens = raw_owner._raw_array(clock_ref, shape=(info['n_bars'],), dtype='int64')
                shift = int(htf.MULTI_TF_SHIFT[tf].value)
                ids = candle._closed_window_ends(opens, issues, shift_ns=shift,
                    window=evidence['seq_len'], warmup=evidence['source_warmup_rows'])
                stop = evidence['right_index_exclusive']
                _require(int(ids[-1]) + 1 == stop
                         and int(ids[0]) + 1 - evidence['seq_len'] == evidence['left_index_inclusive']
                         and np.all(opens[ids] + shift <= issues)
                         and np.array_equal(ids, np.searchsorted(opens + shift, issues, side='right') - 1),
                         'CLOSED_CANDLE_CLOCK:' + tf)
                values = raw_owner._raw_array(feat_ref,
                    shape=(info['n_bars'], len(names)), dtype='float32', stop=stop)
                scores = values[ids, column].astype(np.float64)
                eligible = observed[:, j] & (states[:, j] != state.FLAT)
                binary = np.where(eligible, (states[:, j] == state.UP).astype(np.int8), np.int8(-1))
                ranks = candle._rank_direction(scores, binary, eligible)
                _require(ranks['twice_wins'] == _twice_wins_independent(scores, binary, eligible),
                         'RANK_ORACLE:' + tf)
                _require(ranks['orientation'] in (-1, 1) and ranks['eligible_rows'] > 0,
                         'CANDLE_ORIENTATION:' + tf)
                panels[tf] = {'lead_minutes': int(native_leads[j]), 'field': candle.FIELD,
                    'orientation': ranks['orientation'], 'fit_eligible_up': ranks['up'],
                    'fit_eligible_down': ranks['down'], 'fit_pair_count': ranks['pairs'],
                    'fit_raw_auc': ranks['raw_auc'], 'fit_oriented_auc': ranks['oriented_auc'],
                    'fit_observed_endpoints': int(observed[:, j].sum()),
                    'fit_immature_endpoints': int(((reasons[:, j] & state.IMMATURE) != 0).sum()),
                    'latest_closed_row_indices_sha256': raw_owner._axis_digest(ids),
                    'clock': clock_ref, 'raw_features': feat_ref}
            campaign.capture_verified_file_snapshot()
        _require(_sha(POLICY) == hashlib.sha256(policy_bytes).hexdigest()
                 and _ref(PLAN) == plan_ref
                 and all(_sha(REPO / path) == digest for path, digest in plan['source_files_sha256'].items())
                 and subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == head
                 and subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == '',
                 'SOURCE_OR_POLICY_CHANGED')
        result = delivery._publish(root, 'RESULT.json', {
            'schema_version': 'gx1_dynamic_m5_october_fit_references_result_v1',
            'plan': plan_ref, 'fit_series': plan['fit_series'],
            'development_series': plan['october_series'], 'fit_counts': plan['fit_counts'],
            'fit_prior': plan['fit_prior'], 'prior_replayed_exactly': True,
            'native_target_count_digest': native_digest,
            'candle_panels': panels, 'october_outcomes_read': False,
            'new_prior_fits': 0, 'model_calls': 0, 'test_split_used': False,
            'independent_edge': 'UNKNOWN'})
        delivery._publish(root, 'TERMINAL.json', {'plan': plan_ref, 'result': result,
            'exit_code': 0, 'elapsed_seconds': time.monotonic() - began,
            'source_unchanged': True, 'no_relaunch_under_this_plan': True})
        return result
    except Exception as exc:
        delivery._publish(root, 'FAILED.json', {'plan': plan_ref,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'no_relaunch_under_this_plan': True})
        raise


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True))
