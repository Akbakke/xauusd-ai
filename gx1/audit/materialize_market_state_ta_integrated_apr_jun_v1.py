"""Bind closed M1 path and M5 structure truth to fixed Apr-Jun score rows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import time

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/TA_INTEGRATED_APR_JUN_ISSUE_INPUTS_20261117_002')
PLAN = REPO / 'configs/research/TA_INTEGRATED_APR_JUN_ISSUE_INPUTS_20261117_002.json'
PHYSICAL = Path('/home/andre2/GX1_RUNS/TA_FOLD2_APR_JUN_INPUTS_20261117_001')
STAGE = 'ta_integrated_apr_jun_issue_inputs_20261117_002'
START = 1714521600000000000  # 2024-05-01, same structure replay warmup as fit
SCORE_END = 1751328000000000000  # 2025-07-01 exclusive
M5_NS = 300_000_000_000
MINUTE_NS = 60_000_000_000
MAX_SECONDS = 1800
SOURCES = (
    'gx1/audit/materialize_market_state_ta_integrated_apr_jun_v1.py',
    'gx1/audit/measure_ta_m5_structure_candidate_v1.py',
    'gx1/contracts/market_state_ta_m1_issue_path_v1.py',
)


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_APR_JUN_ISSUE_INPUT_' + reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'REGULAR_FILE')
    return {'path': str(path), 'sha256': sha(path)}


def read(binding):
    need(ref(binding['path']) == binding, 'INPUT_SHA')
    return json.loads(Path(binding['path']).read_text())


def publish(name, value):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    need(len(raw) < 2 * 1024 * 1024, 'JSON_BUDGET')
    _publish_bytes(ROOT / name, raw)
    return ref(ROOT / name)


def build_plan():
    physical_result = ref(PHYSICAL / 'RESULT.json')
    physical_coverage = ref(PHYSICAL / 'COVERAGE.json')
    physical_terminal = ref(PHYSICAL / 'TERMINAL.json')
    result, coverage, terminal = (read(x) for x in
        (physical_result, physical_coverage, physical_terminal))
    need(result['status'] == 'COMPLETE_APR_JUN_TA_SCORE_INPUTS_NO_MODEL'
         and coverage['status'] == terminal['status'] == 'COMPLETE'
         and terminal['exit_code'] == 0 and terminal['result'] == physical_result
         and terminal['coverage'] == physical_coverage and terminal['remaining_invocations'] == 0
         and result['source_unchanged'] is True and result['model_forwards'] == 0
         and result['optimizer_steps'] == 0 and result['test_data_used'] is False
         and coverage['score_parent_start'] == 64890 and coverage['score_rows'] == 17537
         and coverage['score_end_exclusive_ns'] == SCORE_END
         and coverage['batch_indices'] == list(range(1096)), 'PHYSICAL_SCORE_LINEAGE')
    parity = json.loads((REPO / 'docs/TA_M1_PATH_ARCHIVE_PARITY_REVIEW_20261116.json').read_text())
    m1 = parity['source_refs']['S_train_m1_quotes_parquet']
    structure = json.loads((REPO / 'configs/research/TA_M5_STRUCTURE_FOLD2_AUDIT_20261116_001.json').read_text())
    m5 = {'path': structure['input']['path'],
          'sha256': structure['input']['sha256_expected_from_manifest']}
    closure = {key: structure['closure_authority'][key] for key in ('path', 'sha256')}
    need(ref(closure['path']) == closure, 'CLOSURE_SOURCE_SHA')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return {'schema_version': 'gx1_ta_apr_jun_m1_structure_input_plan_v1',
        'stage': STAGE, 'source_base_commit': head,
        'source_owners': {name: sha(REPO / name) for name in SOURCES},
        'physical_result': physical_result, 'physical_coverage': physical_coverage,
        'physical_terminal': physical_terminal, 'm1_source': m1,
        'm5_source': m5, 'closure_authority': closure,
        'scope': {'selected_rows': 17536, 'score_start_ns': 1743465600000000000,
                  'score_end_exclusive_ns': SCORE_END, 'structure_warmup_start_ns': START,
                  'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False},
        'output_root': str(ROOT),
        'execution': {'class': 'audit', 'memory': '4G', 'swap': '512M',
                      'max_seconds': MAX_SECONDS}}


def _groups(parquet, first_ns, end_ns):
    import pandas as pd
    hits = [i for i in range(parquet.metadata.num_row_groups)
        if pd.Timestamp(parquet.metadata.row_group(i).column(0).statistics.max).value >= first_ns
        and pd.Timestamp(parquet.metadata.row_group(i).column(0).statistics.min).value < end_ns]
    need(bool(hits) and hits == list(range(hits[0], hits[-1] + 1)), 'SOURCE_GROUPS')
    return hits


def inspect(plan_sha, *, execute):
    import numpy as np
    import pyarrow.parquet as pq
    plan_ref = ref(PLAN)
    need(plan_ref['sha256'] == plan_sha, 'PLAN_SHA')
    plan = read(plan_ref)
    expected = build_plan()
    expected['source_base_commit'] = plan['source_base_commit']
    need(plan == expected and not ROOT.exists(), 'EXACT_UNUSED_PLAN')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    need(subprocess.run(['git', 'merge-base', '--is-ancestor', plan['source_base_commit'], head],
                        cwd=REPO, capture_output=True).returncode == 0, 'SOURCE_ANCESTRY')
    policy_ref = ref(REPO / 'NEXT_RUN_POLICY.json')
    policy = read(policy_ref)
    need(all(policy[key] is False for key in
         ('training_enabled', 'native_training_enabled', 'full_epoch_training_allowed',
          'native_launch_authorized', 'full_val_allowed')), 'TRAINING_CLOSED')
    if execute:
        gate = policy[STAGE]
        need(gate['status'] == 'ADMITTED' and gate['plan'] == plan_ref
             and gate['execution_authorized'] is True and gate['data_reads_authorized'] is True
             and gate['training_authorized'] is False and gate['remaining_invocations'] == 1
             and gate['relaunch_allowed'] is False, 'EXACT_ADMISSION')
        need(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True),
             'CLEAN_SOURCE')
    coverage = read(plan['physical_coverage'])
    closure = read(plan['closure_authority'])
    need(type(closure.get('intervals')) is list and len(closure['intervals']) > 0,
         'CLOSURE_INTERVALS')
    axis_ref = coverage['full_axes']['source']
    need(ref(axis_ref['path']) == axis_ref, 'AXIS_SHA')
    with np.load(axis_ref['path'], allow_pickle=False) as archive:
        parents = archive['parent_rows'].copy()
        issues = archive['decision_ns'].copy()
    need(parents.dtype == issues.dtype == np.int64
         and parents.shape == issues.shape == (coverage['full_axes']['rows'],), 'AXIS_SHAPE')
    axis_sha = lambda x: hashlib.sha256(x.astype('<i8', copy=False).tobytes()).hexdigest()
    need(axis_sha(parents) == coverage['full_axes']['parent_rows_sha256']
         and axis_sha(issues) == coverage['full_axes']['decision_ns_sha256'], 'FULL_AXIS_HASH')
    selected = (coverage['score_parent_start']
                + np.asarray(coverage['batch_indices'], dtype=np.int64)[:, None] * 16
                + np.arange(16, dtype=np.int64)).reshape(-1)
    issue = issues[selected]
    need(len(selected) == plan['scope']['selected_rows']
         and axis_sha(parents[selected]) == coverage['selected_parent_rows_sha256']
         and axis_sha(issue) == coverage['selected_decision_ns_sha256']
         and bool(np.all((issue >= plan['scope']['score_start_ns']) & (issue < SCORE_END)))
         and bool(np.all(np.diff(issue) > 0)), 'SCORE_AXIS')
    m1_file = pq.ParquetFile(plan['m1_source']['path'])
    m5_file = pq.ParquetFile(plan['m5_source']['path'])
    m1_groups = _groups(m1_file, int(issue[0] - M5_NS), int(issue[-1]))
    m5_groups = _groups(m5_file, START, SCORE_END)
    need(len(m1_groups) == 31 and len(m5_groups) == 21, 'EXACT_SOURCE_GROUPS')
    return plan, plan_ref, policy_ref, head, parents[selected], issue, m1_groups, m5_groups


def bookend(plan, plan_ref, policy_ref, head):
    need(ref(PLAN) == plan_ref and ref(REPO / 'NEXT_RUN_POLICY.json') == policy_ref
         and all(sha(REPO / name) == digest for name, digest in plan['source_owners'].items())
         and all(ref(plan[key]['path']) == plan[key] for key in
                 ('m1_source', 'm5_source', 'closure_authority',
                  'physical_result', 'physical_coverage', 'physical_terminal'))
         and subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == head
         and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True),
         'SOURCE_BOOKEND')


def run(plan_sha):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from gx1.audit.measure_ta_m5_structure_candidate_v1 import measure_candidate
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    from gx1.contracts.market_state_ta_m1_issue_path_v1 import M1IssuePathTape
    cap = require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    began = time.monotonic()
    plan, plan_ref, policy_ref, head, parents, issues, m1_groups, m5_groups = inspect(
        plan_sha, execute=True)
    need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'AMBIENT_TIMER')
    ROOT.mkdir()
    claim = publish('CLAIM.json', {'plan': plan_ref, 'policy': policy_ref,
        'source_commit': head, 'cap': cap, 'claimed_utc': datetime.now(timezone.utc).isoformat()})
    status = 'FAILED_CONSUMED'
    failure = result = None
    def expire(*_):
        raise TimeoutError('TA_APR_JUN_ISSUE_INPUT_MAX_SECONDS')
    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
        need(ref(plan['m1_source']['path']) == plan['m1_source']
             and ref(plan['m5_source']['path']) == plan['m5_source'], 'SOURCE_SHA_BEFORE')
        m1_table = pq.ParquetFile(plan['m1_source']['path']).read_row_groups(
            m1_groups, columns=['time', 'open', 'high', 'low', 'close'])
        m1_time = m1_table['time'].to_numpy(zero_copy_only=False).astype('datetime64[ns]').astype(np.int64)
        m1_ohlc = np.stack([m1_table[key].to_numpy(zero_copy_only=False)
            for key in ('open', 'high', 'low', 'close')], axis=1)
        path = M1IssuePathTape(m1_time, m1_ohlc).gather(issues)
        need(np.array_equal(path['issue_ns'], issues), 'M1_ISSUE_CLOCK')
        del m1_table, m1_time, m1_ohlc
        m5_file = pq.ParquetFile(plan['m5_source']['path'])
        pieces = {key: [] for key in ('time', 'high', 'low', 'close', 'atr')}
        for group in m5_groups:
            table = m5_file.read_row_group(group, columns=list(pieces))
            for key in pieces:
                col = table[key].to_numpy(zero_copy_only=False)
                pieces[key].append(col.astype('datetime64[ns]').astype(np.int64)
                                   if key == 'time' else col)
        bars = {key: np.concatenate(chunks) for key, chunks in pieces.items()}
        stop = int(np.searchsorted(bars['time'], SCORE_END))
        bars = {key: values[:stop] for key, values in bars.items()}
        closure = read(plan['closure_authority'])
        intervals = np.array([[pd.Timestamp(item['closure_start_utc']).value,
                               pd.Timestamp(item['closure_end_utc_exclusive']).value]
                              for item in closure['intervals']
                              if item['successor_across_gap_allowed'] is True],
                             dtype=np.int64).reshape(-1, 2)
        issue_rows = np.flatnonzero(bars['time'] >= START).astype(np.int64)
        measured = measure_candidate(bars['time'], bars['high'], bars['low'],
            bars['close'], bars['atr'], issue_rows,
            fold_target_end_ns=SCORE_END, known_closure_ns=intervals)
        structure_time = bars['time'][issue_rows]
        take = np.searchsorted(structure_time, issues - M5_NS)
        need(bool(np.all(take < len(structure_time)))
             and np.array_equal(structure_time[take], issues - M5_NS),
             'STRUCTURE_SCORE_CLOCK')
        hit = np.stack((measured['high_hit'][take], measured['low_hit'][take]), axis=1)
        observed = np.stack((measured['high_observed'][take],
                             measured['low_observed'][take]), axis=1)
        need(bool(np.all(~hit | observed)), 'STRUCTURE_UNKNOWN_MASK')
        stream = io.BytesIO()
        np.savez_compressed(stream, parent_rows=parents, issue_ns=issues,
            m1_mid_ohlc_bps=path['mid_ohlc_bps'], m1_observed=path['observed'],
            structure_hit=hit, structure_observed=observed)
        _publish_bytes(ROOT / 'INPUTS.npz', stream.getvalue())
        with np.load(ROOT / 'INPUTS.npz', allow_pickle=False) as check:
            need(np.array_equal(check['parent_rows'], parents)
                 and np.array_equal(check['issue_ns'], issues)
                 and np.array_equal(check['m1_mid_ohlc_bps'], path['mid_ohlc_bps'])
                 and np.array_equal(check['m1_observed'], path['observed'])
                 and np.array_equal(check['structure_hit'], hit)
                 and np.array_equal(check['structure_observed'], observed), 'SERIALIZATION')
        bookend(plan, plan_ref, policy_ref, head)
        result = publish('RESULT.json', {
            'status': 'COMPLETE_APR_JUN_M1_STRUCTURE_INPUTS_NO_MODEL',
            'plan': plan_ref, 'claim': claim, 'source_commit': head, 'source_unchanged': True,
            'input_archive': ref(ROOT / 'INPUTS.npz'), 'score_rows': len(issues),
            'issue_axis_sha256': hashlib.sha256(issues.astype('<i8').tobytes()).hexdigest(),
            'm1_row_groups': len(m1_groups), 'm1_observed_slots': int(path['observed'].sum()),
            'm1_complete_paths': int(path['full_path'].sum()),
            'm5_row_groups': len(m5_groups),
            'structure_high_observed': int(observed[:, 0].sum()),
            'structure_low_observed': int(observed[:, 1].sum()),
            'structure_both_observed': int(observed.all(axis=1).sum()),
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'economic_edge_proven': False,
            'elapsed_seconds': time.monotonic() - began})
        bookend(plan, plan_ref, policy_ref, head)
        status = 'COMPLETE'
    except BaseException as error:
        failure = publish('FAILED.json', {'type': type(error).__name__,
            'error': str(error)[:2048], 'plan': plan_ref})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        publish('TERMINAL.json', {'status': status,
            'exit_code': 0 if status == 'COMPLETE' else 1,
            'plan': plan_ref, 'result': result, 'failure': failure,
            'remaining_invocations': 0, 'relaunch_allowed': False,
            'elapsed_seconds': time.monotonic() - began})


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--build-plan', action='store_true')
    modes.add_argument('--preflight', action='store_true')
    modes.add_argument('--run', action='store_true')
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    if args.build_plan:
        need(args.plan_sha256 is None and not PLAN.exists(), 'BUILD_MODE')
        from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
        _publish_bytes(PLAN, (json.dumps(build_plan(), sort_keys=True, indent=2) + '\n').encode())
        print(json.dumps(ref(PLAN)))
        return
    need(type(args.plan_sha256) is str and len(args.plan_sha256) == 64
         and all(c in '0123456789abcdef' for c in args.plan_sha256), 'PLAN_ID')
    if args.preflight:
        plan, plan_ref, _, head, _, issues, m1_groups, m5_groups = inspect(
            args.plan_sha256, execute=False)
        print(json.dumps({'status': 'READY_NO_CLAIM', 'plan': plan_ref,
            'source_commit': head, 'score_rows': len(issues),
            'm1_groups': len(m1_groups), 'm5_groups': len(m5_groups)}))
    else:
        run(args.plan_sha256)


if __name__ == '__main__':
    main()
