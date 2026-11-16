"""Freeze the existing Fold2 later panel's M1 and M5 structure score inputs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess

from gx1.audit.materialize_market_state_ta_integrated_fold2_inputs_v1 import read, sha

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/TA_INTEGRATED_FOLD2_LATER_INPUTS_20261116_001')
TRAIN = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_SPAN_VIEW_20261113_001')
LATER = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_LATER_INPUTS_20261113_002')
START = 1714521600000000000  # 2024-05-01, structure replay warmup
SCORE_START = 1735689600000000000  # 2025-01-01
SCORE_END = 1743465600000000000  # 2025-04-01 exclusive
M5_NS = 300_000_000_000
MINUTE_NS = 60_000_000_000


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_INTEGRATED_LATER_INPUTS_' + reason)


def main():
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
    from gx1.audit.measure_ta_m5_structure_candidate_v1 import measure_candidate
    from gx1.audit.observe_market_state_ta_fold2_fresh_initial_v1 import panel_positions
    from gx1.contracts.gx1_capped_execution_v1 import (require_capped_cpu_audit_execution,
                                                         require_capped_lock_ancestry)
    from gx1.contracts.market_state_ta_m1_issue_path_v1 import M1IssuePathTape

    cap = require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    head = subprocess.check_output(('git', 'rev-parse', 'HEAD'), cwd=REPO, text=True).strip()
    need(subprocess.check_output(('git', 'branch', '--show-current'), cwd=REPO, text=True).strip()
         == 'work/gx1-current'
         and not subprocess.check_output(('git', 'status', '--porcelain'), cwd=REPO, text=True),
         'CLEAN_SOURCE')
    policy = json.loads((REPO / 'NEXT_RUN_POLICY.json').read_text())
    need(policy['training_enabled'] is policy['native_training_enabled'] is False
         and policy['full_epoch_training_allowed'] is False, 'TRAINING_CLOSED')
    need(not ROOT.exists(), 'ONE_USE')
    ROOT.mkdir()
    (ROOT / 'CLAIM.json').write_text(json.dumps({
        'source_commit': head, 'source_sha256': sha(__file__), 'cap': cap,
        'scope': 'fixed_4096_Fold2_later_development_positions_no_fit',
        'claimed_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
    try:
        train = json.loads((TRAIN / 'PLAN.json').read_text())
        later = json.loads((LATER / 'PLAN.json').read_text())
        design = json.loads((REPO / 'docs/TA_FOLD2_INPUT_REVIEW.json').read_text())['fresh_initial_design']
        _, selected = panel_positions(np, design, later)
        need(len(selected) == 4096 and len(np.unique(selected)) == 4096, 'PANEL_SELECTION')
        provider = mature._provider_module()
        provider_plan = read(train['inputs']['provider_plan'])
        info = provider.validate_metadata(provider_plan)
        axes, _ = provider.read_axes(provider_plan, info, np)
        parent = axes['parent_rows'][selected].astype(np.int64, copy=True)
        issues = axes['decision_ns'][selected].astype(np.int64, copy=True)
        need(np.all((issues >= SCORE_START) & (issues < SCORE_END))
             and parent.min() >= later['scope']['score_parent_start']
             and parent.max() <= later['scope']['last_selected_parent_position'], 'SCORE_SCOPE')
        parity = json.loads((REPO / 'docs/TA_M1_PATH_ARCHIVE_PARITY_REVIEW_20261116.json').read_text())
        m1_ref = parity['source_refs']['S_train_m1_quotes_parquet']
        need(sha(m1_ref['path']) == m1_ref['sha256'], 'M1_SOURCE_SHA')
        parquet = pq.ParquetFile(m1_ref['path'])
        groups = []
        wanted_first, wanted_last = int(issues[0] - M5_NS), int(issues[-1] - MINUTE_NS)
        for index in range(parquet.metadata.num_row_groups):
            stats = parquet.metadata.row_group(index).column(0).statistics
            if pd.Timestamp(stats.max).value >= wanted_first and pd.Timestamp(stats.min).value <= wanted_last:
                groups.append(index)
        need(0 < len(groups) <= 40 and groups == list(range(groups[0], groups[-1] + 1)),
             'M1_ROW_GROUP_SCOPE')
        table = parquet.read_row_groups(groups, columns=['time', 'open', 'high', 'low', 'close'])
        time = table['time'].to_numpy(zero_copy_only=False).astype('datetime64[ns]').astype(np.int64)
        ohlc = np.stack([table[key].to_numpy(zero_copy_only=False)
                         for key in ('open', 'high', 'low', 'close')], axis=1)
        path = M1IssuePathTape(time, ohlc).gather(issues)
        need(np.array_equal(path['issue_ns'], issues), 'M1_ISSUE_CLOCK')
        del table, time, ohlc

        structure_plan = json.loads((REPO / 'configs/research/TA_M5_STRUCTURE_FOLD2_AUDIT_20261116_001.json').read_text())
        m5_ref = structure_plan['input']
        need(sha(m5_ref['path']) == m5_ref['sha256_expected_from_manifest'], 'M5_SOURCE_SHA')
        parquet = pq.ParquetFile(m5_ref['path'])
        groups = []
        for index in range(241, parquet.metadata.num_row_groups):
            stats = parquet.metadata.row_group(index).column(0).statistics
            if pd.Timestamp(stats.min).value < SCORE_END:
                groups.append(index)
            else:
                break
        need(12 <= len(groups) <= 20 and groups == list(range(241, groups[-1] + 1)),
             'M5_ROW_GROUP_SCOPE')
        pieces = {key: [] for key in ('time', 'high', 'low', 'close', 'atr')}
        for group in groups:
            table = parquet.read_row_group(group, columns=list(pieces))
            for key in pieces:
                col = table[key].to_numpy(zero_copy_only=False)
                pieces[key].append(col.astype('datetime64[ns]').astype(np.int64)
                                   if key == 'time' else col)
        bars = {key: np.concatenate(chunks) for key, chunks in pieces.items()}
        cutoff = int(np.searchsorted(bars['time'], SCORE_END))
        bars = {key: value[:cutoff] for key, value in bars.items()}
        closure = read(structure_plan['closure_authority'])
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
        need(np.all(take < len(structure_time))
             and np.array_equal(structure_time[take], issues - M5_NS), 'STRUCTURE_SCORE_CLOCK')
        hit = np.stack((measured['high_hit'][take], measured['low_hit'][take]), axis=1)
        observed = np.stack((measured['high_observed'][take],
                             measured['low_observed'][take]), axis=1)
        need(bool((~hit | observed).all()), 'STRUCTURE_UNKNOWN_MASK')
        stream = io.BytesIO()
        np.savez_compressed(stream, parent_rows=parent, issue_ns=issues,
                            m1_mid_ohlc_bps=path['mid_ohlc_bps'],
                            m1_observed=path['observed'], structure_hit=hit,
                            structure_observed=observed)
        with (ROOT / 'INPUTS.npz').open('xb') as output:
            output.write(stream.getvalue())
        with np.load(ROOT / 'INPUTS.npz', allow_pickle=False) as check:
            need(np.array_equal(check['parent_rows'], parent)
                 and np.array_equal(check['issue_ns'], issues)
                 and np.array_equal(check['m1_mid_ohlc_bps'], path['mid_ohlc_bps'])
                 and np.array_equal(check['m1_observed'], path['observed'])
                 and np.array_equal(check['structure_hit'], hit)
                 and np.array_equal(check['structure_observed'], observed), 'SERIALIZATION')
        need(subprocess.check_output(('git', 'rev-parse', 'HEAD'), cwd=REPO, text=True).strip() == head
             and not subprocess.check_output(('git', 'status', '--porcelain'), cwd=REPO, text=True),
             'SOURCE_BOOKEND')
        result = {'status': 'COMPLETE_FOLD2_LATER_PANEL_INPUTS_NO_MODEL_OR_FIT',
                  'source_commit': head, 'source_sha256': sha(__file__),
                  'score_rows': 4096, 'issue_axis_sha256': hashlib.sha256(issues.astype('<i8').tobytes()).hexdigest(),
                  'm1_observed_slots': int(path['observed'].sum()),
                  'm1_complete_paths': int(path['full_path'].sum()),
                  'structure_high_observed': int(observed[:, 0].sum()),
                  'structure_low_observed': int(observed[:, 1].sum()),
                  'structure_both_observed': int(observed.all(axis=1).sum()),
                  'input_archive_sha256': sha(ROOT / 'INPUTS.npz'),
                  'input_archive_bytes': (ROOT / 'INPUTS.npz').stat().st_size,
                  'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False,
                  'training_authorized': False, 'economic_edge_proven': False}
        (ROOT / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
        terminal = {'status': 'COMPLETE', 'exit_code': 0,
                    'result_sha256': sha(ROOT / 'RESULT.json')}
    except Exception as error:
        terminal = {'status': 'FAILED', 'exit_code': 1,
                    'error': type(error).__name__ + ':' + str(error)}
        raise
    finally:
        (ROOT / 'TERMINAL.json').write_text(json.dumps(terminal, indent=2) + '\n')


if __name__ == '__main__':
    main()
