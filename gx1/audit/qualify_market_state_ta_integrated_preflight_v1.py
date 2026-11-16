"""One capped, read-only Fold2 TRAIN-batch preflight for the integrated TA candidate."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/TA_INTEGRATED_PREFLIGHT_20261116_001')
TRAIN = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_SPAN_VIEW_20261113_001')
LATER = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_LATER_INPUTS_20261113_002')
TF = ('M5', 'M15', 'H1', 'H4', 'D1')
LEFT, RIGHT = 128, 144
END = int(__import__('numpy').datetime64('2025-01-01', 'ns').astype('int64'))


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_INTEGRATED_PREFLIGHT_' + reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(ref):
    path = Path(ref['path'])
    need(path.is_file() and not path.is_symlink() and sha(path) == ref['sha256'], 'SOURCE_REF')
    return json.loads(path.read_text())


def head_and_clean():
    head = subprocess.check_output(('git', 'rev-parse', 'HEAD'), cwd=REPO, text=True).strip()
    need(subprocess.check_output(('git', 'branch', '--show-current'), cwd=REPO, text=True).strip()
         == 'work/gx1-current', 'BRANCH')
    need(not subprocess.check_output(('git', 'status', '--porcelain'), cwd=REPO, text=True), 'CLEAN_SOURCE')
    return head


def main():
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
    from gx1.audit.measure_ta_m5_structure_candidate_v1 import measure_candidate
    from gx1.contracts import entry_anchor_span_feature_view_v1 as span
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts.entry_anchor_span_feature_inputs_v1 import build_anchor_span_inputs
    from gx1.contracts.entry_exit_feature_surface_v1 import load_m1_feature_surface_times
    from gx1.contracts.gx1_capped_execution_v1 import (require_capped_cpu_audit_execution,
                                                         require_capped_lock_ancestry)
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_ta_integrated_model_v1 import (IntegratedTAMarketStateModel,
                                                                     masked_structure_hit_sums)
    from gx1.contracts.market_state_ta_m1_issue_path_v1 import M1IssuePathTape
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import combined_loss
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K

    cap = require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    torch.set_num_threads(1)
    head = head_and_clean()
    policy = json.loads((REPO / 'NEXT_RUN_POLICY.json').read_text())
    need(policy['training_enabled'] is policy['native_training_enabled'] is False
         and policy['full_epoch_training_allowed'] is False, 'TRAINING_CLOSED')
    need(not ROOT.exists(), 'ONE_USE')
    ROOT.mkdir()
    claim = {'source_commit': head, 'source_sha256': sha(__file__), 'cap': cap,
             'scope': 'one_read_only_fold2_train_batch_128_143',
             'claimed_utc': datetime.now(timezone.utc).isoformat()}
    (ROOT / 'CLAIM.json').write_text(json.dumps(claim, indent=2) + '\n')
    try:
        train = json.loads((TRAIN / 'PLAN.json').read_text())
        later = json.loads((LATER / 'PLAN.json').read_text())
        packet = read(train['inputs']['packet'])
        old_view = read(later['inputs']['view'])
        constructor = read(train['inputs']['constructor'])
        physical = read(train['inputs']['physical_result'])
        view = span.bind_anchor_span_feature_view(
            prefix_normalization=packet, prefix_normalization_ref=train['inputs']['packet'],
            constructor_metadata=constructor, constructor_metadata_ref=train['inputs']['constructor'],
            source_bindings=old_view['source_bindings'], max_batch_rows=16,
            unobserved_units_profile=None, full_feature_profile=True,
            full_ta_feature_profile=True, market_state_time_query_v2=True,
            dedicated_m5_side=True, dedicated_non_m5_side=True,
            dedicated_tf_activity=True, direct_tf_hidden=True)
        need(view['contract_sha256'] == '8056fe96b4340bccc149da7f22dde85e4ac1b62c5e58c874ec7fbb771c1dff1a',
             'VIEW_IDENTITY')
        provider = mature._provider_module()
        provider_plan = read(train['inputs']['provider_plan'])
        info = provider.validate_metadata(provider_plan)
        axes, _ = provider.read_axes(provider_plan, info, np)
        need(len(axes['parent_rows']) == view['full_axes']['rows'], 'AXES')
        rows = axes['parent_rows'][LEFT:RIGHT]
        issues = axes['decision_ns'][LEFT:RIGHT]
        need(len(rows) == 16 and np.array_equal(rows, np.arange(LEFT, RIGHT, dtype=np.int64))
             and np.all(issues < END), 'TRAIN_SCOPE')
        for key in ('parent_rows', 'bar_start_ns', 'decision_ns'):
            need(hashlib.sha256(axes[key].astype('<i8').tobytes()).hexdigest()
                 == view['full_axes'][key + '_sha256'], 'AXIS_HASH_' + key)
        groups = provider.map_reuse(info['reuse'], np)
        source, current = groups['source050'], groups['current051']
        runtime = later['replay_plan']
        times = load_m1_feature_surface_times(Path(runtime['local_clock_path']),
            context='TA_INTEGRATED_TRAIN_CLOCK', expected_bar_seconds=300).asi8
        positions = np.searchsorted(times, axes['bar_start_ns']).astype(np.int64)
        need(np.all(positions < len(times))
             and np.array_equal(times[positions], axes['bar_start_ns']), 'LOCAL_CLOCK')
        seq = dict(signal=source['signal'], times_ns=times,
                   sample_positions=positions, audit=info['audit'])
        values, clocks, payloads = {}, {}, {}
        for tf in TF:
            values[tf] = np.load(runtime['caches'][tf]['feats']['path'], mmap_mode='r', allow_pickle=False)
            clocks[tf] = np.load(runtime['caches'][tf]['ts']['path'], mmap_mode='r', allow_pickle=False)
        for lane in ('local', *TF):
            artifact = physical['surfaces'][lane]['artifact']
            need(sha(artifact['path']) == artifact['sha256'], 'SPAN_SOURCE_' + lane)
            with np.load(artifact['path'], allow_pickle=False) as archive:
                payloads[lane] = {key: archive[key] for key in archive.files}
        names = packet['parent_prefix_normalization']['base_normalization']['surfaces']
        overlay = build_anchor_span_inputs(view, entry_sequence_source=seq,
            snap_x=current['snap'], ctx_cont=current['ctx_cont'], ctx_cat=current['ctx_cat'],
            parent_signal_names=names['signal']['field_names'],
            parent_mtf_names=names['mtf_m5']['field_names'],
            parent_mtf_values=values, parent_mtf_times=clocks, span_payloads=payloads,
            physical_populations={lane: physical['surfaces'][lane]['population'] for lane in ('local', *TF)},
            unobserved_units_profile=None)
        gathered = views.gather_native_direction_feature_batch(view,
            parent_rows=axes['parent_rows'], bar_start_ns=axes['bar_start_ns'],
            decision_ns=axes['decision_ns'], entry_sequence_source=seq,
            snap_x=current['snap'], ctx_cont=current['ctx_cont'], ctx_cat=current['ctx_cat'],
            multi_tf_sources=overlay.mtf_sources, gc_surfaces=None, gc_layout=None,
            batch_parent_rows=rows, unobserved_units_profile=None, anchor_span_inputs=overlay)
        need(gathered['witness']['parent_rows'] == rows.tolist()
             and gathered['witness']['decision_ns'] == issues.tolist(), 'GATHER_CLOCK')
        features = {key: torch.from_numpy(value) for key, value in gathered['features'].items()}

        parity = json.loads((REPO / 'docs/TA_M1_PATH_ARCHIVE_PARITY_REVIEW_20261116.json').read_text())
        m1_ref = parity['source_refs']['S_train_m1_quotes_parquet']
        need(sha(m1_ref['path']) == m1_ref['sha256'], 'M1_SOURCE_SHA')
        raw = pq.ParquetFile(m1_ref['path']).read_row_group(1820,
            columns=['time', 'open', 'high', 'low', 'close'])
        m1_time = raw['time'].to_numpy(zero_copy_only=False).astype('datetime64[ns]').astype(np.int64)
        m1_ohlc = np.stack([raw[key].to_numpy(zero_copy_only=False)
                            for key in ('open', 'high', 'low', 'close')], axis=1)
        path = M1IssuePathTape(m1_time, m1_ohlc).gather(issues)
        need(bool(path['full_path'].all()) and path['observed'].shape == (16, 5), 'M1_PATH')

        structure_plan = json.loads((REPO / 'configs/research/TA_M5_STRUCTURE_FOLD2_AUDIT_20261116_001.json').read_text())
        m5_ref = structure_plan['input']
        need(sha(m5_ref['path']) == m5_ref['sha256_expected_from_manifest'], 'M5_SOURCE_SHA')
        parquet = pq.ParquetFile(m5_ref['path'])
        pieces = {key: [] for key in ('time', 'high', 'low', 'close', 'atr')}
        for group in range(241, 253):
            table = parquet.read_row_group(group, columns=list(pieces))
            for key in pieces:
                column = table[key].to_numpy(zero_copy_only=False)
                pieces[key].append(column.astype('datetime64[ns]').astype(np.int64)
                                   if key == 'time' else column)
        bars = {key: np.concatenate(chunks) for key, chunks in pieces.items()}
        cutoff = int(np.searchsorted(bars['time'], END))
        bars = {key: value[:cutoff] for key, value in bars.items()}
        close_authority = read(structure_plan['closure_authority'])
        closure = np.array([[pd.Timestamp(item['closure_start_utc']).value,
                             pd.Timestamp(item['closure_end_utc_exclusive']).value]
                            for item in close_authority['intervals']
                            if item['successor_across_gap_allowed'] is True],
                           dtype=np.int64).reshape(-1, 2)
        issue_rows = np.flatnonzero(bars['time'] >= int(np.datetime64('2024-05-01', 'ns').astype(np.int64))).astype(np.int64)
        measured = measure_candidate(bars['time'], bars['high'], bars['low'],
            bars['close'], bars['atr'], issue_rows,
            fold_target_end_ns=END, known_closure_ns=closure)
        actual = np.searchsorted(bars['time'][issue_rows], issues - 300_000_000_000)
        need(np.array_equal(bars['time'][issue_rows[actual]], issues - 300_000_000_000),
             'STRUCTURE_ISSUE_CLOCK')
        hits = np.stack((measured['high_hit'][actual], measured['low_hit'][actual]), axis=1)
        observed = np.stack((measured['high_observed'][actual], measured['low_observed'][actual]), axis=1)

        torch.manual_seed(20260911)
        state = views.build_native_direction_research_model(view,
            prefix_normalization=packet, constructor_metadata=constructor,
            device=torch.device('cpu'), unobserved_units_profile=None)
        model = IntegratedTAMarketStateModel(NativeFutureDualModel(state)).eval()
        query = torch.tensor([NATIVE_K] * 16, dtype=torch.int64)
        output = model(features, query_k=query,
            m1_path_bps=torch.from_numpy(path['mid_ohlc_bps']),
            m1_observed=torch.from_numpy(path['observed']))
        baseline = model.dual(**features, query_k=query)
        need(all(torch.equal(output[key], value) for key, value in baseline.items()),
             'ZERO_GATE_BASELINE_PARITY')
        state_truth = load_bound_five_endpoint_targets().gather(rows, issues, fold_id=2, partition='fit')
        future_truth = load_bound_future_displacement().gather(rows, issues, fold_id=2, partition='fit')
        sf = combined_loss(state_raw=output['state_activity_side_logits'],
            future_raw=output['future_activity_side_logits'],
            state_truth=state_truth, future_truth=future_truth,
            state_query_k=query, future_weight=1.0, ta_only=True,
            dedicated_m5_side=True, dedicated_non_m5_side=True,
            dedicated_tf_activity=True, direct_tf_hidden=True, fold_id=2)
        structure = masked_structure_hit_sums(output['m5_structure_hit_logits'],
            torch.from_numpy(hits), torch.from_numpy(observed))
        need(bool(torch.isfinite(sf['loss'])) and bool(torch.isfinite(structure['loss_sums']).all()),
             'LOSS_FINITE')
        sf['loss'].backward(retain_graph=True)
        sf_gate_grad = int(torch.count_nonzero(model.m1_tf_gate.weight.grad).item())
        need(sf_gate_grad > 0, 'SF_M1_GRADIENT')
        model.zero_grad(set_to_none=True)
        structure['loss_sums'].sum().backward()
        structure_gate_grad = int(torch.count_nonzero(model.m1_tf_gate.weight.grad).item())
        structure_head_grad = int(torch.count_nonzero(model.m5_structure_head.weight.grad).item())
        need(structure_gate_grad > 0 and structure_head_grad > 0, 'STRUCTURE_GRADIENT')
        result = {'status': 'PASS_ONE_ACTUAL_TRAIN_BATCH_FORWARD_AND_GRADIENT_NO_FIT',
                  'source_commit': head, 'view_sha256': view['contract_sha256'],
                  'model_contract_sha256': model.research_objective['contract_sha256'],
                  'batch_positions': [LEFT, RIGHT], 'rows': 16, 'model_forwards': 2,
                  'optimizer_constructions': 0, 'optimizer_steps': 0, 'checkpoint_reads': 0,
                  'M1_full_paths': int(path['full_path'].sum()),
                  'M1_observed_slots': int(path['observed'].sum()),
                  'S_known_sides_by_tf': [int(np.isin(state_truth['allowed'][:, j], (1, 4)).sum()) for j in range(5)],
                  'F_known_sides_by_tf': [int(np.isin(future_truth['allowed'][:, j], (1, 4)).sum()) for j in range(5)],
                  'structure_high_observed_batch': int(observed[:, 0].sum()),
                  'structure_low_observed_batch': int(observed[:, 1].sum()),
                  'corrected_structure_full_fold_high_observed': int(measured['high_observed'].sum()),
                  'corrected_structure_full_fold_low_observed': int(measured['low_observed'].sum()),
                  'corrected_structure_full_fold_pair_class_known': int(np.isin(measured['pair_class'], (0, 1, 2, 3)).sum()),
                  'S_F_diagnostic_loss': float(sf['loss'].detach()),
                  'structure_diagnostic_loss_sum': float(structure['loss_sums'].sum().detach()),
                  'S_F_M1_gate_nonzero_gradient_cells': sf_gate_grad,
                  'structure_M1_gate_nonzero_gradient_cells': structure_gate_grad,
                  'structure_head_nonzero_gradient_cells': structure_head_grad,
                  'test_data_used': False, 'training_authorized': False,
                  'online_delivery_parity_verified': False, 'economic_edge_proven': False,
                  'source_unchanged': head_and_clean() == head}
        need(result['source_unchanged'], 'SOURCE_BOOKEND')
        (ROOT / 'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        terminal = {'status': 'COMPLETE', 'exit_code': 0, 'source_commit': head,
                    'result_sha256': sha(ROOT / 'RESULT.json')}
    except Exception as error:
        terminal = {'status': 'FAILED', 'exit_code': 1, 'source_commit': head,
                    'error': type(error).__name__ + ':' + str(error)}
        raise
    finally:
        (ROOT / 'TERMINAL.json').write_text(json.dumps(terminal, indent=2) + '\n')


if __name__ == '__main__':
    main()
