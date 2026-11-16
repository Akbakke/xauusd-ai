"""One capped, read-only Fold2 TRAIN-batch preflight for the integrated TA candidate."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/TA_INTEGRATED_PREFLIGHT_20261116_004')
TRAIN = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_SPAN_VIEW_20261113_001')
LATER = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_LATER_INPUTS_20261113_002')
TF = ('M5', 'M15', 'H1', 'H4', 'D1')
LEFT, RIGHT = 36000, 36016
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
    import torch
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
    from gx1.contracts import entry_anchor_span_feature_view_v1 as span
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts.entry_anchor_span_feature_inputs_v1 import build_anchor_span_inputs
    from gx1.contracts.entry_exit_feature_surface_v1 import load_m1_feature_surface_times
    from gx1.contracts.gx1_capped_execution_v1 import (require_capped_cpu_audit_execution,
                                                         require_capped_lock_ancestry)
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_ta_integrated_model_v1 import IntegratedTAMarketStateModel
    from gx1.contracts.market_state_ta_integrated_objective_v1 import integrated_loss
    from gx1.contracts.market_state_time_query_binding_v1 import (load_bound_time_query_targets,
                                                                  sampled_query_k)
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel

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
             'scope': 'one_read_only_fold2_train_objective_batch_36000_36015',
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
        for array in (*axes.values(), times, positions, source['signal'],
                      current['snap'], current['ctx_cont'], current['ctx_cat']):
            array.setflags(write=False)
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

        tape_review = json.loads((REPO / 'docs/TA_INTEGRATED_FOLD2_FIT_INPUTS_REVIEW_20261116.json').read_text())
        tape_ref = tape_review['input_archive']
        need(sha(tape_ref['path']) == tape_ref['sha256'], 'FIT_TAPE_SHA')
        with np.load(tape_ref['path'], allow_pickle=False) as tape:
            need(np.array_equal(tape['parent_rows'][LEFT:RIGHT], rows)
                 and np.array_equal(tape['issue_ns'][LEFT:RIGHT], issues),
                 'FIT_TAPE_AXIS')
            m1_bps = tape['m1_mid_ohlc_bps'][LEFT:RIGHT].copy()
            m1_observed = tape['m1_observed'][LEFT:RIGHT].copy()
            hits = tape['structure_hit'][LEFT:RIGHT].copy()
            observed = tape['structure_observed'][LEFT:RIGHT].copy()
        need(m1_bps.shape == (16, 5, 4) and m1_observed.shape == (16, 5)
             and hits.shape == observed.shape == (16, 2), 'TAPE_SHAPES')

        torch.manual_seed(20260911)
        state = views.build_native_direction_research_model(view,
            prefix_normalization=packet, constructor_metadata=constructor,
            device=torch.device('cpu'), unobserved_units_profile=None)
        model = IntegratedTAMarketStateModel(NativeFutureDualModel(state)).eval()
        initial_model_sha = canonical_model_state_sha256(model.state_dict())
        query = torch.from_numpy(sampled_query_k(issues, pass_index=0, seed=20260911))
        output = model(features, query_k=query,
            m1_path_bps=torch.from_numpy(m1_bps),
            m1_observed=torch.from_numpy(m1_observed))
        baseline = model.dual(**features, query_k=query)
        need(all(torch.equal(output[key], value) for key, value in baseline.items()),
             'ZERO_GATE_BASELINE_PARITY')
        state_truth = load_bound_time_query_targets().gather_fit_sampled(
            rows, issues, fold_id=2, pass_index=0, seed=20260911)
        future_truth = load_bound_future_displacement().gather(
            rows, issues, fold_id=2, partition='fit')
        report = integrated_loss(model_output=output,
            state_truth=state_truth, future_truth=future_truth,
            state_query_k=query, structure_hit=hits,
            structure_observed=observed)
        need(not report['skip_update'] and bool(torch.isfinite(report['loss'])),
             'INTEGRATED_LOSS')
        report['loss'].backward()
        gate_grad = int(torch.count_nonzero(model.m1_tf_gate.weight.grad).item())
        state_heads = (model.dual.state_model.m5_side_head,
                       *(model.dual.state_model.non_m5_side_heads[tf] for tf in TF[1:]))
        future_heads = (model.dual.future_m5_side_head,
                        *(model.dual.future_non_m5_side_heads[tf] for tf in TF[1:]))
        state_side_grads = [int(torch.count_nonzero(head.weight.grad).item()) for head in state_heads]
        future_side_grads = [int(torch.count_nonzero(head.weight.grad).item()) for head in future_heads]
        structure_head_grad = int(torch.count_nonzero(model.m5_structure_head.weight.grad).item())
        need(gate_grad > 0 and all(state_side_grads) and all(future_side_grads)
             and structure_head_grad > 0, 'INTEGRATED_GRADIENTS')
        known = lambda truth: [int((truth['eligible'][:, j] & truth['mature'][:, j]
                                  & np.isin(truth['allowed'][:, j], (1, 4))).sum())
                               for j in range(5)]
        result = {'status': 'PASS_ACTUAL_FOLD2_TRAIN_INTEGRATED_OBJECTIVE_NO_FIT',
                  'source_commit': head, 'view_sha256': view['contract_sha256'],
                  'model_contract_sha256': model.research_objective['contract_sha256'],
                  'objective_contract_sha256': report['contract_sha256'],
                  'initial_model_state_sha256_CPU': initial_model_sha,
                  'batch_positions': [LEFT, RIGHT], 'rows': 16, 'model_forwards': 2,
                  'optimizer_constructions': 0, 'optimizer_steps': 0, 'checkpoint_reads': 0,
                  'M1_full_paths': int(m1_observed.all(axis=1).sum()),
                  'M1_observed_slots': int(m1_observed.sum()),
                  'S_known_sides_by_tf': known(state_truth),
                  'F_known_sides_by_tf': known(future_truth),
                  'structure_high_observed_batch': int(observed[:, 0].sum()),
                  'structure_low_observed_batch': int(observed[:, 1].sum()),
                  'integrated_loss': float(report['loss'].detach()),
                  'S_F_loss': float(report['S_F_loss'].detach()),
                  'structure_mean_loss': float(report['structure_loss'].detach()),
                  'M1_gate_nonzero_gradient_cells': gate_grad,
                  'S_side_head_nonzero_gradient_cells_by_tf': state_side_grads,
                  'F_side_head_nonzero_gradient_cells_by_tf': future_side_grads,
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
