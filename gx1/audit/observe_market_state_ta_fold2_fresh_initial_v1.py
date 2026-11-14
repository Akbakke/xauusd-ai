"""One fresh, zero-update fold 2 S/F observation on fixed TRAIN and later panels."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import time

REPO = Path('/home/andre2/src/GX1_CURRENT')
STAGE = 'market_state_ta_fold2_fresh_initial_20261114_003'
ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper()
TRAIN_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_SPAN_VIEW_20261113_001')
LATER_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_LATER_INPUTS_20261113_002')
TF = ('M5', 'M15', 'H1', 'H4', 'D1')
BATCH = 16
PANEL_ROWS = 4096
MAX_SECONDS = 1700
SOURCE_PATHS = (
    'gx1/audit/observe_market_state_ta_fold2_fresh_initial_v1.py',
    'gx1/contracts/entry_anchor_span_feature_inputs_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
    'gx1/contracts/market_state_time_query_native_F_batch_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/market_state_future_displacement_candidate_v1.py',
    'gx1/audit/qualify_market_state_mature_prefix_normalization_v1.py',
    'scripts/gx1_capped_run.sh',
)


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('FOLD2_FRESH_INITIAL_' + reason)


def ref(path):
    path = Path(path)
    need(path.is_absolute() and path.is_file() and not path.is_symlink(), 'REFERENCE_FILE')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def read(binding):
    need(ref(binding['path']) == binding, 'REFERENCE_SHA')
    return json.loads(Path(binding['path']).read_text())


def publish(name, value):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    need(len(data) <= 2 * 1024 * 1024, 'JSON_BYTES')
    path = ROOT / name
    _publish_bytes(path, data)
    return ref(path)


def build_plan(source_base_commit):
    design_ref = ref(REPO / 'docs/TA_FOLD2_INPUT_REVIEW.json')
    design = read(design_ref)['fresh_initial_design']
    train_ref, later_ref = ref(TRAIN_ROOT / 'PLAN.json'), ref(LATER_ROOT / 'PLAN.json')
    train, later = read(train_ref), read(later_ref)
    inputs = {
        'design': design_ref, 'train_plan': train_ref, 'later_plan': later_ref,
        'view': design['model']['feature_view'],
        'normalization': train['inputs']['normalization'],
        'constructor': design['model']['constructor'],
        'priors': design['targets_and_prior']['own_endpoint_priors'],
        'packet': train['inputs']['packet'],
        'provider_plan': train['inputs']['provider_plan'],
        'physical_result': train['inputs']['physical_result'],
        'train_gather_result': later['inputs']['train_gather_result'],
        'later_result': ref(LATER_ROOT / 'RESULT.json'),
        'later_terminal': ref(LATER_ROOT / 'TERMINAL.json'),
        'later_gather_result': ref(LATER_ROOT / 'GATHER_RESULT.json'),
        'later_coverage': ref(LATER_ROOT / 'COVERAGE.json'),
        'train_validation': design['panels']['train']['validation'],
        'later_validation': design['panels']['later']['validation'],
        'target_validation': design['targets_and_prior']['target_count_validation'],
    }
    return {
        'schema_version': 'gx1_ta_fold2_fresh_initial_plan_v1',
        'stage': STAGE, 'source_repo': str(REPO), 'source_base_commit': source_base_commit,
        'output_root': str(ROOT), 'sources': {p: ref(REPO / p) for p in SOURCE_PATHS},
        'inputs': inputs, 'seed': design['model']['seed'], 'batch_size': BATCH,
        'train_rows': PANEL_ROWS, 'later_rows': PANEL_ROWS, 'forwards': 512,
        'max_seconds': MAX_SECONDS, 'memory_bytes': 12 * 1024**3,
        'checkpoint_loads': 0, 'optimizer_constructions': 0, 'backwards': 0,
        'optimizer_updates': 0, 'test_data_used': False,
        'primary': design['predeclared_readout']['primary'],
    }


def gate_fields(binding):
    return dict(status='ADMITTED', plan=binding, max_invocations=1,
                remaining_invocations=1, claimed_invocations=0, completed_invocations=0,
                execution_authorized=True, data_reads_authorized=True,
                checkpoint_reads_authorized=False, training_authorized=False,
                test_data_used=False, relaunch_allowed=False)


def inspect(plan_path, plan_sha, out_dir, *, execute):
    need(Path(plan_path) == ROOT / 'PLAN.json' and Path(out_dir) == ROOT, 'EXACT_PATHS')
    binding = {'path': str(plan_path), 'sha256': plan_sha}
    plan = read(binding)
    need(plan == build_plan(plan['source_base_commit']), 'EXACT_PLAN')
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import (
        contract, require_endpoint_priors,
    )
    design = read(plan['inputs']['design'])['fresh_initial_design']
    need(contract(ta_only=True, dedicated_m5_side=True, m5_equal_observation=True,
                  dedicated_tf_activity=True, fold_id=2)['contract_sha256']
         == design['model']['future_fold2_fit_loss_sha256'], 'FOLD2_FUTURE_FIT_CONTRACT')
    require_endpoint_priors(read(plan['inputs']['priors']), fold_id=2,
                            ta_only=True, dedicated_m5_side=True)
    from gx1.contracts.entry_model_native_input_normalization_v1 import (
        project_ta_input_normalization_contract,
    )
    packet = read(plan['inputs']['packet'])
    normal = read(plan['inputs']['normalization'])
    view = read(plan['inputs']['view'])
    need(normal == project_ta_input_normalization_contract(
        packet['base_normalization'], unobserved_units_profile=None)
         and view['prefix_normalization']['ta_base_normalization_sha256']
         == normal['contract_sha256'], 'TA_NORMALIZATION_PREFLIGHT')
    need(not any((ROOT / n).exists() for n in ('CLAIM.json', 'RESULT.json', 'TERMINAL.json')),
         'ONE_USE')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    need(subprocess.run(['git', 'merge-base', '--is-ancestor', plan['source_base_commit'], head],
                        cwd=REPO, capture_output=True).returncode == 0, 'SOURCE_ANCESTRY')
    policy_ref = ref(REPO / 'NEXT_RUN_POLICY.json')
    policy = read(policy_ref)
    need(all(policy.get(k) is False for k in ('training_enabled', 'native_training_enabled',
         'native_launch_authorized', 'full_epoch_training_allowed', 'full_val_allowed')),
         'TRAINING_CLOSED')
    if execute:
        need(policy.get(STAGE) == gate_fields(binding), 'EXACT_ADMISSION')
        need(subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
             == 'work/gx1-current', 'BRANCH')
        need(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True),
             'CLEAN_SOURCE')
    return dict(plan=plan, plan_ref=binding, policy_ref=policy_ref, head=head)


def panel_positions(np, design, later_plan):
    train = np.concatenate([np.arange(16 * ((i * 2957) // 255),
                                          16 * ((i * 2957) // 255) + 16, dtype=np.int64)
                            for i in range(256)])
    indices = later_plan['scope']['batch_indices']
    need(len(indices) == 256 and later_plan['scope']['fold_id'] == 2, 'LATER_PLAN_PANEL')
    later = np.concatenate([np.arange(16 * b, 16 * b + 16, dtype=np.int64) for b in indices])
    for name, positions in (('train', train), ('later', later)):
        expected = design['panels'][name]['positions_sha256_le_int64']
        need(len(positions) == PANEL_ROWS and len(np.unique(positions)) == PANEL_ROWS
             and hashlib.sha256(positions.astype('<i8').tobytes()).hexdigest() == expected,
             'FIXED_PANEL_' + name.upper())
    return train, later + later_plan['scope']['score_parent_start']


def auc_with_ties(scores, labels, np):
    n = len(labels)
    positives = int(labels.sum())
    if not positives or positives == n:
        return None
    order = np.argsort(scores, kind='stable')
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    _, start, count = np.unique(sorted_scores, return_index=True, return_counts=True)
    midranks = np.repeat(start + (count + 1) / 2, count)
    return float((midranks[sorted_labels].sum(dtype=np.float64)
                  - positives * (positives + 1) / 2) / (positives * (n - positives)))


def direction_readout(raw, truth, issues, np):
    need(raw.shape == (len(issues), 5, 2) and np.isfinite(raw).all(), 'READOUT_SHAPE')
    months = issues.astype('datetime64[ns]').astype('datetime64[M]').astype(str)
    out = {}
    for label, selected in [('all', np.ones(len(issues), dtype=np.bool_)),
                            *((str(m), months == m) for m in np.unique(months))]:
        per_tf = {}
        for j, tf in enumerate(TF):
            known = selected & truth['eligible'][:, j] & truth['mature'][:, j] \
                    & np.isin(truth['allowed'][:, j], (1, 4))
            y = truth['allowed'][known, j] == 4
            x = raw[known, j, 1].astype(np.float64)
            p = 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))
            edges = np.array((0., .2, .4, .6, .8, 1.), dtype=np.float64)
            bins = np.searchsorted(edges[1:-1], p, side='right')
            calibration = [dict(count=int(np.count_nonzero(bins == k)),
                                mean_probability=float(p[bins == k].mean()) if np.any(bins == k) else None,
                                up_fraction=float(y[bins == k].mean()) if np.any(bins == k) else None)
                           for k in range(5)]
            per_tf[tf] = dict(known_side_n=len(y), up_n=int(y.sum()),
                              auc=auc_with_ties(x, y, np),
                              accuracy_at_zero=float(((x >= 0) == y).mean()) if len(y) else None,
                              always_up_accuracy=float(y.mean()) if len(y) else None,
                              brier=float(np.square(p - y).mean()) if len(y) else None,
                              calibration_edges=edges.tolist(), calibration=calibration)
        out[label] = per_tf
    return out


def run(plan_path, plan_sha, out_dir, device):
    need(device == 'cuda', 'CUDA_ONLY')
    q = inspect(plan_path, plan_sha, out_dir, execute=True)
    from gx1.audit.observe_market_state_ta_c171_short_heads_v1 import _runtime_cap
    cap = _runtime_cap()
    import numpy as np
    import torch
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
    from gx1.contracts.entry_anchor_span_feature_inputs_v1 import build_anchor_span_inputs
    from gx1.contracts.entry_exit_feature_surface_v1 import load_m1_feature_surface_times
    from gx1.contracts.entry_model_native_input_normalization_v1 import project_ta_input_normalization_contract
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_endpoint_batch
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

    need(not torch.cuda.is_initialized() and signal.getitimer(signal.ITIMER_REAL) == (0., 0.),
         'INIT_ORDER')
    started = time.monotonic()
    claim = publish('CLAIM.json', dict(plan=q['plan_ref'], policy=q['policy_ref'],
        source_commit=q['head'], cap=cap, no_relaunch_under_this_plan=True,
        test_data_used=False))
    phase = 'INPUTS'
    counts = dict(model_constructions=0, model_forwards=0, feature_batches=0,
                  parent_rows=0, checkpoint_loads=0, optimizer_constructions=0,
                  backwards=0, optimizer_updates=0, test_rows=0)
    groups = None
    mapped = []
    prior_alarm = signal.getsignal(signal.SIGALRM)

    def timeout(*unused):
        raise TimeoutError('FOLD2_FRESH_INITIAL_1800_SECONDS')

    def bookend():
        need(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
             == q['head'] and not subprocess.check_output(
                 ['git', 'status', '--porcelain'], cwd=REPO, text=True), 'SOURCE_BOOKEND')
        for binding in (q['plan_ref'], q['policy_ref'], *q['plan']['sources'].values(),
                        *q['plan']['inputs'].values()):
            need(ref(binding['path']) == binding, 'REFERENCE_BOOKEND')

    try:
        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
        inputs = q['plan']['inputs']
        design = read(inputs['design'])['fresh_initial_design']
        train_plan, later_plan = read(inputs['train_plan']), read(inputs['later_plan'])
        train_result, later_result = read(inputs['physical_result']), read(inputs['later_result'])
        later_terminal, coverage = read(inputs['later_terminal']), read(inputs['later_coverage'])
        train_gather, later_gather = read(inputs['train_gather_result']), read(inputs['later_gather_result'])
        need(train_gather['status'] == 'COMPLETE_FULL_FOLD2_TRAIN_INPUT_GATHER'
             and train_gather['source_unchanged'] is True
             and later_result['status'] == later_terminal['status'] == 'COMPLETE'
             and later_terminal['exit_code'] == 0 and later_terminal['result'] == inputs['later_result']
             and later_terminal['gather_result'] == inputs['later_gather_result']
             and later_gather['status'] == coverage['status'] == 'COMPLETE'
             and later_result['source_unchanged'] is True
             and later_gather['source_unchanged'] is True
             and later_gather['counts']['parent_rows'] == 4096
             and not any(x['test_data_used'] for x in (train_gather, later_gather, later_result)),
             'QUALIFIED_INPUTS')
        view, packet = read(inputs['view']), read(inputs['packet'])
        normalization, constructor, priors = [read(inputs[k]) for k in ('normalization', 'constructor', 'priors')]
        need(view['prefix']['rows'] == 47338 and view['dedicated_m5_side'] is True
             and view['dedicated_tf_activity'] is True
             and view['full_ta_feature_profile'] is True
             and view['prefix_normalization']['ta_base_normalization_sha256'] == normalization['contract_sha256']
             and normalization == project_ta_input_normalization_contract(
                 packet['base_normalization'], unobserved_units_profile=None), 'FOLD2_VIEW')
        train_pos, later_pos = panel_positions(np, design, later_plan)
        provider = mature._provider_module()
        provider_plan = read(inputs['provider_plan'])
        info = provider.validate_metadata(provider_plan)
        axes, _ = provider.read_axes(provider_plan, info, np)
        need(len(axes['parent_rows']) == view['full_axes']['rows'], 'FULL_AXES')
        for array in axes.values():
            array.setflags(write=False)
        for key in ('parent_rows', 'bar_start_ns', 'decision_ns'):
            need(hashlib.sha256(axes[key].astype('<i8').tobytes()).hexdigest()
                 == view['full_axes'][key + '_sha256'], 'FULL_AXIS_' + key)
        state_bound, future_bound = load_bound_five_endpoint_targets(), load_bound_future_displacement()
        oof = state_bound.mature_oof_axes(fold_id=2)
        need(np.array_equal(axes['parent_rows'][:47338], oof['fit_parent_rows'])
             and np.array_equal(axes['parent_rows'][47602:64890], oof['score_parent_rows'])
             and np.array_equal(axes['decision_ns'][:47338], oof['fit_decision_ns'])
             and np.array_equal(axes['decision_ns'][47602:64890], oof['score_decision_ns']),
             'FOLD2_TARGET_AXES')
        runtime = later_plan['replay_plan']
        groups = provider.map_reuse(info['reuse'], np)
        source, current = groups['source050'], groups['current051']
        times = load_m1_feature_surface_times(Path(runtime['local_clock_path']),
            context='FOLD2_FRESH_INITIAL_FULL_PARENT_CLOCK', expected_bar_seconds=300).asi8
        positions = np.searchsorted(times, axes['bar_start_ns']).astype(np.int64)
        need(np.all(positions < len(times))
             and np.array_equal(times[positions], axes['bar_start_ns']), 'LOCAL_CLOCK')
        times.setflags(write=False)
        positions.setflags(write=False)
        seq = dict(signal=source['signal'], times_ns=times,
                   sample_positions=positions, audit=info['audit'])
        values, clocks = {}, {}
        for tf in TF:
            values[tf] = np.load(runtime['caches'][tf]['feats']['path'], mmap_mode='r', allow_pickle=False)
            clocks[tf] = np.load(runtime['caches'][tf]['ts']['path'], mmap_mode='r', allow_pickle=False)
            mapped.extend((values[tf], clocks[tf]))
        names = packet['parent_prefix_normalization']['base_normalization']['surfaces']
        def overlay(result, *, later):
            payloads = {}
            for lane in ('local', *TF):
                binding = result['surfaces'][lane]['artifact']
                need(ref(binding['path']) == binding, 'SIDECAR_' + lane)
                with np.load(binding['path'], allow_pickle=False) as archive:
                    payloads[lane] = {k: archive[k] for k in archive.files}
            return build_anchor_span_inputs(view, entry_sequence_source=seq,
                snap_x=current['snap'], ctx_cont=current['ctx_cont'], ctx_cat=current['ctx_cat'],
                parent_signal_names=names['signal']['field_names'],
                parent_mtf_names=names['mtf_m5']['field_names'],
                parent_mtf_values=values, parent_mtf_times=clocks,
                span_payloads=payloads,
                physical_populations=(coverage['physical_populations'] if later else
                                      {lane: x['population'] for lane, x in result['surfaces'].items()}),
                unobserved_units_profile=None,
                **({'later_coverage': coverage, 'later_axes': axes} if later else {}))
        overlays = {'TRAIN': overlay(train_result, later=False),
                    'LATER': overlay(later_result, later=True)}
        phase = 'FRESH_MODEL'
        cuda = torch.device('cuda', 0)
        trainer._set_deterministic(q['plan']['seed'], cuda, trainer.DETERMINISTIC_FP32)
        backbone = views.build_native_direction_research_model(view,
            prefix_normalization=packet, constructor_metadata=constructor,
            device=torch.device('cpu'), unobserved_units_profile=None)
        model = NativeFutureDualModel(backbone).to(cuda).eval()
        counts['model_constructions'] = 1
        need(model.research_objective['contract_sha256']
             == design['model']['default_model_objective_sha256'], 'MODEL_CONTRACT')
        initial_sha = canonical_model_state_sha256(model.state_dict())
        readouts, measurements, captures = {}, {}, {}
        for label, selected, partition in (('TRAIN', train_pos, 'fit'),
                                           ('LATER', later_pos, 'score')):
            phase = label + '_FORWARDS'
            rows, issues = axes['parent_rows'][selected], axes['decision_ns'][selected]
            truth = {task: bound.gather(rows, issues, fold_id=2, partition=partition)
                     for task, bound in (('state', state_bound), ('future', future_bound))}
            if partition == 'fit':
                need(all(bool(value['mature'].all()) for value in truth.values()),
                     'TRAIN_TARGET_MATURITY')
            raw = {task: np.empty((PANEL_ROWS, 5, 2), dtype=np.float32)
                   for task in ('state', 'future')}
            total = None
            for left in range(0, PANEL_ROWS, BATCH):
                need(time.monotonic() - started < MAX_SECONDS - 30, 'TIME_BUDGET')
                right = left + BATCH
                batch_rows, batch_issues = rows[left:right], issues[left:right]
                gathered = views.gather_native_direction_feature_batch(view,
                    parent_rows=axes['parent_rows'], bar_start_ns=axes['bar_start_ns'],
                    decision_ns=axes['decision_ns'], entry_sequence_source=seq,
                    snap_x=current['snap'], ctx_cont=current['ctx_cont'], ctx_cat=current['ctx_cat'],
                    multi_tf_sources=overlays[label].mtf_sources, gc_surfaces=None, gc_layout=None,
                    batch_parent_rows=batch_rows, unobserved_units_profile=None,
                    anchor_span_inputs=overlays[label])
                need(gathered['witness']['parent_rows'] == batch_rows.tolist()
                     and gathered['witness']['decision_ns'] == batch_issues.tolist(),
                     'GATHER_AXES')
                features = {key: torch.from_numpy(value).to(cuda)
                            for key, value in gathered['features'].items()}
                report = evaluate_endpoint_batch(model, features, state_bound, future_bound,
                    batch_rows, batch_issues, fold_id=2, partition=partition,
                    future_weight=1., frozen_priors=priors)
                total = objective.merge_endpoint_measurements(total, report['measurement'])
                for task in raw:
                    raw[task][left:right] = report[task + '_activity_side_logits']
                counts['feature_batches'] += 1
                counts['model_forwards'] += 1
                counts['parent_rows'] += BATCH
                del gathered, features, report
            need(total['rows'] == PANEL_ROWS and total['batches'] == 256,
                 'MEASUREMENT_COUNTS')
            measurements[label] = objective.finalize_endpoint_measurement(total)
            readouts[label] = {task: direction_readout(raw[task], truth[task], issues, np)
                               for task in raw}
            stream = io.BytesIO()
            np.savez(stream, parents=rows, issues=issues,
                     state_raw=raw['state'], future_raw=raw['future'],
                     **{task + '_' + key: truth[task][key]
                        for task in ('state', 'future')
                        for key in ('allowed', 'eligible', 'mature')})
            blob = stream.getvalue()
            need(len(blob) <= 2 * 1024 * 1024, 'CAPTURE_BYTES')
            path = ROOT / (label + '_READOUT.npz')
            _publish_bytes(path, blob)
            captures[label] = ref(path)
        phase = 'BOOKEND'
        need(counts == dict(model_constructions=1, model_forwards=512,
             feature_batches=512, parent_rows=8192, checkpoint_loads=0,
             optimizer_constructions=0, backwards=0, optimizer_updates=0,
             test_rows=0), 'EXACT_COUNTS')
        need(canonical_model_state_sha256(model.state_dict()) == initial_sha
             and all(p.grad is None for p in model.parameters()), 'ZERO_UPDATE_MODEL_STATE')
        bookend()
        result = publish('RESULT.json', dict(status='COMPLETE_FRESH_INITIAL',
            plan=q['plan_ref'], claim=claim, source_commit=q['head'],
            source_unchanged=True, model_state_sha256=initial_sha,
            optimizer_updates=0, counts=counts, captures=captures,
            measurements=measurements, readouts=readouts,
            elapsed_seconds=time.monotonic() - started,
            train_only_fit_targets=True, later_labels_used_for_fit=False,
            predictive_edge_proven=False, economic_value_measured=False,
            test_data_used=False))
        publish('TERMINAL.json', dict(status='COMPLETE', exit_code=0, result=result,
            source_unchanged=True, counts=counts, relaunch_allowed=False,
            test_data_used=False))
        return dict(result=result, elapsed_seconds=time.monotonic()-started)
    except BaseException as exc:
        publish('TERMINAL.json', dict(status='FAILED_CONSUMED', exit_code=1,
            phase=phase, error=repr(exc), completed_counts=counts,
            relaunch_allowed=False, test_data_used=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_alarm)
        if groups is not None:
            for group in groups.values():
                for array in group.values():
                    provider.flush_pages(array)
        for array in mapped:
            provider.flush_pages(array)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--market-state-ta-fold2-fresh-initial-plan', type=Path, required=True)
    parser.add_argument('--market-state-ta-fold2-fresh-initial-plan-sha256', required=True)
    parser.add_argument('--device', choices=('cuda',), required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(argv)
    kw = dict(plan_path=args.market_state_ta_fold2_fresh_initial_plan,
              plan_sha=args.market_state_ta_fold2_fresh_initial_plan_sha256,
              out_dir=args.out_dir)
    answer = (dict(status='PASS_METADATA_ONLY', plan=inspect(**kw, execute=False)['plan_ref'],
                   model_constructions=0, claims=0)
              if args.preflight else run(**kw, device=args.device))
    print(json.dumps(answer, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
