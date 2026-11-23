"""Fresh V2 initial scoring helpers; old workers and model inputs are unchanged.

No checkpoint, optimizer, construction, launch or publication belongs here.
The caller owns the fresh model and guards. A full score always forwards the
entire authenticated TRAIN clock, including rows without known target labels.
"""
from __future__ import annotations

import numpy as np

from gx1.audit import native_future_state_training_v1 as training
from gx1.audit.build_native_m5_ta_feature_cache_v1 import load_snapshot
from gx1.contracts import market_state_ta_integrated_model_v1 as model_owner
from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
from gx1.contracts.native_future_state_provider_hlc_v1 import require_provider_hlc_state_contract

core = training.core
read = training.read
load_provider_hlc_target_truth = training.load_provider_hlc_target_truth
slice_provider_hlc_truth = training.slice_provider_hlc_truth
ROWS, TRAIN_SHA = training.TRAIN_ROWS, training.TRAIN_SHA
START, UPPER = training.START, training.UPPER
OUTPUT_KEYS = ('activity_logits', 'side_logits')


def need(ok, reason):
    if not bool(ok):
        raise ValueError('NATIVE_PROVIDER_HLC_BASELINE_' + reason)


def load_context(meta):
    """Load the unchanged physical inputs and one authenticated whole V2 truth."""
    need(type(meta) is dict, 'METADATA')
    target = require_provider_hlc_state_contract(meta['target_contract'])
    provider = load_snapshot(meta['source_snapshot'])
    normalization = read(meta['normalization'])
    prior = objective.require_native_future_state_train_prior(read(meta['prior']),
        target_contract=target)
    positions = np.flatnonzero((provider.issue_ns >= START) &
                              (provider.issue_ns < UPPER)).astype(np.int64)
    source_rows = provider.source_row[positions]
    issues = provider.issue_ns[positions]
    need(len(positions) == ROWS and core.digest(issues) == TRAIN_SHA
        and provider.input_ready[positions].all()
        and provider.input_binding(normalization) == meta['input_binding']
        and core.digest(source_rows) == normalization['fit_population']['selected_source_rows_sha256'],
        'PHYSICAL_WHOLE_TRAIN_INPUTS')
    pop = prior['fit_population']
    need(pop['issue_rows'] == ROWS and pop['issue_ns_sha256'] == TRAIN_SHA
        and pop['cutoff_ns'] == UPPER, 'MATCHED_PRIOR_POPULATION')
    truth = load_provider_hlc_target_truth(meta['dataset'], expected_issue_ns=issues,
        target_contract=target, fold_upper_ns=UPPER)
    truth_sha = objective.native_future_state_truth_sha256(truth, target_contract=target)
    need(truth_sha == pop['truth_sha256'], 'MATCHED_PRIOR_TRUTH')
    positions.flags.writeable = False
    source_rows.flags.writeable = False
    return {'provider': provider, 'normalization': normalization,
        'input_binding': meta['input_binding'], 'target_contract': target,
        'prior': prior, 'positions': positions, 'source_rows': source_rows,
        'truth': truth, 'truth_sha256': truth_sha}


def packet(context, take):
    """Gather each genuine feature once; certificate metadata stays in truth."""
    import torch
    need(isinstance(take, np.ndarray) and take.dtype == np.int64 and take.ndim == 1
        and len(take) > 0 and np.all(take >= 0)
        and np.all(take < len(context['source_rows'])) and np.all(np.diff(take) > 0),
        'PACKET_ROWS')
    raw = context['provider'].gather(context['source_rows'][take], max_rows=len(take))
    expected = model_owner._native_bar_feature_keys(context['input_binding'])
    need(type(raw) is dict and len(expected) == 9 and set(raw) == set(expected),
        'FEATURE_KEYS')
    need(all(isinstance(value, np.ndarray) and value.ndim > 0
        and value.shape[0] == len(take) and value.dtype in (np.dtype('float32'), np.dtype('int64'))
        and np.isfinite(value).all() for value in raw.values()), 'FEATURE_ARRAYS')
    features = {key: torch.from_numpy(value) for key, value in raw.items()}
    need(all(value.device.type == 'cpu' for value in features.values()), 'FEATURE_CPU')
    truth = slice_provider_hlc_truth(context['truth'], take,
        target_contract=context['target_contract'], fold_upper_ns=UPPER)
    return {'features': features, 'truth': truth}


def score_initial(model, context, *, batch_rows=256, device='cuda', progress_callback=None):
    """Actual full-clock forwards with exact new-prior output and score parity.

    Only the two CPU logit arrays are collected. The already authenticated full
    truth is reused for canonical scores; metadata is never concatenated.
    """
    import torch
    need(type(batch_rows) is int and batch_rows > 0, 'BATCH_ROWS')
    need(progress_callback is None or callable(progress_callback), 'PROGRESS_CALLBACK')
    requested = torch.device(device)
    parameters = tuple(model.parameters())
    need(requested.type in ('cpu', 'cuda') and len(parameters) > 0
        and all(p.device.type == requested.type and
                (requested.index is None or p.device.index == requested.index) for p in parameters),
        'MODEL_DEVICE')
    need(all(p.grad is None for p in parameters), 'PREEXISTING_GRADIENTS')
    target = require_provider_hlc_state_contract(context['target_contract'])
    prior = objective.require_native_future_state_train_prior(context['prior'], target_contract=target)
    truth = context['truth']
    issues = truth['issue_ns']
    source_rows = context['source_rows']
    need(isinstance(issues, np.ndarray) and issues.dtype == np.int64 and issues.shape == (ROWS,)
        and np.all(np.diff(issues) > 0) and np.all((issues >= START) & (issues < UPPER))
        and core.digest(issues) == TRAIN_SHA and isinstance(source_rows, np.ndarray)
        and source_rows.dtype == np.int64 and source_rows.shape == (ROWS,)
        and np.all(np.diff(source_rows) > 0)
        and core.digest(source_rows) == context['normalization']['fit_population']['selected_source_rows_sha256'],
        'FULL_INITIAL_CLOCK')
    truth_sha = objective.native_future_state_truth_sha256(truth, target_contract=target)
    need(truth_sha == prior['fit_population']['truth_sha256'] == context['truth_sha256']
        and prior['fit_population']['issue_rows'] == ROWS
        and prior['fit_population']['issue_ns_sha256'] == TRAIN_SHA
        and prior['fit_population']['cutoff_ns'] == UPPER, 'FULL_INITIAL_PRIOR_TRUTH')
    objective._native_future_state_truth(truth, rows=ROWS, fold_upper_ns=UPPER,
                                        target_contract=target)
    before_state = core.state_sha(model.state_dict())
    rng = core.training_rng()
    before_rng = core.state_sha(rng)
    original_training = model.training
    outputs = {key: torch.empty((ROWS, 5), dtype=torch.float32, device='cpu')
               for key in OUTPUT_KEYS}
    batches = []
    batch_count = (ROWS + batch_rows - 1) // batch_rows
    model.eval()
    try:
        with torch.no_grad():
            for cursor, left in enumerate(range(0, ROWS, batch_rows), start=1):
                right = min(left + batch_rows, ROWS)
                take = np.arange(left, right, dtype=np.int64)
                current = packet(context, take)
                output, _, activation = core.future_cuda_forward(model, current['features'],
                    grouped=True, equal_tf=True, capture=False)
                objective._native_future_state_output(output)
                cpu = {key: value.detach().cpu() for key, value in output.items()}
                reference = objective.native_future_state_train_prior_logits(prior,
                    rows=len(take), target_contract=target)
                need(all(torch.equal(cpu[key], reference[key]) for key in OUTPUT_KEYS),
                     'INITIAL_EXACT_MATCHED_PRIOR')
                loss = objective.native_future_state_loss(model_output=output,
                    truth=current['truth'], fold_upper_ns=UPPER, target_contract=target)
                for key in OUTPUT_KEYS:
                    outputs[key][left:right].copy_(cpu[key])
                record = {'cursor': cursor, 'rows': len(take), 'rows_done': right,
                    'known_activity_cells': loss['count'], 'partial_NLL': float(loss['loss']),
                    'output_sha256': {key: core.digest(value.numpy()) for key, value in cpu.items()},
                    'activation': activation}
                batches.append(record)
                if progress_callback is not None:
                    progress_callback({'batch_cursor': cursor, 'batch_count': batch_count,
                        'rows_done': right, 'rows_total': ROWS, 'batch': dict(record)})
                del current, output, cpu, reference, loss
            need(len(batches) == batch_count and sum(b['rows'] for b in batches) == ROWS,
                 'FULL_INITIAL_ROW_COVERAGE')
            scores = objective.native_future_state_score_sums(model_output=outputs, truth=truth,
                fold_upper_ns=UPPER, target_contract=target)
            constant = objective.native_future_state_train_prior_logits(prior, rows=1,
                                                                        target_contract=target)
            reference = {key: value.expand(ROWS, 5) for key, value in constant.items()}
            reference_scores = objective.native_future_state_score_sums(model_output=reference,
                truth=truth, fold_upper_ns=UPPER, target_contract=target)
            need(scores == reference_scores, 'INITIAL_FULL_PROPER_SCORE_EXACT')
            count = sum(b['known_activity_cells'] for b in batches)
            need(count == sum(cell['activity']['count'] for cell in scores['per_tf'].values()),
                 'FULL_ACTIVITY_SUPPORT')
            total = sum(cell['partial_log_loss_sum'] for cell in scores['per_tf'].values())
            partial = total / count if count else None
            batch_partial = sum(b['partial_NLL'] * b['known_activity_cells'] for b in batches) / count if count else None
            need(count > 0 and np.isfinite(partial) and np.isfinite(batch_partial)
                and np.isclose(partial, batch_partial, rtol=1e-6, atol=1e-7),
                'PARTIAL_LOSS_CANONICAL_SCORE')
    finally:
        model.train(original_training)
        core.restore_training_rng(rng)
        need(core.state_sha(model.state_dict()) == before_state
            and all(p.grad is None for p in parameters), 'SCORE_STATE_GRAD_MUTATION')
        need(core.state_sha(core.training_rng()) == before_rng, 'SCORE_RNG_RESTORE')
    return {'rows': ROWS, 'batch_count': batch_count, 'batch_rows': batch_rows,
        'batches': batches, 'scores': scores, 'known_activity_cells': count,
        'proper_partial_NLL': partial, 'batch_weighted_partial_NLL': batch_partial,
        'loss_score_abs_delta': abs(partial - batch_partial), 'initial_prior_exact': True,
        'whole_TRAIN_initial': True, 'actual_NN_forward_rows': ROWS,
        'output_sha256': {key: core.digest(value.numpy()) for key, value in outputs.items()},
        'output_arrays_collected': 2,
        'output_ndarray_bytes': sum(value.numel() * value.element_size() for value in outputs.values()),
        'truth_sha256': truth_sha,
        'target_observation_certificate_ref': dict(truth['target_observation_certificate_ref']),
        'state_sha256_before_and_after': before_state, 'rng_sha256_before_and_restored': before_rng,
        'model_training_mode_restored': original_training, 'device': str(requested),
        'score_utility_counters': {'model_forwards': batch_count, 'optimizer_updates': 0,
            'backward_calls': 0, 'fits': 0, 'checkpoint_loads': 0, 'checkpoint_writes': 0},
        'score_scope': 'actual_fresh_v2_whole_TRAIN_initial_forward_not_learning_OOF_or_edge'}
