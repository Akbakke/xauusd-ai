"""Score a frozen fold-1 time-query fit on later TRAIN rows only.

The guarded fit caller publishes its full checkpoint before invoking this
read-only scorer. Existing target, feature-view and matched-score owners retain
all label, clock, normalization and metric authority. No TEST or trade path.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Callable, Mapping

REPO = Path('/home/andre2/src/GX1_CURRENT')
FIT_CHECKPOINT_SCHEMA = 'gx1_market_state_time_query_fold1_fit_state_v1'
FOLD_ID = 1
BATCH_SIZE = 16
MAX_CHECKPOINT_BYTES = 1073741824
VIEW_SHA256 = '7e0d13f1b4f58cc4d3dd294bf869d3195b48a69c022fe570e6b4db02f1581f15'
ZERO_STEP_PLAN = {
    'path': str(REPO / 'configs/research/MARKET_STATE_TIME_QUERY_ZERO_STEP_20261107_002.json'),
    'sha256': '41b94d01e1cae7633946c565ab8975ce9d150d4809506df5dc19f59a77e54764',
}
REFERENCE_RESULT = {
    'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TIME_QUERY_REFERENCES_20261107_001/RESULT.json',
    'sha256': '1ef8e55ebf38a7571cd98e812491c37d2ab7498403734f7701d1f9ee442f4d4b',
}


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('TIME_QUERY_FOLD1_OOF_' + reason)


def _clean_source(source_commit: str) -> None:
    _need(isinstance(source_commit, str) and len(source_commit) == 40
          and all(c in '0123456789abcdef' for c in source_commit), 'SOURCE_COMMIT')
    head = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    branch = subprocess.check_output(['git', '-C', str(REPO), 'branch', '--show-current'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain=v1',
                                     '--untracked-files=all'], text=True)
    _need(head == source_commit and branch == 'work/gx1-current' and dirty == '',
          'SOURCE_CHANGED')


def _frozen_fold_reference() -> dict:
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    ref = campaign.require_binding(REFERENCE_RESULT, label='frozen time-query reference',
                                   verify_file=True)
    result = campaign.read_bound_json(Path(ref['path']), ref['sha256'])
    _need(result.get('status') == 'COMPLETE_FROZEN_TRAIN_REFERENCES_NO_MODEL'
          and result.get('model_fits') == 0 and result.get('test_data_used') is False
          and result.get('read_only') is True, 'REFERENCE_SCOPE')
    folds = result.get('references', {}).get('folds', [])
    _need(isinstance(folds, list) and len(folds) == 5
          and [item.get('fold_id') for item in folds] == [1, 2, 3, 4, 5],
          'REFERENCE_FOLDS')
    return folds[0]


def _prior_arrays(fold: Mapping):
    """Broadcast frozen, separate target-population priors for all 288 ticks."""
    import numpy as np
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TIMEFRAMES
    grid = fold['reference']['per_tf_k']
    _need(set(grid) == set(TIMEFRAMES), 'REFERENCE_TF')
    activity = np.empty((288, 5), dtype=np.float64)
    side = np.empty((288, 5), dtype=np.float64)
    three = np.empty((288, 5, 3), dtype=np.float64)
    for j, tf in enumerate(TIMEFRAMES):
        _need(set(grid[tf]) == {str(k) for k in range(1, int(NATIVE_K[j]) + 1)},
              'REFERENCE_K:' + tf)
        for step in range(1, 289):
            prior = grid[tf][str(min(step, int(NATIVE_K[j])))]['prior']
            _need(set(prior) == {'activity', 'conditional_up', 'three_class'},
                  'REFERENCE_PRIOR:' + tf)
            activity[step - 1, j] = prior['activity']
            side[step - 1, j] = prior['conditional_up']
            three[step - 1, j] = prior['three_class']
    _need(bool(np.isfinite(activity).all() and np.isfinite(side).all()
               and np.isfinite(three).all())
          and bool(np.all((activity > 0) & (activity < 1)))
          and bool(np.all((side > 0) & (side < 1)))
          and bool(np.all((three > 0) & (three < 1)))
          and bool(np.allclose(three.sum(axis=-1), 1, atol=1e-12, rtol=0)),
          'REFERENCE_PROBABILITIES')
    return activity, side, three


def _load_checkpoint(model, *, fit_checkpoint_ref: Mapping[str, str],
                     expected_model_state_sha256: str, source_commit: str,
                     device):
    import torch
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    from gx1.contracts import entry_native_direction_research_state_v1 as full_state
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    _need(isinstance(expected_model_state_sha256, str)
          and len(expected_model_state_sha256) == 64
          and all(c in '0123456789abcdef' for c in expected_model_state_sha256),
          'EXPECTED_MODEL_DIGEST')
    ref = campaign.require_binding(fit_checkpoint_ref, label='frozen fold1 fit state',
                                   verify_file=True)
    path = Path(ref['path'])
    before = common._stat(path)
    _need(0 < before['st_size'] <= MAX_CHECKPOINT_BYTES,
          'CHECKPOINT_SIZE')
    state = torch.load(path, map_location='cpu', weights_only=True)
    _need(common._stat(path) == before and campaign.file_sha256(path) == ref['sha256'],
          'CHECKPOINT_BOOKEND')
    _need(isinstance(state, Mapping)
          and state.get('schema_version') == FIT_CHECKPOINT_SCHEMA
          and state.get('source_commit') == source_commit
          and state.get('fold_id') == FOLD_ID
          and state.get('feature_view_sha256') == VIEW_SHA256
          and state.get('model_state_sha256') == expected_model_state_sha256
          and isinstance(state.get('model_state'), Mapping)
          and isinstance(state.get('binding'), Mapping)
          and state['binding'].get('feature_view_identity') == model.feature_view_identity
          and type(state.get('cursor')) is int and state['cursor'] == 29574
          and type(state.get('optimizer_updates')) is int
          and type(state.get('skipped_batches')) is int
          and state['optimizer_updates'] > 0
          and state['optimizer_updates'] + state['skipped_batches'] == 1849,
          'CHECKPOINT_SCOPE')
    seal = state.get('contract_sha256')
    _need(isinstance(seal, str) and seal == full_state._tree_sha(
        {key: value for key, value in state.items() if key != 'contract_sha256'}),
        'CHECKPOINT_SEAL')
    _need(canonical_model_state_sha256(state['model_state'])
          == expected_model_state_sha256, 'CHECKPOINT_MODEL_DIGEST')
    model.load_state_dict(state['model_state'], strict=True)
    _need(canonical_model_state_sha256(model.state_dict())
          == expected_model_state_sha256
          and model.feature_view_identity['contract_sha256'] == VIEW_SHA256,
          'RESTORED_MODEL_DIGEST')
    model.eval()
    return dict(ref)


def _score_batch(model, features: Mapping, parent_rows, issue_ns, *,
                 bound, scorer, priors, device, check_deadline: Callable[[], None]):
    """Encode one issue batch once; bind all target queries in nine small blocks."""
    import numpy as np
    import torch
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TIMEFRAMES
    _need(parent_rows.dtype == issue_ns.dtype == np.int64
          and parent_rows.ndim == issue_ns.ndim == 1
          and 0 < len(parent_rows) <= BATCH_SIZE
          and len(parent_rows) == len(issue_ns), 'BATCH_AXES')
    batch = len(parent_rows)
    scorer.begin_batch(issue_ns, fold_id=FOLD_ID)
    with torch.inference_mode():
        hidden = model.encode_features(**features)
        for step_start in range(1, 289, 32):
            check_deadline()
            steps = np.arange(step_start, min(step_start + 32, 289), dtype=np.int64)
            q = len(steps)
            k = np.minimum(steps[:, None], np.asarray(NATIVE_K, dtype=np.int64)[None, :])
            query = np.broadcast_to(k[None, :, :], (batch, q, len(TIMEFRAMES))).copy()
            logits = model.decode_query_grid(hidden, torch.from_numpy(query).to(device))
            model_p = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float64)
            # BoundTimeQueryTargets checks every repeated parent/issue pair before
            # the tape is read. One gather handles the whole bounded query block.
            flat_parent = np.repeat(parent_rows, q)
            flat_issue = np.repeat(issue_ns, q)
            truth = bound.gather(flat_parent, flat_issue,
                                 query.reshape(batch * q, len(TIMEFRAMES)),
                                 fold_id=FOLD_ID, partition='score')
            _need(np.array_equal(truth['issue_ns'], flat_issue)
                  and np.array_equal(truth['query_k'], query.reshape(batch * q, 5)),
                  'TARGET_AXIS')
            shape = (batch, q, 5)
            activity, side, three = priors
            scorer.add_block(step_start, query_k=query,
                validity_ns=truth['validity_ns'].reshape(shape), model_p=model_p,
                prior_activity=np.broadcast_to(activity[steps - 1][None, :, :], shape),
                prior_side_up=np.broadcast_to(side[steps - 1][None, :, :], shape),
                prior_three=np.broadcast_to(three[steps - 1][None, :, :, :],
                                            shape + (3,)),
                allowed=truth['allowed'].reshape(shape),
                eligible=truth['eligible'].reshape(shape),
                mature=truth['mature'].reshape(shape))
    scorer.end_batch()


def score_fold1_from_checkpoint(*, fit_checkpoint_ref: Mapping[str, str],
                                expected_model_state_sha256: str,
                                source_commit: str, device,
                                check_deadline: Callable[[], None] | None = None) -> dict:
    """Run one guarded, frozen fold-1 OOF score; return data for caller publication."""
    import numpy as np
    import torch
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import market_state_five_endpoint_target_binding_v1 as old_target
    from gx1.contracts import market_state_time_query_binding_v1 as query_target
    from gx1.contracts.market_state_time_query_scorer_v1 import TimeQueryMatchedScorer
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.contracts import local_random_access_campaign_v2 as campaign

    limits = require_guarded_cuda_trainer_execution()
    _need(isinstance(device, torch.device) and device.type == 'cuda'
          and torch.cuda.is_available(), 'GUARDED_CUDA')
    deadline = check_deadline or (lambda: None)
    _clean_source(source_commit)
    fold = _frozen_fold_reference()
    priors = _prior_arrays(fold)
    zero_plan = zero._read_ref(ZERO_STEP_PLAN)
    receipts = zero._inspect_receipts()
    view = zero._bind_view(zero_plan, receipts, FOLD_ID)
    _need(view['contract_sha256'] == VIEW_SHA256, 'VIEW_IDENTITY')
    bound = query_target.load_bound_time_query_targets()
    old = old_target.load_bound_five_endpoint_targets()
    _need(np.array_equal(bound.parent_rows, old.parent_rows)
          and np.array_equal(bound.decision_ns, old.decision_ns), 'TARGET_MODEL_AXIS')
    axes = old.mature_oof_axes(fold_id=FOLD_ID)
    _need(axes['score_parent_rows_sha256'] == fold['score_parent_rows_sha256']
          and axes['score_decision_ns_sha256'] == fold['score_issue_axis_sha256']
          and len(axes['score_parent_rows']) == fold['score_rows'] == 17753
          and axes['fit_decision_ns_sha256'] == fold['fit_issue_axis_sha256']
          and len(axes['fit_parent_rows']) == fold['fit_rows'] == 29574,
          'FROZEN_FOLD_AXES')
    model = views.build_native_direction_research_model(
        view, prefix_normalization=receipts['fold1']['normal'],
        constructor_metadata=receipts['constructor'], device=device,
        unobserved_units_profile=receipts['fold1']['profile'])
    checkpoint = _load_checkpoint(model, fit_checkpoint_ref=fit_checkpoint_ref,
        expected_model_state_sha256=expected_model_state_sha256,
        source_commit=source_commit, device=device)
    initial_digest = canonical_model_state_sha256(model.state_dict())
    chain = receipts['provider_chain']
    provider = base._provider_module()
    provider_plan = chain['provider_plan']
    info = provider.validate_metadata(provider_plan)
    prepared = base._prepare_inputs(provider, provider_plan, info,
                                    chain['gc_layout'])
    scorer = TimeQueryMatchedScorer()
    try:
        base._validate_axes(receipts['mature_plan']['cohorts'], prepared['axes'])
        all_axes = prepared['axes']
        _need(np.array_equal(all_axes['parent_rows'], bound.parent_rows)
              and np.array_equal(all_axes['decision_ns'], bound.decision_ns),
              'FEATURE_TARGET_AXIS')
        for key in ('parent_rows', 'bar_start_ns', 'decision_ns'):
            all_axes[key].setflags(write=False)
        for key in ('signal', 'times_ns', 'sample_positions'):
            prepared['entry_sequence_source'][key].setflags(write=False)
        for key in ('snap', 'ctx_cont', 'ctx_cat'):
            prepared['current'][key].setflags(write=False)
        score_rows = axes['score_parent_rows']
        score_issues = axes['score_decision_ns']
        for left in range(0, len(score_rows), BATCH_SIZE):
            deadline()
            rows = score_rows[left:left + BATCH_SIZE]
            issues = score_issues[left:left + BATCH_SIZE]
            raw = views.gather_native_direction_feature_batch(
                view, parent_rows=all_axes['parent_rows'],
                bar_start_ns=all_axes['bar_start_ns'],
                decision_ns=all_axes['decision_ns'],
                entry_sequence_source=prepared['entry_sequence_source'],
                snap_x=prepared['current']['snap'],
                ctx_cont=prepared['current']['ctx_cont'],
                ctx_cat=prepared['current']['ctx_cat'],
                multi_tf_sources=prepared['cache'],
                gc_surfaces=prepared['gc']['surfaces'],
                gc_layout=prepared['gc']['layout'],
                batch_parent_rows=rows,
                unobserved_units_profile=receipts['fold1']['profile'])
            _need(raw['witness']['parent_rows'] == rows.tolist()
                  and raw['witness']['decision_ns'] == issues.tolist(),
                  'FEATURE_SCORE_ROWS')
            features = {name: torch.from_numpy(value).to(device)
                        for name, value in raw['features'].items()}
            _score_batch(model, features, rows, issues, bound=bound,
                         scorer=scorer, priors=priors, device=device,
                         check_deadline=deadline)
            del features, raw
        provider.validate_metadata(provider_plan)
    finally:
        base._release_pages(provider, prepared)
    score = scorer.result()
    _need(score['issue_axis']['1']['count'] == fold['score_rows']
          and score['issue_axis']['1']['sha256'] == fold['score_issue_axis_sha256'],
          'SCORED_ISSUE_AXIS')
    m5 = fold['m5_native_endpoint']['score']
    side = score['primary_M5_k1']['certified_side']
    activity = score['primary_M5_k1']['activity']
    _need(side['count'] == m5['side_known']
          and activity['count'] == m5['activity_known']
          and abs(side['prior_brier_sum'] - m5['side_prior_brier_sum']) <= 1e-6
          and abs(activity['prior_brier_sum'] - m5['activity_prior_brier_sum']) <= 1e-6,
          'MATCHED_FROZEN_REFERENCE')
    _need(canonical_model_state_sha256(model.state_dict()) == initial_digest
          and campaign.file_sha256(Path(checkpoint['path'])) == checkpoint['sha256']
          and campaign.file_sha256(Path(REFERENCE_RESULT['path'])) == REFERENCE_RESULT['sha256'],
          'SCORE_BOOKEND')
    _clean_source(source_commit)
    return {'schema_version': 'gx1_market_state_time_query_fold1_oof_result_v1',
            'scope': 'TRAIN_only_frozen_fold1_no_fit_no_TEST_no_trade',
            'checkpoint': checkpoint,
            'model_state_sha256': initial_digest,
            'feature_view_sha256': VIEW_SHA256,
            'frozen_reference': REFERENCE_RESULT,
            'fold_id': FOLD_ID,
            'fit_rows': fold['fit_rows'],
            'score_rows': fold['score_rows'],
            'primary_M5_k1': score['primary_M5_k1'],
            'matched_score': score,
            'limits': limits,
            'test_data_used': False,
            'entry_exit_or_trade_authority': False}
