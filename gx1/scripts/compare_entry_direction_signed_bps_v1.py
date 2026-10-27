"""Fixed signed-bps saved-output comparison. No model, fit or raw-data route."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import types

REPO = Path('/home/andre2/src/GX1_CURRENT')
PROGRAM = Path('/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2')
META_SHA = 'a648c47880b6b195d1213741a0826d34c7e475bd75193313a4c26f7723798c7a'
PROTOCOL_SHA = '25458f5185c55070333729eededff4d33a11dd5b1ee334e74a4358c15a928888'
HORIZONS = (1, 5, 15)
DAY = 86_400_000_000_000
START, PREFIX_END, LATER_END, TRAIN_END = 1714521600000000000, 1735689600000000000, 1743465600000000000, 1767225600000000000
SCHEMES = ('UTC_DAY', 'UTC_WEEK_MONDAY')
ALPHA, REPLICATES, SEED = .05 / 6, 10000, 20261027
BASE_KEYS = {'parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns', 'observed', 'sign', 'logit_up'}
np = None


def require(ok, reason):
    if not ok:
        raise RuntimeError('SIGNED_BPS_COMPARISON_' + reason)


def canonical_panel(arrays, contexts, *, rows, signed_bps):
    keys = BASE_KEYS | {'context__' + c['name'] for c in contexts}
    if signed_bps:
        keys |= {'mean_return_bps', 'mid_return_bps'}
    require(set(arrays) == keys, 'EXACT_PREDICTION_MEMBERS')
    for name in BASE_KEYS:
        dtype = np.bool_ if name == 'observed' else np.int8 if name == 'sign' else np.float32 if name == 'logit_up' else np.int64
        shape = (rows,) if name in ('parent_rows', 'bar_start_ns', 'decision_ns') else (rows, 3)
        require(arrays[name].dtype == np.dtype(dtype) and arrays[name].shape == shape
                and np.isfinite(arrays[name]).all(), 'PREDICTION_HEADER:' + name)
    for name in keys - BASE_KEYS - {'mean_return_bps', 'mid_return_bps'}:
        require(arrays[name].shape == (rows,) and arrays[name].dtype.kind in 'biu',
                'CONTEXT_HEADER:' + name)
    if signed_bps:
        for name, dtype in (('mean_return_bps', np.float32), ('mid_return_bps', np.float64)):
            require(arrays[name].dtype == np.dtype(dtype) and arrays[name].shape == (rows, 3)
                    and np.isfinite(arrays[name]).all(), 'BPS_HEADER:' + name)
    require(len(np.unique(arrays['parent_rows'])) == rows, 'UNIQUE_PARENTS')
    order = np.argsort(arrays['parent_rows'], kind='stable')
    return {name: np.ascontiguousarray(value[order]) for name, value in arrays.items()}


def require_same_panel(candidate, baseline, target, allocation):
    for name in candidate:
        if name in ('logit_up', 'mean_return_bps', 'mid_return_bps'):
            continue
        require(name in baseline and candidate[name].dtype == baseline[name].dtype
                and candidate[name].tobytes() == baseline[name].tobytes(), 'PAIRED_IDENTITIES:' + name)
    selected = allocation.join_targets(candidate, target)
    allocation.join_targets(baseline, target)
    require(candidate['mid_return_bps'].tobytes() == selected['mid_return_bps'].tobytes(),
            'ORIGINAL_FLOAT64_BPS_TARGET_PARITY')
    return selected


def return_error(prediction, target, rms):
    error = prediction - target
    n = len(target)
    return {'count': n, 'sse_bps2': float(np.sum(error * error)),
            'absolute_error_sum_bps': float(np.sum(np.abs(error))),
            'mse_bps2': float(np.mean(error * error)) if n else None,
            'mae_bps': float(np.mean(np.abs(error))) if n else None,
            'prefix_scaled_mse': float(np.mean((error / rms) ** 2)) if n else None,
            'mean_error_bps': float(np.mean(error)) if n else None}


def score_returns(target, mean_return_bps, cutoff, scale, allocation):
    require(mean_return_bps.shape == target['mid_return_bps'].shape
            and np.isfinite(mean_return_bps).all(), 'FINITE_RETURN_PREDICTIONS')
    _, known, binary = allocation.masks(target, cutoff)
    bands = allocation.band_ids(target['mid_return_bps'])
    reports = []
    for j, horizon in enumerate(HORIZONS):
        use = known[:, j]
        actual = target['mid_return_bps'][use, j]
        predictions = {'candidate': mean_return_bps[use, j].astype(np.float64),
                       'own_prefix_mean': np.full(len(actual), scale['mean_bps'][j], np.float64),
                       'zero': np.zeros(len(actual), np.float64)}
        arms = {}
        for name, pred in predictions.items():
            whole = return_error(pred, actual, scale['rms_bps'][j])
            by_band = []
            for b, label in enumerate(allocation.BANDS):
                local = bands[use, j] == b
                by_band.append({'band': label, **return_error(pred[local], actual[local], scale['rms_bps'][j])})
            require(sum(x['count'] for x in by_band) == whole['count'], 'RETURN_BAND_PARTITION')
            arms[name] = {'full': whole, 'bands': by_band}
        reports.append({'horizon_minutes': horizon, 'parents': len(use),
                        'observed_mature_rows': int(use.sum()), 'coverage': float(use.mean()),
                        'mature_zero_rows': int(np.sum(use & (target['sign'][:, j] == 0))),
                        'binary_eligible_rows': int(binary[:, j].sum()), 'arms': arms})
    return reports


def auc_draws(labels, scores, inverse, weights, rank):
    # Direction masks may remove complete calendar blocks. Map their columns
    # without generating different resampling multiplicities.
    present, compact = np.unique(inverse, return_inverse=True)
    summary = rank.block_rank_pairs(np.ascontiguousarray(labels, dtype=np.int8),
        np.ascontiguousarray(scores, dtype=np.float64), np.asarray(compact, np.int64))
    selected_weights = weights[:, present]
    nonempty = selected_weights.sum(axis=1) > 0
    values = np.full(len(weights), np.nan, np.float64)
    pairs = np.zeros(len(weights), np.float64)
    if nonempty.any():
        twice, denominator = rank.weighted_rank_pairs(summary, selected_weights[nonempty])
        local = np.full(len(denominator), np.nan, np.float64)
        np.divide(twice, 2 * denominator, out=local, where=denominator > 0)
        values[nonempty], pairs[nonempty] = local, denominator
    return values, pairs


def interval(values, *, point, alpha, replicates):
    require(len(values) == replicates, 'REPLICATE_COUNT')
    valid = np.isfinite(values)
    limits = np.quantile(values[valid], [alpha/2, 1-alpha/2], method='linear') if valid.any() else (None, None)
    return {'point': float(point), 'low': None if limits[0] is None else float(limits[0]),
            'high': None if limits[1] is None else float(limits[1]), 'alpha': alpha,
            'replicates_requested': replicates, 'replicates_valid': int(valid.sum()),
            'replicates_invalid': int((~valid).sum())}


def paired_intervals(candidate, baseline, target, scale, io, rank, *,
                     replicates=REPLICATES, alpha=ALPHA, seed=SEED, minimum_blocks=8):
    known = target['observed'] & (target['terminal_time_ns'] < LATER_END)
    binary = known & (target['sign'] != 0)
    reports = []
    for sid, scheme in enumerate(SCHEMES):
        days = target['decision_ns'] // DAY
        labels = days if sid == 0 else (days + 3) // 7
        blocks, inverse = np.unique(labels, return_inverse=True)
        k = len(blocks)
        require(k >= minimum_blocks, 'EFFECTIVE_TIME_BLOCKS')
        weights = io.block_weights(k, replicates=replicates,
                                   seed=np.random.SeedSequence([seed, sid]))
        require(weights.shape == (replicates, k) and np.all(weights.sum(axis=1) == k), 'PAIRED_BLOCK_WEIGHTS')
        for j, horizon in enumerate(HORIZONS):
            use = known[:, j]
            y = target['mid_return_bps'][use, j]
            error = candidate['mean_return_bps'][use, j].astype(np.float64) - y
            block_count = np.bincount(inverse[use], minlength=k)
            denominator = weights @ block_count
            contrasts = {}
            for reference, pred in (('own_prefix_mean', scale['mean_bps'][j]), ('zero', 0.0)):
                delta = error ** 2 - (pred - y) ** 2
                sums = np.bincount(inverse[use], weights=delta, minlength=k)
                draws = np.full(replicates, np.nan, np.float64)
                np.divide(weights @ sums, denominator, out=draws, where=denominator > 0)
                contrasts['candidate_minus_' + reference + '_mse_bps2'] = interval(
                    draws, point=np.mean(delta), alpha=alpha, replicates=replicates)
            use = binary[:, j]
            yy = (target['sign'][use, j] == 1).astype(np.int8)
            require((yy == 0).any() and (yy == 1).any(), 'ORIGINAL_AUC_BOTH_CLASSES')
            aa, ap = auc_draws(yy, candidate['logit_up'][use, j], inverse[use], weights, rank)
            bb, bp = auc_draws(yy, baseline['logit_up'][use, j], inverse[use], weights, rank)
            require(np.array_equal(ap, bp), 'PAIRED_AUC_DENOMINATORS')
            one = np.ones((1, k), np.float64)
            a0, _ = auc_draws(yy, candidate['logit_up'][use, j], inverse[use], one, rank)
            b0, _ = auc_draws(yy, baseline['logit_up'][use, j], inverse[use], one, rank)
            contrasts['candidate_minus_native276_auc'] = interval(
                aa - bb, point=a0[0] - b0[0], alpha=alpha, replicates=replicates)
            reports.append({'scheme': scheme, 'horizon_minutes': horizon,
                            'confirmatory': horizon == 5, 'blocks': k,
                            'block_resampling_population': 'all fixed later parents; masks retained within each resample',
                            'seed_sequence': [seed, sid], 'weights_sha256': io.digest(weights, np),
                            'contrasts': contrasts})
    return reports


def interpret_intervals(reports):
    primary = [x for x in reports if x['horizon_minutes'] == 5]
    require(len(primary) == 2 and {x['scheme'] for x in primary} == set(SCHEMES), 'TWO_PRIMARY_SCHEMES')
    valid = all(v['replicates_invalid'] == 0 for x in primary for v in x['contrasts'].values())
    def supports(name, direction):
        return valid and all((x['contrasts'][name]['low'] > 0 if direction > 0
                              else x['contrasts'][name]['high'] < 0) for x in primary)
    mse = all(supports('candidate_minus_' + ref + '_mse_bps2', -1)
              for ref in ('own_prefix_mean', 'zero'))
    auc = supports('candidate_minus_native276_auc', 1)
    return {'all_primary_replicates_defined': valid, 'bps_mse5_improvement_supported': mse,
            'auc5_improvement_supported': auc, 'combined_advance_criterion_met': mse and auc,
            'automatic_multifold_expansion_authorized': False, 'economic_edge_qualified': False,
            'scope': 'Fixed reused TRAIN development split; no test or economic qualification.'}


def check_original_scores(report, direction, returns=None):
    for j, item in enumerate(direction):
        old = report['model_metrics']['per_horizon'][j]
        require(old['counts']['eligible_cells'] == item['eligible_rows'], 'ORIGINAL_BINARY_COUNT')
        for metric in ('logloss', 'brier', 'accuracy', 'balanced_accuracy'):
            require(np.isclose(item['arms']['candidate']['full'][metric], old[metric],
                               rtol=1e-12, atol=1e-12), 'ORIGINAL_SCORE:' + metric)
    if returns is not None:
        original = report['model_metrics']['signed_bps']['per_horizon']
        for new, old in zip(returns, original):
            current = new['arms']['candidate']['full']
            require(current['count'] == old['count'], 'ORIGINAL_RETURN_COUNT')
            for metric in ('sse_bps2', 'absolute_error_sum_bps', 'mse_bps2', 'mae_bps',
                           'prefix_scaled_mse', 'mean_error_bps'):
                require(np.isclose(current[metric], old[metric], rtol=1e-12, atol=1e-12),
                        'ORIGINAL_RETURN_SCORE:' + metric)
            for arm, field in (('zero', 'zero_reference_sse_bps2'),
                               ('own_prefix_mean', 'own_prefix_mean_reference_sse_bps2')):
                require(np.isclose(new['arms'][arm]['full']['sse_bps2'], old[field],
                                   rtol=1e-12, atol=1e-12), 'ORIGINAL_RETURN_REFERENCE:' + arm)


def run_actual(meta, io, guard, owner, allocation, rank, plan, out, paths, pins, fdpaths):
    inputs = plan['inputs']
    proof = meta.read_json(inputs['candidate_root_review'])
    require(proof['status'] == 'COMPLETE_CONSUMED_PAIRED_COMPARISON_PENDING'
            and proof['optimizer_updates'] == 2976 and proof['parent_cursor'] == 47602
            and proof['oof_cursor'] == 17288 and not proof['qualification_weights_reused']
            and proof['comparison_protocol'] == plan['protocol'], 'COMPLETE_FIXED_CANDIDATE')
    protocol = meta.read_json(plan['protocol'])
    require(inputs['baseline_qualification'] == protocol['baseline_qualification'], 'ORIGINAL_BASELINE_QUALIFICATION')
    meta.read_json(inputs['baseline_qualification'])
    scale = meta.read_json(inputs['scale'])
    require(inputs['scale'] == proof['own_prefix_bps_scale'] and scale['prefix_rows'] == 47602
            and scale['cutoff_ns'] == PREFIX_END and scale['horizons_minutes'] == list(HORIZONS)
            and np.isfinite(scale['mean_bps']).all() and np.isfinite(scale['rms_bps']).all()
            and (np.asarray(scale['rms_bps']) > 0).all(), 'OWN_PREFIX_SCALE')
    def load(ref, keys):
        require(ref['path'] in paths, 'DECLARED_INPUT')
        pin = io.pin_artifact(ref)
        pins[ref['path']] = pin
        fdpaths.add('/proc/self/fd/' + str(pin['fd']))
        return io.read_archive(pin, set(keys), np)
    tm, tq = meta.read_json(inputs['target_metadata']), meta.read_json(inputs['target_qualification'])
    require(tq['data_qualified'] is True and tq['metadata'] == inputs['target_metadata']
            and tm['price_targets'] == inputs['price_targets'] and tm['test_data_used'] is False,
            'QUALIFIED_TRAIN_TARGET')
    target = load(inputs['price_targets'], guard.TARGET_DTYPES)
    allocation.validate_targets(target, guard.TARGET_DTYPES, rows=118249)
    candidate_reports = {role: meta.read_json(item['report'])
                         for role, item in inputs['candidate_measurements'].items()}
    baseline_reports = {role: meta.read_json(item['report'])
                        for role, item in inputs['baseline_measurements'].items()}
    require(set(candidate_reports) == set(baseline_reports) == {'INITIAL_TRAIN', 'FINAL_TRAIN', 'OOF'},
            'THREE_FIXED_MEASUREMENTS')
    contexts = candidate_reports['INITIAL_TRAIN']['context_map']['contexts']
    base_keys = BASE_KEYS | {'context__' + x['name'] for x in contexts}
    for reports, refs, signed in ((candidate_reports, inputs['candidate_measurements'], True),
                                  (baseline_reports, inputs['baseline_measurements'], False)):
        for role, report in reports.items():
            require(report['role'] == role and report['fold'] == protocol['fold']
                    and report['context_map']['contexts'] == contexts and not report['test_data_used']
                    and report['predictions'] == {k: refs[role]['payload'][k] for k in ('path', 'sha256')},
                    'MEASUREMENT_BINDING:' + role)
            require(report['schema_version'] == ('gx1_native_direction_bps_prefix_measurement_v1'
                    if signed else 'gx1_native_direction_prefix_measurement_v1'), 'MEASUREMENT_SCHEMA')
    prior = np.asarray(candidate_reports['INITIAL_TRAIN']['own_prefix_prior']['logit_up'], np.float64)
    for report in [*candidate_reports.values(), *baseline_reports.values()]:
        require(np.array_equal(prior, report['own_prefix_prior']['logit_up']), 'SAME_PREFIX_PRIOR')
    # All implementation, protocol and source pins exist before any prediction
    # payload is read. Initial comparison precedes all later prediction reads.
    meta.publish(out, 'COMPARISON_LOCK.json', {'protocol': plan['protocol'],
        'worker': plan['sources']['worker'], 'sources': plan['sources'],
        'primary_direction_score': 'existing UP logit; no selection between output heads',
        'block_universe': 'all fixed later parents', 'bootstrap_seed_sequences': [[SEED,0],[SEED,1]],
        'invalid_replicates_block_claims': True, 'no_saved_prediction_read_before_this_record': True})
    results, initial_rows = {}, None
    for role in ('INITIAL_TRAIN', 'FINAL_TRAIN', 'OOF'):
        n = 17288 if role == 'OOF' else 256
        cutoff = LATER_END if role == 'OOF' else PREFIX_END
        original_candidate = load(inputs['candidate_measurements'][role]['payload'],
                                  base_keys | {'mid_return_bps', 'mean_return_bps'})
        original_baseline = load(inputs['baseline_measurements'][role]['payload'], base_keys)
        require(original_candidate['parent_rows'].tobytes() == original_baseline['parent_rows'].tobytes(),
                'ORIGINAL_PRESENTATION_ORDER:' + role)
        a = canonical_panel(original_candidate, contexts, rows=n, signed_bps=True)
        b = canonical_panel(original_baseline, contexts, rows=n, signed_bps=False)
        t = require_same_panel(a, b, target, allocation)
        if role == 'OOF':
            require(np.array_equal(a['parent_rows'], np.arange(47602,64890))
                    and ((a['decision_ns'] >= PREFIX_END) & (a['decision_ns'] < LATER_END)).all(),
                    'FIXED_LATER_COHORT')
        else:
            require(((a['parent_rows'] >= 0) & (a['parent_rows'] < 47602)).all()
                    and ((a['decision_ns'] >= START) & (a['decision_ns'] < PREFIX_END)).all(), 'OWN256_PREFIX')
        if role == 'INITIAL_TRAIN':
            initial_rows = a['parent_rows'].copy()
            same = a['logit_up'].tobytes() == b['logit_up'].tobytes()
            zero = np.count_nonzero(a['mean_return_bps']) == 0
            meta.publish(out, 'INITIAL_COMPARABILITY.json', {
                'parents': n, 'parent_sha256': io.digest(initial_rows,np),
                'same_initial_direction_bits': same, 'zero_initial_mean_predictions': zero,
                'initial_logit_different_components': int(np.count_nonzero(a['logit_up'] != b['logit_up'])),
                'initial_logit_maximum_absolute_difference': float(np.max(np.abs(
                    a['logit_up'].astype(np.float64)-b['logit_up'].astype(np.float64)))),
                'exact_target_identity_and_original_float64_bps': True,
                'later_prediction_payloads_read': 0})
            require(same and zero, 'SAME_INITIAL_DIRECTION_AND_ZERO_MEAN')
        elif role == 'FINAL_TRAIN':
            require(np.array_equal(a['parent_rows'], initial_rows), 'SAME_INITIAL_FINAL_256')
        logits = {'candidate': a['logit_up'].astype(np.float64), 'native276': b['logit_up'].astype(np.float64),
                  'prior': np.broadcast_to(prior,(n,3)).copy()}
        pp = None
        if role == 'OOF':
            cr, ct = meta.read_json(inputs['controls_report']), meta.read_json(inputs['controls_terminal'])
            require(cr['OOF_predictions'] == inputs['controls'] and ct['status'] == 'COMPLETE'
                    and ct['result'] == inputs['controls_report'], 'QUALIFIED_SAVED_REFERENCES')
            c = load(inputs['controls'], cr['OOF_array_headers'])
            for name,spec in cr['OOF_array_headers'].items():
                require(c[name].dtype == np.dtype(spec['dtype']) and list(c[name].shape) == spec['shape'],
                        'REFERENCE_HEADER:' + name)
            require(np.array_equal(c['parent_rows'], np.arange(29849,118249)), 'REFERENCE_PARENT_RANGE')
            index = np.searchsorted(c['parent_rows'], a['parent_rows'])
            for name in ('parent_rows','decision_ns','terminal_time_ns','sign','observed'):
                require(np.array_equal(a[name], c[name][index]), 'REFERENCE_EXACT_JOIN:' + name)
            require((c['fold_id'][index] == 1).all()
                    and (c['holdout_end_ns_exclusive'][index] == LATER_END).all()
                    and np.array_equal(c['binary_mature'][index], allocation.masks(t,LATER_END)[2]),
                    'REFERENCE_MATURITY_FOLD')
            pp = np.ascontiguousarray(c['saved_ET_FULL_probability_up'][index])
            require(((pp > 0) & (pp < 1)).all()
                    and np.array_equal((pp > .5).astype(np.int8),c['saved_ET_FULL_direction_class'][index]),
                    'ET_ORIGINAL_PROBABILITIES')
            logits['ET252'] = np.log(pp)-np.log1p(-pp)
            logits['R1'] = np.ascontiguousarray(c['saved_R1_logit_up'][index],dtype=np.float64)
        direction,_ = allocation.score_population(t,logits,cutoff,owner,role=role,fold_id=1,et_probability=pp)
        returns = score_returns(t,a['mean_return_bps'],cutoff,scale,allocation)
        check_original_scores(candidate_reports[role],direction,returns)
        baseline_direction = [{**x,'arms':{'candidate':x['arms']['native276']}} for x in direction]
        check_original_scores(baseline_reports[role],baseline_direction)
        results[role] = {'parents':n,'parent_sha256':io.digest(a['parent_rows'],np),
                         'direction':direction,'return_bps':returns,
                         'original_scores_reproduced':True}
        if role == 'OOF':
            intervals = paired_intervals(a,b,t,scale,io,rank)
    return {'measurements':results,'paired_intervals':intervals,'interpretation':interpret_intervals(intervals),
            'model_forwards':0,'fits':0,'optimizer_steps':0,'qualification_weights_reused':False,
            'limitations':[
                'Reused TRAIN development split; not independent TEST or CONTROL generalization.',
                'Endpoint midpoint forecast does not measure early favourable movement or preceding adverse movement.',
                'Magnitude bands are retrospective outcome diagnostics, never executable filters.',
                'No spread, fees, slippage, order policy, realized PnL or economic edge.',
                'Initial output equality does not establish full numerical multi-timeframe feature parity.']}


def synthetic_checks(io, rank, allocation):
    cases = []
    scale = {'rms_bps':[2.,3.,4.], 'mean_bps':[.25,-.5,.75]}
    y = np.array([[0.,.1,50.],[-1.,2.,-3.],[4.,-5.,6.],[9.,9.,9.]],np.float64)
    t = {'mid_return_bps':y,'observed':np.ones((4,3),bool),
         'terminal_time_ns':np.array([[10]*3,[10]*3,[10]*3,[20]*3],np.int64),
         'sign':np.sign(y).astype(np.int8)}
    pred = np.zeros((4,3),np.float32)
    reports = score_returns(t,pred,20,scale,allocation)
    for j in range(3):
        full = reports[j]['arms']['candidate']['full']
        require(full['count']==3 and full['sse_bps2']==float(np.sum(y[:3,j]**2)),
                'SYNTH_RETURN_MSE_DIRECT')
        require(full['absolute_error_sum_bps']==float(np.sum(np.abs(y[:3,j]))),
                'SYNTH_RETURN_MAE_DIRECT')
        require(np.isclose(full['prefix_scaled_mse'],full['mse_bps2']/scale['rms_bps'][j]**2),
                'SYNTH_RETURN_SCALE')
    cases += ['float64_raw_mse_and_mae_match_direct_arithmetic','rms_scaling_does_not_change_reference_units',
              'strict_cutoff_retains_only_earlier_outcomes','observed_zero_included_in_return_error']
    require(reports[0]['mature_zero_rows']==1 and reports[0]['binary_eligible_rows']==2,
            'SYNTH_ZERO_DIRECTION_MASK')
    require(return_error(np.array([0.]),np.array([50.]),1.)['sse_bps2']
            >return_error(np.array([0.]),np.array([.1]),1.)['sse_bps2'],'SYNTH_MAGNITUDE_SUPERVISION')
    cases += ['zero_excluded_only_from_direction','same_sign_different_magnitude_has_different_error']
    labels=np.array([0,1,0,1,0,1],np.int8)
    scores=np.array([.2,.2,-.5,.7,.7,.1],np.float64)
    inverse=np.array([0,0,2,2,3,3],np.int64)
    weights=np.array([[1,0,1,1],[2,0,1,0],[0,3,0,0],[1,0,0,0]],np.float64)
    result,pairs=auc_draws(labels,scores,inverse,weights,rank)
    for b in (0,1,3):
        idx=np.repeat(np.arange(len(labels)),weights[b,inverse].astype(np.int64))
        yy,zz=labels[idx],scores[idx]
        positive,negative=zz[yy==1],zz[yy==0]
        expected=np.mean((positive[:,None]>negative[None,:]).astype(np.float64)
                         +.5*(positive[:,None]==negative[None,:]))
        require(result[b]==expected and pairs[b]==len(positive)*len(negative),
                'SYNTH_WEIGHTED_AUC_DUPLICATES')
    require(np.isnan(result[2]) and pairs[2]==0,'SYNTH_EMPTY_REPLICATE_REPORTED')
    cases += ['weighted_auc_matches_explicit_row_duplication_including_ties',
              'masked_out_calendar_block_uses_same_weight_columns','empty_draw_is_reported_not_filled']
    one_class=np.array([0,0,1,1],np.int8)
    result,pairs=auc_draws(one_class,np.array([0.,1.,2.,3.]),
                           np.array([0,0,1,1],np.int64),np.array([[2.,0.],[0.,2.],[1.,1.]]),rank)
    require(np.isnan(result[:2]).all() and result[2]==1.,'SYNTH_SINGLE_CLASS_INVALID')
    cases.append('single_class_draw_is_reported_not_assigned_chance')
    ci=interval(np.array([-2.,0.,2.,np.nan]),point=0.,alpha=.2,replicates=4)
    require(ci['replicates_invalid']==1 and ci['replicates_valid']==3
            and np.isclose(ci['low'],-1.6) and np.isclose(ci['high'],1.6),'SYNTH_CI')
    cases.append('linear_quantiles_and_invalid_replicate_accounting')
    reports=[]
    for scheme in SCHEMES:
        contrasts={name:{'low':low,'high':high,'replicates_invalid':0}
                   for name,low,high in (('candidate_minus_own_prefix_mean_mse_bps2',-2.,-1.),
                       ('candidate_minus_zero_mse_bps2',-2.,-1.),('candidate_minus_native276_auc',.01,.02))}
        reports.append({'scheme':scheme,'horizon_minutes':5,'contrasts':contrasts})
    require(interpret_intervals(reports)['combined_advance_criterion_met'],'SYNTH_ADVANCE_POSITIVE')
    reports[0]['contrasts']['candidate_minus_native276_auc']['low']=0.
    require(not interpret_intervals(reports)['combined_advance_criterion_met'],'SYNTH_ZERO_NOT_SUPPORT')
    reports[0]['contrasts']['candidate_minus_native276_auc']['low']=.01
    reports[1]['contrasts']['candidate_minus_zero_mse_bps2']['replicates_invalid']=1
    require(not interpret_intervals(reports)['combined_advance_criterion_met'],'SYNTH_INVALID_BLOCKS_CLAIM')
    cases += ['advance_requires_both_references_and_both_schemes','zero_interval_boundary_is_not_supported_gain',
              'invalid_primary_draw_blocks_improvement_claim']
    # Compare complete paired block intervals with explicitly duplicated rows.
    # The last draw contains only a block whose 1-minute positive row is masked.
    time_rows = START + np.repeat(np.arange(4, dtype=np.int64) * 7, 2) * DAY
    target_values = np.tile(np.array([-2.,3.,-1.,1.,-5.,4.,-3.,6.])[:,None], (1,3))
    test_target = {'decision_ns': time_rows, 'mid_return_bps': target_values,
                   'terminal_time_ns': np.broadcast_to(time_rows[:,None]+1,(8,3)).copy(),
                   'observed': np.ones((8,3),bool), 'sign': np.sign(target_values).astype(np.int8)}
    test_target['observed'][7,0] = False
    test_candidate = {'mean_return_bps': np.tile(np.array([-.5,1.,1.,-1.,-2.,2.,0.,1.],np.float32)[:,None],(1,3)),
                      'logit_up': np.tile(np.array([-2.,1.,2.,-1.,-2.,2.,1.,-1.],np.float32)[:,None],(1,3))}
    test_baseline = {'logit_up': np.tile(np.array([-1.,0.,0.,1.,0.,-1.,-1.,2.],np.float32)[:,None],(1,3))}
    fixed_weights = np.array([[1,1,1,1],[2,1,1,0],[0,0,1,3],[0,0,0,4]],np.float64)
    fixed_io = types.SimpleNamespace(
        block_weights=lambda k,replicates,seed: fixed_weights.copy(),
        digest=lambda array,unused: hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())
    paired = paired_intervals(test_candidate,test_baseline,test_target,scale,fixed_io,rank,
                              replicates=4,alpha=.2,minimum_blocks=4)
    for item in paired:
        j = HORIZONS.index(item['horizon_minutes'])
        obs = test_target['observed'][:,j]
        def contrasts(indices):
            valid = indices[obs[indices]]
            yy = test_target['mid_return_bps'][valid,j]
            pred = test_candidate['mean_return_bps'][valid,j].astype(np.float64)
            outcome = (test_target['sign'][valid,j] == 1)
            def direct_auc(scores):
                positive,negative = scores[outcome],scores[~outcome]
                if not len(positive) or not len(negative):
                    return np.nan
                return np.mean((positive[:,None]>negative[None,:]).astype(float)
                               + .5*(positive[:,None]==negative[None,:]))
            return (np.mean((pred-yy)**2-(scale['mean_bps'][j]-yy)**2),
                    np.mean((pred-yy)**2-yy**2),
                    direct_auc(test_candidate['logit_up'][valid,j])-direct_auc(test_baseline['logit_up'][valid,j]))
        original = contrasts(np.arange(8))
        resamples = [contrasts(np.repeat(np.arange(8),weights[np.repeat(np.arange(4),2)].astype(int)))
                     for weights in fixed_weights]
        names = ('candidate_minus_own_prefix_mean_mse_bps2','candidate_minus_zero_mse_bps2',
                 'candidate_minus_native276_auc')
        for column,name in enumerate(names):
            values = np.array([x[column] for x in resamples],np.float64)
            valid = np.isfinite(values)
            lo,hi = np.quantile(values[valid],[.1,.9],method='linear')
            got = item['contrasts'][name]
            require(np.isclose(got['point'],original[column]) and np.isclose(got['low'],lo)
                    and np.isclose(got['high'],hi) and got['replicates_invalid']==int((~valid).sum()),
                    'SYNTH_PAIRED_INTERVAL_DIRECT:' + name)
    cases.append('complete_day_week_paired_intervals_match_explicit_duplicate_rows_and_masks')
    # Reject mismatched target identity before any scored comparison.
    panel = {'parent_rows':np.array([0,1],np.int64),'bar_start_ns':np.array([0,5],np.int64),
             'decision_ns':np.array([5,10],np.int64),
             'terminal_time_ns':np.array([[6,10,20],[11,15,25]],np.int64),
             'observed':np.ones((2,3),bool),'sign':np.ones((2,3),np.int8),
             'logit_up':np.zeros((2,3),np.float32),'mean_return_bps':np.zeros((2,3),np.float32),
             'mid_return_bps':np.ones((2,3),np.float64)}
    canonical = canonical_panel(panel,[],rows=2,signed_bps=True)
    mismatch = {name:value.copy() for name,value in canonical.items()
                if name not in ('mean_return_bps','mid_return_bps')}
    mismatch['sign'][0,0] = -1
    try:
        require_same_panel(canonical,mismatch,canonical,allocation)
    except RuntimeError as exc:
        require(str(exc).endswith('PAIRED_IDENTITIES:sign'),'SYNTH_EXPECTED_MISMATCH')
    else:
        raise RuntimeError('SIGNED_BPS_COMPARISON_SYNTH_ACCEPTED_MISMATCH')
    cases.append('paired_target_sign_mismatch_fails_before_scoring')
    return {'status':'PASS_SYNTHETIC_ONLY','passed':len(cases),'cases':cases,'genuine_payload_reads':0}


def main():
    global np
    sys.dont_write_bytecode = True
    helper_path = PROGRAM/'inspect_direction_gc_parquet_metadata.py'
    raw = helper_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == META_SHA, 'METADATA_HELPER')
    meta = types.ModuleType('signed_bps_comparison_metadata')
    meta.__file__ = str(helper_path)
    exec(compile(raw,str(helper_path),'exec'),meta.__dict__)
    pref = {'path':sys.argv[1], 'sha256':sys.argv[2]}
    plan = meta.read_json(pref)
    polref = {'path':str(REPO/'NEXT_RUN_POLICY.json'), 'sha256':sys.argv[3]}
    policy, head = meta.read_json(polref), sys.argv[4]
    mode, stage = plan['mode'], policy['direction_bps_saved_comparison']
    require(plan['schema_version'] == 'gx1_signed_bps_saved_comparison_execution_v1'
            and mode in ('synthetic','actual'), 'PLAN_MODE')
    require(stage['status']=='ADMITTED' and stage['plan']==pref and stage['mode']==mode
            and stage['execution_authorized'] is True
            and stage['saved_payload_reads_authorized'] is (mode=='actual')
            and stage['max_invocations']==1 and stage['claimed_invocations']==stage['completed_invocations']==0
            and stage['relaunch_allowed'] is False, 'ONE_SHOT_AUTHORITY')
    require(all(policy[k] is False for k in ('training_enabled','native_training_enabled',
            'native_launch_authorized','full_epoch_training_allowed','full_val_allowed')), 'NATIVE_GATES_CLOSED')
    require(meta.git_head()==head and plan['wall_seconds']==(120 if mode=='synthetic' else 180), 'HEAD_WALL')
    require(plan['max_rchar_bytes']==256*1024**2, 'FIXED_IO_BOUND')
    require(plan['protocol']['sha256']==PROTOCOL_SHA, 'FIXED_COMPARISON_PROTOCOL')
    protocol=meta.read_json(plan['protocol'])
    require(protocol['horizons_minutes']==list(HORIZONS) and protocol['primary_horizon_minutes']==5
            and protocol['uncertainty']['bootstrap_replicates']==REPLICATES
            and protocol['uncertainty']['bootstrap_seed']==SEED
            and protocol['uncertainty']['per_interval_alpha']==ALPHA, 'FIXED_COMPARISON_MATH')
    for item in plan['sources'].values():
        meta.read_ref(item)
    require(plan['sources']['worker']['path']==str(Path(__file__).resolve()), 'WORKER_PATH')
    admission=meta.read_json(plan['source_admission'])
    require(str(Path(__file__).relative_to(REPO)) in admission['allowed_repo_files'], 'SOURCE_ADMISSION')
    cap=meta.source_module('bps_comparison_cap',plan['sources']['cap'])
    proof,lock=cap.require_capped_cpu_audit_execution(),cap.require_capped_lock_ancestry()
    require(lock['scope']==proof['cgroup_path'][len('/sys/fs/cgroup'):], 'LOCK_SCOPE')
    io=meta.source_module('bps_comparison_io',plan['sources']['io'])
    guard=meta.source_module('bps_comparison_guard',plan['sources']['read_guard'])
    meta.PUBLISH=meta.source_module('bps_comparison_publication',plan['sources']['publication'])
    out=Path(plan['out_dir'])
    require(out.parent==PROGRAM and out.parent.resolve(strict=True)==PROGRAM, 'OUTPUT_ROOT')
    out.mkdir()
    claim=meta.publish(out,'INVOCATION_CLAIM.json',{'plan':pref,'policy':polref,
        'source_head':head,'mode':mode,'execution':proof,'lock':lock})
    paths=plan['payload_paths']
    require(len(paths)==len(set(paths))==(8 if mode=='actual' else 0), 'EXACT_EIGHT_SAVED_PAYLOADS')
    sys.addaudithook(guard.make_guard(out,paths))
    def expired(*unused):
        raise RuntimeError('SIGNED_BPS_COMPARISON_WALL_LIMIT')
    signal.signal(signal.SIGALRM,expired)
    signal.alarm(plan['wall_seconds'])
    start,before,previous=time.monotonic(),io.io_state(),sys.getprofile()
    pins,fdpaths,counts={},set(),{}
    try:
        sys.path.insert(0,str(REPO))
        import numpy as numpy
        np=numpy
        io.np=np
        probability=meta.source_module('bps_comparison_probability',plan['sources']['probability'])
        rank=meta.source_module('bps_comparison_rank',plan['sources']['rank'])
        allocation=meta.source_module('bps_comparison_allocation',plan['sources']['allocation'])
        allocation.np=np
        roles={np.load.__code__:'payload_load',
               probability.direction_probability_metrics.__code__:'proper',
               probability.direction_rank_diagnostics.__code__:'rank',
               rank.weighted_rank_pairs.__code__:'weighted_rank'}
        maxima={'payload_load':8 if mode=='actual' else 0,'proper':40,'rank':300,'weighted_rank':40}
        counts={k+s:0 for k in maxima for s in ('_calls','_returns')}
        def watch(frame,event,value):
            io.observe_work(frame,event,value)
            role=roles.get(frame.f_code)
            if role and event in ('call','return'):
                if event=='call':
                    counts[role+'_calls']+=1
                    require(counts[role+'_calls']<=maxima[role], 'CALL_CAP:'+role)
                    if role=='payload_load':
                        require(frame.f_locals.get('file') in fdpaths
                                and frame.f_locals.get('allow_pickle') is False, 'PINNED_NUMPY_LOAD')
                elif value is not None:
                    counts[role+'_returns']+=1
        sys.setprofile(watch)
        if mode=='synthetic':
            result=synthetic_checks(io,rank,allocation)
        else:
            qualification=meta.read_json(plan['source_qualification'])
            require(qualification['status']=='PASS_SYNTHETIC_ONLY'
                    and qualification['worker']==plan['sources']['worker'], 'QUALIFIED_WORKER')
            result=run_actual(meta,io,guard,probability,allocation,rank,plan,out,paths,pins,fdpaths)
            require(counts['payload_load_calls']==counts['payload_load_returns']==8
                    and counts['proper_calls']==counts['proper_returns']==33, 'EXACT_ACTUAL_CALLS')
        require(not any(io.FORBIDDEN.values()), 'NO_FORBIDDEN_WORK')
        require(all(counts[k+'_calls']==counts[k+'_returns'] for k in maxima), 'CALL_RETURN_PARITY')
        io.pinned_bookends(pins)
        used=io.io_state()['rchar']-before['rchar']
        require(used<=plan['max_rchar_bytes'], 'IO_BOUND')
        sys.setprofile(previous)
        for path,(identity,digest) in list(meta.PINNED.items()):
            require(meta.regular(path)==identity, 'METADATA_BOOKEND')
            meta.read_ref({'path':path,'sha256':digest})
        require(meta.git_head()==head, 'SOURCE_HEAD_BOOKEND')
        data={'execution_status':'COMPLETE','mode':mode,'plan':pref,'protocol':plan['protocol'],
              'source_head':head,'test_data_used':False,**result}
        require(len(meta.json_bytes(data))<=8*1024**2, 'SUMMARY_BOUND')
        summary=meta.publish(out,'SUMMARY.json',data)
        terminal={'status':'COMPLETE','exit_code':0,'plan':pref,'policy':polref,'claim':claim,'summary':summary,
                  'source_head':head,'source_unchanged':True,'inputs_unchanged':True,
                  'counters':{**counts,**io.FORBIDDEN},'execution':proof,'lock':lock,
                  'rchar_bytes':used,'elapsed_seconds':time.monotonic()-start,'test_data_used':False}
    except Exception as exc:
        sys.setprofile(previous)
        meta.publish(out,'TERMINAL.json',{'status':'FAILED','exit_code':1,'plan':pref,'policy':polref,
            'claim':claim,'source_head':head,'error':type(exc).__name__+': '+str(exc),
            'counters':{**counts,**io.FORBIDDEN},'elapsed_seconds':time.monotonic()-start,'test_data_used':False})
        raise
    finally:
        sys.setprofile(previous)
        signal.alarm(0)
        for pin in pins.values():
            os.close(pin['fd'])
    meta.publish(out,'TERMINAL.json',terminal)
    print(json.dumps({'status':'COMPLETE','mode':mode,'summary':summary,
                      'elapsed_seconds':terminal['elapsed_seconds']}))


if __name__=='__main__':
    main()
