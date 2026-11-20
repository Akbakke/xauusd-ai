"""One fixed Fold2 non-GC S/F plus auxiliary M5 high/low TRAIN loss."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from gx1.contracts.market_state_ta_integrated_model_v1 import masked_structure_hit_sums
from gx1.contracts.market_state_native_m5_targets_v1 import TIMEFRAMES, contract as native_target_contract

STRUCTURE_WEIGHT = 0.25
FUTURE_WEIGHT = 1.0


def contract() -> dict:
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import contract as dual_loss_contract
    base = dual_loss_contract(ta_only=True, dedicated_m5_side=True,
        dedicated_non_m5_side=True, dedicated_tf_activity=True,
        direct_tf_hidden=True, m5_equal_observation=True, fold_id=2)
    value = {
        'schema_version': 'gx1_ta_integrated_fold2_objective_v1',
        'role': 'research_only_no_fit_or_trade_authority',
        'S_F_loss_contract_sha256': base['contract_sha256'],
        'S_F': 'existing_five_TF_activity_and_known_conditional_side_masks_with_fold2_M5_equal_observation',
        'F_weight': FUTURE_WEIGHT,
        'structure_targets': ['next_high_hits_frozen_high', 'next_low_hits_frozen_low'],
        'structure_unknowns': 'masked_zero_gradient_no_imputed_negative_or_flat',
        'structure_loss': 'mean_observed_high_BCE_plus_mean_observed_low_BCE_divided_by_two_per_batch',
        'structure_weight': STRUCTURE_WEIGHT,
        'total': 'S_F_loss_plus_structure_weight_times_structure_loss',
        'test_data_used': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def integrated_loss(*, model_output: dict, state_truth: dict, future_truth: dict,
                    state_query_k: torch.Tensor, structure_hit: np.ndarray,
                    structure_observed: np.ndarray) -> dict:
    from gx1.contracts.market_state_time_query_model_v2 import factorized_log_probabilities
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import combined_loss
    if not (type(model_output) is dict and set(model_output) == {
            'state_activity_side_logits', 'state_log_probs',
            'future_activity_side_logits', 'future_log_probs',
            'm5_structure_hit_logits'}):
        raise ValueError('TA_INTEGRATED_LOSS_MODEL_OUTPUT')
    raw = model_output['m5_structure_hit_logits']
    for task in ('state', 'future'):
        logits = model_output[task + '_activity_side_logits']
        log_probs = model_output[task + '_log_probs']
        if not (isinstance(logits, torch.Tensor) and logits.dtype == torch.float32
                and logits.ndim == 3 and logits.shape[1:] == (5, 2)
                and isinstance(log_probs, torch.Tensor)
                and log_probs.shape == (logits.shape[0], 5, 3)
                and log_probs.device == logits.device
                and torch.equal(log_probs, factorized_log_probabilities(logits))):
            raise ValueError('TA_INTEGRATED_LOSS_FACTOR_PROBABILITIES')
    if not (isinstance(raw, torch.Tensor) and raw.dtype == torch.float32
            and raw.ndim == 2 and raw.shape[1] == 2
            and isinstance(structure_hit, np.ndarray)
            and isinstance(structure_observed, np.ndarray)
            and structure_hit.dtype == structure_observed.dtype == np.bool_
            and structure_hit.shape == structure_observed.shape == tuple(raw.shape)
            and bool(np.all(~structure_hit | structure_observed))):
        raise ValueError('TA_INTEGRATED_LOSS_STRUCTURE')
    sf = combined_loss(state_raw=model_output['state_activity_side_logits'],
        future_raw=model_output['future_activity_side_logits'],
        state_truth=state_truth, future_truth=future_truth,
        state_query_k=state_query_k, future_weight=FUTURE_WEIGHT,
        ta_only=True, dedicated_m5_side=True, dedicated_non_m5_side=True,
        dedicated_tf_activity=True, direct_tf_hidden=True,
        m5_equal_observation=True, fold_id=2)
    structure = masked_structure_hit_sums(raw,
        torch.from_numpy(structure_hit).to(device=raw.device),
        torch.from_numpy(structure_observed).to(device=raw.device))
    per_side = structure['loss_sums'] / structure['observed_counts'].clamp(min=1)
    auxiliary = per_side.mean()
    total = sf['loss'] + STRUCTURE_WEIGHT * auxiliary
    if not bool(torch.isfinite(total)):
        raise ValueError('TA_INTEGRATED_LOSS_NONFINITE')
    return {'loss': total, 'S_F_loss': sf['loss'], 'structure_loss': auxiliary,
            'structure_loss_per_side': per_side,
            'structure_observed_counts': structure['observed_counts'],
            'state_per_tf_counts': sf['state']['per_tf_counts'],
            'future_per_tf_counts': sf['future']['per_tf_counts'],
            'skip_update': sf['skip_update'] and not bool(structure['observed_counts'].any()),
            'contract_sha256': contract()['contract_sha256']}


def _native_target_spec(target_contract):
    if target_contract is None:
        return None
    from gx1.contracts.market_state_native_m5_targets_v1 import require_noise_margin_dataset_contract
    return require_noise_margin_dataset_contract(target_contract)


def _native_noise_class_fields(spec):
    identity={'class_order':spec['class_order'],'class_names':spec['class_names']}
    raw=json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return {'class_names':list(spec['class_names']),'class_names_sha256':hashlib.sha256(raw).hexdigest()}


def _native_label_field(task,target_contract):
    return task+'_sign' if target_contract is None else target_contract['tasks'][task]['sign']


def native_bar_objective_contract(*,target_contract=None):
    value = {
        'schema_version': 'gx1_native_m5_bar_direction_objective_v1',
        'target_contract_sha256': native_target_contract()['contract_sha256'],
        'timeframes': list(TIMEFRAMES), 'class_order': [-1, 0, 1],
        'close_mask': 'target_present_and_mature_and_observed_previous',
        'open_prediction_mask': 'close_mask_and_not_open_known_at_issue',
        'known_open': 'reported_as_observed_context_separately_no_prediction_credit',
        'loss': 'unweighted_cross_entropy_sum_over_predictive_cells_divided_by_predictive_count',
        'aggregation': 'per_comparison_per_TF_loss_sums_and_counts_before_division',
        'scores': 'same_masks_three_class_log_loss_Brier_confusion_and_unique_target_rows',
        'future_metadata_as_model_input': False,
    }
    spec=_native_target_spec(target_contract)
    if spec is not None:
        value.update(schema_version='gx1_native_m5_bar_noise_margin_objective_v1',
            target_contract_sha256=spec['contract_sha256'],raw_target_contract_sha256=spec['raw_target_contract_sha256'],
            close_mask=spec['tasks']['close']['eligible'],open_prediction_mask=spec['tasks']['open']['eligible'],
            known_open='noise_open_known_and_raw_open_known_at_issue_observed_context_only',
            target_contract=spec,**_native_noise_class_fields(spec))
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def _native_truth(truth, *, rows, fold_upper_ns, target_contract_sha256,target_contract=None):
    def need(ok, reason):
        if not bool(ok):
            raise ValueError('NATIVE_BAR_OBJECTIVE_' + reason)
    if target_contract is not None:
        spec=_native_target_spec(target_contract)
        need(target_contract_sha256==spec['contract_sha256'],'NOISE_TARGET_CONTRACT')
        raw_masks=_native_truth(truth,rows=rows,fold_upper_ns=fold_upper_ns,
            target_contract_sha256=spec['raw_target_contract_sha256'])
        return _native_noise_truth(truth,rows=rows,raw_masks=raw_masks,spec=spec)
    need(target_contract_sha256 == native_target_contract()['contract_sha256'],
         'TARGET_CONTRACT')
    need(type(truth) is dict and type(fold_upper_ns) is int and fold_upper_ns > 0, 'TRUTH')
    shape = (rows, len(TIMEFRAMES))
    for key in ('target_present', 'observed_previous', 'mature', 'eligible', 'open_known_at_issue'):
        need(isinstance(truth.get(key), np.ndarray) and truth[key].dtype == np.bool_
             and truth[key].shape == shape, 'MASK_' + key)
    for key in ('open_sign', 'close_sign'):
        need(isinstance(truth.get(key), np.ndarray) and truth[key].dtype == np.int8
             and truth[key].shape == shape and np.isin(truth[key], (-1, 0, 1)).all(),
             'SIGN_' + key)
    for key in ('target_row', 'validity_ns'):
        need(isinstance(truth.get(key), np.ndarray) and truth[key].dtype == np.int64
             and truth[key].shape == shape, 'AXIS_' + key)
    issues = truth.get('issue_ns')
    need(isinstance(issues, np.ndarray) and issues.dtype == np.int64
         and issues.shape == (rows,) and np.all(issues > 0)
         and np.all(issues < fold_upper_ns) and np.all(issues % 300_000_000_000 == 0)
         and fold_upper_ns % 300_000_000_000 == 0, 'ISSUE_AXIS')
    present, mature = truth['target_present'], truth['mature']
    need(np.all(truth['validity_ns'][present] % 300_000_000_000 == 0)
         and np.all(truth['validity_ns'][present] > np.broadcast_to(issues[:, None], shape)[present])
         and np.array_equal(mature, present & (truth['validity_ns'] < fold_upper_ns))
         and np.array_equal(truth['eligible'], mature & truth['observed_previous'])
         and np.array_equal(truth['target_row'] >= 0, mature)
         and not np.any(truth['open_known_at_issue'] & ~present), 'OBSERVABILITY')
    return {'close': truth['eligible'],
            'open': truth['eligible'] & ~truth['open_known_at_issue'],
            'open_known': truth['eligible'] & truth['open_known_at_issue']}


def native_bar_direction_loss(*, model_output, truth, fold_upper_ns, target_contract_sha256,target_contract=None):
    if not (type(model_output) is dict and set(model_output) == {'open_logits', 'close_logits'}):
        raise ValueError('NATIVE_BAR_OBJECTIVE_OUTPUT')
    logits = model_output['close_logits']
    if not (isinstance(logits, torch.Tensor) and logits.dtype == torch.float32
            and logits.ndim == 3 and logits.shape[0] > 0 and logits.shape[1:] == (len(TIMEFRAMES), 3)):
        raise ValueError('NATIVE_BAR_OBJECTIVE_LOGITS')
    masks = _native_truth(truth, rows=len(logits), fold_upper_ns=fold_upper_ns,
                          target_contract_sha256=target_contract_sha256,target_contract=target_contract)
    sums, counts = {}, {}
    for name in ('open', 'close'):
        raw = model_output[name + '_logits']
        if not (isinstance(raw, torch.Tensor) and raw.shape == logits.shape
                and raw.dtype == torch.float32 and raw.device == logits.device
                and bool(torch.isfinite(raw).all())):
            raise ValueError('NATIVE_BAR_OBJECTIVE_LOGITS')
        labels = torch.as_tensor(truth[_native_label_field(name,target_contract)].astype(np.int64) + 1, device=raw.device)
        mask = torch.as_tensor(masks[name], device=raw.device)
        cells = torch.nn.functional.cross_entropy(
            raw.reshape(-1, 3), labels.reshape(-1), reduction='none').reshape(labels.shape)
        sums[name] = torch.where(mask, cells, 0.0).sum(dim=0)
        counts[name] = mask.sum(dim=0)
    total_count = counts['open'].sum() + counts['close'].sum()
    total_sum = sums['open'].sum() + sums['close'].sum()
    if not bool(torch.isfinite(total_sum)):
        raise ValueError('NATIVE_BAR_OBJECTIVE_NONFINITE')
    return {'loss': total_sum / total_count.clamp(min=1),
            'loss_sums': sums, 'counts': counts, 'skip_update': not bool(total_count),
            'contract_sha256': native_bar_objective_contract(target_contract=target_contract)['contract_sha256']}


def native_bar_direction_score_sums(*, model_output, truth, fold_upper_ns,
                                    target_contract_sha256,target_contract=None):
    """Add proper-score sums on the same predictive cells used by the loss."""
    if not (type(model_output) is dict and set(model_output) == {'open_logits', 'close_logits'}):
        raise ValueError('NATIVE_BAR_OBJECTIVE_OUTPUT')
    raw_by_task = {}
    for name in ('open', 'close'):
        raw = model_output[name + '_logits']
        raw = raw.detach().cpu().numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw)
        if not (raw.dtype == np.float32 and raw.ndim == 3 and raw.shape[0] > 0
                and raw.shape[1:] == (len(TIMEFRAMES), 3) and np.isfinite(raw).all()):
            raise ValueError('NATIVE_BAR_OBJECTIVE_LOGITS')
        raw_by_task[name] = raw
    if raw_by_task['open'].shape != raw_by_task['close'].shape:
        raise ValueError('NATIVE_BAR_OBJECTIVE_LOGITS')
    masks = _native_truth(truth, rows=len(raw_by_task['close']), fold_upper_ns=fold_upper_ns,
                          target_contract_sha256=target_contract_sha256,target_contract=target_contract)
    report = {}
    for name, mask in masks.items():
        task = 'close' if name == 'close' else 'open'
        logits = raw_by_task[task].astype(np.float64)
        shifted = logits - logits.max(axis=2, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=2, keepdims=True))
        probs = np.exp(log_probs)
        labels = truth[_native_label_field(task,target_contract)].astype(np.int64) + 1
        task_report = {}
        for j, tf in enumerate(TIMEFRAMES):
            selected = mask[:, j]
            target = labels[selected, j]
            predictions = logits[selected, j].argmax(axis=1)
            confusion = np.zeros((3, 3), dtype=np.int64)
            np.add.at(confusion, (target, predictions), 1)
            one_hot = np.eye(3)[target]
            maximum = logits[selected, j].max(axis=1, keepdims=True)
            tied = (logits[selected, j] == maximum).sum(axis=1) > 1
            task_report[tf] = {
                'count': int(selected.sum()),
                'unique_target_rows': int(np.unique(truth['target_row'][selected, j]).size),
                'log_loss_sum': float(-log_probs[selected, j, target].sum()),
                'brier_sum': float(np.square(probs[selected, j] - one_hot).sum()),
                'confusion_truth_by_prediction': confusion.tolist(),
                'max_logit_ties': int(tied.sum()),
            }
        report[name] = task_report
    return {**(_native_noise_class_fields(_native_target_spec(target_contract)) if target_contract is not None else {}),
            'schema_version': ('gx1_native_m5_bar_noise_margin_score_sums_v1' if target_contract is not None else 'gx1_native_m5_bar_direction_score_sums_v1'),
            'contract_sha256': native_bar_objective_contract(target_contract=target_contract)['contract_sha256'],
            'class_order': [-1, 0, 1], 'open_known_role': 'observed_context_diagnostic',
            'per_task': report}


def _native_prior_seal(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return {**value,'contract_sha256':hashlib.sha256(raw).hexdigest()}


def _native_prior_array_sha(arrays):
    digest=hashlib.sha256()
    for name,array in sorted(arrays.items()):
        value=np.ascontiguousarray(array)
        digest.update(json.dumps([name,value.dtype.str,list(value.shape)],separators=(',',':')).encode()+b'\0')
        digest.update(value.tobytes())
    return digest.hexdigest()


def native_bar_train_prior_contract(*,target_contract=None):
    value={'schema_version':'gx1_native_m5_train_threeclass_prior_contract_v1',
        'target_contract_sha256':native_target_contract()['contract_sha256'],
        'objective_contract_sha256':native_bar_objective_contract(target_contract=target_contract)['contract_sha256'],
        'class_order':[-1,0,1],'timeframes':list(TIMEFRAMES),'pseudocount_per_class':1,
        'formula':'(TRAIN_predictive_issue_cell_class_count+1)/(TRAIN_predictive_issue_cell_count+3)',
        'fit_masks':'same_conditional_open_and_close_masks_as_native_bar_direction_loss',
        'known_open_role':'excluded_from_prior_fit_separate_observed_context_diagnostic',
        'zero_observation_cell':'explicit_unfitted_uniform_pseudocount_reference',
        'logits':'float32_log_probabilities_reused_by_existing_native_bar_direction_score_sums',
        'class_or_smoothing_selection_from_control':False}
    target_spec=_native_target_spec(target_contract)
    if target_spec is not None:
        value.update(schema_version='gx1_native_m5_noise_margin_train_threeclass_prior_contract_v1',
            target_contract_sha256=target_spec['contract_sha256'],target_contract=target_spec,
            fit_masks='same_per_task_noise_predictive_masks_as_native_bar_direction_loss',
            **_native_noise_class_fields(target_spec))
    return _native_prior_seal(value)


def require_native_bar_train_prior(value,*,target_contract=None):
    from gx1.contracts.entry_model_native_input_normalization_v1 import NATIVE_M5_TRAIN_FIT_CUTOFF_NS
    def need(ok,reason):
        if not bool(ok):raise ValueError('NATIVE_BAR_TRAIN_PRIOR_'+reason)
    def sha(text):
        return isinstance(text,str) and len(text)==64 and all(c in '0123456789abcdef' for c in text)
    keys={'schema_version','prior_contract_sha256','target_contract_sha256','objective_contract_sha256',
          'class_order','timeframes','fit_population','per_task','contract_sha256'}
    spec=native_bar_train_prior_contract(target_contract=target_contract)
    target_spec=_native_target_spec(target_contract)
    if target_spec is not None:keys|={'target_contract','class_names','class_names_sha256'}
    schema=('gx1_native_m5_noise_margin_train_threeclass_prior_v1' if target_spec is not None
            else 'gx1_native_m5_train_threeclass_prior_v1')
    need(type(value)is dict and set(value)==keys and value['schema_version']==schema,'SCHEMA')
    if target_spec is not None:
        need(all(value[key]==spec[key]for key in ('target_contract','class_names','class_names_sha256')),'NOISE_IDENTITY')
    need(value['prior_contract_sha256']==spec['contract_sha256']
        and value['target_contract_sha256']==spec['target_contract_sha256']
        and value['objective_contract_sha256']==spec['objective_contract_sha256']
        and value['class_order']==[-1,0,1] and value['timeframes']==list(TIMEFRAMES),'IDENTITY')
    population=value['fit_population']
    need(type(population)is dict and set(population)=={'scope','cutoff_ns','issue_rows','first_issue_ns',
        'last_issue_ns','issue_ns_sha256','truth_sha256'} and population['scope']=='TRAIN','POPULATION')
    need(type(population['cutoff_ns'])is int and 0<population['cutoff_ns']<=NATIVE_M5_TRAIN_FIT_CUTOFF_NS
        and population['cutoff_ns']%300_000_000_000==0 and type(population['issue_rows'])is int
        and population['issue_rows']>0 and type(population['first_issue_ns'])is int
        and type(population['last_issue_ns'])is int
        and 0<population['first_issue_ns']<=population['last_issue_ns']<population['cutoff_ns']
        and population['first_issue_ns']%300_000_000_000==population['last_issue_ns']%300_000_000_000==0
        and sha(population['issue_ns_sha256']) and sha(population['truth_sha256']),'TRAIN_CUTOFF_OR_AXIS')
    need(type(value['per_task'])is dict and set(value['per_task'])=={'open','close'},'TASKS')
    for task,cells in value['per_task'].items():
        need(type(cells)is dict and set(cells)==set(TIMEFRAMES),'TIMEFRAMES')
        for tf,cell in cells.items():
            need(type(cell)is dict and set(cell)=={'class_counts','count','unique_target_rows',
                'mask_sha256','population_sha256','probabilities','train_observations_available'},'CELL_SCHEMA')
            counts=cell['class_counts'];count=cell['count'];unique=cell['unique_target_rows']
            need(type(counts)is list and len(counts)==3 and all(type(n)is int and n>=0 for n in counts)
                and type(count)is int and count==sum(counts) and count<=population['issue_rows']
                and type(unique)is int and 0<=unique<=count and (unique>0)==(count>0)
                and type(cell['train_observations_available'])is bool
                and cell['train_observations_available']==(count>0)
                and sha(cell['mask_sha256']) and sha(cell['population_sha256']),'CELL_COUNTS')
            probabilities=cell['probabilities']
            need(type(probabilities)is list and len(probabilities)==3
                and all(type(p)is float and np.isfinite(p) and 0<p<=1 for p in probabilities)
                and probabilities==[(n+1)/(count+3)for n in counts],'FIXED_LAPLACE_ONE')
    expected=_native_prior_seal({k:v for k,v in value.items()if k!='contract_sha256'})
    need(value==expected,'HASH')
    return json.loads(json.dumps(value))


def fit_native_bar_train_prior(*,truth,fold_upper_ns,target_contract_sha256,target_contract=None):
    """Freeze one predefined threeclass reference on the exact TRAIN issue cells."""
    from gx1.contracts.entry_model_native_input_normalization_v1 import NATIVE_M5_TRAIN_FIT_CUTOFF_NS
    if (type(truth)is not dict or not isinstance(truth.get('issue_ns'),np.ndarray)
        or type(fold_upper_ns)is not int or not 0<fold_upper_ns<=NATIVE_M5_TRAIN_FIT_CUTOFF_NS):
        raise ValueError('NATIVE_BAR_TRAIN_PRIOR_TRAIN_CUTOFF_OR_AXIS')
    issues=truth['issue_ns'];rows=len(issues)
    if rows==0 or not np.all(np.diff(issues)>0):
        raise ValueError('NATIVE_BAR_TRAIN_PRIOR_ISSUE_ORDER')
    masks=_native_truth(truth,rows=rows,fold_upper_ns=fold_upper_ns,
                        target_contract_sha256=target_contract_sha256,target_contract=target_contract)
    target_spec=_native_target_spec(target_contract)
    per_task={}
    for task in ('open','close'):
        cells={}
        for j,tf in enumerate(TIMEFRAMES):
            mask=masks[task][:,j];sign=truth[_native_label_field(task,target_contract)][mask,j]
            counts=np.bincount(sign.astype(np.int64)+1,minlength=3).tolist();count=int(mask.sum())
            targets=truth['target_row'][mask,j]
            cells[tf]={'class_counts':counts,'count':count,'unique_target_rows':int(np.unique(targets).size),
                'mask_sha256':hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),
                'population_sha256':_native_prior_array_sha({'issue_ns':issues[mask],'target_row':targets,'sign':sign}),
                'probabilities':[(n+1)/(count+3)for n in counts],
                'train_observations_available':count>0}
        per_task[task]=cells
    keys=('issue_ns','target_row','validity_ns','target_present','observed_previous','mature',
          'eligible','open_known_at_issue','open_sign','close_sign')
    extra={}
    if target_spec is not None:
        keys=target_spec['required_array_fields']
        extra={'target_contract':target_spec,**_native_noise_class_fields(target_spec)}
    prior=_native_prior_seal({**extra,'schema_version':('gx1_native_m5_noise_margin_train_threeclass_prior_v1'
        if target_spec is not None else 'gx1_native_m5_train_threeclass_prior_v1'),
        'prior_contract_sha256':native_bar_train_prior_contract(target_contract=target_contract)['contract_sha256'],
        'target_contract_sha256':target_contract_sha256,
        'objective_contract_sha256':native_bar_objective_contract(target_contract=target_contract)['contract_sha256'],
        'class_order':[-1,0,1],'timeframes':list(TIMEFRAMES),
        'fit_population':{'scope':'TRAIN','cutoff_ns':fold_upper_ns,'issue_rows':rows,
            'first_issue_ns':int(issues[0]),'last_issue_ns':int(issues[-1]),
            'issue_ns_sha256':hashlib.sha256(np.ascontiguousarray(issues).tobytes()).hexdigest(),
            'truth_sha256':_native_prior_array_sha({key:truth[key]for key in keys})},
        'per_task':per_task})
    return require_native_bar_train_prior(prior,target_contract=target_contract)


def native_bar_train_prior_logits(prior,*,rows,target_contract=None):
    """Constant frozen reference logits; never a learned model input."""
    prior=require_native_bar_train_prior(prior,target_contract=target_contract)
    if type(rows)is not int or rows<=0:raise ValueError('NATIVE_BAR_TRAIN_PRIOR_SCORE_ROWS')
    return {task+'_logits':np.broadcast_to(np.log(np.asarray([
        prior['per_task'][task][tf]['probabilities']for tf in TIMEFRAMES],np.float64)).astype(np.float32),
        (rows,len(TIMEFRAMES),3)).copy()for task in ('open','close')}


def _native_noise_truth(truth,*,rows,raw_masks,spec):
    def need(ok,reason):
        if not bool(ok):raise ValueError('NATIVE_BAR_OBJECTIVE_NOISE_'+reason)
    shape=(rows,len(TIMEFRAMES))
    need(set(spec['required_array_fields']).issubset(truth),'ARRAY_FIELDS')
    if 'noise_margin_contract' in truth:
        need(truth['noise_margin_contract']==spec['noise_margin_contract'],'PROJECTION_METADATA')
    for name,dtype in (('m5_count',np.int64),('source_gap',np.bool_),
        ('noise_margin_price',np.float64),('noise_atr14_asof',np.float64),
        ('noise_atr_asof_row',np.int64),('noise_atr_asof_available_ns',np.int64),
        *[(f'noise_{task}_{field}',dtype)for task in ('open','close')for field,dtype in
            (('delta',np.float64),('sign',np.int8),('known',np.bool_),('eligible',np.bool_))]):
        value=truth.get(name)
        need(isinstance(value,np.ndarray)and value.dtype==dtype and value.shape==shape,'HEADER_'+name)
    need((truth['m5_count']>=0).all(),'M5_COUNTS')
    atr=truth['noise_atr14_asof'];margin=truth['noise_margin_price']
    at=truth['noise_atr_asof_row'];available=truth['noise_atr_asof_available_ns']
    present=at>=0
    need((at>=-1).all()and np.array_equal(available>=0,present)
        and (available[~present]==-1).all()and np.isnan(atr[~present]).all()
        and (available[present]<=np.broadcast_to(truth['issue_ns'][:,None],shape)[present]).all()
        and (available[present]%300_000_000_000==0).all(),'ATR_ASOF')
    need((np.isnan(atr)|(np.isfinite(atr)&(atr>=0))).all(),'ATR_DOMAIN')
    with np.errstate(over='ignore',invalid='ignore'):
        expected_margin=atr*spec['noise_margin_contract']['margin_atr']
    margin_known=np.isfinite(expected_margin)&(expected_margin>=0)
    expected_margin=np.where(margin_known,expected_margin,np.nan)
    need(np.array_equal(margin,expected_margin,equal_nan=True),'MARGIN_VALUES')
    masks={}
    for task in ('open','close'):
        fields=spec['tasks'][task];delta=truth['noise_'+task+'_delta']
        known=truth[fields['known']];eligible=truth[fields['eligible']];sign=truth[fields['sign']]
        endpoints=raw_masks['close'];finite=np.isfinite(delta);zero=delta==0
        need(np.isnan(delta[~endpoints]).all(),'IMMATURE_DELTA_'+task)
        need(np.array_equal(np.sign(delta[endpoints&finite]).astype(np.int8),
            truth[task+'_sign'][endpoints&finite]),'RAW_DELTA_SIGN_'+task)
        expected_known=endpoints&finite&(zero|margin_known)
        expected_eligible=expected_known if task=='close'else expected_known&~truth['open_known_at_issue']
        expected_sign=np.zeros(shape,np.int8)
        directional=expected_known&~zero&(np.abs(delta)>=margin)
        expected_sign[directional]=np.sign(delta[directional]).astype(np.int8)
        need(np.array_equal(known,expected_known)and np.array_equal(eligible,expected_eligible),'MASKS_'+task)
        need(np.array_equal(sign,expected_sign),'SIGNS_'+task)
        masks[task]=eligible
    masks['open_known']=truth[spec['tasks']['open']['known']]&truth['open_known_at_issue']
    return masks
