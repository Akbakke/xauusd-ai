"""Fixed CPU LightGBM diagnostic for the five market-state endpoints.

The caller binds causal feature rows and target masks, freezes the TRAIN prior,
then separately admits the CPU research fit. This module has no file or launch
side effects and never inspects future labels when building features.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from gx1.contracts.market_state_five_endpoint_contract_v1 import TIMEFRAMES
from gx1.contracts.market_state_tabular_target_score_v1 import combine_binary_probabilities

EXPECTED_WIDTH = 2727 + 1 + 5 * 255
EXPECTED_CATEGORICAL = 7
EXPECTED_CATEGORICAL_DOMAINS = {
    'signal.smc_swing_state': (0, 1, 2, 3, 4),
    'ctx_cat.session_id': (0, 1, 2, 3),
    **{f'mtf_{tf.lower()}.mtf_smc_swing_state': (0, 1, 2, 3, 4)
       for tf in TIMEFRAMES},
}
LIGHTGBM_VERSION = '4.6.0'
BOOST_ROUNDS = 64
RECIPE = {
    'objective': 'binary',
    'metric': 'None',
    'device_type': 'cpu',
    'num_threads': 1,
    'deterministic': True,
    'force_col_wise': True,
    'zero_as_missing': False,
    'use_missing': True,
    'num_leaves': 15,
    'min_data_in_leaf': 512,
    'max_bin': 63,
    'learning_rate': 0.05,
    'lambda_l2': 1.0,
    'feature_fraction': 1.0,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'seed': 20261107,
    'verbosity': -1,
}


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError('MARKET_STATE_TABULAR_LIGHTGBM_' + code)


def _import_lightgbm():
    import lightgbm
    _require(lightgbm.__version__ == LIGHTGBM_VERSION, 'VERSION')
    return lightgbm


def fixed_recipe() -> dict:
    """Return a fresh copy; score windows cannot tune any parameter."""
    return {'library': 'lightgbm', 'version': LIGHTGBM_VERSION,
            'boost_rounds': BOOST_ROUNDS, 'params': dict(RECIPE),
            'heads': ['activity', 'conditional_side'],
            'timeframes': list(TIMEFRAMES),
            'early_stopping': False, 'class_weighting': False,
            'score_driven_selection': False}


def _categorical_indices(feature_names: tuple[str, ...], categorical: tuple[dict, ...],
                         train_matrix: np.ndarray, score_matrix: np.ndarray) -> list[int]:
    _require(isinstance(feature_names, tuple) and len(feature_names) == EXPECTED_WIDTH
             and len(set(feature_names)) == len(feature_names)
             and all(isinstance(name, str) and name for name in feature_names),
             'FEATURE_NAMES')
    _require(isinstance(categorical, tuple) and len(categorical) == EXPECTED_CATEGORICAL,
             'CATEGORICAL_COUNT')
    indices: list[int] = []
    for item in categorical:
        _require(isinstance(item, Mapping) and set(item) == {'index', 'name', 'domain'},
                 'CATEGORICAL_SCHEMA')
        index, domain = item['index'], item['domain']
        _require(type(index) is int and 0 <= index < EXPECTED_WIDTH
                 and item['name'] == feature_names[index]
                 and isinstance(domain, (tuple, list)) and len(domain) >= 2
                 and all(type(value) is int and 0 <= value < 2**31 for value in domain)
                 and len(set(domain)) == len(domain), 'CATEGORICAL_DOMAIN')
        for matrix in (train_matrix, score_matrix):
            _require(np.isin(matrix[:, index], domain).all(),
                     'CATEGORICAL_VALUES:' + item['name'])
        indices.append(index)
    _require(len(set(indices)) == EXPECTED_CATEGORICAL, 'CATEGORICAL_DUPLICATE')
    _require(indices == sorted(indices)
             and {item['name']: tuple(item['domain']) for item in categorical}
                 == EXPECTED_CATEGORICAL_DOMAINS, 'CATEGORICAL_IDENTITY')
    return indices


def _validate_inputs(train_matrix: np.ndarray, score_matrix: np.ndarray,
                     binary: dict, feature_names: tuple[str, ...],
                     categorical: tuple[dict, ...]) -> list[int]:
    _require(isinstance(train_matrix, np.ndarray) and isinstance(score_matrix, np.ndarray)
             and train_matrix.dtype == score_matrix.dtype == np.float32
             and train_matrix.ndim == score_matrix.ndim == 2
             and 0 < train_matrix.shape[0] and 0 < score_matrix.shape[0]
             and train_matrix.shape[1] == score_matrix.shape[1] == EXPECTED_WIDTH
             and not np.isinf(train_matrix).any() and not np.isinf(score_matrix).any(),
             'MATRIX')
    _require(isinstance(binary, Mapping)
             and all(key in binary for key in ('activity_mask', 'activity_label',
                                                'certified_side_mask', 'side_up_label')),
             'BINARY_KEYS')
    shape = (len(train_matrix), len(TIMEFRAMES))
    for mask_key, label_key in (('activity_mask', 'activity_label'),
                                ('certified_side_mask', 'side_up_label')):
        mask, label = binary[mask_key], binary[label_key]
        _require(isinstance(mask, np.ndarray) and mask.dtype == np.bool_
                 and isinstance(label, np.ndarray) and label.dtype == np.uint8
                 and mask.shape == label.shape == shape
                 and np.isin(label, (0, 1)).all()
                 and np.all(label[~mask] == 0), 'BINARY_TARGETS:' + mask_key)
    _require(np.all(binary['certified_side_mask'] <= binary['activity_mask'])
             and np.all(binary['activity_label'][binary['certified_side_mask']] == 1),
             'SIDE_REQUIRES_ACTIVE')
    return _categorical_indices(feature_names, categorical, train_matrix, score_matrix)


def fit_predict_fold(train_matrix: np.ndarray, score_matrix: np.ndarray,
                     binary: dict, *, feature_names: tuple[str, ...],
                     categorical: tuple[dict, ...]) -> dict:
    """Fit fixed binary heads on masked TRAIN rows; predict every score row.

    The caller owns exact chronology, output publication and one-use policy.
    Models are returned so the caller can save them only under its claimed RUNS
    directory. No score target enters this function.
    """
    category_indices = _validate_inputs(train_matrix, score_matrix, binary,
                                         feature_names, categorical)
    lgb = _import_lightgbm()
    score_rows = len(score_matrix)
    p_active = np.empty((score_rows, len(TIMEFRAMES)), dtype=np.float64)
    p_up = np.empty_like(p_active)
    boosters: dict[str, object] = {}
    fit_counts: dict[str, dict[str, dict[str, int]]] = {}
    for tf_index, tf in enumerate(TIMEFRAMES):
        fit_counts[tf] = {}
        for name, mask_key, label_key, output in (
                ('activity', 'activity_mask', 'activity_label', p_active),
                ('conditional_side', 'certified_side_mask', 'side_up_label', p_up)):
            mask = binary[mask_key][:, tf_index]
            label = binary[label_key][mask, tf_index]
            counts = np.bincount(label, minlength=2)
            _require(len(label) >= 2 and len(counts) == 2
                     and int(counts[0]) > 0 and int(counts[1]) > 0,
                     'TWO_CLASS_SUPPORT:' + tf + ':' + name)
            selected = np.ascontiguousarray(train_matrix[mask], dtype=np.float32)
            dataset = lgb.Dataset(selected, label=label,
                                  feature_name=list(feature_names),
                                  categorical_feature=category_indices,
                                  free_raw_data=True)
            booster = lgb.train(dict(RECIPE), dataset, num_boost_round=BOOST_ROUNDS)
            predictions = np.asarray(booster.predict(score_matrix, num_threads=1),
                                     dtype=np.float64)
            _require(predictions.shape == (score_rows,)
                     and np.isfinite(predictions).all()
                     and np.all((predictions > 0) & (predictions < 1)),
                     'PREDICTION:' + tf + ':' + name)
            output[:, tf_index] = predictions
            boosters[tf + ':' + name] = booster
            fit_counts[tf][name] = {'rows': len(label), 'negative': int(counts[0]),
                                    'positive': int(counts[1])}
            del selected, dataset
    probabilities = combine_binary_probabilities(p_active, p_up)
    _require(probabilities.shape == (score_rows, len(TIMEFRAMES), 3), 'OUTPUT_SHAPE')
    return {'probabilities': probabilities, 'boosters': boosters,
            'fit_counts': fit_counts, 'recipe': fixed_recipe()}
