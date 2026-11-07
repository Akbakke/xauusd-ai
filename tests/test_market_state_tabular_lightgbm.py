"""Small fake-backend tests; never fit LightGBM on project or synthetic rows."""
from types import SimpleNamespace

import numpy as np
import pytest

from gx1.contracts import market_state_tabular_lightgbm_v1 as model
from gx1.contracts.market_state_tabular_target_score_v1 import binary_fit_targets


def _inputs():
    names = [f'field_{i}' for i in range(model.EXPECTED_WIDTH)]
    nominal = ['signal.smc_swing_state', 'ctx_cat.session_id'] + [
        f'mtf_{tf.lower()}.mtf_smc_swing_state' for tf in ('M5', 'M15', 'H1', 'H4', 'D1')]
    for i, name in enumerate(nominal):
        names[i] = name
    categories = tuple({'index': i, 'name': name,
                        'domain': tuple(range(4 if i == 1 else 5))}
                       for i, name in enumerate(nominal))
    train = np.zeros((10, model.EXPECTED_WIDTH), dtype=np.float32)
    score = np.zeros((3, model.EXPECTED_WIDTH), dtype=np.float32)
    allowed = np.array([1, 2, 4, 5, 7, 1, 4, 2, 1, 4], dtype=np.uint8)
    allowed = np.repeat(allowed[:, None], 5, axis=1)
    ready = np.ones_like(allowed, dtype=np.bool_)
    binary = binary_fit_targets(allowed, ready, ready)
    return train, score, binary, tuple(names), categories


def _fake_backend(calls, *, invalid=False):
    def dataset(matrix, *, label, feature_name, categorical_feature, free_raw_data):
        calls.append(('dataset', matrix.shape, label.copy(), len(feature_name),
                      tuple(categorical_feature), free_raw_data))
        return SimpleNamespace(label=label.copy())

    def train(params, data, *, num_boost_round):
        calls.append(('train', dict(params), num_boost_round))
        probability = (int(data.label.sum()) + 1) / (len(data.label) + 2)
        return SimpleNamespace(predict=lambda matrix, *, num_threads: np.full(
            len(matrix), np.nan if invalid else probability, dtype=np.float64))

    return SimpleNamespace(Dataset=dataset, train=train, __version__=model.LIGHTGBM_VERSION)


def test_fixed_two_head_five_tf_recipe_uses_exact_masks_and_no_eval(monkeypatch):
    calls = []
    monkeypatch.setattr(model, '_import_lightgbm', lambda: _fake_backend(calls))
    train, score, binary, names, categories = _inputs()
    result = model.fit_predict_fold(train, score, binary,
                                    feature_names=names, categorical=categories)
    datasets = [item for item in calls if item[0] == 'dataset']
    training = [item for item in calls if item[0] == 'train']
    assert len(datasets) == len(training) == len(result['boosters']) == 10
    assert [item[1][0] for item in datasets] == [9, 6] * 5
    assert all(item[1][1] == model.EXPECTED_WIDTH and item[3] == model.EXPECTED_WIDTH
               and item[4] == tuple(range(7)) and item[5] is True for item in datasets)
    assert all(params == model.RECIPE and rounds == 64 for _, params, rounds in training)
    assert result['probabilities'].shape == (3, 5, 3)
    assert np.allclose(result['probabilities'].sum(axis=-1), 1.0)
    assert result['fit_counts']['M5']['activity'] == {'rows': 9, 'negative': 2, 'positive': 7}
    assert result['fit_counts']['M5']['conditional_side'] == {'rows': 6, 'negative': 3, 'positive': 3}
    assert result['recipe']['early_stopping'] is False
    assert result['recipe']['class_weighting'] is False
    assert result['recipe']['score_driven_selection'] is False
    assert not any(key in model.RECIPE for key in ('scale_pos_weight', 'is_unbalance', 'class_weight'))


def test_rejects_changed_categorical_domain_and_nonfinite_prediction(monkeypatch):
    train, score, binary, names, categories = _inputs()
    changed = list(categories)
    changed[0] = {**changed[0], 'domain': (1, 2, 3, 4)}
    with pytest.raises(ValueError, match='CATEGORICAL_VALUES'):
        model.fit_predict_fold(train, score, binary,
                               feature_names=names, categorical=tuple(changed))
    changed[0] = {**changed[0], 'domain': (0, 1, 2, 3)}
    with pytest.raises(ValueError, match='CATEGORICAL_IDENTITY'):
        model.fit_predict_fold(train, score, binary,
                               feature_names=names, categorical=tuple(changed))
    train[0, 0] = 7.
    with pytest.raises(ValueError, match='CATEGORICAL_VALUES'):
        model.fit_predict_fold(train, score, binary,
                               feature_names=names, categorical=categories)
    train[0, 0] = 0.
    monkeypatch.setattr(model, '_import_lightgbm', lambda: _fake_backend([], invalid=True))
    with pytest.raises(ValueError, match='PREDICTION'):
        model.fit_predict_fold(train, score, binary,
                               feature_names=names, categorical=categories)


def test_rejects_unmasked_unknown_labels_and_one_class_support(monkeypatch):
    train, score, binary, names, categories = _inputs()
    changed = dict(binary)
    changed['side_up_label'] = binary['side_up_label'].copy()
    changed['side_up_label'][4, 0] = 1
    with pytest.raises(ValueError, match='BINARY_TARGETS'):
        model.fit_predict_fold(train, score, changed,
                               feature_names=names, categorical=categories)
    changed = dict(binary)
    changed['certified_side_mask'] = binary['certified_side_mask'].copy()
    changed['certified_side_mask'][:, 0] = False
    changed['side_up_label'] = binary['side_up_label'].copy()
    changed['side_up_label'][:, 0] = 0
    monkeypatch.setattr(model, '_import_lightgbm', lambda: _fake_backend([]))
    with pytest.raises(ValueError, match='TWO_CLASS_SUPPORT'):
        model.fit_predict_fold(train, score, changed,
                               feature_names=names, categorical=categories)


def test_numeric_nan_is_missing_but_infinity_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(model, '_import_lightgbm', lambda: _fake_backend(calls))
    train, score, binary, names, categories = _inputs()
    train[0, 7] = np.nan
    score[0, 7] = np.nan
    output = model.fit_predict_fold(train, score, binary,
                                    feature_names=names, categorical=categories)
    assert output['probabilities'].shape == (3, 5, 3)
    assert len([item for item in calls if item[0] == 'train']) == 10
    train[0, 7] = np.inf
    with pytest.raises(ValueError, match='MATRIX'):
        model.fit_predict_fold(train, score, binary,
                               feature_names=names, categorical=categories)


def test_library_version_is_exact(monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'lightgbm',
                        SimpleNamespace(__version__='4.5.0'))
    with pytest.raises(ValueError, match='VERSION'):
        model._import_lightgbm()
