import numpy as np

from gx1.audit.qualify_market_state_fold2_initial_mask_observation_v1 import (
    classify_pairs,
)


def test_classify_physical_masks_for_zero_gradient_channels():
    values = np.array([[[7, 0, 4, 1, 2], [8, 0, 0, 0, 2]]], dtype=np.float32)
    names = ['missing_value', 'missing_observed',
             'seen_value', 'seen_observed', 'category']
    result = classify_pairs(values, names, ((0, 1), (2, 3)), {0, 2, 4})
    assert result['zero_gradient_unobserved_pairs'] == 1
    assert result['zero_gradient_observed_pairs'] == [{
        'index': 2, 'name': 'seen_value',
        'observed_elements': 1, 'observed_nonzero_value_elements': 1}]
    assert result['unpaired_zero_gradient_fields'] == [
        {'index': 4, 'name': 'category'}]
