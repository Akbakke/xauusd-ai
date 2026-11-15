"""Full Fold 2 support binding and masked multiminute side gradients."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gx1.contracts.market_state_fold2_multiminute_path_v1 import QUERY_MINUTES
from gx1.contracts.market_state_fold2_multiminute_side_loss_v1 import (
    FIT_BATCHES, FIT_INFORMATIVE, FIT_KNOWN_SIDE, SUPPORT_REVIEW_SHA256,
    allowed_set_loss, contract,
)


def test_exact_full_fit_support_is_bound_to_immutable_review():
    p = Path(__file__).parents[1] / 'docs/S_FOLD2_MULTIMINUTE_FIT_SUPPORT_20261115.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == SUPPORT_REVIEW_SHA256
    support = json.loads(p.read_text())
    assert support['schema_version'] == 'gx1_fold2_multiminute_fit_support_v1'
    assert support['query_sha256'] == contract()['query_sha256']
    assert support['fit_rows'] == 47338
    assert support['fit_batches_b16'] == FIT_BATCHES == 2959
    assert tuple(map(tuple, support['informative_support_query_by_TF'])) == FIT_INFORMATIVE
    assert tuple(map(tuple, support['known_side_support_query_by_TF'])) == FIT_KNOWN_SIDE
    assert support['model_forwards'] == support['optimizer_steps'] == 0
    assert support['training_authority'] is False
    assert contract()['fresh_initial_baseline_required_before_fit'] is True


def test_loss_uses_separate_activity_and_known_side_support():
    query = torch.tensor(np.broadcast_to(QUERY_MINUTES[None], (2, 5, 5)).copy(),
                         dtype=torch.int64)
    raw = torch.zeros((2, 5, 5, 2), dtype=torch.float32, requires_grad=True)
    allowed = torch.full((2, 5, 5), 7, dtype=torch.uint8)
    allowed[0, 0, 0] = 5  # activity known, side unknown
    allowed[0, 1, 0] = 4
    allowed[0, 2, 0] = 2  # flat has no side gradient
    allowed[0, 3, 0] = 1
    eligible = torch.ones((2, 5, 5), dtype=torch.bool)
    mature = torch.ones((2, 5, 5), dtype=torch.bool)
    mature[1, 0, 0] = False
    eligible[1, 0, 0] = False
    result = allowed_set_loss(raw, allowed, eligible, mature, query)
    expected = torch.log(torch.tensor(2.0)) * FIT_BATCHES / 25 * (
        1/FIT_INFORMATIVE[0][0] + 1/FIT_INFORMATIVE[1][0]
        + 1/FIT_INFORMATIVE[2][0] + 1/FIT_INFORMATIVE[3][0]
        + 1/FIT_KNOWN_SIDE[1][0] + 1/FIT_KNOWN_SIDE[3][0])
    torch.testing.assert_close(result['loss'], expected, rtol=1e-6, atol=1e-7)
    assert result['known_side_counts'][:, 0].tolist() == [0, 1, 0, 1, 0]
    result['loss'].backward()
    assert raw.grad[0, 0, 0, 0] != 0
    assert raw.grad[0, 0, 0, 1] == 0
    assert raw.grad[0, 1, 0, 1] != 0
    assert raw.grad[0, 2, 0, 1] == 0
    assert torch.count_nonzero(raw.grad[1]) == 0


def test_query_or_mask_change_is_rejected_without_optimizer():
    query = torch.tensor(np.broadcast_to(QUERY_MINUTES[None], (1, 5, 5)).copy(),
                         dtype=torch.int64)
    raw = torch.zeros((1, 5, 5, 2), dtype=torch.float32)
    allowed = torch.full((1, 5, 5), 7, dtype=torch.uint8)
    ready = torch.ones((1, 5, 5), dtype=torch.bool)
    query[0, 1, 0] = 5
    with pytest.raises(ValueError, match='SHAPE_MASK_OR_QUERY'):
        allowed_set_loss(raw, allowed, ready, ready, query)
    query[0, 1, 0] = 2
    mature = ready.clone()
    mature[0, 0, 0] = False
    with pytest.raises(ValueError, match='SHAPE_MASK_OR_QUERY'):
        allowed_set_loss(raw, allowed, ready, mature, query)
