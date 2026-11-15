"""Input-channel gradient reporting for the one-use initial witness."""
import torch

from gx1.audit.qualify_market_state_fold2_multiminute_initial_v1 import (
    SCOPE, gradient_summary,
)
from gx1.contracts.market_state_fold2_multiminute_path_v1 import QUERY_MINUTES, QUERY_SHA256
from gx1.contracts.market_state_fold2_multiminute_side_loss_v1 import contract


def test_gradient_summary_reports_unused_channels_without_inventing_consumption():
    x = torch.ones((2, 3, 4), requires_grad=True)
    (x[..., 0] + 2 * x[..., 2]).sum().backward()
    got = gradient_summary(x, feature_channels=True)
    assert got['present'] and got['finite']
    assert got['feature_channels'] == 4
    assert got['nonzero_channels'] == 2
    assert got['zero_channel_indices'] == [1, 3]
    assert got['nonzero_elements'] == 12
    y = torch.ones((2, 3, 4), requires_grad=True)
    missing = gradient_summary(y, feature_channels=True)
    assert missing['present'] is False
    assert missing['zero_channel_indices'] == [0, 1, 2, 3]


def test_read_only_scope_is_exactly_the_fixed_path_and_loss():
    assert SCOPE['query_minutes'] == QUERY_MINUTES.tolist()
    assert SCOPE['query_sha256'] == QUERY_SHA256
    assert SCOPE['loss_contract_sha256'] == contract()['contract_sha256']
    assert SCOPE['model_constructions'] == 1
    assert SCOPE['backbone_encode_calls'] == SCOPE['head_decode_calls'] == 1
    assert SCOPE['optimizer_steps'] == SCOPE['checkpoint_reads'] == 0
    assert SCOPE['test_data_used'] is False
