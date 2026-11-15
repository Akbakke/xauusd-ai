import hashlib
import json

import numpy as np
import pytest

from gx1.contracts.market_state_fold2_multiminute_path_v1 import QUERY_MINUTES
from gx1.contracts.market_state_fold2_multiminute_prior_v1 import (
    fit_exact_fold2_prior, prior_grid, require_exact_fold2_prior, SCHEMA,
)


def test_fixed_path_prior_keeps_activity_and_unknown_side_separate():
    allowed = np.full((4, 5, 5), 2, dtype=np.uint8)
    allowed[:, 0, 0] = [1, 4, 5, 2]
    allowed[:, 1, 0] = 2
    allowed[0, 2, 1] = 7
    eligible = np.ones_like(allowed, dtype=bool)
    mature = np.ones_like(allowed, dtype=bool)
    query = np.broadcast_to(QUERY_MINUTES, allowed.shape).copy()
    result = prior_grid(allowed, eligible, mature, query)
    assert result['raw_logits'].shape == (5, 5, 2)
    assert result['activity_support'][0, 0] == 4
    assert result['known_side_support'][0, 0] == 2
    assert result['counts'][0]['M5'] == {
        'activity': 4, 'active': 3, 'certified_side': 2, 'up': 1}
    assert np.isclose(result['raw_logits'][0, 0, 0], np.log(4 / 2))
    assert np.isclose(result['raw_logits'][0, 0, 1], 0)
    assert result['known_side_support'][1, 0] == 0
    assert np.isclose(result['raw_logits'][1, 0, 1], 0)
    assert result['activity_support'][2, 1] == 3
    with pytest.raises(ValueError, match='FIT_AXIS'):
        fit_exact_fold2_prior(
            {'allowed': allowed, 'eligible': eligible, 'mature': mature,
             'query_minutes': query, 'issue_ns': np.arange(4, dtype=np.int64),
             'query_sha256': 'wrong'}, np.arange(4, dtype=np.int64),
            expected_parent_rows_sha256='0' * 64,
            expected_issue_ns_sha256='0' * 64)


def test_saved_prior_logits_must_match_sealed_support_counts():
    from gx1.contracts.market_state_fold2_multiminute_path_v1 import QUERY_SHA256
    from gx1.contracts.market_state_fold2_multiminute_side_loss_v1 import (
        FIT_INFORMATIVE, FIT_KNOWN_SIDE, FIT_ROWS, contract,
    )
    from gx1.contracts.market_state_time_query_target_v1 import TIMEFRAMES

    counts, raw = [], []
    for q in range(5):
        row, logits = {}, []
        for j, tf in enumerate(TIMEFRAMES):
            activity, side = FIT_INFORMATIVE[q][j], FIT_KNOWN_SIDE[q][j]
            up = side // 2
            row[tf] = {'activity': activity, 'active': side,
                       'certified_side': side, 'up': up}
            p_active = (side + 1) / (activity + 2)
            p_up = (up + 1) / (side + 2)
            logits.append([np.log(p_active / (1 - p_active)),
                           np.log(p_up / (1 - p_up))])
        counts.append(row)
        raw.append(logits)
    value = {'schema_version': SCHEMA, 'fold_id': 2, 'partition': 'fit',
             'fit_rows': FIT_ROWS, 'query_sha256': QUERY_SHA256,
             'parent_rows_sha256': 'a' * 64, 'issue_ns_sha256': 'b' * 64,
             'loss_contract_sha256': contract()['contract_sha256'],
             'raw_logits': raw,
             'activity_support': [list(x) for x in FIT_INFORMATIVE],
             'known_side_support': [list(x) for x in FIT_KNOWN_SIDE],
             'counts': counts,
             'smoothing': 'laplace_plus_one_separate_binary_activity_and_conditional_side',
             'test_data_used': False}
    def seal(item):
        encoded = json.dumps(item, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()
        return {**item, 'contract_sha256': hashlib.sha256(encoded).hexdigest()}
    assert require_exact_fold2_prior(
        seal(value), parent_rows_sha256='a' * 64,
        issue_ns_sha256='b' * 64).shape == (5, 5, 2)
    value['raw_logits'][0][0][0] += 0.01
    with pytest.raises(ValueError, match='COUNT_LOGITS'):
        require_exact_fold2_prior(
            seal(value), parent_rows_sha256='a' * 64,
            issue_ns_sha256='b' * 64)
