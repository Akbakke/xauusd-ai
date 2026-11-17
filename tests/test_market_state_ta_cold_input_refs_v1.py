"""Cold input owner binds both already qualified prefix sidecar receipts."""
from __future__ import annotations

import pytest
import torch

from gx1.audit.market_state_ta_apr_jun_campaign_inputs_v1 import (
    COLD_PREFIX_SPANS, cold_campaign_input_refs, prepare, read, ref,
)


def test_cold_inputs_bind_prefix_receipts_before_gather():
    refs = cold_campaign_input_refs()
    assert refs['prefix_span_result'] == ref(COLD_PREFIX_SPANS / 'RESULT.json')
    assert refs['prefix_span_result']['sha256'] == (
        '8878d17a5bf114fcf9cd5eb0e74792ddc5df068b95b67951b594759d5b2b05bb')
    assert refs['prefix_span_terminal'] == ref(COLD_PREFIX_SPANS / 'TERMINAL.json')
    assert refs['prefix_span_terminal']['sha256'] == (
        '9414bc5dd6e07ff565e4cb0f3db7c2364bbea3d954a9708c03fc91c131034007')
    terminal = read(refs['prefix_span_terminal'])
    assert terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
    assert terminal['result'] == refs['prefix_span_result']
    with pytest.raises(RuntimeError, match='COLD_INPUT_REFS'):
        prepare(device=torch.device('cpu'),
                cold_inputs={k: v for k, v in refs.items() if k != 'prefix_span_result'})
