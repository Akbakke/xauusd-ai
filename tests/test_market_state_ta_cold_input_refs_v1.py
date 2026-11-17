"""Cold input owner binds both already qualified prefix sidecar receipts."""
from __future__ import annotations

import pytest
import torch

from gx1.audit.market_state_ta_apr_jun_campaign_inputs_v1 import (
    COLD_PREFIX_SPANS, COLD_PRIOR_ROOT, COLD_PACKET, COLD_ROOT,
    cold_campaign_input_refs, prepare, read, ref,
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
    assert refs['priors'] == ref(COLD_PRIOR_ROOT / 'ENDPOINT_PRIORS.json')
    prior_result = read(refs['prior_result'])
    prior_terminal = read(refs['prior_terminal'])
    assert prior_result['status'] == prior_terminal['status'] == 'COMPLETE'
    assert prior_result['endpoint_priors'] == refs['priors']
    assert prior_terminal['result'] == refs['prior_result']
    for missing in ('prefix_span_result', 'prior_result', 'prior_terminal'):
        with pytest.raises(RuntimeError, match='COLD_INPUT_REFS'):
            prepare(device=torch.device('cpu'),
                    cold_inputs={k: v for k, v in refs.items() if k != missing})


def test_cold_units_profile_is_explicit_and_matches_runtime_normalization():
    from gx1.contracts.entry_native_direction_feature_view_v1 import _require_view
    from gx1.contracts.entry_model_native_input_normalization_v1 import (
        project_ta_input_normalization_contract,
    )
    packet = read(ref(COLD_PACKET))
    view = read(ref(COLD_ROOT / 'FEATURE_VIEW.json'))
    units = packet['unobserved_units_profile']
    assert units == view['unobserved_units_profile']
    with pytest.raises(RuntimeError, match='ANCHOR_SPAN_EXPLICIT_UNITS'):
        _require_view(view, unobserved_units_profile=None)
    _require_view(view, unobserved_units_profile=units)
    runtime = project_ta_input_normalization_contract(
        packet['base_normalization'], unobserved_units_profile=units)
    assert runtime['contract_sha256'] == view['prefix_normalization']['ta_base_normalization_sha256']
