"""The new prior must use the current loss and qualified Fold1 axes."""
import copy
import json
from pathlib import Path

import pytest

from gx1.audit import qualify_ta_ema_delta_fold1_priors_v1 as owner
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective


def actual():
    refs = owner.input_refs()
    values = {key: owner.read(binding) for key, binding in refs.items()}
    return refs, values


def test_old_prior_is_rejected_by_current_loss():
    old = json.loads(Path(
        '/home/andre2/GX1_RUNS/TA_M5_COLD_FOLD1_ENDPOINT_PRIORS_20261117_001/ENDPOINT_PRIORS.json'
    ).read_text())
    assert old['loss_contract_sha256'] != owner.LOSS_SHA
    assert objective.contract(ta_only=True, dedicated_m5_side=True)['contract_sha256'] == owner.LOSS_SHA
    with pytest.raises(ValueError, match='NATIVE_F_ENDPOINT_PRIOR_BINDING'):
        objective.require_endpoint_priors(old, fold_id=1, ta_only=True, dedicated_m5_side=True)


def test_genuine_gather_chain_is_only_new_view():
    refs, values = actual()
    panel = owner.validate_receipts(refs, values['gather_result'],
        values['gather_coverage'], values['gather_terminal'], values['view'])
    assert panel['prefix_rows'] == owner.FIT_ROWS
    assert values['gather_result']['fit']['parent_rows_sha256'] == owner.old.PARENT_SHA
    assert values['gather_coverage']['qualification'] == values['gather_result']['materialized_result']
    bad = copy.deepcopy(values['gather_coverage'])
    bad['qualification'] = refs['gather_result']
    with pytest.raises(RuntimeError, match='QUALIFIED_INPUT_CHAIN'):
        owner.validate_receipts(refs, values['gather_result'], bad,
                                values['gather_terminal'], values['view'])


def test_plan_binds_fit_only_current_contract():
    plan = owner.build_plan()
    assert plan['stage'] == owner.STAGE
    assert plan['inputs']['gather_result']['sha256'] == owner.GATHER_SHA['result']
    assert plan['scope']['fit_rows'] == 29574
    assert plan['scope']['loss_contract_sha256'] == owner.LOSS_SHA
    assert plan['scope']['later_target_values_in_fit'] is False
    assert plan['scope']['model_forwards'] == plan['scope']['optimizer_steps'] == 0
    assert plan['scope']['test_data_used'] is False
