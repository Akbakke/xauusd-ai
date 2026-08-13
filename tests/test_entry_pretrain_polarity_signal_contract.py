from __future__ import annotations

import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_PRETRAIN_POLARITY_SIGNAL_CONTRACT,
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_pretrain_polarity_signal_v1 import (
    PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS,
    SUPPORT_STACK_FEATURE,
    pretrain_polarity_signal_contract_metadata,
)
from gx1.scripts.audit_xau_direction_repair_pretrain_v1 import (
    REQUIRED_POLARITY_FEATURES,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields


def test_every_pretrain_polarity_input_is_code_owned_and_mandatory() -> None:
    mandatory = set(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
    assert REQUIRED_POLARITY_FEATURES == PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS
    assert set(PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS) <= mandatory
    assert MODEL_NATIVE_PRETRAIN_POLARITY_SIGNAL_CONTRACT == (
        pretrain_polarity_signal_contract_metadata(
            MODEL_NATIVE_MANDATORY_SELECTED_FIELDS
        )
    )
    signal_contract = model_native_signal_contract_metadata(
        canonical_model_native_selected_fields()
    )
    assert signal_contract["pretrain_polarity_signal_contract"] == (
        MODEL_NATIVE_PRETRAIN_POLARITY_SIGNAL_CONTRACT
    )


def test_pretrain_polarity_contract_rejects_ranking_owned_input() -> None:
    mandatory = list(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
    mandatory.remove(SUPPORT_STACK_FEATURE)
    with pytest.raises(
        RuntimeError,
        match="PRETRAIN_POLARITY_SIGNAL_REQUIREMENTS_NOT_MANDATORY",
    ):
        pretrain_polarity_signal_contract_metadata(mandatory)
