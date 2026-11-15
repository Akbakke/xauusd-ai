"""Source-only Fold2 25-cell S private optimizer identity; no run admission.

Reuses the qualified B16/2-update AdamW values and exact selected 32 TRAIN
parents solely to qualify a future bit-exact S-only resume. No Entry/F loss,
full FIT, later score, TEST, broker, or trading authority follows.
"""
from __future__ import annotations

import hashlib
import json

from gx1.contracts.entry_model_native_train_recipe_v1 import native_direction_research_recipe
from gx1.contracts.market_state_fold2_multiminute_side_loss_v1 import contract as s_loss_contract
from gx1.contracts.market_state_time_query_model_v2 import factorized_time_query_model_contract
from gx1.contracts.market_state_time_query_private_campaign_v1 import PARENT_ROWS

SCHEMA = 'gx1_market_state_fold2_25_cell_S_private_numerical_v1'
BATCH_SIZE = 16
MAX_UPDATES = 2


def numerical_recipe() -> dict:
    """One exact S-only numerical contract, reusing the existing private values."""
    base = dict(native_direction_research_recipe(
        batch_size=BATCH_SIZE, max_updates=MAX_UPDATES))
    base.pop('contract_sha256')
    base['schema_version'] = SCHEMA
    base['research_objective_sha256'] = (
        factorized_time_query_model_contract()['contract_sha256'])
    base['fold2_25_cell_s_loss_contract_sha256'] = (
        s_loss_contract()['contract_sha256'])
    value = json.dumps(base, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=True, allow_nan=False).encode()
    return {**base, 'contract_sha256': hashlib.sha256(value).hexdigest()}


def require_private_scope(value: dict) -> dict:
    """Reject mode, cohort, or objective substitution before optimizer setup."""
    if not (isinstance(value, dict) and value == numerical_recipe()
            and tuple(PARENT_ROWS) == tuple(range(2, 34))):
        raise ValueError('FOLD2_25_CELL_S_PRIVATE_SCOPE')
    return value
