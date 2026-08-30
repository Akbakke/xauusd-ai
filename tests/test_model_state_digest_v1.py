from __future__ import annotations

import torch

from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


def test_canonical_model_state_digest_matches_trainer_and_changes_with_tensor_state() -> None:
    model = torch.nn.Linear(3, 2)
    digest = canonical_model_state_sha256(model.state_dict())

    assert digest == trainer._model_state_sha256(model)

    changed = {name: value.detach().clone() for name, value in model.state_dict().items()}
    changed["bias"][0] += 1.0
    assert canonical_model_state_sha256(changed) != digest
