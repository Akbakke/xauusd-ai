import sys
import types

import pytest

from gx1.contracts.unified_exit_legacy_state_v1 import RETIRED_STATIC_EXIT_STATE_KEYS
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import _guard_no_rl


def test_retired_checkpoint_field_metadata_is_allowed():
    assert RETIRED_STATIC_EXIT_STATE_KEYS
    _guard_no_rl()


@pytest.mark.parametrize("name", [
    "gx1.rl",
    "gx1.rl.actor",
    "gx1.legacy.runner",
    "gx1.contracts.unified_exit_legacy_state_v1_other",
])
def test_execution_imports_remain_forbidden(monkeypatch, name):
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    with pytest.raises(RuntimeError, match="ENTRY_V10_CTX_(RL|LEGACY)_FORBIDDEN"):
        _guard_no_rl()
