from __future__ import annotations

import pytest

from gx1.scripts.run_unified_exit_random_access_val_v1 import main


def test_val_cli_help_imports_without_loading_checkpoint(capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    output = capsys.readouterr().out
    assert "Strict weight-EMA full-cohort VAL executor" in output
    assert "--final-train-checkpoint-authority" in output
    assert "--rollout-progress-path" in output
