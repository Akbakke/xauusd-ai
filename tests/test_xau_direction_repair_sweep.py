from pathlib import Path

from gx1.scripts.sweep_entry_smart_seq520_direction_repair_v1 import (
    FIXED_ENV,
    lint_trial_env,
    sample_trials,
    trial_command,
)


def test_xau_direction_repair_sweep_samples_xau_learning_knobs_only() -> None:
    trials = sample_trials(trials=4, seed=123)

    assert len(trials) == 4
    for env in trials:
        assert env["ENTRY_FOUNDATION_CANDIDATE_ANCHOR_GATE_INIT"] == "0.0"
        assert env["ENTRY_FOUNDATION_CANDIDATE_BAD_PATH_PROB_PENALTY"] == "0.0"
        assert "ENTRY_FOUNDATION_CANDIDATE_LR" in env
        assert "ENTRY_FOUNDATION_CANDIDATE_MULTI_TF_SCALE" in env
        assert float(env["ENTRY_FOUNDATION_CANDIDATE_DIRECTION_CE_SCALE"]) >= 2.0
        assert 0.45 <= float(env["ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_ALPHA"]) <= 0.50
        assert float(env["ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT"]) >= 2.5
        assert 0.0 < float(env["ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE"]) <= 0.50
        assert float(env["ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT"]) >= 0.0
        assert float(env["ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_RECALL_LOSS_WEIGHT"]) >= 0.0
        assert float(env["ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_BALANCED_CE_WEIGHT"]) >= 2.0
        assert env["ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE"] == "0.10"
        assert env["ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS"] == "8"
        assert env["ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_LOSS_AGGREGATION"] == "mean"
        assert lint_trial_env(env) == []
        assert not any("EUR" in key.upper() for key in env)
    assert FIXED_ENV["ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_CLASS_WEIGHTS"] == "1.0,1.0,4.0"
    assert FIXED_ENV["ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_TARGET"] == "label"
    assert FIXED_ENV["ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_CTX_CAT_INDICES"] == "0,1,2,3,4"


def test_xau_direction_repair_sweep_command_uses_smart_wrapper_and_dry_run() -> None:
    env = sample_trials(trials=1, seed=7)[0]
    cmd = trial_command(
        trial_idx=1,
        trial_env=env,
        dataset_dir=Path("/tmp/xau_dataset"),
        out_bundle_dir=Path("/tmp/xau_bundle_trial_001"),
        vedtak="SMART_SEQ520_XAU_DIRECTION_REPAIR_SWEEP_DRY_RUN",
        epochs=3,
        batch_size=96,
        subsample_rows=90000,
        seed=20260713,
        dry_run=True,
    )
    text = " ".join(cmd)

    assert "scripts/run_entry_foundation_seq146_candidate_train.sh" in text
    assert "--smart-seq520" in cmd
    assert "--dry-run" in cmd
    assert "--dataset-dir" in cmd
    assert "--out-bundle-dir" in cmd
    assert "--subsample-rows" in cmd
    assert "ENTRY_FOUNDATION_CANDIDATE_LR=" in text
    assert "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT=" in text
    assert "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_BALANCED_CE_WEIGHT=" in text
    assert "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE=0.10" in text
    assert "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS=8" in text
    assert "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_LOSS_AGGREGATION=mean" in text
    assert "FOREIGN_FX" not in text.upper()


def test_xau_direction_repair_sweep_lints_invalid_contract_values() -> None:
    env = sample_trials(trials=1, seed=11)[0]
    env["ENTRY_FOUNDATION_CANDIDATE_DIRECTION_CE_SCALE"] = "1.75"
    env["ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_ALPHA"] = "0.35"
    env["ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT"] = "2.0"
    env["ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE"] = "1.0"
    env["ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_BALANCED_CE_WEIGHT"] = "1.0"
    env["ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE"] = "0.05"
    env["ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS"] = "4"
    env["ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_LOSS_AGGREGATION"] = "sqrt"

    failures = lint_trial_env(env)

    assert any("DIRECTION_CE_SCALE" in item for item in failures)
    assert any("PRED_BALANCE_ALPHA" in item for item in failures)
    assert any("DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT" in item for item in failures)
    assert any("DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE" in item for item in failures)
    assert any("DIRECTION_SLICE_BALANCED_CE_WEIGHT" in item for item in failures)
    assert any("DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE" in item for item in failures)
    assert any("DIRECTION_SLICE_BALANCED_CE_MIN_ROWS" in item for item in failures)
    assert any("DIRECTION_SLICE_LOSS_AGGREGATION" in item for item in failures)
