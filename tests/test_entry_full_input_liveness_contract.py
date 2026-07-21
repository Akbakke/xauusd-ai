from __future__ import annotations

from pathlib import Path

from gx1.contracts.entry_full_input_liveness_v1 import (
    PASS_DECISION,
    sha256_file,
    validate_full_input_liveness_artifact,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
)
from tests.entry_full_input_liveness_support import (
    full_input_field_order,
    write_full_input_liveness_fixture,
)


def test_full_input_liveness_validates_all_660_fields_on_all_splits(tmp_path) -> None:
    path, artifact, _ = write_full_input_liveness_fixture(tmp_path)

    result = validate_full_input_liveness_artifact(
        path,
        expected_sha256=sha256_file(path),
        expected_dataset_dir=tmp_path / "smart_dataset",
        expected_field_order=artifact["field_order"],
    )

    assert result["ok"] is True
    assert result["field_counts"] == {"signal": 513, "ctx_cont": 142, "ctx_cat": 5}
    assert result["field_status_row_count"] == 3 * (513 + 142 + 5)
    assert artifact["decision"] == PASS_DECISION
    assert artifact["atr_ood_drift"]["status"] == "STABLE"


def test_full_input_liveness_allowlist_is_exact_and_has_no_prefix_pass_through(tmp_path) -> None:
    order = full_input_field_order()
    order["signal"][-1] = "p_unreviewed_direction_hint"

    def make_unreviewed_field_constant(stats) -> None:
        for split in ("train", "val", "test"):
            stats[split]["signal"]["p_unreviewed_direction_hint"].update(
                {"mean": 0.0, "std": 0.0, "active_count": 0, "active_rate": 0.0}
            )

    path, artifact, _ = write_full_input_liveness_fixture(
        tmp_path,
        field_order=order,
        mutate_stats=make_unreviewed_field_constant,
    )
    result = validate_full_input_liveness_artifact(path)

    assert artifact["decision"] == "FAIL"
    assert result["ok"] is False
    assert any(
        row.get("field") == "p_unreviewed_direction_hint"
        and row.get("code") == "field_liveness_fail"
        for row in result["failures"]
    )


def test_oos_single_regime_state_is_observed_but_train_constant_fails(tmp_path) -> None:
    order = full_input_field_order()
    numeric_field = order["signal"][-1]
    categorical_field = order["ctx_cat"][0]

    def oos_single_state(stats) -> None:
        for split in ("val", "test"):
            stats[split]["signal"][numeric_field].update(
                {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "value_range": 0.0,
                    "active_count": 0,
                    "active_rate": 0.0,
                }
            )
            stats[split]["ctx_cat"][categorical_field].update(
                {"unique_count": 1, "unique_values": [1]}
            )

    path, artifact, _ = write_full_input_liveness_fixture(
        tmp_path / "oos_single_state",
        field_order=order,
        mutate_stats=oos_single_state,
    )
    assert artifact["decision"] == "PASS"
    assert validate_full_input_liveness_artifact(path)["ok"] is True
    observed = {
        (row["split"], row["surface"], row["field"]): row["status"]
        for row in artifact["field_status"]
    }
    assert observed[("val", "signal", numeric_field)] == "OBSERVED_SINGLE_STATE"
    assert observed[("test", "ctx_cat", categorical_field)] == "OBSERVED_SINGLE_STATE"

    def train_constant(stats) -> None:
        stats["train"]["signal"][numeric_field].update(
            {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "value_range": 0.0,
                "active_count": 0,
                "active_rate": 0.0,
            }
        )

    train_path, train_artifact, _ = write_full_input_liveness_fixture(
        tmp_path / "train_constant",
        field_order=order,
        mutate_stats=train_constant,
    )
    assert train_artifact["decision"] == "FAIL"
    assert validate_full_input_liveness_artifact(train_path)["ok"] is False


def test_oos_categorical_value_outside_train_vocabulary_fails(tmp_path) -> None:
    order = full_input_field_order()
    field = order["ctx_cat"][0]

    def unseen_category(stats) -> None:
        stats["val"]["ctx_cat"][field].update(
            {"unique_count": 4, "unique_values": [0, 1, 2, 9]}
        )

    path, artifact, _ = write_full_input_liveness_fixture(
        tmp_path,
        field_order=order,
        mutate_stats=unseen_category,
    )
    assert artifact["decision"] == "FAIL"
    result = validate_full_input_liveness_artifact(path)
    assert result["ok"] is False
    assert any(
        row.get("code") == "categorical_oos_value_outside_train_support"
        and row.get("field") == field
        and row.get("unseen_values") == [9]
        for row in artifact["failures"]
    )


def test_full_input_liveness_rejects_every_retired_bridge_field(tmp_path) -> None:
    for index, forbidden_field in enumerate(FORBIDDEN_LEGACY_BRIDGE_FIELDS):
        order = full_input_field_order()
        order["signal"][-1] = forbidden_field
        path, artifact, _ = write_full_input_liveness_fixture(
            tmp_path / f"forbidden_{index}",
            field_order=order,
        )

        result = validate_full_input_liveness_artifact(path)

        assert artifact["decision"] == "FAIL"
        assert result["ok"] is False
        assert any(
            row.get("code") == "forbidden_legacy_bridge_fields_present"
            and forbidden_field in row.get("fields", [])
            for row in artifact["failures"]
        )


def test_full_input_liveness_enforces_exact_rare_event_support_floor(tmp_path) -> None:
    def at_floor(stats) -> None:
        stats["train"]["signal"]["smc_choch"].update(
            {"std": 0.2, "active_count": 32, "active_rate": 0.0032}
        )

    pass_path, pass_artifact, _ = write_full_input_liveness_fixture(
        tmp_path / "at_floor",
        mutate_stats=at_floor,
    )
    pass_row = next(
        row
        for row in pass_artifact["field_status"]
        if row["split"] == "train" and row["field"] == "smc_choch"
    )
    assert pass_row["status"] == "ALLOWED_RARE_EVENT"
    assert validate_full_input_liveness_artifact(pass_path)["ok"] is True

    def below_floor(stats) -> None:
        stats["train"]["signal"]["smc_choch"].update(
            {"std": 0.2, "active_count": 31, "active_rate": 0.0031}
        )

    fail_path, fail_artifact, _ = write_full_input_liveness_fixture(
        tmp_path / "below_floor",
        mutate_stats=below_floor,
    )
    assert fail_artifact["decision"] == "FAIL"
    assert validate_full_input_liveness_artifact(fail_path)["ok"] is False


def test_full_input_liveness_fails_on_missing_field_and_hash_tamper_but_records_atr_shift(tmp_path) -> None:
    missing_order = full_input_field_order()
    missing_order["ctx_cat"].pop()
    missing_path, missing_artifact, _ = write_full_input_liveness_fixture(
        tmp_path / "missing",
        field_order=missing_order,
    )
    assert missing_artifact["decision"] == "FAIL"
    assert validate_full_input_liveness_artifact(missing_path)["ok"] is False

    def red_atr(stats) -> None:
        stats["val"]["signal"]["ctx_cont.d1_atr14_canon_v2"].update(
            {"mean": 10.0, "std": 0.05, "min": 9.5, "max": 10.5, "value_range": 1.0}
        )

    drift_path, drift_artifact, _ = write_full_input_liveness_fixture(
        tmp_path / "drift",
        mutate_stats=red_atr,
    )
    assert drift_artifact["atr_ood_drift"]["status"] == "SHIFT_OBSERVED"
    assert drift_artifact["decision"] == "PASS"
    assert validate_full_input_liveness_artifact(drift_path)["ok"] is True

    valid_path, _, _ = write_full_input_liveness_fixture(tmp_path / "hash")
    tampered_binding = validate_full_input_liveness_artifact(valid_path, expected_sha256="0" * 64)
    assert tampered_binding["ok"] is False
    assert any(row["code"] == "artifact_sha256_mismatch" for row in tampered_binding["failures"])


def test_full_input_liveness_fails_when_bound_manifest_or_scanned_parquet_changes(tmp_path) -> None:
    manifest_path, _, manifest_bindings = write_full_input_liveness_fixture(tmp_path / "manifest")
    bound_manifest = Path(manifest_bindings["train"]["path"])
    bound_manifest.write_text('{"split":"train","changed":true}\n', encoding="utf-8")

    manifest_result = validate_full_input_liveness_artifact(manifest_path)

    assert manifest_result["ok"] is False
    assert any(
        row["code"] == "split_manifest_binding_invalid" and row.get("split") == "train"
        for row in manifest_result["failures"]
    )

    parquet_path, artifact, _ = write_full_input_liveness_fixture(tmp_path / "parquet")
    bound_parquet = Path(
        artifact["input_bindings"]["fullscan_proof"]["test"]["parquet_path"]
    )
    bound_parquet.write_bytes(b"changed-after-fullscan")

    parquet_result = validate_full_input_liveness_artifact(parquet_path)

    assert parquet_result["ok"] is False
    assert any(
        row["code"] == "fullscan_proof_invalid" and row.get("split") == "test"
        for row in parquet_result["failures"]
    )
