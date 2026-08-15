import argparse
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from tests.htf_v29_registry_test_support import synthetic_v29_registry_constants
from tests.volatility_squeeze_test_support import (
    make_volatility_squeeze_artifact_set,
)

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
    model_native_context_contract_metadata,
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_model_native_state_v2 import (
    MODEL_NATIVE_HISTORY_MODE,
    MODEL_NATIVE_STATE_SCHEMA_VERSION,
    RETIRED_RANK_STATE_FIELDS,
)
from gx1.contracts.entry_fitted_q_v1 import require_entry_fitted_q_contract
from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    _log_label_distribution_proof,
    _model_native_state_contract,
    entry_fitted_q_dataset_contract,
    write_manifest,
)
from gx1.utils.artifact_primitives_v1 import canonical_json_sha256
from tests.model_native_signal_support import canonical_model_native_selected_fields


_TAPE_PROVENANCE_FIXTURE = {
    "schema_version": "xau_tape_current_snapshot_v1",
    "instrument": "XAU_USD",
    "entry_run_id": "MODEL_NATIVE_DATASET_BUILD_PYTEST",
}


def _splits() -> dict[str, dict[str, str]]:
    return {
        "train": {
            "start": "2020-11-09 00:00:00+00:00",
            "end": "2025-09-30 23:59:59+00:00",
        },
        "val": {
            "start": "2025-10-01 00:00:00+00:00",
            "end": "2025-12-31 23:59:59+00:00",
        },
        "test": {
            "start": "2026-01-01 00:00:00+00:00",
            "end": "2026-06-26 03:25:00+00:00",
        },
    }


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "canonical_source.parquet"
    source.write_bytes(b"canonical-source-proof")
    return source


def _extra(tmp_path: Path, *, artifact_label: str = "default") -> dict:
    _source(tmp_path)
    signal_contract = model_native_signal_contract_metadata(
        canonical_model_native_selected_fields(
            remainder_prefix="session_regime.split_writer_fixture"
        )
    )
    cache_dir = tmp_path / "mtf_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_manifest = cache_dir / "manifest.json"
    cache_manifest.write_text('{"fixture":"v4"}\n', encoding="utf-8")
    cache_manifest_sha = hashlib.sha256(cache_manifest.read_bytes()).hexdigest()
    return {
        "xau_tape_provenance": dict(_TAPE_PROVENANCE_FIXTURE),
        # The retired direction/hierarchical label contracts are replaced by
        # the fitted-Q dataset contract; the writer now rejects a manifest that
        # still carries `entry_direction_target_policy`.
        **entry_fitted_q_dataset_contract(),
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
        "model_native_signal_contract": signal_contract,
        "signal_bridge": {
            "id": signal_contract["schema_version"],
            "fields": list(signal_contract["fields"]),
            "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
            "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
            "bridge_dim": 0,
            "bridge_source": None,
        },
        # The writer compares this against its owner byte for byte.
        "ctx_contract": model_native_context_contract_metadata(),
        "model_native_state_contract": {
            "schema_version": MODEL_NATIVE_STATE_SCHEMA_VERSION,
            "feature_history_start_utc": "2020-11-01T00:00:00Z",
            "feature_history_mode": MODEL_NATIVE_HISTORY_MODE,
            "split_reset_allowed": False,
            "runtime_rule_free": True,
            "entry_run_id": "MODEL_NATIVE_DATASET_BUILD_PYTEST",
        },
        "multi_tf_cache_binding": {
            "cache_dir": str(cache_dir.resolve()),
            "manifest_path": str(cache_manifest.resolve()),
            "manifest_sha256": cache_manifest_sha,
            "cache_identity_sha256": "d" * 64,
            "m5_prebuilt_source": str((tmp_path / "source.parquet").resolve()),
            "m5_prebuilt_source_sha256": "e" * 64,
            # V29 split manifests freeze the TRAIN-fitted registry constants
            # inside the binding; the writer validates them via their owner.
            "v29_registry_constants": synthetic_v29_registry_constants(),
            "volatility_squeeze_artifact_set": (
                make_volatility_squeeze_artifact_set(
                    tmp_path / f"squeeze_{artifact_label}"
                ).binding()
            ),
        },
    }


def test_canonical_writer_stamps_exact_seq513_schema_on_all_split_manifests(
    tmp_path: Path,
) -> None:
    for split_name in ("train", "val", "test"):
        output_path = tmp_path / f"model_native_seq513_{split_name}.parquet"
        extra = _extra(tmp_path, artifact_label=split_name)
        manifest_path = write_manifest(
            output_path=output_path,
            build_command=[
                "python",
                "-m",
                "gx1.scripts.build_entry_v10_ctx_training_dataset_v3",
            ],
            source_parquet=_source(tmp_path),
            tape_root=tmp_path / "tape",
            splits=_splits(),
            extra=extra,
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION
        assert manifest["manifest_variant"] == MODEL_NATIVE_CONTRACT_MODE
        assert manifest["expected_seq_snap_width"] == MODEL_NATIVE_SIGNAL_DIM
        assert manifest["output_data_path"] == str(output_path)
        assert manifest["splits"] == _splits()
        assert manifest["inputs"] == {
            "source_parquet": str(_source(tmp_path)),
            "tape_root": str(tmp_path / "tape"),
        }
        # The Entry action target is owned by the fitted-Q contract; the
        # manifest declares the binding and never materializes a label copy.
        require_entry_fitted_q_contract(
            manifest["extra"]["entry_fitted_q"],
            context="SPLIT_MANIFEST_WRITER_TEST",
        )
        assert manifest["extra"][
            "entry_action_target_materialized_in_feature_parquet"
        ] is False
        assert manifest["extra"]["episode_fill_binding_required"] is True
        assert manifest["extra"]["pathwise_hindsight_target"] is False
        assert manifest["extra"]["xau_tape_provenance"] == _TAPE_PROVENANCE_FIXTURE
        assert canonical_json_sha256(
            manifest["extra"]["xau_tape_provenance"]
        ) == canonical_json_sha256(_TAPE_PROVENANCE_FIXTURE)
        # The retired direction-target policy may never reappear in extra.
        assert "entry_direction_target_policy" not in manifest["extra"]
        assert not (
            RETIRED_RANK_STATE_FIELDS
            & set(manifest["extra"]["model_native_state_contract"])
        )
        assert manifest["feature_contract"]["ctx_cat_names"] == list(
            MODEL_NATIVE_CTX_CAT_FIELDS
        )
        assert manifest["feature_contract"]["ctx_cat_dim"] == len(
            MODEL_NATIVE_CTX_CAT_FIELDS
        )
        assert manifest["feature_contract"]["ctx_cont_names"] == list(
            MODEL_NATIVE_CTX_CONT_FIELDS
        )


def test_test_label_distribution_is_withheld_before_final_evaluation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = pd.DataFrame({"y_direction": [0, 0, 1, 2]})

    with caplog.at_level("INFO"):
        _log_label_distribution_proof(frame, split="test")

    assert "withheld_until_final_candidate_evaluation" in caplog.text
    assert "long=2" not in caplog.text


def test_canonical_writer_records_one_exact_source(tmp_path: Path) -> None:
    source = _source(tmp_path)
    manifest_path = write_manifest(
        output_path=tmp_path / "model_native_seq513.parquet",
        build_command=["builder"],
        source_parquet=source,
        tape_root=tmp_path / "tape",
        extra=_extra(tmp_path),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["inputs"]["source_parquet"] == str(source)


def test_canonical_writer_rejects_missing_exact_source(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="MODEL_NATIVE_SOURCE_PARQUET_MISSING"):
        write_manifest(
            output_path=tmp_path / "model_native_seq513.parquet",
            build_command=["builder"],
            source_parquet=tmp_path / "missing.parquet",
            tape_root=tmp_path / "tape",
            extra=_extra(tmp_path),
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda extra: extra.update({"model_native_signal_contract": None}),
            "MODEL_NATIVE_MANIFEST_SIGNAL_CONTRACT_MISSING",
        ),
        (
            lambda extra: extra["signal_bridge"].update({"bridge_dim": 7}),
            "MODEL_NATIVE_MANIFEST_BRIDGE_DIM_INVALID",
        ),
        (
            lambda extra: extra["signal_bridge"].update(
                {"fields": list(reversed(extra["signal_bridge"]["fields"]))}
            ),
            "MODEL_NATIVE_MANIFEST_ORDERED_FIELDS_MISMATCH",
        ),
        (
            # The retired direction-target policy is rejected outright, so a
            # stale builder cannot smuggle a second label authority back in.
            lambda extra: extra.update({"entry_direction_target_policy": {}}),
            "MODEL_NATIVE_MANIFEST_RETIRED_DIRECTION_POLICY_PRESENT",
        ),
        (
            lambda extra: extra.pop("entry_fitted_q"),
            "MODEL_NATIVE_DATASET_MANIFEST",
        ),
        (
            lambda extra: extra["ctx_contract"].update({"ctx_cat_dim": 5}),
            "MODEL_NATIVE_MANIFEST_CTX_CONTRACT_INVALID",
        ),
        (
            lambda extra: extra.pop("xau_tape_provenance"),
            "MODEL_NATIVE_MANIFEST_XAU_TAPE_PROVENANCE_MISSING",
        ),
    ],
)
def test_canonical_split_writer_rejects_soft_or_legacy_contracts(
    tmp_path: Path,
    mutation,
    error: str,
) -> None:
    extra = copy.deepcopy(_extra(tmp_path))
    mutation(extra)

    with pytest.raises(RuntimeError, match=error):
        write_manifest(
            output_path=tmp_path / "model_native_seq513_train.parquet",
            build_command=["builder"],
            source_parquet=_source(tmp_path),
            tape_root=tmp_path / "tape",
            splits=_splits(),
            extra=extra,
        )

    assert not list(tmp_path.glob("*.manifest.json"))


def test_builder_state_contract_is_the_exact_live_common_history_surface(
    tmp_path: Path,
) -> None:
    """The builder emits the v2 state contract: history only, no rank artifact."""
    contract = _model_native_state_contract(
        args=argparse.Namespace(run_id="MODEL_NATIVE_DATASET_BUILD_PYTEST"),
        feature_history_start=pd.Timestamp("2020-11-01T00:00:00Z"),
        train_start=pd.Timestamp("2020-11-09T00:00:00Z"),
        train_end=pd.Timestamp("2025-09-30T23:59:59Z"),
    )

    assert contract["schema_version"] == MODEL_NATIVE_STATE_SCHEMA_VERSION
    assert contract["feature_history_mode"] == MODEL_NATIVE_HISTORY_MODE
    assert contract["split_reset_allowed"] is False
    assert contract["runtime_rule_free"] is True
    assert contract["entry_run_id"] == "MODEL_NATIVE_DATASET_BUILD_PYTEST"
    # The retired fixed top-k rank reference must not be re-emitted here.
    assert not (RETIRED_RANK_STATE_FIELDS & set(contract))


def test_builder_state_contract_requires_run_id_and_ordered_window(
    tmp_path: Path,
) -> None:
    with pytest.raises(Exception):
        _model_native_state_contract(
            args=argparse.Namespace(run_id=""),
            feature_history_start=pd.Timestamp("2020-11-01T00:00:00Z"),
            train_start=pd.Timestamp("2020-11-09T00:00:00Z"),
            train_end=pd.Timestamp("2025-09-30T23:59:59Z"),
        )

    with pytest.raises(RuntimeError, match="STATE_CONTRACT_TIME_ORDER_INVALID"):
        _model_native_state_contract(
            args=argparse.Namespace(run_id="MODEL_NATIVE_DATASET_BUILD_PYTEST"),
            # history starts AFTER train_start: the causal prefix would be cut.
            feature_history_start=pd.Timestamp("2021-01-01T00:00:00Z"),
            train_start=pd.Timestamp("2020-11-09T00:00:00Z"),
            train_end=pd.Timestamp("2025-09-30T23:59:59Z"),
        )
