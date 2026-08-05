from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.features.htf_features import (
    HTF_V4_CACHE_BUILDER_VERSION,
    HTF_V4_MATRIX_CONTRACT,
    HTF_V4_CACHE_SCHEMA_VERSION,
    MULTI_TF_FEATURE_COUNT_V4,
    MULTI_TF_PER_BAR_FEATURES_V4,
    MULTI_TF_RESAMPLE_RULES,
    build_multi_tf_v4_liveness_contract,
)
from gx1.contracts.entry_foundation_audit_policy_v1 import (
    FOUNDATION_AUDIT_DATA_SPLITS,
    FOUNDATION_AUDIT_SMOKE_SPLITS,
    foundation_audit_policy_binding,
    foundation_audit_policy_enforcement,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_BASE_SIGNAL_DIM,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    model_native_aux_target_contract_metadata,
)
from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    CORE_ARTIFACTS as BUNDLE_COMMIT_CORE_ARTIFACTS,
    write_bundle_commit_manifest,
)
from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_ACTIVE_HEADS,
    MODEL_NATIVE_BLOCKED_HEADS,
    MODEL_NATIVE_REQUIRED_SPECIALISTS,
    model_native_readiness_contract_metadata,
)
from gx1.contracts.entry_model_native_smoke_bundle_audit_v1 import (
    SCHEMA_VERSION as SMOKE_BUNDLE_AUDIT_SCHEMA_VERSION,
)
from gx1.contracts.entry_model_native_state_v2 import (
    TRAIN_RANK_SOURCE_MARKET_IDENTITY_COLUMNS,
    TRAIN_RANK_SOURCE_MARKET_IDENTITY_CONTRACT,
)
from gx1.contracts.entry_model_native_train_launch_v1 import (
    RECIPE_AUDIT_SCHEMA,
    REQUIRED_RAIL_FEATURES,
    REQUIRED_SPECIALISTS,
    artifact_binding,
    canonical_json_sha256,
    recipe_env_sha256,
    recipe_source_bindings,
)
from gx1.contracts.entry_model_native_train_recipe_v1 import (
    MODEL_NATIVE_RECIPE_ENV,
    model_native_recipe_env_contract_metadata,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    REQUIRED_POSITIVE_LOSS_WEIGHTS,
    training_objective_contract_metadata,
)
from gx1.contracts.unified_exit_lifecycle_v1 import (
    UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION,
    UNIFIED_EXIT_M1_AUTHORITY_SCHEMA_VERSION,
)
from gx1.contracts.entry_exit_feature_base_v1 import (
    entry_exit_shared_feature_base_contract,
)
from gx1.contracts.entry_exit_feature_surface_v1 import (
    ENTRY_M5_FEATURE_SURFACE_CONSUMPTION_MODE,
)
from gx1.models.entry_v10.direction_decision_contract import (
    model_direction_decision_contract_metadata,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    PREDICTION_EVIDENCE_SCHEMA_VERSION,
)
from tests.entry_full_input_liveness_support import write_full_input_liveness_fixture
from tests.entry_model_native_smoke_audit_support import passing_smoke_audit_splits
from tests.model_native_signal_support import canonical_model_native_selected_fields
from tests.model_native_offline_rl_support import (
    model_native_target_audit_evidence,
)
from tests.model_native_test_seal_support import (
    write_prefreeze_test_seal_fixture,
)


REPO = Path(__file__).resolve().parents[1]
RUN_ID = "MODEL_NATIVE_SEQ513_PYTEST_V1"
DATASET_RUN_ID = "MODEL_NATIVE_SEQ513_DATASET_PYTEST_V1"
STAMP = "20260716T010203123456Z"


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path.resolve()


def _bindings(paths: dict[str, Path]) -> dict[str, dict]:
    return {key: artifact_binding(path) for key, path in paths.items()}


def _multi_tf_v4_liveness_fixture() -> dict[str, object]:
    frames: dict[str, pd.DataFrame] = {}
    row = np.arange(1, 5, dtype=np.float32)[:, None]
    feature_scale = np.arange(
        1,
        MULTI_TF_FEATURE_COUNT_V4 + 1,
        dtype=np.float32,
    )[None, :]
    for tf_position, tf_name in enumerate(MULTI_TF_RESAMPLE_RULES):
        values = np.ascontiguousarray(
            row * feature_scale + np.float32(tf_position / 10.0),
            dtype=np.float32,
        )
        frame = pd.DataFrame(
            values,
            columns=MULTI_TF_PER_BAR_FEATURES_V4,
        )
        frame.attrs["feats_np"] = values
        frame.attrs["causal_warmup_rows"] = 0
        frame.attrs["htf_feature_contract"] = HTF_V4_MATRIX_CONTRACT
        frames[tf_name] = frame
    return build_multi_tf_v4_liveness_contract(frames)


def _split_manifest(
    path: Path,
    parquet: Path,
    *,
    m5_prebuilt: Path,
    rank_source: Path,
    profile: str,
) -> Path:
    selected = canonical_model_native_selected_fields(
        remainder_prefix="session_regime.train_wrapper_fixture"
    )
    contract = model_native_signal_contract_metadata(selected)
    return _write_json(
        path,
        {
            "schema_version": MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
            "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
            "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
            "output_data_path": str(parquet.resolve()),
            "inputs": {"source_parquet": str(m5_prebuilt.resolve())},
            "extra": {
                "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
                "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
                "model_native_signal_contract": contract,
                "signal_bridge": {
                    "fields": contract["fields"],
                    "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                    "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                },
                "aux_head_target_contract": {
                    **model_native_aux_target_contract_metadata(),
                    "incomplete_tail_rows_total": 96,
                    "candidate_rows_before_completeness": 100,
                    "incomplete_candidate_rows_excluded": 96,
                    "complete_rows_emitted": 4,
                },
                "model_native_state_contract": {
                    "entry_run_id": DATASET_RUN_ID,
                    "feature_history_start_utc": "2021-01-05T00:00:00Z",
                    "rank_fit_end_utc": "2026-05-31T23:59:59Z",
                    "rank_reference_source_parquet": str(rank_source.resolve()),
                    "rank_reference_source_parquet_sha256": artifact_binding(
                        rank_source
                    )["sha256"],
                    "rank_reference_model_source_market_identity": {
                        "contract": TRAIN_RANK_SOURCE_MARKET_IDENTITY_CONTRACT,
                        "rank_source_parquet": str(rank_source.resolve()),
                        "rank_source_sha256": artifact_binding(rank_source)[
                            "sha256"
                        ],
                        "model_source_parquet": str(m5_prebuilt.resolve()),
                        "model_source_sha256": artifact_binding(m5_prebuilt)[
                            "sha256"
                        ],
                        "history_start_utc": "2021-01-05T00:00:00+00:00",
                        "fit_end_utc": "2026-05-31T23:59:59+00:00",
                        "compared_rows": 4,
                        "columns": list(
                            TRAIN_RANK_SOURCE_MARKET_IDENTITY_COLUMNS
                        ),
                    },
                },
                "entry_run_id": DATASET_RUN_ID,
            },
        },
    )


def build_wrapper_contract(tmp_path: Path, *, profile: str, wrapper: Path) -> tuple[list[str], dict[str, Path]]:
    assert profile in {"smoke", "candidate"}
    dataset_dir = (tmp_path / f"dataset_{STAMP}").resolve()
    dataset_dir.mkdir(parents=True)
    gx1_data_root = (tmp_path / "GX1_DATA").resolve()
    gx1_data_root.mkdir()
    output_parent = (tmp_path / "bundles").resolve()
    output_parent.mkdir()
    out_bundle = output_parent / f"entry_{profile}_{STAMP}"

    artifacts: dict[str, Path] = {}
    m5_dir = (tmp_path / f"m5_{STAMP}").resolve()
    m5_dir.mkdir()
    m5_path = m5_dir / "xau_m5.parquet"
    m5_path.write_bytes(b"xau-m5-fixture")
    artifacts["m5_prebuilt_path"] = m5_path.resolve()
    rank_source = m5_dir / "canonical_rank_source.parquet"
    rank_source.write_bytes(b"canonical-rank-source-fixture")

    mtf_cache_dir = (tmp_path / f"MULTI_TF_V4_CACHE_{STAMP}").resolve()
    mtf_cache_dir.mkdir()
    artifacts["multi_tf_cache_manifest_json"] = _write_json(
        mtf_cache_dir / "manifest.json",
        {
            "schema_version": HTF_V4_CACHE_SCHEMA_VERSION,
            "builder_version": HTF_V4_CACHE_BUILDER_VERSION,
            "feature_count": MULTI_TF_FEATURE_COUNT_V4,
            "feature_names": list(MULTI_TF_PER_BAR_FEATURES_V4),
            "m5_prebuilt_source": str(m5_path.resolve()),
            "full_input_liveness": _multi_tf_v4_liveness_fixture(),
        },
    )

    for split in ("train", "val", "test"):
        parquet = dataset_dir / f"xau_seq513_{split}.parquet"
        parquet.write_bytes(f"{profile}-{split}-parquet".encode())
        manifest = dataset_dir / f"xau_seq513_{split}.manifest.json"
        artifacts[f"{split}_parquet"] = parquet.resolve()
        artifacts[f"{split}_manifest_json"] = _split_manifest(
            manifest,
            parquet,
            m5_prebuilt=m5_path,
            rank_source=rank_source,
            profile=profile,
        )

    lifecycle_dir = (tmp_path / f"exit_lifecycle_{STAMP}").resolve()
    lifecycle_dir.mkdir()
    lifecycle_m1 = lifecycle_dir / f"xau_m1_{STAMP}.parquet"
    lifecycle_m1.write_bytes(b"xau-m1-lifecycle-fixture")
    lifecycle_feature_base = lifecycle_dir / f"xau_m1_feature_base_{STAMP}.parquet"
    lifecycle_feature_base.write_bytes(b"xau-m1-feature-base-fixture")
    lifecycle_feature_manifest = Path(
        str(lifecycle_feature_base) + ".manifest.json"
    )
    lifecycle_feature_manifest.write_text(
        json.dumps(
            {
                "schema_version": "gx1_entry_exit_m1_feature_surface_v1",
                "decision": "PASS",
                "dataset_run_id": DATASET_RUN_ID,
                "output_parquet": str(lifecycle_feature_base),
                "output_parquet_sha256": artifact_binding(
                    lifecycle_feature_base
                )["sha256"],
                "shared_feature_base_contract": (
                    entry_exit_shared_feature_base_contract()
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    lifecycle_authority = {
        "schema_version": UNIFIED_EXIT_M1_AUTHORITY_SCHEMA_VERSION,
        "pair_manifest_path": str(
            tmp_path / "pair_generation" / "PAIR_MANIFEST.json"
        ),
        "pair_generation_root": str(tmp_path / "pair_generations"),
        "m1_source_path": str(lifecycle_m1),
        "m1_source_sha256": artifact_binding(lifecycle_m1)["sha256"],
    }
    lifecycle_splits = {
        split: {
            "entry_dataset_path": str(artifacts[f"{split}_parquet"]),
            "entry_dataset_sha256": artifact_binding(
                artifacts[f"{split}_parquet"]
            )["sha256"],
            "lifecycle_parquet": (
                f"{split}_unified_exit_lifecycle.parquet"
            ),
            "lifecycle_parquet_sha256": "1" * 64,
            "lifecycle_manifest": (
                f"{split}_unified_exit_lifecycle.manifest.json"
            ),
            "lifecycle_manifest_sha256": "2" * 64,
            "episode_rows": 2,
            "target_counts": {
                "HOLD": 1,
                "EXIT_NOW": 1,
                "TIED_OMITTED": 0,
            },
            "target_stream_sha256": "3" * 64,
        }
        for split in ("train", "val", "test")
    }
    artifacts["unified_exit_lifecycle_manifest_json"] = _write_json(
        lifecycle_dir / "UNIFIED_EXIT_LIFECYCLE_MANIFEST.json",
        {
            "schema_version": (
                UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION
            ),
            "decision": "PASS",
            "entry_run_id": DATASET_RUN_ID,
            "m1_source_path": str(lifecycle_m1),
            "m1_source_sha256": artifact_binding(lifecycle_m1)["sha256"],
            "m1_feature_base_path": str(lifecycle_feature_base),
            "m1_feature_base_sha256": artifact_binding(
                lifecycle_feature_base
            )["sha256"],
            "m1_feature_base_manifest_path": str(lifecycle_feature_manifest),
            "m1_feature_base_manifest_sha256": artifact_binding(
                lifecycle_feature_manifest
            )["sha256"],
            "m1_authority": lifecycle_authority,
            "m1_authority_sha256": canonical_json_sha256(
                lifecycle_authority
            ),
            "path_state_count": 512,
            "target_lookahead_m1_steps": 3,
            "m1_row_clock": (
                "consecutive_authoritative_closed_m1_source_rows"
            ),
            "shared_feature_base_contract": (
                entry_exit_shared_feature_base_contract()
            ),
            "side_order": ["long", "short"],
            "action_order": ["HOLD", "EXIT_NOW"],
            "splits": lifecycle_splits,
        },
    )

    evidence_dir = (tmp_path / f"evidence_{STAMP}").resolve()
    liveness_dir = evidence_dir / "liveness"
    liveness_path, _, _ = write_full_input_liveness_fixture(
        liveness_dir,
        dataset_dir=dataset_dir,
    )
    artifacts["full_input_liveness_audit_json"] = liveness_path.resolve()

    feature_selected = canonical_model_native_selected_fields(
        remainder_prefix="session_regime.train_wrapper_fixture"
    )
    feature_signal_contract = model_native_signal_contract_metadata(feature_selected)
    ranked_remainder = feature_selected[
        MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT:
    ]
    artifacts["feature_audit_json"] = _write_json(
        evidence_dir / f"ENTRY_FEATURE_AUDIT_{STAMP}.json",
        {
            "schema_version": "entry_feature_foundation_audit_v1",
            **foundation_audit_policy_binding(),
            "foundation_audit_policy_enforcement": (
                foundation_audit_policy_enforcement("feature")
            ),
            "decision": "PASS",
            "failures": [],
            "dataset_dir": str(dataset_dir),
            "data_splits": list(FOUNDATION_AUDIT_DATA_SPLITS),
            "model_native_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
            "base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
            "base_signal_fields": list(MODEL_NATIVE_BASE_FIELDS),
            "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
            "manifest_selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
            "mandatory_selected_feature_count": (
                MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
            ),
            "manifest_mandatory_selected_feature_count": (
                MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
            ),
            "ranked_remainder_feature_count": (
                MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT
            ),
            "manifest_ranked_remainder_feature_count": (
                MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT
            ),
            "ranked_remainder_fields_sha256": canonical_json_sha256(
                ranked_remainder
            ),
            "feature_ranking_fit_scope": "train_only",
            "feature_ranking_sha256": "a" * 64,
            "model_native_signal_contract": feature_signal_contract,
            "ctx_cont_dim_v3": 142,
            "ctx_cat_dim_v3": 5,
        },
    )
    artifacts["target_audit_json"] = _write_json(
        evidence_dir / f"ENTRY_TARGET_AUDIT_{STAMP}.json",
        {
            "schema_version": "entry_target_foundation_audit_v2",
            **foundation_audit_policy_binding(),
            "foundation_audit_policy_enforcement": (
                foundation_audit_policy_enforcement("target")
            ),
            "decision": "PASS",
            "failures": [],
            "dataset_dir": str(dataset_dir),
            "data_splits": list(FOUNDATION_AUDIT_DATA_SPLITS),
            **model_native_target_audit_evidence(),
        },
    )
    artifacts["specialist_audit_json"] = _write_json(
        evidence_dir / f"ENTRY_SPECIALIST_AUDIT_{STAMP}.json",
        {
            "schema_version": "entry_specialist_feature_group_audit_v1",
            **foundation_audit_policy_binding(),
            "foundation_audit_policy_enforcement": (
                foundation_audit_policy_enforcement("specialist")
            ),
            "decision": "PASS",
            "failures": [],
            "dataset_dir": str(dataset_dir),
            "data_splits": list(FOUNDATION_AUDIT_DATA_SPLITS),
            "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "signal_field_count": 513,
            "selected_feature_count": 479,
            "required_training_specialists": list(REQUIRED_SPECIALISTS),
            "specialist_model_contract_valid": True,
            "signal_routing_all_mapped": True,
        },
    )
    artifacts["trainability_readiness_json"] = _write_json(
        evidence_dir / f"ENTRY_TRAINABILITY_READINESS_{STAMP}.json",
        {
            "schema_version": "entry_model_native_seq513_trainability_readiness_v1",
            "decision": "READY_FOR_MODEL_NATIVE_SEQ513_TRAINABILITY_REVIEW",
            "failures": [],
            "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
            "expected_signal_dim": 513,
            "required_training_specialists": list(REQUIRED_SPECIALISTS),
            "future_train_contract": {
                "profile": "smoke",
                "control_route": "model-native-smoke-train",
                "wrapper_path": "scripts/run_entry_model_native_seq513_train.sh",
            },
        },
    )

    if profile == "smoke":
        embedded = {
            "schema_version": "entry_model_native_seq513_smoke_dataset_v3",
            "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
            "expected_seq_snap_width": 513,
            "out_dir": str(dataset_dir),
            "splits": {
                split: {
                    "out_parquet": str(artifacts[f"{split}_parquet"]),
                    "out_manifest": str(artifacts[f"{split}_manifest_json"]),
                    "out_parquet_sha256": artifact_binding(artifacts[f"{split}_parquet"])["sha256"],
                    "out_manifest_sha256": artifact_binding(artifacts[f"{split}_manifest_json"])["sha256"],
                }
                for split in ("train", "val")
            },
        }
        artifacts["smoke_manifest_json"] = _write_json(
            evidence_dir / f"ENTRY_SMOKE_MANIFEST_{STAMP}.json",
            {
                "schema_version": "entry_model_native_seq513_smoke_manifest_v3",
                "decision": "READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST_REVIEW",
                "failures": [],
                "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
                "expected_seq_snap_width": 513,
                "smoke_manifest": embedded,
                "manifest_sha256": canonical_json_sha256(embedded),
            },
        )
        artifacts["smoke_readiness_json"] = _write_json(
            evidence_dir / f"ENTRY_SMOKE_READINESS_{STAMP}.json",
            {
                "schema_version": "entry_model_native_seq513_smoke_readiness_v3",
                "decision": "READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_READINESS_REVIEW",
                "failures": [],
                "smart_candidate": {
                    "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
                    "expected_signal_dim": 513,
                    "expected_selected_feature_count": 479,
                },
            },
        )
    else:
        pass

    sealed_parquet_sha256 = {
        split: artifact_binding(artifacts[f"{split}_parquet"])["sha256"]
        for split in ("train", "val", "test")
    }
    test_seal_path, test_seal_sha256, test_seal_lineage = (
        write_prefreeze_test_seal_fixture(
            authority_dir=evidence_dir / "rebuild_authority",
            dataset_dir=dataset_dir,
            dataset_run_id=DATASET_RUN_ID,
            manifest_path=artifacts["test_manifest_json"],
            manifest_sha256=artifact_binding(artifacts["test_manifest_json"])[
                "sha256"
            ],
            parquet_path=artifacts["test_parquet"],
            parquet_sha256=sealed_parquet_sha256["test"],
        )
    )
    artifacts["prefreeze_test_seal_json"] = test_seal_path
    large_artifact_sha256 = {
        key: artifact_binding(artifacts[key])["sha256"]
        for key in sorted(("train_parquet", "val_parquet", "m5_prebuilt_path"))
    }
    tape_provenance = {
        "schema_version": "xau_tape_current_snapshot_v1",
        "instrument": "XAU_USD",
        "entry_run_id": DATASET_RUN_ID,
    }
    pretrain_splits = []
    for split in ("train", "val"):
        manifest = json.loads(
            artifacts[f"{split}_manifest_json"].read_text(encoding="utf-8")
        )
        signal_contract = manifest["extra"]["model_native_signal_contract"]
        pretrain_splits.append(
            {
                "split": split,
                "manifest_path": str(artifacts[f"{split}_manifest_json"]),
                "parquet_path": str(artifacts[f"{split}_parquet"]),
                "feature_count": MODEL_NATIVE_SIGNAL_DIM,
                "core_target_policy": "future_path_and_utility_outcomes_only",
                "seq_structure_extension_mode": (
                    ENTRY_M5_FEATURE_SURFACE_CONSUMPTION_MODE
                ),
                "missing_polarity_features": [],
                "missing_target_columns": [],
                "provenance": {
                    "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
                    "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
                    "xau_tape_provenance": tape_provenance,
                    "fields": signal_contract["fields"],
                    "model_native_signal_contract": signal_contract,
                },
            }
        )
    artifacts["pretrain_audit_json"] = _write_json(
        evidence_dir / f"XAU_PRETRAIN_AUDIT_{STAMP}.json",
        {
            "schema_version": "xau_direction_repair_pretrain_audit_v2",
            "decision": "PASS",
            "failures": [],
            "dataset_dir": str(dataset_dir),
            "data_splits": ["train", "val"],
            "require_rail_features": True,
            "require_inline_seq_structure": True,
            "require_xau_provenance": True,
            "required_rail_features": list(REQUIRED_RAIL_FEATURES),
            "missing_rail_features": [],
            "tape_provenance": {
                split: tape_provenance for split in ("train", "val")
            },
            "splits": pretrain_splits,
        },
    )
    artifacts["post_rebuild_readiness_json"] = _write_json(
        evidence_dir / f"ENTRY_POST_REBUILD_READINESS_{STAMP}.json",
        {
            "schema_version": "entry_model_native_seq513_post_rebuild_readiness_v2",
            "decision": "READY_FOR_MODEL_NATIVE_SEQ513_POST_REBUILD_REVIEW",
            "failures": [],
            "entry_run_id": DATASET_RUN_ID,
            "dataset_dir": str(dataset_dir),
            "smoke_dataset_dir": str(dataset_dir),
            "report_only": True,
            "training_allowed": False,
            "side_effects_started": {
                "dataset_rebuild": False,
                "training": False,
                "replay": False,
                "iql_distillation": False,
                "shadow": False,
                "live": False,
            },
            "full_input_liveness_contract": {
                "path": str(artifacts["full_input_liveness_audit_json"]),
                "sha256": artifact_binding(
                    artifacts["full_input_liveness_audit_json"]
                )["sha256"],
            },
            "pretrain_audit": {
                "path": str(artifacts["pretrain_audit_json"]),
                "sha256": artifact_binding(artifacts["pretrain_audit_json"])[
                    "sha256"
                ],
            },
            "split_artifacts": {
                **{
                    split: {
                        "manifest_path": str(artifacts[f"{split}_manifest_json"]),
                        "manifest_sha256": artifact_binding(
                            artifacts[f"{split}_manifest_json"]
                        )["sha256"],
                        "parquet_path": str(artifacts[f"{split}_parquet"]),
                        "parquet_sha256": sealed_parquet_sha256[split],
                        "rows": 1,
                    }
                    for split in ("train", "val")
                },
                "test": {
                    "manifest_path": test_seal_lineage["test_manifest"]["path"],
                    "manifest_sha256": test_seal_lineage["test_manifest"]["sha256"],
                    "parquet_path": test_seal_lineage["test_parquet"]["path"],
                    "parquet_sha256": test_seal_lineage["test_parquet"]["sha256"],
                    "schema_version": test_seal_lineage["test_manifest"]["schema_version"],
                    "manifest_variant": test_seal_lineage["test_manifest"]["manifest_variant"],
                    "rows": test_seal_lineage["rows"],
                    "entry_run_id": test_seal_lineage["dataset_run_id"],
                    "instrument": test_seal_lineage["test_manifest"]["instrument"],
                    "access_mode": test_seal_lineage["access_policy"],
                    "seal_path": test_seal_lineage["seal_event"]["path"],
                    "seal_sha256": test_seal_lineage["seal_event"]["sha256"],
                },
            },
            "test_isolation": {
                "schema_version": test_seal_lineage["seal_event"]["schema_version"],
                "decision": test_seal_lineage["seal_event"]["decision"],
                "access_policy": test_seal_lineage["access_policy"],
                "authority": {
                    "path": test_seal_lineage["seal_event"]["path"],
                    "sha256": test_seal_lineage["seal_event"]["sha256"],
                },
                "content_binding_sha256": test_seal_lineage["seal_event"][
                    "content_binding_sha256"
                ],
                "rebuild_terminal": {
                    key: test_seal_lineage["rebuild_terminal"][key]
                    for key in (
                        "path",
                        "sha256",
                        "schema_version",
                        "content_binding_sha256",
                    )
                },
                "pair_lineage_sha256": test_seal_lineage["pair_lineage_sha256"],
                "source_lineage_sha256": test_seal_lineage["source_lineage_sha256"],
                "disclosure_count": 0,
                "test_dataset_bytes_read": False,
                "test_manifest_bytes_read": False,
                "test_paths_resolved_or_statted": False,
            },
        },
    )

    if profile == "candidate":
        training_objective = training_objective_contract_metadata(
            {key: 1.0 for key in REQUIRED_POSITIVE_LOSS_WEIGHTS}
        )
        smoke_bundle_dir = (tmp_path / f"smoke_bundle_{STAMP}").resolve()
        smoke_bundle_dir.mkdir()
        smoke_metadata = _write_json(
            smoke_bundle_dir / "bundle_metadata.json",
            {"fixture": "metadata"},
        )
        smoke_lock = _write_json(
            smoke_bundle_dir / "MASTER_TRANSFORMER_LOCK.json",
            {"fixture": "lock"},
        )
        smoke_state = smoke_bundle_dir / "model_state_dict.pt"
        smoke_state.write_bytes(b"smoke model state")
        write_bundle_commit_manifest(
            bundle_dir=smoke_bundle_dir,
            artifact_names=BUNDLE_COMMIT_CORE_ARTIFACTS,
            bundle_kind="trained",
            created_at_utc="2026-07-23T12:00:00+00:00",
        )
        prediction_report = _write_json(
            evidence_dir / f"ENTRY_CANDIDATE_SELECTIVE_EDGE_{STAMP}.json",
            {"fixture": "prediction report"},
        )
        prediction_path = (
            evidence_dir / f"selective_edge_predictions_{STAMP}.parquet"
        )
        prediction_path.write_bytes(b"wrapper-prediction-evidence")

        def simple_binding(path: Path) -> dict[str, str]:
            row = artifact_binding(path)
            return {"path": row["path"], "sha256": row["sha256"]}

        artifacts["smoke_bundle_audit_json"] = _write_json(
            evidence_dir / f"ENTRY_SMOKE_BUNDLE_AUDIT_{STAMP}.json",
            {
                "schema_version": SMOKE_BUNDLE_AUDIT_SCHEMA_VERSION,
                **foundation_audit_policy_binding(),
                "decision": "PASS",
                "failures": [],
                "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
                "sequence_length": 96,
                "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
                "bundle_dir": str(smoke_bundle_dir),
                "dataset_dir": str(dataset_dir),
                "data_splits": list(FOUNDATION_AUDIT_SMOKE_SPLITS),
                "model_native_readiness_contract": (
                    model_native_readiness_contract_metadata()
                ),
                "direction_decision_contract": (
                    model_direction_decision_contract_metadata()
                ),
                "bundle_artifacts": {
                    "bundle_commit": simple_binding(
                        smoke_bundle_dir
                        / "ENTRY_MODEL_NATIVE_BUNDLE_COMMIT.json"
                    ),
                    "bundle_metadata": simple_binding(smoke_metadata),
                    "master_transformer_lock": simple_binding(smoke_lock),
                    "model_state_dict": simple_binding(smoke_state),
                },
                "input_audits": {
                    "target": {
                        **simple_binding(artifacts["target_audit_json"]),
                        **foundation_audit_policy_binding(),
                        "foundation_audit_policy_enforcement": (
                            foundation_audit_policy_enforcement("target")
                        ),
                        "schema_version": "entry_target_foundation_audit_v2",
                        "decision": "PASS",
                        "failures": [],
                        "data_splits": list(FOUNDATION_AUDIT_DATA_SPLITS),
                    },
                    "specialist": {
                        **simple_binding(artifacts["specialist_audit_json"]),
                        **foundation_audit_policy_binding(),
                        "foundation_audit_policy_enforcement": (
                            foundation_audit_policy_enforcement("specialist")
                        ),
                        "schema_version": "entry_specialist_feature_group_audit_v1",
                        "decision": "PASS",
                        "failures": [],
                        "data_splits": list(FOUNDATION_AUDIT_DATA_SPLITS),
                    },
                    "pretrain": {
                        **simple_binding(artifacts["pretrain_audit_json"]),
                        "schema_version": "xau_direction_repair_pretrain_audit_v2",
                        "decision": "PASS",
                        "failures": [],
                    },
                },
                "model_native_training_objective_contract": {
                    "decision": "PASS",
                    "failures": [],
                    "meta_lock_exact": True,
                    "objective": training_objective,
                    "metadata_path": str(smoke_metadata),
                    "metadata_sha256": simple_binding(smoke_metadata)["sha256"],
                    "lock_path": str(smoke_lock),
                    "lock_sha256": simple_binding(smoke_lock)["sha256"],
                },
                "head_contract": {
                    "decision": "PASS",
                    "failures": [],
                    "active_heads": list(MODEL_NATIVE_ACTIVE_HEADS),
                    "blocked_heads": list(MODEL_NATIVE_BLOCKED_HEADS),
                },
                "specialist_contract": {
                    "decision": "PASS",
                    "failures": [],
                    "specialists": list(MODEL_NATIVE_REQUIRED_SPECIALISTS),
                    "gate_liveness_proven": True,
                },
                "liveness_contract": {
                    "decision": "PASS",
                    "failures": [],
                    "all_active_head_predictions_live": True,
                    "all_specialist_gates_live": True,
                    "strict_bundle_components_live": True,
                },
                "edge_contract": {
                    "decision": "PASS",
                    "failures": [],
                    "direction_edge_proven": True,
                    "context_slice_edge_proven": True,
                    "path_quality_edge_proven": True,
                    "bad_path_edge_proven": True,
                    "turning_point_edge_proven": True,
                    "offline_rl_edge_proven": True,
                },
                "splits": passing_smoke_audit_splits(),
                "prediction_evidence": {
                    "schema_version": (
                        PREDICTION_EVIDENCE_SCHEMA_VERSION
                    ),
                    "evidence_stage": "pre_calibration",
                    "authoritative": False,
                    "runtime_head_evidence_authoritative": False,
                    "path": str(prediction_path),
                    "sha256": simple_binding(prediction_path)["sha256"],
                    "splits": ["val"],
                    "models": ["entry_model_native_smoke"],
                },
                "prediction_evidence_stage": "pre_calibration",
                "prediction_report_json": str(prediction_report),
                "prediction_report_sha256": simple_binding(prediction_report)["sha256"],
                "promotion_shadow_live_allowed": False,
                "activation_authority": False,
            },
        )
        readiness_bindings = {
            "smoke_bundle_audit": simple_binding(
                artifacts["smoke_bundle_audit_json"]
            ),
            "specialist_audit": simple_binding(artifacts["specialist_audit_json"]),
            "trainability_readiness": simple_binding(
                artifacts["trainability_readiness_json"]
            ),
        }
        artifacts["candidate_readiness_json"] = _write_json(
            evidence_dir / f"ENTRY_CANDIDATE_READINESS_{STAMP}.json",
            {
                "schema_version": "entry_candidate_readiness_model_native_v1",
                "decision": "READY_FOR_CANDIDATE_TRAINING",
                "failures": [],
                "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
                "sequence_length": 96,
                "expected_signal_dim": 513,
                "required_specialist_groups": list(REQUIRED_SPECIALISTS),
                "candidate_training_allowed": True,
                "promotion_shadow_live_allowed": False,
                "activation_authority": False,
                "input_bindings": readiness_bindings,
                "input_bindings_sha256": canonical_json_sha256(readiness_bindings),
            },
        )

    recipe_binding_keys = [
        "train_manifest_json",
        "val_manifest_json",
        "train_parquet",
        "val_parquet",
        "unified_exit_lifecycle_manifest_json",
        "m5_prebuilt_path",
        "multi_tf_cache_manifest_json",
        "post_rebuild_readiness_json",
        "prefreeze_test_seal_json",
        "full_input_liveness_audit_json",
        "feature_audit_json",
        "target_audit_json",
        "specialist_audit_json",
        "pretrain_audit_json",
        "trainability_readiness_json",
        *(
            ["smoke_manifest_json", "smoke_readiness_json"]
            if profile == "smoke"
            else ["candidate_readiness_json", "smoke_bundle_audit_json"]
        ),
    ]
    recipe_bindings = _bindings({key: artifacts[key] for key in recipe_binding_keys})
    source_bindings = recipe_source_bindings(
        repo=REPO,
        wrapper_path=wrapper.resolve(),
    )
    env_map = dict(MODEL_NATIVE_RECIPE_ENV)
    trainer_cli = {
        "device": "cpu",
        "seed": 1337,
        "epochs": 2,
        "batch_size": 8,
        "early_stop_patience": 1,
        "subsample_rows": 0,
        "num_workers": 0,
        "learning_rate": 0.0003,
        "early_stop_min_delta": 0.0,
        "grad_clip_norm": 1.0,
        "weight_decay": 0.00001,
        "dropout": 0.05,
        "multi_tf_scale": 0.5,
        "specialist_fusion_scale": 0.25,
        "cross_family_fusion_scale": 0.25,
        "multi_tf_num_layers": 2,
        "specialist_num_layers": 1,
        "grad_accum_steps": 1,
        "per_tf_seq_len_m5": 16,
        "per_tf_seq_len_m15": 64,
        "per_tf_seq_len_h1": 96,
        "per_tf_seq_len_h4": 96,
        "per_tf_seq_len_d1": 252,
        "memory_cap": "2G",
        "swap_cap": "512M",
        "gx1_data_root": str(gx1_data_root),
    }
    recipe_path = evidence_dir / f"ENTRY_TRAIN_RECIPE_AUDIT_{STAMP}.json"
    artifacts["recipe_audit_json"] = _write_json(
        recipe_path,
        {
            "schema_version": RECIPE_AUDIT_SCHEMA,
            "created_utc": "2026-07-16T01:02:03.123456+00:00",
            "decision": "PASS",
            "failures": [],
            "profile": profile,
            "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
            "expected_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
            "expected_selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
            "execution_allowed": True,
            "activation_authority": False,
            "report_only": True,
            "side_effects_started": {
                "training": False,
                "replay": False,
                "iql_distillation": False,
                "shadow": False,
                "live": False,
            },
            "run_id": RUN_ID,
            "dataset_run_id": DATASET_RUN_ID,
            "dataset_dir": str(dataset_dir),
            "out_bundle_dir": str(out_bundle),
            "prefreeze_test_seal_lineage": test_seal_lineage,
            "prefreeze_test_seal_lineage_sha256": canonical_json_sha256(
                test_seal_lineage
            ),
            "source_commit": subprocess.check_output(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
            ).strip(),
            "trainer_cli": trainer_cli,
            "trainer_cli_sha256": canonical_json_sha256(trainer_cli),
            "trainer_env": env_map,
            "trainer_env_sha256": recipe_env_sha256(env_map),
            "trainer_env_contract": model_native_recipe_env_contract_metadata(),
            "artifact_bindings": recipe_bindings,
            "artifact_bindings_sha256": canonical_json_sha256(recipe_bindings),
            "large_artifact_sha256": large_artifact_sha256,
            "source_bindings": source_bindings,
            "source_bindings_sha256": canonical_json_sha256(source_bindings),
        },
    )

    args = [
        "--run-id", RUN_ID,
        "--dataset-dir", str(dataset_dir),
        "--train-manifest-json", str(artifacts["train_manifest_json"]),
        "--val-manifest-json", str(artifacts["val_manifest_json"]),
        "--train-parquet", str(artifacts["train_parquet"]),
        "--val-parquet", str(artifacts["val_parquet"]),
        "--unified-exit-lifecycle-manifest-json",
        str(artifacts["unified_exit_lifecycle_manifest_json"]),
        "--m5-prebuilt-path", str(artifacts["m5_prebuilt_path"]),
        "--multi-tf-cache-manifest-json", str(artifacts["multi_tf_cache_manifest_json"]),
        "--post-rebuild-readiness-json", str(artifacts["post_rebuild_readiness_json"]),
        "--prefreeze-test-seal-json", str(artifacts["prefreeze_test_seal_json"]),
        "--prefreeze-test-seal-sha256", test_seal_sha256,
        "--full-input-liveness-audit-json", str(artifacts["full_input_liveness_audit_json"]),
        "--feature-audit-json", str(artifacts["feature_audit_json"]),
        "--target-audit-json", str(artifacts["target_audit_json"]),
        "--specialist-audit-json", str(artifacts["specialist_audit_json"]),
        "--pretrain-audit-json", str(artifacts["pretrain_audit_json"]),
        "--recipe-audit-json", str(artifacts["recipe_audit_json"]),
        "--trainability-readiness-json", str(artifacts["trainability_readiness_json"]),
        "--out-bundle-dir", str(out_bundle),
        "--gx1-data-root", str(gx1_data_root),
        "--device", "cpu",
        "--seed", "1337",
        "--epochs", "2",
        "--batch-size", "8",
        "--learning-rate", "0.0003",
        "--early-stop-patience", "1",
        "--early-stop-min-delta", "0.0",
        "--grad-clip-norm", "1.0",
        "--weight-decay", "0.00001",
        "--dropout", "0.05",
        "--multi-tf-scale", "0.5",
        "--specialist-fusion-scale", "0.25",
        "--cross-family-fusion-scale", "0.25",
        "--subsample-rows", "0",
        "--num-workers", "0",
        "--multi-tf-num-layers", "2",
        "--specialist-num-layers", "1",
        "--grad-accum-steps", "1",
        "--per-tf-seq-len-m5", "16",
        "--per-tf-seq-len-m15", "64",
        "--per-tf-seq-len-h1", "96",
        "--per-tf-seq-len-h4", "96",
        "--per-tf-seq-len-d1", "252",
        "--memory-cap", "2G",
        "--swap-cap", "512M",
    ]
    if profile == "smoke":
        args.extend(
            [
                "--smoke-manifest-json", str(artifacts["smoke_manifest_json"]),
                "--smoke-readiness-json", str(artifacts["smoke_readiness_json"]),
            ]
        )
    else:
        args.extend(
            [
                "--candidate-readiness-json", str(artifacts["candidate_readiness_json"]),
                "--smoke-bundle-audit-json", str(artifacts["smoke_bundle_audit_json"]),
            ]
        )
    return args, {**artifacts, "dataset_dir": dataset_dir, "out_bundle_dir": out_bundle}
