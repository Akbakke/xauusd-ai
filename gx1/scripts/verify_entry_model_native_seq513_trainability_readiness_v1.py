#!/usr/bin/env python3
"""Report-only model-native seq513 trainability readiness evidence gate.

This gate is deliberately stricter than smoke-readiness. It does not train,
replay, distill IQL, rebuild data, or touch shadow/live paths. It only proves
whether the model-native seq513 candidate has a fully wired train/proof lane before a
future trainer can be reviewed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.contracts.entry_full_input_liveness_v1 import (
    SCHEMA_VERSION as FULL_INPUT_LIVENESS_SCHEMA,
    validate_full_input_liveness_artifact,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CONTEXT_TAG,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.entry_model_native_train_launch_v1 import (
    RECIPE_AUDIT_SCHEMA,
    TRAIN_WRAPPER_RELATIVE_PATH,
)
from gx1.contracts.entry_model_native_train_recipe_v1 import (
    DIRECTION_BALANCE_ENV_TEMPLATE as DIRECTION_BALANCE_ENV_TEMPLATE,
    DIRECTION_BALANCE_ENV_KEYS,
    DIRECTION_BALANCE_RECIPE_CONTRACT,
    DIRECTION_CONTEXT_SLICE_CONTRACT,
    MODEL_NATIVE_RECIPE_ENV_KEYS,
    PATH_CALIBRATION_ENV_KEYS,
    PATH_CALIBRATION_RECIPE_CONTRACT,
    TAIL_DIRECTION_ENV_KEYS,
    TAIL_DIRECTION_RECIPE_CONTRACT,
)
from gx1.contracts.entry_model_native_post_rebuild_v1 import (
    READY_DECISION as POST_REBUILD_READY_DECISION,
    SCHEMA_VERSION as POST_REBUILD_SCHEMA_VERSION,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    REQUIRED_POSITIVE_LOSS_WEIGHTS,
    SCHEMA_VERSION as TRAINING_OBJECTIVE_SCHEMA,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.features.entry_specialist_feature_groups_v1 import (
    SPECIALIST_CONTRACT_MODES,
    required_training_specialists_for_mode,
    specialist_contract_training_allowed_for_mode,
)
from gx1.models.entry_v10.direction_decision_contract import (
    model_direction_decision_contract_metadata,
)
CONTRACT_MODE = MODEL_NATIVE_CONTRACT_MODE
EXPECTED_SIGNAL_DIM = MODEL_NATIVE_SIGNAL_DIM
EXPECTED_SPECIALIST_COUNT = 8
EXPECTED_CTX_TAG = MODEL_NATIVE_CONTEXT_TAG
# V30 (2026-08-13): 164 = 142 + H4_range_compression_ratio (package 1) + the
# 9 adopted swing V29 ctx fields + the 3 momentum-G3 raw-RSI canon scalars
# (package 2) + the 3 quote/spread-dynamics fields (package 4) + the 6
# emission-only swing additions of package 8A (two missing run counters, two
# level-intact flags, two normalized swing ages); independent
# cross-check literal against the derived contract tag/dim.
EXPECTED_CTX_CONT_DIM = 164
EXPECTED_CTX_CAT_DIM = 5
READY_DECISION = "READY_FOR_MODEL_NATIVE_SEQ513_TRAINABILITY_REVIEW"
BLOCKED_DECISION = "BLOCKED_MODEL_NATIVE_SEQ513_TRAINABILITY_READINESS"
EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SEQ513_TRAINABILITY_READINESS"
_TIMESTAMPED_JSON_RE = re.compile(
    r"^.+_\d{8}T\d{6}(?:\d{6})?Z\.json$"
)
CANONICAL_DIRECTION_DECISION_CONTRACT = model_direction_decision_contract_metadata()

SIDE_EFFECTS_CLOSED = {
    "dataset_rebuild": False,
    "training": False,
    "replay": False,
    "iql_distillation": False,
    "shadow": False,
    "live": False,
    "promotion": False,
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": bool(path.exists()),
        "size_bytes": int(path.stat().st_size) if path.exists() else None,
        "sha256": _sha256_file(path),
    }


def _require_timestamped_evidence_path(path: Path, *, label: str) -> None:
    if path.name.endswith("_latest.json") or not _TIMESTAMPED_JSON_RE.fullmatch(
        path.name
    ):
        raise RuntimeError(
            f"{label} must be an explicit timestamped JSON evidence event, got {path}"
        )


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _source_model_native_contract_binding_review(text: str) -> dict[str, Any]:
    """Prove that a downstream owner imports and uses the exact SSOT constants.

    A raw search for the resolved contract string or integer is the wrong
    boundary: correctly factored consumers import these values from the signal
    contract and therefore should not duplicate either literal in source.
    """

    required_names = {
        "MODEL_NATIVE_CONTRACT_MODE",
        "MODEL_NATIVE_SIGNAL_DIM",
    }
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {
            "ok": False,
            "parse_error": f"{exc.__class__.__name__}: {exc}",
            "required_module": "gx1.contracts.entry_model_native_signal_v1",
            "required_names": sorted(required_names),
            "imported_names": [],
            "used_names": [],
        }

    imported_names: set[str] = set()
    used_names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module == "gx1.contracts.entry_model_native_signal_v1"
        ):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used_names.add(node.id)

    imported_exact = required_names.issubset(imported_names)
    used_exact = required_names.issubset(used_names)
    return {
        "ok": bool(imported_exact and used_exact),
        "parse_error": None,
        "required_module": "gx1.contracts.entry_model_native_signal_v1",
        "required_names": sorted(required_names),
        "imported_names": sorted(imported_names),
        "used_names": sorted(required_names.intersection(used_names)),
        "imports_exact_contract_owner": imported_exact,
        "uses_imported_contract_constants": used_exact,
        "resolved_contract_mode": CONTRACT_MODE,
        "resolved_signal_dim": EXPECTED_SIGNAL_DIM,
    }


def _check(name: str, ok: bool, details: Any = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details if details is not None else {}}


def _future_train_contract(smoke_readiness: dict[str, Any]) -> dict[str, Any]:
    contracts = smoke_readiness.get("future_command_contracts")
    if not isinstance(contracts, dict):
        return {}
    contract = contracts.get("smart_smoke_train")
    return contract if isinstance(contract, dict) else {}


def _path_calibration_recipe_review(contract: dict[str, Any]) -> dict[str, Any]:
    recipe = contract.get("path_calibration_recipe_contract")
    recipe_exact = isinstance(recipe, dict) and recipe == PATH_CALIBRATION_RECIPE_CONTRACT
    recipe_keys = set(contract.get("recipe_env_keys") or ())
    recipe_keys_exact = recipe_keys == set(MODEL_NATIVE_RECIPE_ENV_KEYS)
    required_rank_keys_present = {
        "ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT",
        "ENTRY_PATH_QUALITY_RANK_WEIGHT",
    }.issubset(recipe_keys)
    argv = contract.get("wrapper_argv_template")
    argv_text = " ".join(str(part) for part in argv) if isinstance(argv, list) else ""
    argv_declares_recipe_audit = "--recipe-audit-json" in argv_text
    return {
        "ok": bool(
            contract.get("requires_path_calibration_recipe_contract") is True
            and recipe_exact
            and recipe_keys_exact
            and required_rank_keys_present
            and argv_declares_recipe_audit
        ),
        "requires_path_calibration_recipe_contract": contract.get("requires_path_calibration_recipe_contract"),
        "recipe_exact": recipe_exact,
        "recipe_keys_exact": recipe_keys_exact,
        "required_rank_keys_present": required_rank_keys_present,
        "wrapper_argv_declares_recipe_audit": argv_declares_recipe_audit,
        "expected_recipe": PATH_CALIBRATION_RECIPE_CONTRACT,
        "observed_recipe": recipe,
    }


def _direction_balance_recipe_review(contract: dict[str, Any]) -> dict[str, Any]:
    recipe = contract.get("direction_balance_recipe_contract")
    recipe_exact = isinstance(recipe, dict) and recipe == DIRECTION_BALANCE_RECIPE_CONTRACT
    recipe_keys = set(contract.get("recipe_env_keys") or ())
    recipe_keys_exact = recipe_keys == set(MODEL_NATIVE_RECIPE_ENV_KEYS)
    required_objective_keys_present = set(REQUIRED_POSITIVE_LOSS_WEIGHTS).issubset(
        recipe_keys
    )
    argv = contract.get("wrapper_argv_template")
    argv_text = " ".join(str(part) for part in argv) if isinstance(argv, list) else ""
    argv_declares_recipe_audit = "--recipe-audit-json" in argv_text
    argv_uses_exact_wrapper = (
        contract.get("control_route") == "model-native-smoke-train"
        and "gx1.models.entry_v10.entry_v10_ctx_train_v3" not in argv_text
        and "--anchor-gate-init" not in argv_text
    )
    return {
        "ok": bool(
            contract.get("requires_direction_balance_recipe_contract") is True
            and recipe_exact
            and recipe_keys_exact
            and required_objective_keys_present
            and argv_declares_recipe_audit
            and argv_uses_exact_wrapper
        ),
        "requires_direction_balance_recipe_contract": contract.get("requires_direction_balance_recipe_contract"),
        "recipe_exact": recipe_exact,
        "recipe_keys_exact": recipe_keys_exact,
        "required_objective_keys_present": required_objective_keys_present,
        "wrapper_argv_declares_recipe_audit": argv_declares_recipe_audit,
        "wrapper_argv_uses_exact_route": argv_uses_exact_wrapper,
        "expected_recipe": DIRECTION_BALANCE_RECIPE_CONTRACT,
        "observed_recipe": recipe,
    }


def _tail_direction_recipe_review(contract: dict[str, Any]) -> dict[str, Any]:
    recipe = contract.get("tail_direction_recipe_contract")
    recipe_exact = isinstance(recipe, dict) and recipe == TAIL_DIRECTION_RECIPE_CONTRACT
    recipe_keys = set(contract.get("recipe_env_keys") or ())
    recipe_keys_exact = recipe_keys == set(MODEL_NATIVE_RECIPE_ENV_KEYS)
    argv = contract.get("wrapper_argv_template")
    argv_text = " ".join(str(part) for part in argv) if isinstance(argv, list) else ""
    argv_declares_recipe_audit = "--recipe-audit-json" in argv_text
    return {
        "ok": bool(
            contract.get("requires_tail_direction_recipe_contract") is True
            and recipe_exact
            and recipe_keys_exact
            and "ENTRY_TAIL_DIRECTION_CE_WEIGHT" in recipe_keys
            and argv_declares_recipe_audit
        ),
        "requires_tail_direction_recipe_contract": contract.get("requires_tail_direction_recipe_contract"),
        "recipe_exact": recipe_exact,
        "recipe_keys_exact": recipe_keys_exact,
        "wrapper_argv_declares_recipe_audit": argv_declares_recipe_audit,
        "expected_recipe": TAIL_DIRECTION_RECIPE_CONTRACT,
        "observed_recipe": recipe,
    }


def _training_objective_future_review(contract: dict[str, Any]) -> dict[str, Any]:
    recipe_keys = set(contract.get("recipe_env_keys") or ())
    required = set(contract.get("required_positive_loss_weights") or ())
    ok = bool(
        contract.get("recipe_audit_schema") == RECIPE_AUDIT_SCHEMA
        and contract.get("training_objective_schema") == TRAINING_OBJECTIVE_SCHEMA
        and contract.get("requires_exact_model_native_training_objective") is True
        and recipe_keys == set(MODEL_NATIVE_RECIPE_ENV_KEYS)
        and required == set(REQUIRED_POSITIVE_LOSS_WEIGHTS)
        and required.issubset(recipe_keys)
    )
    return {
        "ok": ok,
        "recipe_audit_schema": contract.get("recipe_audit_schema"),
        "training_objective_schema": contract.get("training_objective_schema"),
        "recipe_env_keys_exact": recipe_keys == set(MODEL_NATIVE_RECIPE_ENV_KEYS),
        "required_positive_loss_weights_exact": required
        == set(REQUIRED_POSITIVE_LOSS_WEIGHTS),
    }


def _direction_context_slice_review(contract: dict[str, Any]) -> dict[str, Any]:
    observed = contract.get("direction_context_slice_contract")
    exact = isinstance(observed, dict) and observed == DIRECTION_CONTEXT_SLICE_CONTRACT
    return {
        "ok": bool(contract.get("requires_direction_context_slice_contract") is True and exact),
        "requires_direction_context_slice_contract": contract.get("requires_direction_context_slice_contract"),
        "contract_exact": exact,
        "expected_contract": DIRECTION_CONTEXT_SLICE_CONTRACT,
        "observed_contract": observed,
    }


def _canonical_direction_decision_review(contract: dict[str, Any]) -> dict[str, Any]:
    observed = contract.get("canonical_direction_decision_contract")
    exact = (
        isinstance(observed, dict)
        and observed == CANONICAL_DIRECTION_DECISION_CONTRACT
    )
    return {
        "ok": bool(
            contract.get("requires_canonical_direction_decision_contract")
            is True
            and exact
        ),
        "requires_canonical_direction_decision_contract": contract.get(
            "requires_canonical_direction_decision_contract"
        ),
        "contract_exact": exact,
        "expected_contract": CANONICAL_DIRECTION_DECISION_CONTRACT,
        "observed_contract": observed,
    }


def _wrapper_recipe_audit_review(text: str, required_env_keys: tuple[str, ...]) -> dict[str, Any]:
    recipe_keys = set(MODEL_NATIVE_RECIPE_ENV_KEYS)
    missing_recipe_keys = [key for key in required_env_keys if key not in recipe_keys]
    required_wiring = (
        "--recipe-audit-json",
        "--pretrain-audit-json",
        "--full-input-liveness-audit-json",
        "--post-rebuild-readiness-json",
        "--prefreeze-test-seal-json",
        "--prefreeze-test-seal-sha256",
        "--trainability-readiness-json",
        "gx1.contracts.entry_model_native_train_launch_v1",
        "--run-id",
        "--execute",
    )
    missing_wiring = [fragment for fragment in required_wiring if fragment not in text]
    forbidden_inline_contracts = [
        fragment
        for fragment in (
            "ENTRY_FOUNDATION_",
            "--anchor-gate-init",
            "GX1_ALLOW_LEGACY_ENTRY_V10_RESEARCH",
        )
        if fragment in text
    ]
    return {
        "recipe_audit_schema": RECIPE_AUDIT_SCHEMA,
        "required_recipe_env_keys": list(required_env_keys),
        "missing_recipe_env_keys": missing_recipe_keys,
        "required_wrapper_wiring": list(required_wiring),
        "missing_wrapper_wiring": missing_wiring,
        "forbidden_inline_contracts": forbidden_inline_contracts,
        "ok": not missing_recipe_keys and not missing_wiring and not forbidden_inline_contracts,
    }


def _wrapper_path_calibration_env_review(text: str) -> dict[str, Any]:
    return _wrapper_recipe_audit_review(
        text,
        (
            "ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT",
            "ENTRY_PATH_QUALITY_RANK_WEIGHT",
        ),
    )


def _wrapper_direction_balance_env_review(text: str) -> dict[str, Any]:
    return _wrapper_recipe_audit_review(
        text,
        (
            "ENTRY_DIRECTION_CE_SCALE",
            "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT",
            "ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT",
            "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT",
            "ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT",
            "ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT",
            "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT",
            "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT",
            "ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT",
            "ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT",
            "ENTRY_HIER_SIDE_VALIDITY_WEIGHT",
            "ENTRY_TRENDLINE_RAIL_AUX_WEIGHT",
            "ENTRY_OFFLINE_RL_Q_WEIGHT",
            "ENTRY_OFFLINE_RL_V_WEIGHT",
            "ENTRY_OFFLINE_RL_RANK_WEIGHT",
        ),
    )


def _wrapper_tail_direction_env_review(text: str) -> dict[str, Any]:
    return _wrapper_recipe_audit_review(
        text,
        ("ENTRY_TAIL_DIRECTION_CE_WEIGHT",),
    )


def _walk_json(value: Any, *, path: str = "$"):
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_json(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            yield from _walk_json(item, path=f"{path}[{idx}]")


def _contains_exact_string(payloads: dict[str, Any], needle: str) -> list[str]:
    matches: list[str] = []
    for root, payload in payloads.items():
        for path, value in _walk_json(payload, path=root):
            if value == needle:
                matches.append(path)
    return matches


def _path_str(raw: object) -> str | None:
    if isinstance(raw, str) and raw.strip():
        return str(Path(raw).expanduser().resolve())
    return None


def _argv_value(argv: object, flag: str) -> str | None:
    if not isinstance(argv, list):
        return None
    for idx, value in enumerate(argv[:-1]):
        if value == flag:
            return _path_str(argv[idx + 1])
    return None


def _fresh_source_identity_contract(post_rebuild: dict[str, Any], smoke_readiness: dict[str, Any], future_train: dict[str, Any]) -> dict[str, Any]:
    post_contract = (
        post_rebuild.get("post_rebuild_refresh_command_contract")
        if isinstance(post_rebuild.get("post_rebuild_refresh_command_contract"), dict)
        else {}
    )
    smoke_inputs = smoke_readiness.get("inputs") if isinstance(smoke_readiness.get("inputs"), dict) else {}
    source_dataset = _path_str(post_rebuild.get("dataset_dir"))
    post_smoke_dataset = _path_str(post_contract.get("smoke_dataset_dir"))
    readiness_source_dataset = _path_str(smoke_inputs.get("smart_dataset_dir"))
    readiness_smoke_dataset = _path_str(smoke_inputs.get("smart_smoke_dataset_dir"))
    train_argv = future_train.get("wrapper_argv_template")
    train_dataset = _argv_value(train_argv, "--dataset-dir")
    train_out_bundle = _argv_value(train_argv, "--out-bundle-dir")
    source_root = _path_str(str(Path(source_dataset).parent)) if source_dataset else None
    smoke_root = _path_str(str(Path(post_smoke_dataset).parent)) if post_smoke_dataset else None
    out_root = _path_str(str(Path(train_out_bundle).parent)) if train_out_bundle else None
    return {
        "source_dataset": source_dataset,
        "post_rebuild_smoke_dataset": post_smoke_dataset,
        "smoke_readiness_source_dataset": readiness_source_dataset,
        "smoke_readiness_smoke_dataset": readiness_smoke_dataset,
        "future_train_dataset": train_dataset,
        "future_train_out_bundle": train_out_bundle,
        "source_rebuild_root": source_root,
        "smoke_rebuild_root": smoke_root,
        "future_train_out_root": out_root,
        "source_matches_smoke_readiness": bool(source_dataset) and source_dataset == readiness_source_dataset,
        "smoke_matches_smoke_readiness": bool(post_smoke_dataset) and post_smoke_dataset == readiness_smoke_dataset,
        "future_train_dataset_matches_smoke": bool(post_smoke_dataset) and post_smoke_dataset == train_dataset,
        "future_train_out_under_source_root": bool(source_root) and source_root == out_root,
        "source_and_smoke_share_rebuild_root": bool(source_root) and source_root == smoke_root,
    }


def _ctx_contract_rows(payloads: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root, payload in payloads.items():
        for path, value in _walk_json(payload, path=root):
            if not isinstance(value, dict):
                continue
            if "ctx_contract" in value and isinstance(value.get("ctx_contract"), dict):
                ctx = value["ctx_contract"]
                rows.append(
                    {
                        "path": f"{path}.ctx_contract",
                        "tag": ctx.get("tag") or ctx.get("ctx_tag"),
                        "ctx_cont_dim": ctx.get("ctx_cont_dim"),
                        "ctx_cat_dim": ctx.get("ctx_cat_dim"),
                    }
                )
            elif ("ctx_tag" in value or "tag" in value) and (
                "ctx_cont_dim" in value or "ctx_cat_dim" in value
            ):
                rows.append(
                    {
                        "path": path,
                        "tag": value.get("ctx_tag") or value.get("tag"),
                        "ctx_cont_dim": value.get("ctx_cont_dim"),
                        "ctx_cat_dim": value.get("ctx_cat_dim"),
                    }
                )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = (row.get("path"), row.get("tag"), row.get("ctx_cont_dim"), row.get("ctx_cat_dim"))
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def _ctx_metadata_contract(payloads: dict[str, Any]) -> dict[str, Any]:
    rows = _ctx_contract_rows(payloads)
    stale_ctx6cat6_paths = _contains_exact_string(payloads, "CTX6CAT6")
    declared_rows = [row for row in rows if row.get("tag") or row.get("ctx_cont_dim") or row.get("ctx_cat_dim")]
    mismatched_rows = [
        row
        for row in declared_rows
        if not (
            row.get("tag") == EXPECTED_CTX_TAG
            and int(row.get("ctx_cont_dim") or 0) == EXPECTED_CTX_CONT_DIM
            and int(row.get("ctx_cat_dim") or 0) == EXPECTED_CTX_CAT_DIM
        )
    ]
    return {
        "expected": {
            "ctx_tag": EXPECTED_CTX_TAG,
            "ctx_cont_dim": EXPECTED_CTX_CONT_DIM,
            "ctx_cat_dim": EXPECTED_CTX_CAT_DIM,
        },
        "declared_ctx_contract_count": int(len(declared_rows)),
        "declared_ctx_contracts": declared_rows,
        "mismatched_ctx_contracts": mismatched_rows,
        "stale_ctx6cat6_paths": stale_ctx6cat6_paths,
        "no_stale_ctx6cat6": not stale_ctx6cat6_paths,
        "declared_ctx_contracts_match_expected": not mismatched_rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    post_rebuild_json = Path(args.smart_post_rebuild_readiness_json).expanduser().resolve()
    smoke_readiness_json = Path(args.smart_smoke_readiness_json).expanduser().resolve()
    control_script = Path(args.control_script).expanduser().resolve()
    trainer_source = Path(args.trainer_source).expanduser().resolve()
    train_wrapper = Path(args.train_wrapper).expanduser().resolve()
    candidate_readiness_script = Path(args.candidate_readiness_script).expanduser().resolve()
    selective_edge_script = Path(args.selective_edge_script).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    full_input_liveness_json = Path(args.full_input_liveness_json).expanduser().resolve()
    for label, path in (
        ("post-rebuild readiness", post_rebuild_json),
        ("smoke readiness", smoke_readiness_json),
        ("full-input liveness", full_input_liveness_json),
    ):
        _require_timestamped_evidence_path(path, label=label)

    post_rebuild = _read_json_or_empty(post_rebuild_json)
    smoke_readiness = _read_json_or_empty(smoke_readiness_json)
    post_liveness_meta = (
        post_rebuild.get("full_input_liveness_contract")
        if isinstance(post_rebuild.get("full_input_liveness_contract"), dict)
        else {}
    )
    smoke_inputs = (
        smoke_readiness.get("inputs")
        if isinstance(smoke_readiness.get("inputs"), dict)
        else {}
    )
    smoke_liveness_meta = (
        smoke_inputs.get("full_input_liveness_contract")
        if isinstance(smoke_inputs.get("full_input_liveness_contract"), dict)
        else {}
    )
    smoke_liveness_validation = (
        smoke_readiness.get("full_input_liveness_validation")
        if isinstance(smoke_readiness.get("full_input_liveness_validation"), dict)
        else {}
    )
    full_input_liveness_validation = validate_full_input_liveness_artifact(
        full_input_liveness_json,
        expected_sha256=str(post_liveness_meta.get("sha256") or ""),
        expected_dataset_dir=post_rebuild.get("dataset_dir") or "",
        expected_contract_mode=CONTRACT_MODE,
        expected_field_order_sha256=(
            post_liveness_meta.get("field_order_sha256")
            if isinstance(post_liveness_meta.get("field_order_sha256"), dict)
            else {}
        ),
    )
    future_train = _future_train_contract(smoke_readiness)
    fresh_source_identity_contract = _fresh_source_identity_contract(post_rebuild, smoke_readiness, future_train)
    source_metadata_contract = _ctx_metadata_contract(
        {
            "smart_post_rebuild_readiness": post_rebuild,
            "smart_smoke_readiness": smoke_readiness,
        }
    )

    control_text = _read_text(control_script)
    trainer_text = _read_text(trainer_source)
    train_wrapper_text = _read_text(train_wrapper)
    candidate_readiness_text = _read_text(candidate_readiness_script)
    selective_edge_text = _read_text(selective_edge_script)
    candidate_readiness_contract_review = (
        _source_model_native_contract_binding_review(candidate_readiness_text)
    )
    selective_edge_contract_review = _source_model_native_contract_binding_review(
        selective_edge_text
    )

    try:
        required_specialists = tuple(required_training_specialists_for_mode(CONTRACT_MODE))
    except Exception:
        required_specialists = ()
    try:
        registry_training_allowed = specialist_contract_training_allowed_for_mode(CONTRACT_MODE)
    except Exception:
        registry_training_allowed = False
    path_calibration_review = _path_calibration_recipe_review(future_train)
    direction_balance_review = _direction_balance_recipe_review(future_train)
    tail_direction_review = _tail_direction_recipe_review(future_train)
    training_objective_review = _training_objective_future_review(future_train)
    direction_context_slice_review = _direction_context_slice_review(future_train)
    canonical_direction_decision_review = _canonical_direction_decision_review(
        future_train
    )
    train_wrapper_path_calibration_review = _wrapper_path_calibration_env_review(train_wrapper_text)
    train_wrapper_direction_balance_review = _wrapper_direction_balance_env_review(train_wrapper_text)
    train_wrapper_tail_direction_review = _wrapper_tail_direction_env_review(train_wrapper_text)

    checks = [
        _check(
            "model-native post-rebuild dataset audit is ready",
            post_rebuild.get("schema_version") == POST_REBUILD_SCHEMA_VERSION
            and post_rebuild.get("decision") == POST_REBUILD_READY_DECISION,
            {
                "schema_version": post_rebuild.get("schema_version"),
                "decision": post_rebuild.get("decision"),
            },
        ),
        _check(
            "model-native seq513 smoke readiness is ready",
            smoke_readiness.get("decision")
            == "READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_READINESS_REVIEW",
            smoke_readiness.get("decision"),
        ),
        _check(
            "smart post-rebuild binds canonical full-input liveness artifact",
            post_liveness_meta.get("schema_version") == FULL_INPUT_LIVENESS_SCHEMA
            and _path_str(post_liveness_meta.get("path")) == str(full_input_liveness_json)
            and post_liveness_meta.get("decision") == "PASS"
            and post_liveness_meta.get("atr_ood_status")
            == full_input_liveness_validation.get("atr_ood_status"),
            post_liveness_meta,
        ),
        _check(
            "smart smoke readiness revalidates the same full-input liveness bytes",
            smoke_liveness_meta.get("path") == str(full_input_liveness_json)
            and smoke_liveness_meta.get("sha256") == post_liveness_meta.get("sha256")
            and smoke_liveness_validation.get("ok") is True
            and smoke_liveness_validation.get("sha256") == post_liveness_meta.get("sha256")
            and smoke_liveness_validation.get("schema_version") == FULL_INPUT_LIVENESS_SCHEMA
            and smoke_liveness_validation.get("field_order_sha256")
            == post_liveness_meta.get("field_order_sha256"),
            {
                "post_rebuild_binding": post_liveness_meta,
                "smoke_input_binding": smoke_liveness_meta,
                "smoke_validation": smoke_liveness_validation,
            },
        ),
        _check(
            "full-input liveness artifact hash schema fields and ATR shift observation validate for trainability",
            bool(full_input_liveness_validation["ok"]),
            full_input_liveness_validation,
        ),
        _check(
            "smart smoke readiness uses same source dataset as post-rebuild readiness",
            fresh_source_identity_contract["source_matches_smoke_readiness"],
            fresh_source_identity_contract,
        ),
        _check(
            "smart smoke readiness uses same smoke dataset as post-rebuild contract",
            fresh_source_identity_contract["smoke_matches_smoke_readiness"],
            fresh_source_identity_contract,
        ),
        _check(
            "smart future train dataset matches fresh smoke dataset",
            fresh_source_identity_contract["future_train_dataset_matches_smoke"],
            fresh_source_identity_contract,
        ),
        _check(
            "smart future train output stays under fresh source rebuild root",
            fresh_source_identity_contract["future_train_out_under_source_root"],
            fresh_source_identity_contract,
        ),
        _check(
            "smart source and smoke datasets share rebuild root",
            fresh_source_identity_contract["source_and_smoke_share_rebuild_root"],
            fresh_source_identity_contract,
        ),
        _check("smart specialist mode is accepted by trainer contract modes", CONTRACT_MODE in SPECIALIST_CONTRACT_MODES, list(SPECIALIST_CONTRACT_MODES)),
        _check(
            "smart specialist registry is trainable only through explicit candidate gate",
            registry_training_allowed is True,
            {
                "registry_training_allowed": registry_training_allowed,
                "candidate_training_allowed_by_this_report": False,
                "requires_candidate_readiness_and_run_lineage": True,
            },
        ),
        _check("smart required specialist count is eight", len(required_specialists) == EXPECTED_SPECIALIST_COUNT, list(required_specialists)),
        _check("trainer CLI can accept smart specialist contract mode", CONTRACT_MODE in SPECIALIST_CONTRACT_MODES and "specialist-contract-mode" in trainer_text, _artifact_meta(trainer_source)),
        _check(
            "trainer ctx contract is not hard-coded to stale CTX6CAT6 for smart",
            "CTX6CAT6" not in trainer_text,
            _artifact_meta(trainer_source),
        ),
        _check(
            "smart source metadata has no stale CTX6CAT6 ctx contract",
            source_metadata_contract["no_stale_ctx6cat6"],
            source_metadata_contract,
        ),
        _check(
            f"declared smart source ctx metadata matches {EXPECTED_CTX_TAG}",
            source_metadata_contract["declared_ctx_contracts_match_expected"],
            source_metadata_contract,
        ),
        _check(
            "canonical train wrapper exposes both explicit model-native profiles",
            CONTRACT_MODE in train_wrapper_text
            and "--profile" in train_wrapper_text
            and "smoke|candidate" in train_wrapper_text
            and "--smoke-manifest-json" in train_wrapper_text
            and "--candidate-readiness-json" in train_wrapper_text
            and "--anchor-gate-init" not in train_wrapper_text,
            _artifact_meta(train_wrapper),
        ),
        _check(
            "canonical train wrapper exposes path calibration rank env",
            bool(train_wrapper_path_calibration_review["ok"]),
            train_wrapper_path_calibration_review,
        ),
        _check(
            "canonical train wrapper exposes direction balance env",
            bool(train_wrapper_direction_balance_review["ok"]),
            train_wrapper_direction_balance_review,
        ),
        _check(
            "canonical train wrapper exposes tail direction env",
            bool(train_wrapper_tail_direction_review["ok"]),
            train_wrapper_tail_direction_review,
        ),
        _check(
            "both model-native profiles use the canonical wrapper in control surface",
            "model-native-smoke-train)" in control_text
            and "model-native-candidate-train)" in control_text
            and f'{Path(TRAIN_WRAPPER_RELATIVE_PATH).name}" --profile smoke'
            in control_text
            and f'{Path(TRAIN_WRAPPER_RELATIVE_PATH).name}" --profile candidate'
            in control_text,
            _artifact_meta(control_script),
        ),
        _check("smart smoke future contract is implemented in control surface", future_train.get("implemented_in_control_surface") is True, future_train),
        _check(
            "smart smoke future contract uses only the compact wrapper route",
            future_train.get("profile") == "smoke"
            and future_train.get("control_route") == "model-native-smoke-train"
            and future_train.get("wrapper_path")
            == TRAIN_WRAPPER_RELATIVE_PATH
            and future_train.get("wrapper_argv_template")
            == future_train.get("argv_template")
            and "gx1.models.entry_v10.entry_v10_ctx_train_v3"
            not in " ".join(future_train.get("argv_template") or ())
            and "audit-smoke-bundle"
            not in (future_train.get("argv_template") or ()),
            future_train,
        ),
        _check(
            "smart smoke future contract exposes immutable recipe prediction and smoke audit routes",
            future_train.get("requires_edge_audit") is True
            and future_train.get("recipe_audit_control_route_exposed") is True
            and future_train.get("recipe_audit_control_route")
            == "model-native-train-recipe-audit"
            and bool(future_train.get("recipe_audit_argv_template"))
            and future_train.get("post_smoke_prediction_control_route_exposed")
            is True
            and future_train.get("post_smoke_prediction_control_route")
            == "model-native-selective-edge"
            and bool(future_train.get("post_smoke_prediction_argv_template"))
            and future_train.get("post_smoke_audit_control_route_exposed") is True
            and future_train.get("post_smoke_audit_control_route")
            == "model-native-smoke-bundle-audit"
            and bool(future_train.get("post_smoke_audit_argv_template"))
            and "model-native-train-recipe-audit)" in control_text
            and "model-native-selective-edge)" in control_text
            and "model-native-smoke-bundle-audit)" in control_text,
            future_train,
        ),
        _check(
            "smart smoke future contract declares path calibration rank recipe",
            bool(path_calibration_review["ok"]),
            path_calibration_review,
        ),
        _check(
            "smart smoke future contract declares direction balance recipe",
            bool(direction_balance_review["ok"]),
            direction_balance_review,
        ),
        _check(
            "smart smoke future contract declares tail direction recipe",
            bool(tail_direction_review["ok"]),
            tail_direction_review,
        ),
        _check(
            "smart smoke future contract declares exact positive training objective",
            bool(training_objective_review["ok"]),
            training_objective_review,
        ),
        _check(
            "smart smoke future contract declares direction context slice audit",
            bool(direction_context_slice_review["ok"]),
            direction_context_slice_review,
        ),
        _check(
            "smart smoke future contract declares canonical derived direction pair",
            bool(canonical_direction_decision_review["ok"]),
            canonical_direction_decision_review,
        ),
        _check(
            "trainer supports path calibration rank env",
            all(key in trainer_text for key in PATH_CALIBRATION_ENV_KEYS),
            {"required_env_keys": list(PATH_CALIBRATION_ENV_KEYS), "trainer_source": _artifact_meta(trainer_source)},
        ),
        _check(
            "trainer supports direction balance env",
            all(key in trainer_text for key in DIRECTION_BALANCE_ENV_KEYS),
            {"required_env_keys": list(DIRECTION_BALANCE_ENV_KEYS), "trainer_source": _artifact_meta(trainer_source)},
        ),
        _check(
            "trainer supports tail direction env",
            all(key in trainer_text for key in TAIL_DIRECTION_ENV_KEYS),
            {"required_env_keys": list(TAIL_DIRECTION_ENV_KEYS), "trainer_source": _artifact_meta(trainer_source)},
        ),
        _check(
            "candidate-readiness supports model-native seq513",
            bool(candidate_readiness_contract_review["ok"]),
            {
                **_artifact_meta(candidate_readiness_script),
                "contract_binding": candidate_readiness_contract_review,
            },
        ),
        _check(
            "selective-edge supports model-native seq513",
            bool(selective_edge_contract_review["ok"]),
            {
                **_artifact_meta(selective_edge_script),
                "contract_binding": selective_edge_contract_review,
            },
        ),
        _check("side effects remain closed", all(value is False for value in SIDE_EFFECTS_CLOSED.values()), SIDE_EFFECTS_CLOSED),
    ]
    failures = [row for row in checks if not row["ok"]]
    ready = not failures
    decision = READY_DECISION if ready else BLOCKED_DECISION
    report = {
        "schema_version": "entry_model_native_seq513_trainability_readiness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "report_only": True,
        "manifest_variant": CONTRACT_MODE,
        "expected_signal_dim": EXPECTED_SIGNAL_DIM,
        "required_training_specialists": list(required_specialists),
        "inputs": {
            "smart_post_rebuild_readiness": _artifact_meta(post_rebuild_json),
            "smart_smoke_readiness": _artifact_meta(smoke_readiness_json),
            "full_input_liveness_contract": _artifact_meta(full_input_liveness_json),
            "control_script": _artifact_meta(control_script),
            "trainer_source": _artifact_meta(trainer_source),
            "train_wrapper": _artifact_meta(train_wrapper),
            "candidate_readiness_script": _artifact_meta(candidate_readiness_script),
            "selective_edge_script": _artifact_meta(selective_edge_script),
        },
        "future_train_contract": future_train,
        "fresh_source_identity_contract": fresh_source_identity_contract,
        "source_metadata_contract": source_metadata_contract,
        "full_input_liveness_validation": full_input_liveness_validation,
        "checks": checks,
        "failures": failures,
        "blockers": [row["name"] for row in failures],
        "training_allowed": False,
        "candidate_training_allowed": False,
        "replay_allowed": False,
        "iql_allowed": False,
        "shadow_live_promotion_allowed": False,
        "execution_allowed_now": False,
        "side_effects_started": dict(SIDE_EFFECTS_CLOSED),
        "next_required_gate": (
            "review explicit smart smoke-train implementation package"
            if ready
            else "wire model-native seq513 trainer/wrapper/control/candidate/replay surfaces before any training run_id"
        ),
    }
    report["evidence_binding_sha256"] = _sha256_json(report["inputs"])
    _, report = write_immutable_json_event(out_dir, EVENT_PREFIX, report)
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    if failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smart-post-rebuild-readiness-json", required=True)
    ap.add_argument("--smart-smoke-readiness-json", required=True)
    ap.add_argument("--full-input-liveness-json", required=True)
    ap.add_argument("--control-script", required=True)
    ap.add_argument("--trainer-source", required=True)
    ap.add_argument("--train-wrapper", required=True)
    ap.add_argument("--candidate-readiness-script", required=True)
    ap.add_argument("--selective-edge-script", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
