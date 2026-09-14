"""Materialize TRAIN-owned Exit economics readiness from an immutable rate receipt."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gx1.contracts.unified_exit_economics_objective_v2 import (
    FROZEN_CAPITAL_HURDLE_SCHEMA_VERSION,
    MARK_TO_MARKET_REWARD_ACCOUNTING,
    LIQUIDATION_ADVANTAGE_REWARD_ACCOUNTING,
    SECONDS_PER_YEAR,
    build_unified_exit_economics_objective_contract,
    seal_frozen_capital_hurdle_owner_artifact,
)
from gx1.contracts.unified_exit_fitted_q_v1 import (
    require_unified_exit_unbounded_training_readiness,
)
from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import (
    canonical_sha256,
    file_sha256,
)


CAPITAL_HURDLE_METHOD_RECEIPT_SCHEMA_VERSION = (
    "gx1_exit_capital_hurdle_source_method_receipt_v1"
)
VAL_ECONOMICS_REFERENCE_SCHEMA_VERSION = "gx1_unified_exit_val_economics_reference_v1"
OPERATOR_EFFECTIVE_ANNUAL_RETURN_HURDLE = 0.10


def seal_capital_hurdle_method_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    observed = dict(value)
    if "receipt_sha256" in observed:
        raise RuntimeError("EXIT_HURDLE_METHOD_RECEIPT_ALREADY_SEALED")
    observed["receipt_sha256"] = canonical_sha256(observed)
    return observed


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"EXIT_HURDLE_{label}_SHA_INVALID")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeError(f"EXIT_HURDLE_{label}_PATH_INVALID")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"EXIT_HURDLE_{label}_INVALID") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"EXIT_HURDLE_{label}_INVALID")
    return value


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require_capital_hurdle_method_receipt(
    value: Mapping[str, Any],
    *,
    expected_train_split_sha256: str,
    expected_train_fold_sha256: str,
    expected_source_lineage_sha256: str,
    expected_policy_sha256: str,
) -> dict[str, Any]:
    keys = {
        "schema_version",
        "decision",
        "method",
        "owner_kind",
        "applicable_splits",
        "fitted_splits",
        "validation_or_test_used",
        "train_split_sha256",
        "train_fold_sha256",
        "source_lineage_sha256",
        "policy_sha256",
        "effective_annual_return_hurdle",
        "annual_continuous_hurdle_rate",
        "rate_conversion_formula",
        "rate_unit",
        "seconds_per_year",
        "source_method_artifact_path",
        "source_method_artifact_sha256",
        "test_data_used",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeError("EXIT_HURDLE_METHOD_RECEIPT_INVALID")
    observed = dict(value)
    declared_sha = observed.pop("receipt_sha256")
    source_path = Path(str(observed["source_method_artifact_path"] or ""))
    try:
        rho = float(observed["annual_continuous_hurdle_rate"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("EXIT_HURDLE_METHOD_RECEIPT_INVALID") from exc
    expected = {
        "train_split_sha256": _require_sha(expected_train_split_sha256, "TRAIN_SPLIT"),
        "train_fold_sha256": _require_sha(expected_train_fold_sha256, "TRAIN_FOLD"),
        "source_lineage_sha256": _require_sha(
            expected_source_lineage_sha256, "SOURCE_LINEAGE"
        ),
        "policy_sha256": _require_sha(expected_policy_sha256, "POLICY"),
    }
    if (
        observed["schema_version"] != CAPITAL_HURDLE_METHOD_RECEIPT_SCHEMA_VERSION
        or observed["decision"] != "PASS"
        or observed["owner_kind"] != "project_preregistered_prospective_policy"
        or observed["applicable_splits"] != ["train", "val"]
        or observed["fitted_splits"] != []
        or observed["validation_or_test_used"] is not False
        or observed["test_data_used"] is not False
        or not isinstance(observed["method"], str)
        or not observed["method"]
        or not math.isfinite(rho)
        or rho <= 0.0
        or observed["effective_annual_return_hurdle"]
        != OPERATOR_EFFECTIVE_ANNUAL_RETURN_HURDLE
        or observed["rate_conversion_formula"] != "rho=ln(1+effective_annual_return)"
        or not math.isclose(
            rho,
            math.log1p(OPERATOR_EFFECTIVE_ANNUAL_RETURN_HURDLE),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or observed["rate_unit"] != "continuous_per_wall_clock_year"
        or observed["seconds_per_year"] != SECONDS_PER_YEAR
        or any(observed[key] != digest for key, digest in expected.items())
        or not source_path.is_absolute()
        or not source_path.is_file()
        or source_path.is_symlink()
        or file_sha256(source_path) != observed["source_method_artifact_sha256"]
        or declared_sha != canonical_sha256(observed)
    ):
        raise RuntimeError("EXIT_HURDLE_METHOD_RECEIPT_INVALID")
    _require_sha(declared_sha, "METHOD_RECEIPT")
    _require_sha(observed["source_method_artifact_sha256"], "SOURCE_METHOD")
    return {
        **observed,
        "annual_continuous_hurdle_rate": rho,
        "receipt_sha256": declared_sha,
    }


def build_training_economics_readiness(
    *,
    output_path: Path,
    method_receipt_path: Path,
    train_split_sha256: str,
    train_fold_sha256: str,
    source_lineage_sha256: str,
    policy_sha256: str,
    publish: bool,
    reward_accounting: str = "terminal_cash_v2",
) -> dict[str, Any]:
    """Create readiness; rho is accepted only from the bound TRAIN method receipt."""

    if type(publish) is not bool:
        raise RuntimeError("EXIT_HURDLE_INVOCATION_INVALID")
    receipt_path = method_receipt_path.expanduser().resolve()
    receipt = require_capital_hurdle_method_receipt(
        _read_json(receipt_path, "METHOD_RECEIPT"),
        expected_train_split_sha256=train_split_sha256,
        expected_train_fold_sha256=train_fold_sha256,
        expected_source_lineage_sha256=source_lineage_sha256,
        expected_policy_sha256=policy_sha256,
    )
    hurdle = seal_frozen_capital_hurdle_owner_artifact(
        {
            "schema_version": FROZEN_CAPITAL_HURDLE_SCHEMA_VERSION,
            "decision": "PASS",
            "owner_kind": "project_preregistered_prospective_policy",
            "applicable_splits": ["train", "val"],
            "fitted_splits": [],
            "validation_or_test_used": False,
            "train_split_sha256": train_split_sha256,
            "train_fold_sha256": train_fold_sha256,
            "source_lineage_sha256": source_lineage_sha256,
            "effective_annual_return_hurdle": receipt["effective_annual_return_hurdle"],
            "annual_continuous_hurdle_rate": receipt["annual_continuous_hurdle_rate"],
            "rate_conversion_formula": "rho=ln(1+effective_annual_return)",
            "rate_unit": "continuous_per_wall_clock_year",
            "seconds_per_year": SECONDS_PER_YEAR,
            "source_method": receipt["method"],
            "source_method_artifact_sha256": receipt["receipt_sha256"],
        }
    )
    objective = build_unified_exit_economics_objective_contract(
        capital_hurdle_artifact=hurdle,
        expected_train_split_sha256=train_split_sha256,
        expected_train_fold_sha256=train_fold_sha256,
        expected_source_lineage_sha256=source_lineage_sha256,
        policy_sha256=policy_sha256,
        reward_accounting=reward_accounting,
    )
    readiness = {
        "schema_version": "gx1_unified_exit_training_economics_readiness_v3",
        "mode": ("economics_objective_v4" if reward_accounting == LIQUIDATION_ADVANTAGE_REWARD_ACCOUNTING
                 else "economics_objective_v3" if reward_accounting == MARK_TO_MARKET_REWARD_ACCOUNTING
                 else "economics_objective_v2"),
        "capital_hurdle_artifact": hurdle,
        "economics_objective_contract": objective,
        "expected_train_split_sha256": train_split_sha256,
        "expected_train_fold_sha256": train_fold_sha256,
        "expected_source_lineage_sha256": source_lineage_sha256,
        "policy_sha256": policy_sha256,
        "proper_policy_certificate_sha256": None,
        "test_data_used": False,
    }
    checked = require_unified_exit_unbounded_training_readiness(
        readiness, context="BUILD_TRAINING_ECONOMICS_READINESS"
    )
    if publish:
        _write_atomic(output_path.expanduser().resolve(), readiness)
    return {
        "decision": "PASS" if publish else "PASS_NOT_PUBLISHED",
        "output_path": str(output_path.expanduser().resolve()),
        "method_receipt_path": str(receipt_path),
        "method_receipt_file_sha256": file_sha256(receipt_path),
        "readiness_sha256": canonical_sha256(readiness),
        "readiness": readiness,
        "validated_annual_continuous_hurdle_rate": checked[
            "validated_annual_continuous_hurdle_rate"
        ],
    }


def build_val_economics_reference(
    *,
    output_path: Path,
    train_readiness_path: Path,
    val_split_sha256: str,
    publish: bool,
) -> dict[str, Any]:
    """Bind VAL to the exact TRAIN-frozen economics owner without refitting rho."""

    if type(publish) is not bool:
        raise RuntimeError("EXIT_HURDLE_VAL_REFERENCE_INVOCATION_INVALID")
    train_path = train_readiness_path.expanduser().resolve()
    readiness = require_unified_exit_unbounded_training_readiness(
        _read_json(train_path, "TRAIN_READINESS"),
        context="BUILD_VAL_ECONOMICS_REFERENCE",
    )
    val_split = _require_sha(val_split_sha256, "VAL_SPLIT")
    payload = {
        "schema_version": VAL_ECONOMICS_REFERENCE_SCHEMA_VERSION,
        "decision": "PASS",
        "split": "val",
        "val_split_sha256": val_split,
        "train_readiness_path": str(train_path),
        "train_readiness_file_sha256": file_sha256(train_path),
        "train_readiness_identity_sha256": canonical_sha256(
            {
                key: value
                for key, value in readiness.items()
                if key != "validated_annual_continuous_hurdle_rate"
            }
        ),
        "train_capital_hurdle_artifact_sha256": readiness["capital_hurdle_artifact"][
            "artifact_sha256"
        ],
        "economics_objective_contract_sha256": readiness[
            "economics_objective_contract"
        ]["contract_sha256"],
        "annual_continuous_hurdle_rate": readiness[
            "validated_annual_continuous_hurdle_rate"
        ],
        "rho_refit_for_val": False,
        "validation_or_test_used_to_fit_hurdle": False,
        "test_data_used": False,
    }
    payload["reference_sha256"] = canonical_sha256(payload)
    if publish:
        _write_atomic(output_path.expanduser().resolve(), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method-receipt", required=True, type=Path)
    parser.add_argument("--train-split-sha256", required=True)
    parser.add_argument("--train-fold-sha256", required=True)
    parser.add_argument("--source-lineage-sha256", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--reward-accounting", default="terminal_cash_v2",
                        choices=("terminal_cash_v2", MARK_TO_MARKET_REWARD_ACCOUNTING, LIQUIDATION_ADVANTAGE_REWARD_ACCOUNTING))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    result = build_training_economics_readiness(
        output_path=args.output,
        method_receipt_path=args.method_receipt,
        train_split_sha256=args.train_split_sha256,
        train_fold_sha256=args.train_fold_sha256,
        source_lineage_sha256=args.source_lineage_sha256,
        policy_sha256=args.policy_sha256,
        publish=args.publish,
        reward_accounting=args.reward_accounting,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
