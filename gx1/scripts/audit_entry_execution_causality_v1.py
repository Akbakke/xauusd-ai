#!/usr/bin/env python3
"""Audit the Entry decision-to-fill boundary before any trainer launch.

This is intentionally a metadata/manifest audit.  It neither materializes a
dataset nor opens a model, so an execution-causality failure is discovered
before GPU allocation or a multi-day training run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from gx1.contracts.entry_execution_causality_v1 import (
    ENTRY_EXECUTION_CAUSALITY_REQUIRED_SPLITS,
    build_entry_execution_causality_audit,
    canonical_json_sha256,
    is_sha256,
    legacy_same_close_target_contract_failures,
)
from gx1.contracts.entry_fitted_q_v1 import require_entry_fitted_q_contract
from gx1.contracts.entry_causal_m1_position_size_target_policy_v1 import (
    require_causal_m1_position_size_target_manifest_binding,
)
from gx1.contracts.entry_causal_m1_target_policy_v1 import (
    causal_m1_direction_diagnostic_outcome_contract,
    require_causal_m1_target_policy,
)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError(f"ENTRY_EXECUTION_CAUSALITY_{label}_MISSING: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"ENTRY_EXECUTION_CAUSALITY_{label}_INVALID: {resolved}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"ENTRY_EXECUTION_CAUSALITY_{label}_OBJECT_REQUIRED")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_row(
    *,
    split: str,
    dataset_manifest_path: Path,
    lifecycle_manifest_path: Path,
    expected_run_id: str,
    expected_direction_policy_sha256: str,
    require_causal_auxiliary: bool,
    expected_causal_direction_policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    dataset_manifest = _read_json(
        dataset_manifest_path, label=f"{split.upper()}_DATASET_MANIFEST"
    )
    extra = dataset_manifest.get("extra")
    if not isinstance(extra, Mapping):
        raise RuntimeError(
            f"ENTRY_EXECUTION_CAUSALITY_{split.upper()}_DATASET_EXTRA_MISSING"
        )
    if extra.get("entry_run_id") != expected_run_id:
        raise RuntimeError(
            f"ENTRY_EXECUTION_CAUSALITY_{split.upper()}_DATASET_RUN_ID_MISMATCH"
        )
    require_entry_fitted_q_contract(
        extra.get("entry_fitted_q"),
        context=f"ENTRY_EXECUTION_CAUSALITY_{split.upper()}",
    )
    diagnostic = extra.get("diagnostic_outcome_labels")
    if not isinstance(diagnostic, Mapping):
        raise RuntimeError(
            f"ENTRY_EXECUTION_CAUSALITY_{split.upper()}_AUXILIARY_POLICY_LINEAGE_INVALID"
        )
    if require_causal_auxiliary:
        if not isinstance(expected_causal_direction_policy, Mapping):
            raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_CAUSAL_TARGET_POLICY_MISSING")
        expected_diagnostic = causal_m1_direction_diagnostic_outcome_contract(
            expected_causal_direction_policy
        )
        if any(
            diagnostic.get(field) != expected
            for field, expected in expected_diagnostic.items()
        ):
            raise RuntimeError(
                f"ENTRY_EXECUTION_CAUSALITY_{split.upper()}_CAUSAL_DIAGNOSTIC_CONTRACT_INVALID"
            )
        position = require_causal_m1_position_size_target_manifest_binding(
            extra,
            expected_source_parquet_sha256=expected_causal_direction_policy[
                "source_parquet_sha256"
            ],
            expected_tape_provenance_sha256=expected_causal_direction_policy[
                "tape_provenance_sha256"
            ],
            expected_m1_source_sha256=expected_causal_direction_policy[
                "m1_source_sha256"
            ],
            expected_direction_policy_sha256=expected_direction_policy_sha256,
        )
    else:
        # Legacy artifacts are still reportable as BLOCK; they must not make
        # the audit crash before it can state why their M5-close contract is
        # unfit for training.
        position = extra.get("entry_position_size_target_policy")
    position_direction_hash = (
        position.get("entry_causal_m1_target_policy_sha256")
        if isinstance(position, Mapping)
        else None
    )
    if position_direction_hash is None and isinstance(position, Mapping):
        position_direction_hash = position.get("entry_direction_target_policy_sha256")
    if (
        not isinstance(position, Mapping)
        or diagnostic.get("diagnostic_outcome_policy_sha256")
        != expected_direction_policy_sha256
        or extra.get("diagnostic_outcome_policy_sha256")
        != expected_direction_policy_sha256
        or position_direction_hash != expected_direction_policy_sha256
    ):
        raise RuntimeError(
            f"ENTRY_EXECUTION_CAUSALITY_{split.upper()}_AUXILIARY_POLICY_LINEAGE_INVALID"
        )
    lifecycle = _read_json(
        lifecycle_manifest_path, label=f"{split.upper()}_LIFECYCLE_MANIFEST"
    )
    lifecycle_bound = (
        lifecycle.get("decision") == "PASS"
        and lifecycle.get("entry_run_id") == expected_run_id
        and lifecycle.get("entry_side_selection")
        == "both_sides_for_every_causal_entry_snapshot"
        and lifecycle.get("first_state_post_fill_closed_bars") == 1
        and lifecycle.get("state_row_timestamp_semantics")
        == "authoritative_m1_bar_start"
        and lifecycle.get("decision_time_semantics")
        == "authoritative_m1_bar_close_available_at"
        and lifecycle.get("future_outcomes_used_as_model_inputs") is False
        and lifecycle.get("sample_selection_depends_on_future_target") is False
        and lifecycle.get("exit_supervision_authority")
        == "executable_exit_now_reward_plus_train_fitted_q"
    )
    return {
        "split": split,
        "dataset_manifest_path": str(dataset_manifest_path.resolve()),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "lifecycle_manifest_path": str(lifecycle_manifest_path.resolve()),
        "lifecycle_manifest_sha256": _sha256_file(lifecycle_manifest_path),
        "entry_fitted_q_m1_fill_lifecycle_bound": lifecycle_bound,
        # The immutable ranking target decides whether every current active
        # diagnostic/sizing auxiliary has a causal quote definition.  The
        # top-level report derives this one, so it cannot be silently set True
        # in a split row.
        "active_auxiliary_targets_m1_fill_bound": False,
    }


def build_report(
    *,
    dataset_dir: Path,
    signal_manifest_path: Path,
    split_paths: Mapping[str, tuple[Path, Path]],
) -> dict[str, Any]:
    signal = _read_json(signal_manifest_path, label="SIGNAL_MANIFEST")
    run_id = signal.get("entry_run_id")
    ranking = signal.get("feature_ranking")
    if not isinstance(run_id, str) or not run_id or not isinstance(ranking, Mapping):
        raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_SIGNAL_LINEAGE_INVALID")
    policy_sha256 = ranking.get("entry_direction_target_policy_sha256")
    if not is_sha256(policy_sha256):
        raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_DIRECTION_POLICY_HASH_INVALID")
    target_contract = ranking.get("target_contract")
    require_causal_auxiliary = not legacy_same_close_target_contract_failures(
        target_contract
    )
    causal_direction_policy: Mapping[str, Any] | None = None
    if require_causal_auxiliary:
        source_sha256 = ranking.get("source_sha256")
        if not is_sha256(source_sha256):
            raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_CAUSAL_RANKING_SOURCE_INVALID")
        causal_direction_policy = require_causal_m1_target_policy(
            ranking.get("entry_direction_target_policy"),
            expected_source_parquet_sha256=source_sha256,
        )
        # `feature_ranking.target_contract` describes the ranking objective,
        # while the causal policy's `target_contract` owns the lower-level M1
        # outcome construction.  They intentionally have different schemas:
        # the former adds ranking fit semantics and the latter adds gap/path
        # semantics.  Requiring whole-object equality would therefore reject a
        # correctly-bound causal ranking.  Require agreement on every field
        # they deliberately share instead; each complete contract is already
        # validated by its respective owner above.
        causal_target_contract = causal_direction_policy["target_contract"]
        shared_target_contract_fields = set(causal_target_contract).intersection(
            target_contract
        )
        if (
            causal_direction_policy["policy_sha256"] != policy_sha256
            or not shared_target_contract_fields
            or any(
                causal_target_contract[field] != target_contract[field]
                for field in shared_target_contract_fields
            )
        ):
            raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_CAUSAL_RANKING_POLICY_MISMATCH")
    split_rows = [
        _split_row(
            split=split,
            dataset_manifest_path=split_paths[split][0],
            lifecycle_manifest_path=split_paths[split][1],
            expected_run_id=run_id,
            expected_direction_policy_sha256=policy_sha256,
            require_causal_auxiliary=require_causal_auxiliary,
            expected_causal_direction_policy=causal_direction_policy,
        )
        for split in ENTRY_EXECUTION_CAUSALITY_REQUIRED_SPLITS
    ]
    auxiliary_bound = not legacy_same_close_target_contract_failures(target_contract)
    fitted_q_bound = all(
        row["entry_fitted_q_m1_fill_lifecycle_bound"] for row in split_rows
    )
    for row in split_rows:
        row["entry_fitted_q_m1_fill_lifecycle_bound"] = fitted_q_bound
        row["active_auxiliary_targets_m1_fill_bound"] = auxiliary_bound
    return build_entry_execution_causality_audit(
        dataset_dir=str(dataset_dir.resolve()),
        entry_run_id=run_id,
        signal_manifest_path=str(signal_manifest_path.resolve()),
        signal_manifest_sha256=_sha256_file(signal_manifest_path),
        ranking_target_contract=target_contract,
        split_rows=split_rows,
    )


def _write_fresh_json(path: Path, payload: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve()
    if output.suffix != ".json" or not output.parent.is_dir():
        raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_OUTPUT_PATH_INVALID")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short report write: {output}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--signal-manifest", required=True)
    for split in ENTRY_EXECUTION_CAUSALITY_REQUIRED_SPLITS:
        parser.add_argument(f"--{split}-manifest", required=True)
        parser.add_argument(f"--{split}-lifecycle-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_DATASET_DIR_INVALID")
    split_paths = {
        split: (
            Path(getattr(args, f"{split}_manifest")).expanduser().resolve(),
            Path(getattr(args, f"{split}_lifecycle_manifest")).expanduser().resolve(),
        )
        for split in ENTRY_EXECUTION_CAUSALITY_REQUIRED_SPLITS
    }
    report = build_report(
        dataset_dir=dataset_dir,
        signal_manifest_path=Path(args.signal_manifest).expanduser().resolve(),
        split_paths=split_paths,
    )
    _write_fresh_json(Path(args.output), report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "training_authorized": report["training_authorized"],
                "failures": report["failures"],
                "report_sha256": canonical_json_sha256(report),
                "output": str(Path(args.output).expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if report["training_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
