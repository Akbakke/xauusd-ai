#!/usr/bin/env python3
"""Bind one completed seq513 rebuild into an immutable smoke-readiness input.

This is a report-only boundary.  It revalidates the exact split bytes, the
complete liveness artifact, the pretrain target audit and the on-disk XAU tape
lineage.  It never copies a dataset and never starts training or serving.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.contracts.entry_dataset_split_artifacts_v1 import (
    ENTRY_DATASET_SPLIT_ARTIFACTS_SCHEMA_VERSION,
    require_dataset_split_artifacts,
)
from gx1.contracts.entry_full_input_liveness_v1 import (
    EXPECTED_FIELD_COUNTS,
    SCHEMA_VERSION as LIVENESS_SCHEMA_VERSION,
    validate_full_input_liveness_artifact,
)
from gx1.contracts.entry_model_native_post_rebuild_v1 import (
    EVENT_PREFIX,
    READY_DECISION,
    REQUIRED_PROOF_CHECKS,
    SCHEMA_VERSION,
    SIDE_EFFECT_KEYS,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
    require_model_native_signal_contract,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.contracts.xau_tape_provenance_v1 import (
    CANONICAL_NATIVE_SOURCE_SCHEMA,
    CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
    CURRENT_SNAPSHOT_SCHEMA,
    XAU_INSTRUMENT,
    validate_xau_tape_provenance_v1,
)


SPLITS = ("train", "val", "test")
SPLIT_MANIFEST_SCHEMA = MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION
PREFLIGHT_SCHEMA = "entry_model_native_seq513_rebuild_preflight_v9"
PREFLIGHT_DECISION = "READY_FOR_MODEL_NATIVE_SEQ513_REBUILD"
PRETRAIN_SCHEMA = "xau_direction_repair_pretrain_audit_v2"
CHAIN_SCHEMA = "seq513_rebuild_chain_status_v4"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label}: missing regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"{label}: invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label}: JSON root must be an object: {path}")
    return value


def _path(raw: str, *, label: str, directory: bool = False) -> Path:
    path = Path(str(raw or "")).expanduser()
    valid = path.is_dir() if directory else path.is_file()
    if (
        not path.is_absolute()
        or path.resolve() != path
        or path.is_symlink()
        or not valid
        or any("latest" in part.lower() for part in path.parts)
    ):
        raise RuntimeError(f"{label}: invalid immutable path: {path}")
    return path


def _artifact(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema_version": payload.get("schema_version"),
        "decision": payload.get("decision"),
    }


def _check(name: str, ok: bool, details: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details}


def _git_identity(repo_dir: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if head.returncode != 0 or status.returncode != 0:
        raise RuntimeError("cannot establish post-rebuild producer Git identity")
    dirty = [line for line in status.stdout.splitlines() if line.strip()]
    if dirty:
        raise RuntimeError(
            f"post-rebuild readiness requires a clean worktree: {dirty[:20]}"
        )
    observed_head = head.stdout.strip().lower()
    if len(observed_head) != 40:
        raise RuntimeError("invalid post-rebuild producer Git head")
    return {"repo_dir": str(repo_dir), "head": observed_head, "status_short": []}


def _manifest_contract(
    *,
    split: str,
    binding: dict[str, str],
    expected_run_id: str,
    expected_xau_provenance: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    manifest_path = Path(binding["manifest_path"])
    manifest = _json(manifest_path, label=f"{split} manifest")
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    feature = (
        manifest.get("feature_contract")
        if isinstance(manifest.get("feature_contract"), dict)
        else {}
    )
    signal_contract = (
        extra.get("model_native_signal_contract")
        if isinstance(extra.get("model_native_signal_contract"), dict)
        else {}
    )
    failures: list[str] = []
    if manifest.get("schema_version") != SPLIT_MANIFEST_SCHEMA:
        failures.append(f"{split}: split manifest schema mismatch")
    if manifest.get("manifest_variant") != MODEL_NATIVE_CONTRACT_MODE:
        failures.append(f"{split}: contract mode mismatch")
    if manifest.get("output_data_path") != binding["parquet_path"]:
        failures.append(f"{split}: parquet self-path mismatch")
    if manifest.get("expected_seq_snap_width") != MODEL_NATIVE_SIGNAL_DIM:
        failures.append(f"{split}: expected signal width mismatch")
    signal_fields = feature.get("signal_bridge_fields")
    if (
        not isinstance(signal_fields, list)
        or len(signal_fields) != MODEL_NATIVE_SIGNAL_DIM
    ):
        failures.append(f"{split}: signal dimension mismatch")
    try:
        require_model_native_signal_contract(
            signal_contract,
            context=f"post-rebuild {split} split manifest",
        )
    except Exception as exc:
        failures.append(f"{split}: model-native signal contract invalid: {exc}")
    if signal_fields != signal_contract.get("fields"):
        failures.append(f"{split}: feature/signal field order mismatch")
    if feature.get("ctx_cont_dim") != EXPECTED_FIELD_COUNTS["ctx_cont"]:
        failures.append(f"{split}: continuous context dimension mismatch")
    if feature.get("ctx_cat_dim") != EXPECTED_FIELD_COUNTS["ctx_cat"]:
        failures.append(f"{split}: categorical context dimension mismatch")
    if extra.get("entry_run_id") != expected_run_id:
        failures.append(f"{split}: run id mismatch")
    if extra.get("xau_tape_provenance") != expected_xau_provenance:
        failures.append(f"{split}: XAU tape provenance mismatch")
    return {
        **binding,
        "schema_version": manifest.get("schema_version"),
        "manifest_variant": manifest.get("manifest_variant"),
        "rows": int(extra.get("rows") or 0),
        "entry_run_id": extra.get("entry_run_id"),
        "instrument": (extra.get("xau_tape_provenance") or {}).get("instrument"),
    }, failures


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_id = str(args.run_id or "").strip()
    if not run_id:
        raise RuntimeError("entry run id is required")
    event_root = _path(args.event_root, label="event root", directory=True)
    repo_dir = _path(args.repo_dir, label="repository", directory=True)
    producer_git = _git_identity(repo_dir)
    dataset_dir = _path(args.dataset_dir, label="dataset dir", directory=True)
    smoke_dataset_dir = _path(
        args.smoke_dataset_dir,
        label="smoke dataset dir",
        directory=True,
    )
    if dataset_dir != smoke_dataset_dir:
        raise RuntimeError(
            "separate smoke dataset is forbidden; smoke must consume the exact rebuild splits"
        )
    if dataset_dir.parent != event_root:
        raise RuntimeError("dataset directory is not owned by the declared event root")

    terminal_path = _path(args.chain_terminal_json, label="chain terminal")
    preflight_path = _path(args.rebuild_preflight_json, label="rebuild preflight")
    liveness_path = _path(args.full_input_liveness_json, label="full-input liveness")
    pretrain_path = _path(args.pretrain_audit_json, label="pretrain audit")
    for label, path in (
        ("chain terminal", terminal_path),
        ("rebuild preflight", preflight_path),
        ("full-input liveness", liveness_path),
        ("pretrain audit", pretrain_path),
    ):
        if event_root not in path.parents:
            raise RuntimeError(f"{label} is outside the declared event root: {path}")

    terminal = _json(terminal_path, label="chain terminal")
    preflight = _json(preflight_path, label="rebuild preflight")
    liveness = _json(liveness_path, label="full-input liveness")
    pretrain = _json(pretrain_path, label="pretrain audit")

    supplied_bindings = {
        split: {
            "manifest_path": str(
                _path(getattr(args, f"{split}_manifest_json"), label=f"{split} manifest")
            ),
            "manifest_sha256": str(getattr(args, f"{split}_manifest_sha256")),
            "parquet_sha256": str(getattr(args, f"{split}_parquet_sha256")),
        }
        for split in SPLITS
    }
    resolved_splits = require_dataset_split_artifacts(
        dataset_dir,
        supplied_bindings,
        expected_splits=SPLITS,
        context="MODEL_NATIVE_POST_REBUILD_SPLITS",
    )
    for split in SPLITS:
        supplied_parquet = _path(
            getattr(args, f"{split}_parquet"),
            label=f"{split} parquet",
        )
        if str(supplied_parquet) != resolved_splits[split]["parquet_path"]:
            raise RuntimeError(f"{split}: explicit parquet differs from manifest binding")

    first_manifest = _json(
        Path(resolved_splits["train"]["manifest_path"]),
        label="train manifest",
    )
    first_extra = (
        first_manifest.get("extra")
        if isinstance(first_manifest.get("extra"), dict)
        else {}
    )
    declared_xau = first_extra.get("xau_tape_provenance")
    if not isinstance(declared_xau, dict):
        raise RuntimeError("train manifest lacks XAU tape provenance")
    xau_provenance = validate_xau_tape_provenance_v1(
        declared_xau.get("tape_root"),
        expected_run_id=run_id,
        require_current=True,
    )
    if declared_xau != xau_provenance:
        raise RuntimeError("train manifest XAU binding differs from disk revalidation")

    split_artifacts: dict[str, dict[str, Any]] = {}
    split_failures: list[str] = []
    for split in SPLITS:
        split_artifacts[split], failures = _manifest_contract(
            split=split,
            binding=resolved_splits[split],
            expected_run_id=run_id,
            expected_xau_provenance=xau_provenance,
        )
        split_failures.extend(failures)

    liveness_validation = validate_full_input_liveness_artifact(
        liveness_path,
        expected_sha256=_sha256_file(liveness_path),
        expected_dataset_dir=dataset_dir,
        expected_contract_mode=MODEL_NATIVE_CONTRACT_MODE,
        expected_manifest_bindings={
            split: {
                "path": split_artifacts[split]["manifest_path"],
                "sha256": split_artifacts[split]["manifest_sha256"],
            }
            for split in SPLITS
        },
    )

    terminal_ok = (
        terminal.get("schema_version") == CHAIN_SCHEMA
        and terminal.get("state") == "GREEN"
        and terminal.get("step") == "chain-complete"
        and terminal.get("exit_code") == 0
        and terminal.get("entry_run_id") == run_id
        and Path(str(terminal.get("event_root") or "")).resolve() == event_root
        and Path(str(terminal.get("terminal_event_path") or "")).resolve()
        == terminal_path
    )
    terminal_preflight = (
        terminal.get("preflight")
        if isinstance(terminal.get("preflight"), dict)
        else {}
    )
    preflight_ok = (
        preflight.get("schema_version") == PREFLIGHT_SCHEMA
        and preflight.get("decision") == PREFLIGHT_DECISION
        and preflight.get("entry_run_id") == run_id
        and preflight.get("training_allowed") is False
        and terminal_preflight.get("json_path") == str(preflight_path)
        and terminal_preflight.get("sha256") == _sha256_file(preflight_path)
    )
    liveness_ok = (
        liveness.get("schema_version") == LIVENESS_SCHEMA_VERSION
        and liveness.get("decision") == "PASS"
        and not liveness.get("failures")
        and bool(liveness_validation.get("ok"))
    )
    pretrain_tape = (
        pretrain.get("tape_provenance")
        if isinstance(pretrain.get("tape_provenance"), dict)
        else {}
    )
    pretrain_ok = (
        pretrain.get("schema_version") == PRETRAIN_SCHEMA
        and pretrain.get("decision") == "PASS"
        and not pretrain.get("failures")
        and pretrain.get("require_xau_provenance") is True
        and tuple(pretrain.get("data_splits") or ()) == SPLITS
        and all(pretrain_tape.get(split) == xau_provenance for split in SPLITS)
    )
    provenance_schema = xau_provenance.get("schema_version")
    if provenance_schema in {
        CANONICAL_NATIVE_SOURCE_SCHEMA,
        CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
    }:
        # A strict native source root is complete tape provenance by
        # construction; run identity is bound by the consuming split
        # manifests, which _manifest_contract validates per split above.
        tape_identity_ok = xau_provenance.get("instrument") == XAU_INSTRUMENT
    else:
        tape_identity_ok = (
            provenance_schema == CURRENT_SNAPSHOT_SCHEMA
            and xau_provenance.get("instrument") == XAU_INSTRUMENT
            and xau_provenance.get("entry_run_id") == run_id
        )
    splits_ok = (
        not split_failures
        and tape_identity_ok
        and all(split_artifacts[split]["rows"] > 0 for split in SPLITS)
    )

    checks = [
        _check(REQUIRED_PROOF_CHECKS[0], terminal_ok, _artifact(terminal_path, terminal)),
        _check(REQUIRED_PROOF_CHECKS[1], preflight_ok, _artifact(preflight_path, preflight)),
        _check(REQUIRED_PROOF_CHECKS[2], liveness_ok, liveness_validation),
        _check(REQUIRED_PROOF_CHECKS[3], pretrain_ok, _artifact(pretrain_path, pretrain)),
        _check(
            REQUIRED_PROOF_CHECKS[4],
            splits_ok,
            {"failures": split_failures, "splits": split_artifacts},
        ),
    ]
    failures = [row for row in checks if not row["ok"]]
    side_effects = {key: False for key in SIDE_EFFECT_KEYS}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": READY_DECISION if not failures else "BLOCKED_MODEL_NATIVE_SEQ513_POST_REBUILD",
        "entry_run_id": run_id,
        "event_root": str(event_root),
        "producer_git": producer_git,
        "source_git_head": terminal.get("git_head"),
        "dataset_dir": str(dataset_dir),
        "smoke_dataset_dir": str(smoke_dataset_dir),
        "report_only": True,
        "full_input_liveness_contract": {
            **_artifact(liveness_path, liveness),
            "field_order_sha256": liveness.get("field_order_sha256"),
            "field_counts": liveness.get("expected_field_counts"),
            "atr_ood_status": (liveness.get("atr_ood_drift") or {}).get("status"),
        },
        "pretrain_audit": _artifact(pretrain_path, pretrain),
        "chain_terminal": _artifact(terminal_path, terminal),
        "rebuild_preflight": _artifact(preflight_path, preflight),
        "split_artifacts_schema_version": ENTRY_DATASET_SPLIT_ARTIFACTS_SCHEMA_VERSION,
        "split_artifacts": split_artifacts,
        "xau_tape_provenance": xau_provenance,
        "post_rebuild_refresh_command_contract": {
            "smoke_dataset_dir": str(smoke_dataset_dir),
            "all_commands_avoid_training_replay_iql_shadow_live": True,
        },
        "checks": checks,
        "failures": failures,
        "side_effects_started": side_effects,
        "training_allowed": False,
        "replay_allowed": False,
        "shadow_live_allowed": False,
    }
    out_dir = Path(args.out_dir).expanduser().resolve()
    path, payload = write_immutable_json_event(out_dir, EVENT_PREFIX, payload)
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": payload["decision"],
                    "entry_run_id": run_id,
                    "split_rows": {
                        split: split_artifacts[split]["rows"] for split in SPLITS
                    },
                    "json_path": str(path),
                    "failures": failures,
                },
                indent=2,
                sort_keys=True,
            )
        )
    if failures:
        raise SystemExit(2)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--event-root", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--chain-terminal-json", required=True)
    parser.add_argument("--rebuild-preflight-json", required=True)
    parser.add_argument("--full-input-liveness-json", required=True)
    parser.add_argument("--pretrain-audit-json", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--smoke-dataset-dir", required=True)
    for split in SPLITS:
        parser.add_argument(f"--{split}-manifest-json", required=True)
        parser.add_argument(f"--{split}-manifest-sha256", required=True)
        parser.add_argument(f"--{split}-parquet", required=True)
        parser.add_argument(f"--{split}-parquet-sha256", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
