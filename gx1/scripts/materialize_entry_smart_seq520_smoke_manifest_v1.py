#!/usr/bin/env python3
"""Materialize a report-only smart_seq520 smoke dataset manifest.

This script verifies an already-built smart_seq520 smoke dataset and writes a
manifest/report under the report directory only. It does not rebuild data, copy
parquets, start training, run replay, distill IQL, or touch shadow/live paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from gx1.scripts.verify_entry_foundation_state_v1 import FOUNDATION_DATASET_DIR, REPORTS_ROOT


SPLITS = ("train", "val", "test")
SCHEMA_VERSION = "entry_smart_seq520_smoke_dataset_v1"
SPLIT_SCHEMA_VERSION = "entry_smart_seq520_smoke_split_manifest_v1"
REPORT_SCHEMA_VERSION = "entry_smart_seq520_smoke_manifest_readiness_v1"
MANIFEST_VARIANT = "smart_seq520_candidate"
EXPECTED_SEQ_SNAP_WIDTH = 520
DEFAULT_STEM = "v10_smart_seq520_smoke__HOLD_03B"
SMART_SPECIALIST_CONTRACT_MODE = "smart_seq520_candidate"
DEFAULT_SMART_SMOKE_DATASET_DIR = FOUNDATION_DATASET_DIR.parent / "v10_dataset_smart_seq520_smoke_20260630"
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_smart_seq520_smoke_manifest_20260630_v1"
DEFAULT_POST_REBUILD_READINESS_JSON = (
    REPORTS_ROOT
    / "entry_smart_dataset_post_rebuild_readiness_20260630_v1"
    / "ENTRY_SMART_DATASET_POST_REBUILD_READINESS_latest.json"
)
POST_REBUILD_READY_DECISION = "ENTRY_SMART_DATASET_READY_FOR_TRAIN_READINESS_REVIEW"
RAM_CAP_RUNNER = "scripts/gx1_capped_run.sh"
DEFAULT_MEMORY_CAP = "22G"
DEFAULT_SWAP_CAP = "2G"
SIDE_EFFECTS_STARTED = {
    "dataset_rebuild": False,
    "training": False,
    "replay": False,
    "iql_distillation": False,
    "shadow": False,
    "live": False,
}
REQUIRED_POST_REBUILD_SIDE_EFFECT_KEYS = tuple(SIDE_EFFECTS_STARTED)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _artifact_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": bool(path.exists()),
        "size_bytes": int(path.stat().st_size) if path.exists() else None,
        "sha256": _sha256_file(path),
    }


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _check(name: str, ok: bool, details: Any = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details if details is not None else {}}


def _explicit_vedtak_id_ok(vedtak_id: str) -> bool:
    value = str(vedtak_id or "").strip()
    placeholders = {
        "",
        "<id>",
        "<vedtak-id>",
        "<SMART_SEQ520_SMOKE_VEDTAK_ID>",
        "TODO",
        "TBD",
    }
    return value not in placeholders and "<" not in value and ">" not in value


def _split_candidates(dataset_dir: Path, split: str) -> tuple[list[Path], list[Path]]:
    parquets = sorted(dataset_dir.glob(f"*_{split}.parquet")) if dataset_dir.exists() else []
    manifests = sorted(dataset_dir.glob(f"*_{split}.manifest.json")) if dataset_dir.exists() else []
    return parquets, manifests


def _resolve_manifest_output_path(manifest: dict[str, Any], fallback: Path | None) -> Path | None:
    raw = str(manifest.get("output_data_path") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return fallback.resolve() if fallback is not None else None


def _stack_list_column(values: Any, dtype: np.dtype) -> np.ndarray:
    items = list(values)
    if not items:
        return np.asarray([], dtype=dtype)
    try:
        return np.stack(items).astype(dtype, copy=False)
    except ValueError:
        return np.stack([np.stack(item) for item in items]).astype(dtype, copy=False)


def _sample_seq_snap_shapes(path: Path, *, sample_rows: int, batch_size: int) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "errors": ["missing parquet"], "rows": 0}
    errors: list[str] = []
    try:
        pf = pq.ParquetFile(path)
    except Exception as exc:
        return {"ok": False, "errors": [f"open parquet failed: {type(exc).__name__}: {exc}"], "rows": 0}
    schema_names = set(pf.schema_arrow.names)
    missing = [name for name in ("seq", "snap") if name not in schema_names]
    if missing:
        return {
            "ok": False,
            "errors": [f"missing columns: {missing}"],
            "rows": int(pf.metadata.num_rows or 0),
            "schema_columns": sorted(schema_names),
        }
    total_rows = int(pf.metadata.num_rows or 0)
    if total_rows <= 0:
        return {"ok": False, "errors": ["empty parquet"], "rows": 0, "schema_columns": sorted(schema_names)}

    scan_limit = min(total_rows, int(sample_rows))
    scanned = 0
    shape_examples: dict[str, Any] = {}
    nonfinite = {"seq": 0, "snap": 0}
    for batch in pf.iter_batches(batch_size=int(batch_size), columns=["seq", "snap"]):
        if scanned >= scan_limit:
            break
        pdf = batch.to_pandas()
        remaining = scan_limit - scanned
        if len(pdf) > remaining:
            pdf = pdf.iloc[:remaining].copy()
        if pdf.empty:
            continue
        try:
            seq = _stack_list_column(pdf["seq"], np.float32)
            snap = _stack_list_column(pdf["snap"], np.float32)
        except Exception as exc:
            errors.append(f"seq/snap stack failed: {type(exc).__name__}: {exc}")
            scanned += int(len(pdf))
            continue
        shape_examples = {"seq": list(seq.shape), "snap": list(snap.shape)}
        if seq.ndim != 3 or int(seq.shape[-1]) != EXPECTED_SEQ_SNAP_WIDTH:
            errors.append(
                f"seq width mismatch got={list(seq.shape)} expected_last_dim={EXPECTED_SEQ_SNAP_WIDTH}"
            )
        if snap.ndim != 2 or int(snap.shape[-1]) != EXPECTED_SEQ_SNAP_WIDTH:
            errors.append(
                f"snap width mismatch got={list(snap.shape)} expected_last_dim={EXPECTED_SEQ_SNAP_WIDTH}"
            )
        if not errors:
            nonfinite["seq"] += int((~np.isfinite(seq)).sum())
            nonfinite["snap"] += int((~np.isfinite(snap)).sum())
        scanned += int(len(pdf))

    if scanned <= 0:
        errors.append("no seq/snap rows scanned")
    if any(count != 0 for count in nonfinite.values()):
        errors.append(f"nonfinite seq/snap values: {nonfinite}")
    return {
        "ok": bool(scanned > 0 and not errors),
        "rows": total_rows,
        "scanned_rows": int(scanned),
        "sample_rows": int(sample_rows),
        "shape_examples": shape_examples,
        "nonfinite_counts": nonfinite,
        "errors": errors,
    }


def _split_summary(dataset_dir: Path, split: str, *, sample_rows: int, batch_size: int) -> dict[str, Any]:
    parquets, manifests = _split_candidates(dataset_dir, split)
    parquet_path = parquets[0].resolve() if len(parquets) == 1 else None
    manifest_path = manifests[0].resolve() if len(manifests) == 1 else None
    manifest = _read_json_or_empty(manifest_path) if manifest_path is not None else {}
    output_path = _resolve_manifest_output_path(manifest, parquet_path)
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    signal_bridge = extra.get("signal_bridge") if isinstance(extra.get("signal_bridge"), dict) else {}
    fields = [str(x) for x in signal_bridge.get("fields", []) if str(x).strip()]
    shape_probe = (
        _sample_seq_snap_shapes(output_path, sample_rows=sample_rows, batch_size=batch_size)
        if output_path is not None
        else {"ok": False, "errors": ["missing output_data_path"], "rows": 0}
    )
    return {
        "split": split,
        "parquet_candidates": [str(path) for path in parquets],
        "manifest_candidates": [str(path) for path in manifests],
        "parquet_candidate_count": int(len(parquets)),
        "manifest_candidate_count": int(len(manifests)),
        "parquet_path": str(parquet_path) if parquet_path is not None else "",
        "manifest_path": str(manifest_path) if manifest_path is not None else "",
        "parquet_exists": bool(parquet_path is not None and parquet_path.exists()),
        "manifest_exists": bool(manifest_path is not None and manifest_path.exists()),
        "output_data_path": str(output_path) if output_path is not None else "",
        "output_data_exists": bool(output_path is not None and output_path.exists()),
        "manifest_output_matches_split_parquet": bool(
            output_path is not None and parquet_path is not None and output_path == parquet_path
        ),
        "rows": int(shape_probe.get("rows") or 0),
        "manifest_sha256": _sha256_file(manifest_path) if manifest_path is not None else None,
        "parquet_sha256": _sha256_file(output_path) if output_path is not None else None,
        "manifest_variant": str(manifest.get("manifest_variant") or ""),
        "expected_seq_snap_width": int(manifest.get("expected_seq_snap_width") or 0),
        "schema_version": str(manifest.get("schema_version") or ""),
        "seq_input_dim": int(signal_bridge.get("seq_input_dim") or 0),
        "snap_input_dim": int(signal_bridge.get("snap_input_dim") or 0),
        "field_count": int(len(fields)),
        "shape_probe": shape_probe,
    }


def _future_command_contracts(
    *,
    dataset_dir: Path,
    vedtak_id: str,
    memory_cap: str,
    swap_cap: str,
) -> dict[str, Any]:
    train_parquet = dataset_dir / f"{DEFAULT_STEM}_train.parquet"
    train_inner = [
        ".venv/bin/python",
        "-m",
        "gx1.models.entry_v10.entry_v10_ctx_train_v3",
        "--train",
        "--vedtak",
        vedtak_id,
        "--dataset_dir",
        str(dataset_dir),
        "--dataset_train_parquet",
        str(train_parquet),
        "--epochs",
        "1",
        "--batch_size",
        "64",
        "--num-workers",
        "0",
        "--enable-specialist-fusion",
        "--specialist-contract-mode",
        SMART_SPECIALIST_CONTRACT_MODE,
    ]
    return {
        "smart_smoke_manifest": {
            "mode": "report_only_manifest_materialization",
            "requires_explicit_vedtak": True,
            "explicit_vedtak_id": vedtak_id,
            "mutates_only_report_dir": True,
            "starts_training": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
        },
        "smart_smoke_train": {
            "mode": "future_train_contract_not_executed",
            "implemented_in_control_surface": False,
            "requires_trainer_surface_enablement": True,
            "argv_template": [
                RAM_CAP_RUNNER,
                "--mem",
                memory_cap,
                "--swap",
                swap_cap,
                "--",
                *train_inner,
            ],
            "inner_train_argv_template": train_inner,
            "requires_explicit_vedtak": True,
            "explicit_vedtak_id": vedtak_id,
            "requires_ram_cap": True,
            "ram_cap_runner": RAM_CAP_RUNNER,
            "memory_cap": memory_cap,
            "swap_cap": swap_cap,
            "num_workers": 0,
            "started_by_this_report": False,
            "starts_training_if_executed": True,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "expected_seq_snap_width": EXPECTED_SEQ_SNAP_WIDTH,
            "manifest_variant": MANIFEST_VARIANT,
            "specialist_contract_mode": SMART_SPECIALIST_CONTRACT_MODE,
        },
    }


def _build_smoke_manifest(
    *,
    dataset_dir: Path,
    vedtak_id: str,
    splits: dict[str, dict[str, Any]],
    future_command_contracts: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "report_only": True,
        "manifest_variant": MANIFEST_VARIANT,
        "expected_seq_snap_width": EXPECTED_SEQ_SNAP_WIDTH,
        "explicit_vedtak_id": vedtak_id,
        "out_dir": str(dataset_dir),
        "dataset_dir": str(dataset_dir),
        "stem": DEFAULT_STEM,
        "splits": {
            split: {
                "rows": int(row["rows"]),
                "out_parquet": row["output_data_path"],
                "out_manifest": row["manifest_path"],
                "out_parquet_sha256": row["parquet_sha256"],
                "out_manifest_sha256": row["manifest_sha256"],
                "split_manifest_schema_version": row["schema_version"],
                "seq_input_dim": int(row["seq_input_dim"]),
                "snap_input_dim": int(row["snap_input_dim"]),
                "field_count": int(row["field_count"]),
            }
            for split, row in splits.items()
        },
        "future_command_contracts": future_command_contracts,
        "side_effects_started": dict(SIDE_EFFECTS_STARTED),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Smart Seq520 Smoke Manifest",
        "",
        f"- decision: `{report['decision']}`",
        f"- report_only: `{report['report_only']}`",
        f"- manifest_variant: `{MANIFEST_VARIANT}`",
        f"- expected_seq_snap_width: `{EXPECTED_SEQ_SNAP_WIDTH}`",
        f"- manifest_written: `{report['manifest_written']}`",
        f"- failures: `{len(report['failures'])}`",
        "",
        "## Failures",
        "",
    ]
    if report["failures"]:
        lines.extend([f"- `{failure['name']}`" for failure in report["failures"]])
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.smart_smoke_dataset_dir).expanduser().resolve()
    post_rebuild_readiness_json = Path(args.post_rebuild_readiness_json).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    vedtak_id = str(args.vedtak_id or "").strip()
    memory_cap = str(args.memory_cap or DEFAULT_MEMORY_CAP)
    swap_cap = str(args.swap_cap or DEFAULT_SWAP_CAP)
    sample_rows = int(args.sample_rows)
    batch_size = int(args.batch_size)

    splits = {
        split: _split_summary(dataset_dir, split, sample_rows=sample_rows, batch_size=batch_size)
        for split in SPLITS
    }
    post_rebuild_readiness = _read_json_or_empty(post_rebuild_readiness_json)
    post_rebuild_refresh_contract = (
        post_rebuild_readiness.get("post_rebuild_refresh_command_contract")
        if isinstance(post_rebuild_readiness.get("post_rebuild_refresh_command_contract"), dict)
        else {}
    )
    post_rebuild_side_effects = (
        post_rebuild_readiness.get("side_effects_started")
        if isinstance(post_rebuild_readiness.get("side_effects_started"), dict)
        else {}
    )
    post_rebuild_side_effects_closed = (
        all(key in post_rebuild_side_effects for key in REQUIRED_POST_REBUILD_SIDE_EFFECT_KEYS)
        and all(post_rebuild_side_effects.get(key) is False for key in REQUIRED_POST_REBUILD_SIDE_EFFECT_KEYS)
    )
    future_command_contracts = _future_command_contracts(
        dataset_dir=dataset_dir,
        vedtak_id=vedtak_id,
        memory_cap=memory_cap,
        swap_cap=swap_cap,
    )
    checks: list[dict[str, Any]] = [
        _check(
            "explicit smart seq520 smoke vedtak id is provided",
            _explicit_vedtak_id_ok(vedtak_id),
            {"vedtak_id": vedtak_id},
        ),
        _check(
            "smart post-rebuild readiness report exists",
            post_rebuild_readiness_json.exists(),
            _artifact_meta(post_rebuild_readiness_json),
        ),
        _check(
            "smart post-rebuild readiness decision is ready",
            post_rebuild_readiness.get("decision") == POST_REBUILD_READY_DECISION,
            {"decision": post_rebuild_readiness.get("decision"), "expected": POST_REBUILD_READY_DECISION},
        ),
        _check(
            "smart post-rebuild refresh contract points at this smoke dataset",
            str(Path(str(post_rebuild_refresh_contract.get("smart_smoke_dataset_dir") or "")).expanduser().resolve())
            == str(dataset_dir),
            {
                "reported_smart_smoke_dataset_dir": post_rebuild_refresh_contract.get("smart_smoke_dataset_dir"),
                "actual_smart_smoke_dataset_dir": str(dataset_dir),
            },
        ),
        _check(
            "smart post-rebuild refresh contract starts no trainer replay iql shadow live",
            post_rebuild_refresh_contract.get("all_commands_avoid_training_replay_iql_shadow_live") is True
            and post_rebuild_side_effects_closed,
            {
                "all_commands_avoid_training_replay_iql_shadow_live": post_rebuild_refresh_contract.get(
                    "all_commands_avoid_training_replay_iql_shadow_live"
                ),
                "required_side_effect_keys": list(REQUIRED_POST_REBUILD_SIDE_EFFECT_KEYS),
                "side_effects_started": post_rebuild_side_effects,
            },
        ),
        _check("smart smoke dataset directory exists", dataset_dir.exists(), {"dataset_dir": str(dataset_dir)}),
        _check(
            "exact train val test split parquets exist",
            all(row["parquet_candidate_count"] == 1 for row in splits.values()),
            splits,
        ),
        _check(
            "exact train val test split manifests exist",
            all(row["manifest_candidate_count"] == 1 for row in splits.values()),
            splits,
        ),
        _check(
            "split manifest output_data_path exists and matches split parquet",
            all(row["output_data_exists"] and row["manifest_output_matches_split_parquet"] for row in splits.values()),
            splits,
        ),
        _check(
            "split hashes are present for train val test",
            all(_is_sha256(row["parquet_sha256"]) and _is_sha256(row["manifest_sha256"]) for row in splits.values()),
            splits,
        ),
        _check(
            "split manifests use smart seq520 split schema",
            all(row["schema_version"] == SPLIT_SCHEMA_VERSION for row in splits.values()),
            {split: row["schema_version"] for split, row in splits.items()},
        ),
        _check(
            "split manifests pin smart_seq520_candidate",
            all(row["manifest_variant"] == MANIFEST_VARIANT for row in splits.values()),
            {split: row["manifest_variant"] for split, row in splits.items()},
        ),
        _check(
            "split manifests pin expected seq snap width 520",
            all(row["expected_seq_snap_width"] == EXPECTED_SEQ_SNAP_WIDTH for row in splits.values()),
            {split: row["expected_seq_snap_width"] for split, row in splits.items()},
        ),
        _check(
            "split signal_bridge seq and snap dims are 520",
            all(row["seq_input_dim"] == EXPECTED_SEQ_SNAP_WIDTH and row["snap_input_dim"] == EXPECTED_SEQ_SNAP_WIDTH for row in splits.values()),
            splits,
        ),
        _check(
            "split signal field counts are 520",
            all(row["field_count"] == EXPECTED_SEQ_SNAP_WIDTH for row in splits.values()),
            {split: row["field_count"] for split, row in splits.items()},
        ),
        _check(
            "split parquet seq and snap samples have width 520",
            all(bool(row["shape_probe"].get("ok")) for row in splits.values()),
            {split: row["shape_probe"] for split, row in splits.items()},
        ),
        _check("side effects remain closed", all(value is False for value in SIDE_EFFECTS_STARTED.values()), SIDE_EFFECTS_STARTED),
        _check(
            "future train contract requires RAM cap and num_workers zero",
            future_command_contracts["smart_smoke_train"]["requires_ram_cap"] is True
            and future_command_contracts["smart_smoke_train"]["ram_cap_runner"] == RAM_CAP_RUNNER
            and future_command_contracts["smart_smoke_train"]["num_workers"] == 0
            and "--num-workers" in future_command_contracts["smart_smoke_train"]["inner_train_argv_template"],
            future_command_contracts["smart_smoke_train"],
        ),
    ]
    failures = [check for check in checks if not check["ok"]]
    ready = not failures
    decision = (
        "READY_FOR_SMART_SEQ520_SMOKE_MANIFEST_REVIEW"
        if ready
        else "BLOCKED_SMART_SEQ520_SMOKE_MANIFEST_READINESS"
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = out_dir / "SMOKE_DATASET_MANIFEST.json"
    json_path = out_dir / f"ENTRY_SMART_SEQ520_SMOKE_MANIFEST_READINESS_{stamp}.json"
    md_path = out_dir / f"ENTRY_SMART_SEQ520_SMOKE_MANIFEST_READINESS_{stamp}.md"
    smoke_manifest = (
        _build_smoke_manifest(
            dataset_dir=dataset_dir,
            vedtak_id=vedtak_id,
            splits=splits,
            future_command_contracts=future_command_contracts,
        )
        if ready
        else {}
    )
    stale_manifest_quarantined_path: str | None = None
    if ready:
        manifest_path.write_text(
            json.dumps(smoke_manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
    elif manifest_path.exists():
        quarantine_path = out_dir / f"SMOKE_DATASET_MANIFEST_STALE_BLOCKED_{stamp}.json"
        manifest_path.replace(quarantine_path)
        stale_manifest_quarantined_path = str(quarantine_path)

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "report_only": True,
        "manifest_variant": MANIFEST_VARIANT,
        "expected_seq_snap_width": EXPECTED_SEQ_SNAP_WIDTH,
        "explicit_vedtak_id": vedtak_id if _explicit_vedtak_id_ok(vedtak_id) else None,
        "smart_smoke_dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "manifest_written": bool(ready),
        "manifest_sha256": _sha256_file(manifest_path) if ready else None,
        "stale_manifest_quarantined_path": stale_manifest_quarantined_path,
        "post_rebuild_readiness": _artifact_meta(post_rebuild_readiness_json),
        "split_artifacts": splits,
        "future_command_contracts": future_command_contracts,
        "checks": checks,
        "failures": failures,
        "blockers": [row["name"] for row in failures],
        "training_allowed": False,
        "replay_allowed": False,
        "iql_allowed": False,
        "shadow_live_allowed": False,
        "control_surface_mutated": False,
        "mutations_outside_report_dir": False,
        "side_effects_started": dict(SIDE_EFFECTS_STARTED),
        "next_required_gate": (
            "review the materialized smoke manifest and integrate a separate vedtak-gated control path"
            if ready
            else "repair missing or invalid smart_seq520 smoke split artifacts before any train/replay/IQL/shadow/live path"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    (out_dir / "ENTRY_SMART_SEQ520_SMOKE_MANIFEST_READINESS_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_SMART_SEQ520_SMOKE_MANIFEST_READINESS_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    if failures and bool(args.fail_on_not_ready):
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smart-smoke-dataset-dir", default=str(DEFAULT_SMART_SMOKE_DATASET_DIR))
    ap.add_argument("--post-rebuild-readiness-json", default=str(DEFAULT_POST_REBUILD_READINESS_JSON))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--vedtak-id", "--vedtak", dest="vedtak_id", required=True)
    ap.add_argument("--memory-cap", default=DEFAULT_MEMORY_CAP)
    ap.add_argument("--swap-cap", default=DEFAULT_SWAP_CAP)
    ap.add_argument("--sample-rows", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
