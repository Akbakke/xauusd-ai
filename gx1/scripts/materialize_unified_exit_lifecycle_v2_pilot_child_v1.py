#!/usr/bin/env python3
"""Build compact lifecycle v2 from a verified pilot Entry child view."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.contracts.unified_exit_lifecycle_v2 import (
    UNIFIED_EXIT_CHUNK_ROWS,
    unified_exit_lifecycle_v2_contract,
)
from gx1.scripts.materialize_unified_exit_lifecycle_v2 import (
    COMPACT_LIFECYCLE_SCHEMA_VERSION,
    COMPACT_ROOT_SCHEMA_VERSION,
    _EconomicAuthorityBlocked,
    _canonical_sha256,
    _compact_pointer_stream_sha256,
    _load_terminal_counts,
    _sha256_file,
    _validate_m1_source,
    build_compact_split,
    pair_schedule_coverage,
)
from gx1.scripts.prepare_unified_exit_lifecycle_v2_pilot_v1 import (
    PILOT_EPOCHS,
    TRAIN_END,
    VAL_END,
)
from gx1.scripts.validate_lifecycle_v2_pilot_child_view_v1 import (
    validate_pilot_child_view,
)


PRODUCER_SCHEMA_VERSION = "gx1_unified_exit_pilot_child_compact_producer_v1"


def _producer_source_identity() -> dict[str, Any]:
    source = Path(__file__).resolve()
    return {
        "schema_version": PRODUCER_SCHEMA_VERSION,
        "module": "gx1.scripts.materialize_unified_exit_lifecycle_v2_pilot_child_v1",
        "source_sha256": _sha256_file(source),
    }


def _build_child_split(
    *,
    split: str,
    child_witness: dict[str, Any],
    economic_authority_path: Path,
    dataset_run_id: str,
    split_end: str,
    m1_times: pd.DatetimeIndex,
    m1_source_sha256: str,
    m1_source_manifest_sha256: str,
    planned_epochs: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    child = child_witness["splits"][split]
    entry_path = Path(child["parquet_path"])
    entry_times = pd.read_parquet(entry_path, columns=["time"])["time"]
    if len(entry_times) != child["rows"]:
        raise RuntimeError("PILOT_COMPACT_ENTRY_ROW_COUNT_CHANGED")
    mapping, authority, authority_sha = _load_terminal_counts(
        economic_authority_path,
        split=split,
        dataset_run_id=dataset_run_id,
        entry_rows=len(entry_times),
    )
    entry_binding_sha = _canonical_sha256(
        {
            "child_admission_witness_sha256": child_witness["witness_sha256"],
            "split": split,
            "child_split": child,
        }
    )
    compact = build_compact_split(
        entry_times=entry_times,
        m1_times=m1_times,
        split=split,
        split_end=split_end,
        terminal_state_count_by_entry_side=mapping,
        m1_source_sha256=m1_source_sha256,
        gap_classification_source_sha256=m1_source_manifest_sha256,
        entry_binding_sha256=entry_binding_sha,
    )
    lineage = _canonical_sha256(
        {
            "dataset_run_id": dataset_run_id,
            "split": split,
            "entry_sha256": child["parquet_sha256"],
            "entry_manifest_sha256": child["manifest_file_sha256"],
            "entry_adoption_witness_sha256": entry_binding_sha,
            "m1_source_sha256": m1_source_sha256,
            "economic_authority_sha256": authority_sha,
        }
    )
    coverage = pair_schedule_coverage(
        chunk_count_by_entry={
            int(row.entry_row_index): int(row.pair_chunk_count)
            for row in compact.itertuples(index=False)
        },
        planned_epochs=planned_epochs,
        lineage_sha256=lineage,
        split=split,
        claim_full_coverage=False,
    )
    manifest = {
        **unified_exit_lifecycle_v2_contract(),
        "compact_schema_version": COMPACT_LIFECYCLE_SCHEMA_VERSION,
        "decision": "PASS",
        "dataset_run_id": dataset_run_id,
        "split": split,
        "split_end_utc": pd.Timestamp(split_end).isoformat(),
        "test_accessed": False,
        "entry_parquet_path": child["parquet_path"],
        "entry_parquet_sha256": child["parquet_sha256"],
        "entry_manifest_path": child["manifest_path"],
        "entry_manifest_sha256": child["manifest_file_sha256"],
        "entry_adoption_witness": {
            "schema_version": "gx1_unified_exit_pilot_child_adoption_witness_v1",
            "split": split,
            "child_admission_witness_sha256": child_witness["witness_sha256"],
            "child_split": child,
            "test_accessed": False,
            "witness_sha256": entry_binding_sha,
        },
        "entry_binding_sha256": entry_binding_sha,
        "gap_classification_source_sha256": m1_source_manifest_sha256,
        "economic_authority_path": str(economic_authority_path),
        "economic_authority_sha256": authority_sha,
        "economic_authority": authority,
        "m1_source_sha256": m1_source_sha256,
        "compact_rows": len(compact),
        "compact_pointer_stream_sha256": _compact_pointer_stream_sha256(compact),
        "successor_pointer_binding": "per_entry_side_chunk_pointer_stream_sha256",
        "target_q_stored": False,
        "producer_source": _producer_source_identity(),
        "schedule_lineage_sha256": lineage,
        "schedule_coverage": coverage,
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    return compact, manifest


def materialize_pilot_child_lifecycle_v2(
    *,
    source_recipe_path: Path,
    source_recipe_sha256: str,
    pilot_root: Path,
    child_root_path: Path,
    output_dir: Path,
    m1_source_path: Path,
    m1_source_manifest_path: Path,
    train_economic_authority_path: Path | None,
    val_economic_authority_path: Path | None,
    planned_epochs: int = PILOT_EPOCHS,
    publish: bool,
) -> dict[str, Any]:
    """Validate or atomically publish the child-view compact lifecycle."""

    if type(publish) is not bool or planned_epochs != PILOT_EPOCHS:
        raise RuntimeError("PILOT_COMPACT_INVOCATION_INVALID")
    root = pilot_root.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    if output != root / "LIFECYCLE_V2":
        raise RuntimeError("PILOT_COMPACT_OUTPUT_PATH_INVALID")
    child_witness = validate_pilot_child_view(
        source_recipe_path=source_recipe_path,
        source_recipe_sha256=source_recipe_sha256,
        pilot_root=root,
        child_root_path=child_root_path,
    )
    m1_path = m1_source_path.expanduser().resolve()
    m1_manifest_path = m1_source_manifest_path.expanduser().resolve()
    m1_times, _manifest, m1_manifest_sha, m1_sha = _validate_m1_source(
        m1_path, m1_manifest_path
    )
    if child_witness["m1_source_binding"] != {
        "parquet_path": str(m1_path),
        "parquet_sha256": m1_sha,
        "manifest_path": str(m1_manifest_path),
        "manifest_sha256": m1_manifest_sha,
    }:
        raise RuntimeError("PILOT_COMPACT_M1_BINDING_INVALID")
    authorities = {
        "train": train_economic_authority_path,
        "val": val_economic_authority_path,
    }
    missing = [split for split, path in authorities.items() if path is None]
    if missing:
        if publish:
            raise RuntimeError("PILOT_COMPACT_PUBLISH_BLOCKED")
        return {
            "mode": "validate_no_publish",
            "decision": "BLOCKED",
            "published": False,
            "output_dir": str(output),
            "blockers": [f"missing_{split}_economic_authority" for split in missing],
            "child_admission_witness_sha256": child_witness["witness_sha256"],
            "m1_source_sha256": m1_sha,
            "test_accessed": False,
        }
    built: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    try:
        for split, split_end in (("train", TRAIN_END), ("val", VAL_END)):
            authority = Path(authorities[split]).expanduser().resolve()  # type: ignore[arg-type]
            if authority.is_symlink() or not authority.is_file():
                raise RuntimeError("PILOT_COMPACT_AUTHORITY_FILE_INVALID")
            built[split] = _build_child_split(
                split=split,
                child_witness=child_witness,
                economic_authority_path=authority,
                dataset_run_id=child_witness["child_dataset_run_id"],
                split_end=split_end,
                m1_times=m1_times,
                m1_source_sha256=m1_sha,
                m1_source_manifest_sha256=m1_manifest_sha,
                planned_epochs=planned_epochs,
            )
    except _EconomicAuthorityBlocked as exc:
        if publish:
            raise RuntimeError("PILOT_COMPACT_PUBLISH_BLOCKED") from exc
        return {
            "mode": "validate_no_publish",
            "decision": "BLOCKED",
            "published": False,
            "output_dir": str(output),
            "blockers": [str(exc)],
            "child_admission_witness_sha256": child_witness["witness_sha256"],
            "test_accessed": False,
        }
    root_manifest = {
        "schema_version": COMPACT_ROOT_SCHEMA_VERSION,
        "decision": "PASS",
        "dataset_run_id": child_witness["child_dataset_run_id"],
        "allowed_splits": ["train", "val"],
        "test_accessed": False,
        "child_admission_witness_sha256": child_witness["witness_sha256"],
        "m1_source_path": str(m1_path),
        "m1_source_sha256": m1_sha,
        "m1_source_manifest_path": str(m1_manifest_path),
        "m1_source_manifest_sha256": m1_manifest_sha,
        "chunk_state_capacity": UNIFIED_EXIT_CHUNK_ROWS,
        "compact_storage_complexity": "one_row_per_entry_pair_O_entry",
        "successor_pointer_binding": "hashed_static_pointer_stream_per_side",
        "target_q_stored": False,
        "producer_source": _producer_source_identity(),
        "split_manifests": {
            split: {
                "manifest_sha256": manifest["manifest_sha256"],
                "compact_rows": len(frame),
                "schedule_coverage_sha256": manifest["schedule_coverage"][
                    "contract_sha256"
                ],
            }
            for split, (frame, manifest) in built.items()
        },
        "publish_requested": publish,
    }
    root_manifest["manifest_sha256"] = _canonical_sha256(root_manifest)
    if not publish:
        return {
            "mode": "validate_no_publish",
            "decision": "PASS",
            "published": False,
            "output_dir": str(output),
            "root_manifest": root_manifest,
            "splits": {
                split: {
                    "compact_rows": len(frame),
                    "compact_pointer_stream_sha256": manifest[
                        "compact_pointer_stream_sha256"
                    ],
                    "schedule_coverage": manifest["schedule_coverage"],
                }
                for split, (frame, manifest) in built.items()
            },
        }
    if output.exists() or output.is_symlink():
        raise RuntimeError("PILOT_COMPACT_OUTPUT_EXISTS")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        for split, (frame, manifest) in built.items():
            parquet = staging / f"{split}_unified_exit_lifecycle_v2.parquet"
            frame.to_parquet(parquet, index=False)
            manifest = {
                **manifest,
                "compact_parquet": parquet.name,
                "compact_parquet_sha256": _sha256_file(parquet),
            }
            manifest["manifest_sha256"] = _canonical_sha256(
                {key: value for key, value in manifest.items() if key != "manifest_sha256"}
            )
            manifest_path = staging / f"{split}_unified_exit_lifecycle_v2.manifest.json"
            manifest_path.write_bytes(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode()
                + b"\n"
            )
            root_manifest["split_manifests"][split].update(
                {
                    "manifest": manifest_path.name,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "compact_parquet": parquet.name,
                    "compact_parquet_sha256": manifest["compact_parquet_sha256"],
                }
            )
        root_manifest["manifest_sha256"] = _canonical_sha256(
            {
                key: value
                for key, value in root_manifest.items()
                if key != "manifest_sha256"
            }
        )
        (staging / "UNIFIED_EXIT_LIFECYCLE_V2_MANIFEST.json").write_bytes(
            json.dumps(root_manifest, indent=2, sort_keys=True, allow_nan=False).encode()
            + b"\n"
        )
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "mode": "publish",
        "decision": "PASS",
        "published": True,
        "output_dir": str(output),
        "root_manifest_sha256": root_manifest["manifest_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-recipe", required=True, type=Path)
    parser.add_argument("--source-recipe-sha256", required=True)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--child-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--m1-source-parquet", required=True, type=Path)
    parser.add_argument("--m1-source-manifest", required=True, type=Path)
    parser.add_argument("--train-economic-authority", type=Path)
    parser.add_argument("--val-economic-authority", type=Path)
    parser.add_argument("--planned-epochs", type=int, default=PILOT_EPOCHS)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    result = materialize_pilot_child_lifecycle_v2(
        source_recipe_path=args.source_recipe,
        source_recipe_sha256=args.source_recipe_sha256,
        pilot_root=args.pilot_root,
        child_root_path=args.child_root,
        output_dir=args.output_dir,
        m1_source_path=args.m1_source_parquet,
        m1_source_manifest_path=args.m1_source_manifest,
        train_economic_authority_path=args.train_economic_authority,
        val_economic_authority_path=args.val_economic_authority,
        planned_epochs=args.planned_epochs,
        publish=args.publish,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
