#!/usr/bin/env python3
"""Publish final TRAIN/VAL normalization and first-state pilot bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from gx1.contracts.unified_exit_market_closure_authority_v1 import (
    m1_clock_sha256,
    require_market_closure_authority,
)
from gx1.contracts.unified_exit_pilot_final_bindings_v1 import (
    build_composite_normalization_binding,
    build_split_sequence_binding,
    require_composite_normalization_binding,
)
from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    build_first_state_entry_bridge_witness,
    build_sampler_benchmark_candidate_set,
    canonical_sha256,
    require_lifetime_summary_normalization,
)
from gx1.contracts.unified_exit_lifetime_summary_v1 import (
    lifetime_summary_registry,
)


RECIPE_SCHEMA_VERSION = "gx1_unified_exit_pilot_final_bindings_recipe_v1"
BUNDLE_SCHEMA_VERSION = "gx1_unified_exit_pilot_final_bindings_bundle_v1"


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("UNIFIED_EXIT_FINAL_JSON_INVALID")
    return value


def _binding(value: Mapping[str, Any], label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise RuntimeError(f"UNIFIED_EXIT_FINAL_{label}_BINDING_INVALID")
    path = Path(str(value["path"]))
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
        or _file_sha(path) != value["sha256"]
    ):
        raise RuntimeError(f"UNIFIED_EXIT_FINAL_{label}_BINDING_INVALID")
    return path


def _verify_canonical(value: Mapping[str, Any], hash_key: str, label: str) -> None:
    unsigned = dict(value)
    claimed = unsigned.pop(hash_key, None)
    if claimed != canonical_sha256(unsigned):
        raise RuntimeError(f"UNIFIED_EXIT_FINAL_{label}_HASH_INVALID")


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def build_bundle(recipe_path: Path) -> dict[str, Any]:
    recipe = _json(recipe_path)
    if recipe.get("schema_version") != RECIPE_SCHEMA_VERSION:
        raise RuntimeError("UNIFIED_EXIT_FINAL_RECIPE_INVALID")
    _verify_canonical(recipe, "recipe_sha256", "RECIPE")
    if recipe.get("test_accessed") is not False:
        raise RuntimeError("UNIFIED_EXIT_FINAL_RECIPE_INVALID")

    admission_path = _binding(recipe["child_admission"], "ADMISSION")
    view_path = _binding(recipe["normalization_view"], "NORMALIZATION_VIEW")
    base_path = _binding(recipe["base_normalization"], "BASE_NORMALIZATION")
    _binding(recipe["state_view_source"], "STATE_SOURCE")
    admission = _json(admission_path)
    view = _json(view_path)
    base = _json(base_path)
    _verify_canonical(admission, "witness_sha256", "ADMISSION")
    _verify_canonical(view, "contract_sha256", "NORMALIZATION_VIEW")
    if (
        admission.get("decision") != "PASS"
        or view.get("decision") != "PASS"
        or view.get("child_admission", {}).get("file_sha256")
        != recipe["child_admission"]["sha256"]
        or base.get("decision") != "PASS"
        or base.get("val_fit_rows") != 0
        or base.get("test_fit_rows") != 0
        or base.get("test_accessed") is not False
    ):
        raise RuntimeError("UNIFIED_EXIT_FINAL_PARENT_INVALID")

    summaries: dict[str, dict[str, Any]] = {}
    split_data: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        spec = recipe["splits"][split]
        summary_path = _binding(spec["summary_manifest"], f"{split}_SUMMARY")
        counts_path = _binding(spec["successor_counts"], f"{split}_COUNTS")
        m1_path = _binding(spec["m1_source"], f"{split}_M1")
        m1_manifest_path = _binding(spec["m1_manifest"], f"{split}_M1_MANIFEST")
        closure_path = _binding(spec["closure_authority"], f"{split}_CLOSURE")
        summary = _json(summary_path)
        _verify_canonical(summary, "manifest_sha256", f"{split}_SUMMARY")
        child = admission["splits"][split]
        child_path = Path(child["parquet_path"])
        child_manifest_path = Path(child["manifest_path"])
        if (
            summary.get("decision") != "PASS"
            or summary.get("split") != split
            or summary.get("entry_pair_population") != child["rows"]
            or summary.get("child_admission_sha256")
            != recipe["child_admission"]["sha256"]
            or summary.get("child_parquet_sha256") != child["parquet_sha256"]
            or _file_sha(child_path) != child["parquet_sha256"]
            or _file_sha(child_manifest_path) != child["manifest_file_sha256"]
        ):
            raise RuntimeError(f"UNIFIED_EXIT_FINAL_{split.upper()}_SUMMARY_INVALID")
        counts = np.load(counts_path, allow_pickle=False)
        if (
            counts.dtype != np.dtype("<i8")
            or counts.shape != (child["rows"],)
            or hashlib.sha256(counts.tobytes()).hexdigest()
            != summary["successor_counts_sha256"]
        ):
            raise RuntimeError(f"UNIFIED_EXIT_FINAL_{split.upper()}_COUNTS_INVALID")
        m1_manifest = _json(m1_manifest_path)
        m1_table = pq.read_table(m1_path, columns=["time", "bid_open", "ask_open"])
        m1_times = m1_table["time"].to_pandas()
        closure = _json(closure_path)
        checked_closure = require_market_closure_authority(
            closure,
            expected_m1_source_sha256=spec["m1_source"]["sha256"],
            expected_m1_clock_sha256=m1_clock_sha256(m1_times),
        )
        if (
            m1_manifest.get("parquet_sha256") != spec["m1_source"]["sha256"]
            or m1_manifest.get("clock_sha256") != m1_clock_sha256(m1_times)
            or summary.get("m1_source_sha256") != spec["m1_source"]["sha256"]
            or summary.get("m1_manifest_sha256") != spec["m1_manifest"]["sha256"]
            or summary.get("closure_authority_sha256")
            != checked_closure["artifact_sha256"]
            or summary.get("closure_authority_file_sha256")
            != spec["closure_authority"]["sha256"]
        ):
            raise RuntimeError(f"UNIFIED_EXIT_FINAL_{split.upper()}_SOURCE_INVALID")
        entry_times = pq.read_table(child_path, columns=["time"])["time"].to_pandas()
        sequence = build_split_sequence_binding(
            split=split,
            entry_times=entry_times,
            m1_times=m1_times,
            successor_transition_counts=[int(item) for item in counts],
            child_admission_file_sha256=recipe["child_admission"]["sha256"],
            child_admission_witness_sha256=admission["witness_sha256"],
            child_parquet_sha256=child["parquet_sha256"],
            child_manifest_file_sha256=child["manifest_file_sha256"],
            child_manifest_contract_sha256=child["manifest_contract_sha256"],
            m1_source_sha256=spec["m1_source"]["sha256"],
            m1_manifest_file_sha256=spec["m1_manifest"]["sha256"],
            closure_authority_file_sha256=spec["closure_authority"]["sha256"],
            closure_authority_sha256=checked_closure["artifact_sha256"],
        )
        if sequence["entry_clock_sha256"] != child["clock_sha256"]:
            raise RuntimeError(f"UNIFIED_EXIT_FINAL_{split.upper()}_CLOCK_INVALID")
        summaries[split] = summary
        split_data[split] = {
            "child": child,
            "m1_times": m1_times,
            "m1_bid_open": m1_table["bid_open"].to_numpy(zero_copy_only=False),
            "m1_ask_open": m1_table["ask_open"].to_numpy(zero_copy_only=False),
            "sequence": sequence,
            "closure": checked_closure,
            "spec": spec,
        }

    train_summary = summaries["train"]
    train_normalization = require_lifetime_summary_normalization(
        train_summary["lifetime_summary_normalization"],
        expected_sample_authority_sha256=train_summary[
            "summary_sample_authority"
        ]["authority_sha256"],
    )
    if summaries["val"].get("lifetime_summary_normalization") is not None:
        raise RuntimeError("UNIFIED_EXIT_FINAL_VAL_FIT_FORBIDDEN")
    composite = build_composite_normalization_binding(
        base_artifact=base,
        base_path=str(base_path),
        base_file_sha256=recipe["base_normalization"]["sha256"],
        summary_normalization=train_normalization,
        summary_manifest_path=recipe["splits"]["train"]["summary_manifest"]["path"],
        summary_manifest_file_sha256=recipe["splits"]["train"]["summary_manifest"]["sha256"],
        summary_manifest_sha256=train_summary["manifest_sha256"],
    )
    require_composite_normalization_binding(composite)
    candidates = build_sampler_benchmark_candidate_set(
        source_lineage_sha256=train_summary["source_lineage_sha256"],
        entry_pair_population=admission["splits"]["train"]["rows"],
    )
    bridges: dict[str, Any] = {}
    for split in ("train", "val"):
        item = split_data[split]
        child = item["child"]
        entry_times = pq.read_table(child["parquet_path"], columns=["time"])[
            "time"
        ].to_pandas()
        bridges[split] = build_first_state_entry_bridge_witness(
            split=split,
            entry_times=entry_times,
            m1_times=item["m1_times"],
            child_admission_sha256=recipe["child_admission"]["sha256"],
            child_parquet_sha256=child["parquet_sha256"],
            entry_sequence_audit_sha256=item["sequence"]["binding_sha256"],
            m1_source_sha256=item["spec"]["m1_source"]["sha256"],
            closure_authority_sha256=item["closure"]["artifact_sha256"],
            state_view_source_sha256=recipe["state_view_source"]["sha256"],
            lifetime_summary_registry_sha256=lifetime_summary_registry()[
                "registry_sha256"
            ],
            train_normalization_sha256=composite[
                "composite_normalization_sha256"
            ],
            m1_bid_open=item["m1_bid_open"],
            m1_ask_open=item["m1_ask_open"],
        )
    value = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "decision": "BLOCKED_PENDING_TRAIN_ONLY_SAMPLER_BENCHMARK",
        "recipe_path": str(recipe_path),
        "recipe_file_sha256": _file_sha(recipe_path),
        "recipe_sha256": recipe["recipe_sha256"],
        "child_admission": recipe["child_admission"],
        "normalization_view": recipe["normalization_view"],
        "state_view_source": recipe["state_view_source"],
        "composite_normalization": composite,
        "sampler_benchmark_candidates": candidates,
        "split_sequence_bindings": {
            split: split_data[split]["sequence"] for split in ("train", "val")
        },
        "summary_fit_manifests": {
            split: {
                **recipe["splits"][split]["summary_manifest"],
                "manifest_sha256": summaries[split]["manifest_sha256"],
                "successor_counts_sha256": summaries[split][
                    "successor_counts_sha256"
                ],
                "successor_transition_total": summaries[split][
                    "successor_transition_total"
                ],
            }
            for split in ("train", "val")
        },
        "first_state_entry_bridges": bridges,
        "val_mode": "apply_frozen_train_transforms_only",
        "val_fit_rows": 0,
        "test_fit_rows": 0,
        "test_accessed": False,
    }
    value["bundle_sha256"] = canonical_sha256(value)
    return value


def materialize(*, recipe_path: Path, output_dir: Path, publish: bool) -> dict[str, Any]:
    bundle = build_bundle(recipe_path)
    if not publish:
        return {"mode": "validate_no_publish", "published": False, "bundle": bundle}
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_FINAL_OUTPUT_EXISTS")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent)
    )
    try:
        _write(staging / "FINAL_BINDINGS_BUNDLE.json", bundle)
        _write(staging / "COMPOSITE_NORMALIZATION.json", bundle["composite_normalization"])
        _write(staging / "SAMPLER_BENCHMARK_CANDIDATES.json", bundle["sampler_benchmark_candidates"])
        for split in ("train", "val"):
            _write(
                staging / f"FIRST_STATE_ENTRY_BRIDGE_{split.upper()}.json",
                bundle["first_state_entry_bridges"][split],
            )
            _write(
                staging / f"SPLIT_SEQUENCE_BINDING_{split.upper()}.json",
                bundle["split_sequence_bindings"][split],
            )
        os.rename(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "mode": "publish",
        "published": True,
        "output_dir": str(output_dir),
        "bundle_sha256": bundle["bundle_sha256"],
        "composite_normalization_sha256": bundle["composite_normalization"][
            "composite_normalization_sha256"
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    result = materialize(
        recipe_path=args.recipe.resolve(),
        output_dir=args.output_dir.resolve(),
        publish=args.publish,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
