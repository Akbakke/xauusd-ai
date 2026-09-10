#!/usr/bin/env python3
"""Materialize TRAIN-only lifetime normalization and Entry bridge evidence.

The producer consumes canonical, separately verified state-summary values.  It
never derives economics or samples from price outcomes.  Epoch sampler budgets
remain benchmark candidates until measured throughput and memory select one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from gx1.contracts.unified_exit_lifetime_summary_v1 import lifetime_summary_registry
from gx1.contracts.unified_exit_market_closure_authority_v1 import (
    m1_clock_sha256,
    require_market_closure_authority,
)
from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    build_first_state_entry_bridge_witness,
    build_physical_summary_sample_authority,
    build_sampler_benchmark_candidate_set,
    canonical_sha256,
    fit_lifetime_summary_normalization,
)


BUNDLE_SCHEMA_VERSION = "gx1_unified_exit_pilot_train_normalization_bundle_v1"
RECIPE_SCHEMA_VERSION = "gx1_unified_exit_pilot_train_normalization_recipe_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("PILOT_TRAIN_NORMALIZATION_JSON_INVALID")
    return value


def _exact_file_binding(raw: Mapping[str, Any], label: str) -> Path:
    if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
        raise RuntimeError(f"PILOT_TRAIN_NORMALIZATION_{label}_BINDING_INVALID")
    path = Path(str(raw["path"]))
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
        or _sha256_file(path) != raw["sha256"]
    ):
        raise RuntimeError(f"PILOT_TRAIN_NORMALIZATION_{label}_BINDING_INVALID")
    return path


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def build_bundle(*, recipe_path: Path) -> dict[str, Any]:
    recipe_file_sha = _sha256_file(recipe_path)
    recipe = _read_json(recipe_path)
    claimed = recipe.get("recipe_sha256")
    unsigned = dict(recipe)
    unsigned.pop("recipe_sha256", None)
    if (
        recipe.get("schema_version") != RECIPE_SCHEMA_VERSION
        or claimed != canonical_sha256(unsigned)
        or recipe.get("test_accessed") is not False
    ):
        raise RuntimeError("PILOT_TRAIN_NORMALIZATION_RECIPE_INVALID")

    admission_path = _exact_file_binding(recipe["child_admission"], "ADMISSION")
    input_view_path = _exact_file_binding(recipe["normalization_inputs"], "INPUT_VIEW")
    closure_path = _exact_file_binding(recipe["closure_authority"], "CLOSURE")
    base_path = _exact_file_binding(recipe["base_feature_normalization"], "BASE")
    _exact_file_binding(recipe["state_view_source"], "STATE_SOURCE")
    m1_path = _exact_file_binding(recipe["m1_source"], "M1_SOURCE")
    counts_path = _exact_file_binding(recipe["train_successor_counts"], "COUNTS")
    summaries_path = _exact_file_binding(recipe["train_summary_values"], "SUMMARIES")

    admission = _read_json(admission_path)
    input_view = _read_json(input_view_path)
    closure = _read_json(closure_path)
    base = _read_json(base_path)
    if admission.get("decision") != "PASS" or input_view.get("decision") != "PASS":
        raise RuntimeError("PILOT_TRAIN_NORMALIZATION_PARENT_NOT_ADMITTED")
    unsigned_admission = dict(admission)
    admission_witness = unsigned_admission.pop("witness_sha256", None)
    unsigned_view = dict(input_view)
    view_contract = unsigned_view.pop("contract_sha256", None)
    if (
        admission_witness != canonical_sha256(unsigned_admission)
        or view_contract != canonical_sha256(unsigned_view)
    ):
        raise RuntimeError("PILOT_TRAIN_NORMALIZATION_PARENT_CONTRACT_INVALID")
    if input_view.get("child_admission", {}).get("file_sha256") != recipe["child_admission"]["sha256"]:
        raise RuntimeError("PILOT_TRAIN_NORMALIZATION_VIEW_SPLIT_BRAIN")
    registry = lifetime_summary_registry()
    if input_view.get("lifetime_summary_registry", {}).get("registry_sha256") != registry["registry_sha256"]:
        raise RuntimeError("PILOT_TRAIN_NORMALIZATION_REGISTRY_SPLIT_BRAIN")

    counts = np.load(counts_path, allow_pickle=False)
    if counts.dtype != np.dtype("<i8") or counts.shape != (admission["splits"]["train"]["rows"],):
        raise RuntimeError("PILOT_TRAIN_NORMALIZATION_COUNTS_INVALID")
    lineage = canonical_sha256(
        {
            "child_admission_sha256": recipe["child_admission"]["sha256"],
            "normalization_inputs_sha256": recipe["normalization_inputs"]["sha256"],
            "closure_authority_sha256": recipe["closure_authority"]["sha256"],
            "m1_source_sha256": recipe["m1_source"]["sha256"],
        }
    )
    sample_authority = build_physical_summary_sample_authority(
        successor_transition_count_by_entry=[int(value) for value in counts],
        source_lineage_sha256=lineage,
    )
    summaries = np.load(summaries_path, allow_pickle=False)
    summary_normalization = fit_lifetime_summary_normalization(
        values=summaries,
        sample_authority=sample_authority,
    )
    candidates = build_sampler_benchmark_candidate_set(
        source_lineage_sha256=lineage,
        entry_pair_population=int(admission["splits"]["train"]["rows"]),
    )

    m1_table = pq.read_table(m1_path, columns=["time", "bid_open", "ask_open"])
    m1_times = m1_table["time"].to_pandas()
    m1_bid_open = m1_table["bid_open"].to_numpy(zero_copy_only=False)
    m1_ask_open = m1_table["ask_open"].to_numpy(zero_copy_only=False)
    require_market_closure_authority(
        closure,
        expected_m1_source_sha256=recipe["m1_source"]["sha256"],
        expected_m1_clock_sha256=m1_clock_sha256(m1_times),
    )
    bridges: dict[str, Any] = {}
    for split in ("train", "val"):
        child = admission["splits"][split]
        child_path = Path(child["parquet_path"])
        if _sha256_file(child_path) != child["parquet_sha256"]:
            raise RuntimeError("PILOT_TRAIN_NORMALIZATION_CHILD_BYTES_INVALID")
        entry_times = pq.read_table(child_path, columns=["time"])["time"].to_pandas()
        bridges[split] = build_first_state_entry_bridge_witness(
            split=split,
            entry_times=entry_times,
            m1_times=m1_times,
            child_admission_sha256=recipe["child_admission"]["sha256"],
            child_parquet_sha256=child["parquet_sha256"],
            entry_sequence_audit_sha256=input_view["child_sequence_reconstruction_audit"]["file_sha256"],
            m1_source_sha256=recipe["m1_source"]["sha256"],
            closure_authority_sha256=recipe["closure_authority"]["sha256"],
            state_view_source_sha256=recipe["state_view_source"]["sha256"],
            lifetime_summary_registry_sha256=registry["registry_sha256"],
            train_normalization_sha256=summary_normalization["normalization_sha256"],
            m1_bid_open=m1_bid_open,
            m1_ask_open=m1_ask_open,
        )

    value = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "decision": "BLOCKED_PENDING_TRAIN_ONLY_SAMPLER_BENCHMARK",
        "recipe_path": str(recipe_path),
        "recipe_file_sha256": recipe_file_sha,
        "recipe_sha256": claimed,
        "source_lineage_sha256": lineage,
        "child_admission": recipe["child_admission"],
        "normalization_inputs": recipe["normalization_inputs"],
        "closure_authority": {**recipe["closure_authority"], "artifact_sha256": closure.get("artifact_sha256")},
        "base_feature_normalization": {**recipe["base_feature_normalization"], "contract_sha256": base.get("contract_sha256")},
        "state_view_source": recipe["state_view_source"],
        "m1_source": recipe["m1_source"],
        "summary_sample_authority": sample_authority,
        "lifetime_summary_normalization": summary_normalization,
        "sampler_benchmark_candidates": candidates,
        "first_state_entry_bridge": bridges,
        "train_fit_rows": {
            "base_physical_population": input_view["train_normalization_population_witness"]["contract_sha256"],
            "lifetime_summary": sample_authority["fit_row_count"],
        },
        "val_mode": "apply_frozen_train_transforms_only",
        "val_fit_rows": 0,
        "test_fit_rows": 0,
        "test_accessed": False,
    }
    value["bundle_sha256"] = canonical_sha256(value)
    return value


def materialize(*, recipe_path: Path, output_dir: Path, publish: bool) -> dict[str, Any]:
    bundle = build_bundle(recipe_path=recipe_path)
    if not publish:
        return {"mode": "validate_no_publish", "published": False, "bundle": bundle}
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError("PILOT_TRAIN_NORMALIZATION_OUTPUT_EXISTS")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent))
    try:
        (staging / "TRAIN_NORMALIZATION_BUNDLE.json").write_bytes(_json_bytes(bundle))
        for split, witness in bundle["first_state_entry_bridge"].items():
            (staging / f"FIRST_STATE_ENTRY_BRIDGE_{split.upper()}.json").write_bytes(_json_bytes(witness))
        for path in staging.iterdir():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        os.rename(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"mode": "publish", "published": True, "output_dir": str(output_dir), "bundle_sha256": bundle["bundle_sha256"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(materialize(recipe_path=args.recipe.resolve(), output_dir=args.output_dir.resolve(), publish=args.publish), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
