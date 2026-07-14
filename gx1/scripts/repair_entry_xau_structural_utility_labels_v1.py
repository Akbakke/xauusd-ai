#!/usr/bin/env python3
"""Repair XAU Entry V10 structural side-utility labels in an existing dataset.

This is a dataset target materializer, not a runtime trading rule. It keeps
feature tensors unchanged and rewrites only side-specific utility, bad-path and
MAE targets so expected-utility training no longer learns that SHORT is valid in
rising/support trap pockets or LONG is valid in falling/resistance trap pockets.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    XAU_STRUCTURAL_UTILITY_REPAIR_MAE_MARGIN_BPS,
    XAU_STRUCTURAL_UTILITY_REPAIR_MARGIN_BPS,
    _apply_structural_side_repair,
    _apply_structural_utility_repair,
    _position_size_target_from_repaired_path,
    _repaired_scalar_bad_path_target,
    hierarchical_direction_label_contract,
)

REQUIRED_COLUMNS = (
    "y_direction",
    "y_trade",
    "y_tradable",
    "y_side",
    "y_side_mask",
    "y_bad_path",
    "mae_first_n_bps",
    "mfe_first_n_bps",
    "path_quality_bps",
    "y_quality_score",
    "y_position_size_target",
    "atr_bps",
    "mfe_long_first_n_bps",
    "mae_long_first_n_bps",
    "mfe_short_first_n_bps",
    "mae_short_first_n_bps",
    "y_long_path_utility_bps",
    "y_short_path_utility_bps",
    "y_long_bad_path",
    "y_short_bad_path",
    "y_long_expected_mae_bps",
    "y_short_expected_mae_bps",
    "y_rising_channel_support_touch",
    "y_falling_channel_resistance_touch",
    "y_support_retest_continuation",
    "y_resistance_retest_continuation",
    "y_countertrend_short_trap",
    "y_countertrend_long_trap",
    "y_long_high_mae_low_mfe_early_failure",
    "y_short_high_mae_low_mfe_early_failure",
)

REPAIRED_COLUMNS = (
    "y_direction",
    "y_trade",
    "y_tradable",
    "y_side",
    "y_side_mask",
    "y_bad_path",
    "mae_first_n_bps",
    "mfe_first_n_bps",
    "path_quality_bps",
    "y_quality_score",
    "y_position_size_target",
    "y_direction_long_score_bps",
    "y_direction_short_score_bps",
    "y_long_path_utility_bps",
    "y_short_path_utility_bps",
    "y_long_bad_path",
    "y_short_bad_path",
    "y_long_expected_mae_bps",
    "y_short_expected_mae_bps",
)


def _as_numpy(table: pa.Table, name: str) -> np.ndarray:
    return table[name].combine_chunks().to_numpy(zero_copy_only=False)


def _replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx < 0:
        raise RuntimeError(f"missing column while replacing {name}")
    field = table.schema.field(idx)
    return table.set_column(idx, field, pa.array(values, type=field.type))


def _direction_to_internal_side(values: np.ndarray) -> np.ndarray:
    direction = np.asarray(values, dtype=np.int32)
    side = np.full(direction.shape, -1, dtype=np.int8)
    side[direction == 0] = 0
    side[direction == 1] = 1
    return side


def _harvest_side_from_trade(y_trade: np.ndarray, y_side: np.ndarray, y_side_mask: np.ndarray) -> np.ndarray:
    trade = np.asarray(y_trade, dtype=np.float32)
    side_raw = np.asarray(y_side, dtype=np.int32)
    side_mask = np.asarray(y_side_mask, dtype=np.float32)
    out = np.full(trade.shape, -1, dtype=np.int8)
    active = (trade > 0.5) & (side_mask > 0.5)
    out[active & (side_raw == 0)] = 0
    out[active & (side_raw == 1)] = 1
    return out


def _internal_side_to_direction(values: np.ndarray) -> np.ndarray:
    side = np.asarray(values, dtype=np.int8)
    direction = np.full(side.shape, 2, dtype=np.int32)
    direction[side == 0] = 0
    direction[side == 1] = 1
    return direction


def _selected_side_path_targets(
    table: pa.Table,
    repaired_harvest_side: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    side = np.asarray(repaired_harvest_side, dtype=np.int8)
    mfe_long = _as_numpy(table, "mfe_long_first_n_bps").astype(np.float32, copy=False)
    mae_long = _as_numpy(table, "mae_long_first_n_bps").astype(np.float32, copy=False)
    mfe_short = _as_numpy(table, "mfe_short_first_n_bps").astype(np.float32, copy=False)
    mae_short = _as_numpy(table, "mae_short_first_n_bps").astype(np.float32, copy=False)
    mfe = np.zeros(side.shape, dtype=np.float32)
    mae = np.zeros(side.shape, dtype=np.float32)
    long_rows = side == 0
    short_rows = side == 1
    mfe[long_rows] = mfe_long[long_rows]
    mae[long_rows] = mae_long[long_rows]
    mfe[short_rows] = mfe_short[short_rows]
    mae[short_rows] = mae_short[short_rows]
    path = (mfe - mae).astype(np.float32)
    path[side == -1] = 0.0
    return mfe, mae, path


def _validate_repair_invariants(
    *,
    y_direction: np.ndarray,
    y_trade: np.ndarray,
    y_side: np.ndarray,
    y_side_mask: np.ndarray,
    anti_short: np.ndarray,
    anti_long: np.ndarray,
    support_continue: np.ndarray,
    resistance_continue: np.ndarray,
    split_source: Path,
) -> dict[str, int]:
    anti_short = np.asarray(anti_short, dtype=bool)
    anti_long = np.asarray(anti_long, dtype=bool)
    conflict = anti_short & anti_long
    anti_short_only = anti_short & (~conflict)
    anti_long_only = anti_long & (~conflict)
    direction = np.asarray(y_direction, dtype=np.int32)
    trade = np.asarray(y_trade, dtype=np.float32)
    side = np.asarray(y_side, dtype=np.int8)
    side_mask = np.asarray(y_side_mask, dtype=np.float32)
    failures: list[str] = []
    anti_short_wrong = int(np.sum(anti_short_only & (direction == 1)))
    anti_long_wrong = int(np.sum(anti_long_only & (direction == 0)))
    support_not_long = int(np.sum(np.asarray(support_continue, dtype=bool) & (~anti_long) & (direction != 0)))
    resistance_not_short = int(np.sum(np.asarray(resistance_continue, dtype=bool) & (~anti_short) & (direction != 1)))
    conflict_not_flat = int(np.sum(conflict & ((direction != 2) | (trade > 0.0) | (side_mask > 0.0))))
    masked_wrong_short = int(np.sum(anti_short_only & (side_mask > 0.0) & (side == 1)))
    masked_wrong_long = int(np.sum(anti_long_only & (side_mask > 0.0) & (side == 0)))
    if anti_short_wrong:
        failures.append(f"anti_short_only rows still labeled SHORT: {anti_short_wrong}")
    if anti_long_wrong:
        failures.append(f"anti_long_only rows still labeled LONG: {anti_long_wrong}")
    if support_not_long:
        failures.append(f"support continuation rows not LONG outside conflict: {support_not_long}")
    if resistance_not_short:
        failures.append(f"resistance continuation rows not SHORT outside conflict: {resistance_not_short}")
    if conflict_not_flat:
        failures.append(f"conflict rows not FLAT/no-trade: {conflict_not_flat}")
    if masked_wrong_short:
        failures.append(f"side-mask anti_short_only rows still teach SHORT: {masked_wrong_short}")
    if masked_wrong_long:
        failures.append(f"side-mask anti_long_only rows still teach LONG: {masked_wrong_long}")
    if failures:
        raise RuntimeError(f"[XAU_STRUCTURAL_REPAIR_INVARIANT_FAIL] {split_source}: " + "; ".join(failures))
    return {
        "anti_short_wrong_side_rows": anti_short_wrong,
        "anti_long_wrong_side_rows": anti_long_wrong,
        "support_not_long_rows": support_not_long,
        "resistance_not_short_rows": resistance_not_short,
        "conflict_not_flat_rows": conflict_not_flat,
        "masked_wrong_short_rows": masked_wrong_short,
        "masked_wrong_long_rows": masked_wrong_long,
    }


def _repair_table(table: pa.Table) -> tuple[pa.Table, dict[str, int]]:
    rising = _as_numpy(table, "y_rising_channel_support_touch") > 0.5
    falling = _as_numpy(table, "y_falling_channel_resistance_touch") > 0.5
    support_continue = _as_numpy(table, "y_support_retest_continuation") > 0.5
    resistance_continue = _as_numpy(table, "y_resistance_retest_continuation") > 0.5
    short_trap = _as_numpy(table, "y_countertrend_short_trap") > 0.5
    long_trap = _as_numpy(table, "y_countertrend_long_trap") > 0.5
    long_early_fail = _as_numpy(table, "y_long_high_mae_low_mfe_early_failure") > 0.5
    short_early_fail = _as_numpy(table, "y_short_high_mae_low_mfe_early_failure") > 0.5
    anti_short = (
        rising
        | support_continue
        | short_trap
        | short_early_fail
    )
    anti_long = (
        falling
        | resistance_continue
        | long_trap
        | long_early_fail
    )
    direction_side = _direction_to_internal_side(_as_numpy(table, "y_direction"))
    harvest_side = _harvest_side_from_trade(
        _as_numpy(table, "y_trade"),
        _as_numpy(table, "y_side"),
        _as_numpy(table, "y_side_mask"),
    )
    repaired_side, repaired_harvest_side, side_masks = _apply_structural_side_repair(
        direction_side,
        harvest_side,
        rising,
        short_trap,
        support_continue,
        falling,
        long_trap,
        resistance_continue,
        short_early_fail,
        long_early_fail,
    )
    repaired_direction = _internal_side_to_direction(repaired_side)
    repaired_trade = (repaired_harvest_side != -1).astype(np.float32)
    repaired_y_side = np.where(repaired_harvest_side == 1, 1, 0).astype(np.int8)
    repaired_side_mask = repaired_trade.astype(np.float32)
    selected_mfe, selected_mae, selected_path = _selected_side_path_targets(table, repaired_harvest_side)
    repaired_position_size = _position_size_target_from_repaired_path(
        selected_mfe,
        selected_mae,
        _as_numpy(table, "atr_bps"),
        repaired_trade,
    )
    (
        long_util,
        short_util,
        long_bad,
        short_bad,
        long_mae,
        short_mae,
        masks,
    ) = _apply_structural_utility_repair(
        _as_numpy(table, "y_long_path_utility_bps"),
        _as_numpy(table, "y_short_path_utility_bps"),
        _as_numpy(table, "y_long_bad_path"),
        _as_numpy(table, "y_short_bad_path"),
        _as_numpy(table, "y_long_expected_mae_bps"),
        _as_numpy(table, "y_short_expected_mae_bps"),
        anti_short,
        anti_long,
    )
    scalar_bad = _repaired_scalar_bad_path_target(repaired_harvest_side, long_bad, short_bad)
    replacements = {
        "y_direction": repaired_direction.astype(np.int32, copy=False),
        "y_trade": repaired_trade.astype(np.float32, copy=False),
        "y_tradable": repaired_trade.astype(np.float32, copy=False),
        "y_side": repaired_y_side.astype(np.int8, copy=False),
        "y_side_mask": repaired_side_mask.astype(np.float32, copy=False),
        "y_bad_path": scalar_bad.astype(np.float32, copy=False),
        "mfe_first_n_bps": selected_mfe.astype(np.float32, copy=False),
        "mae_first_n_bps": selected_mae.astype(np.float32, copy=False),
        "path_quality_bps": selected_path.astype(np.float32, copy=False),
        "y_quality_score": np.maximum(0.0, selected_path).astype(np.float32, copy=False),
        "y_position_size_target": repaired_position_size.astype(np.float32, copy=False),
        "y_long_path_utility_bps": long_util.astype(np.float32, copy=False),
        "y_short_path_utility_bps": short_util.astype(np.float32, copy=False),
        "y_long_bad_path": long_bad.astype(np.float32, copy=False),
        "y_short_bad_path": short_bad.astype(np.float32, copy=False),
        "y_long_expected_mae_bps": long_mae.astype(np.float32, copy=False),
        "y_short_expected_mae_bps": short_mae.astype(np.float32, copy=False),
    }
    if "y_direction_long_score_bps" in table.schema.names:
        replacements["y_direction_long_score_bps"] = long_util.astype(np.float32, copy=False)
    if "y_direction_short_score_bps" in table.schema.names:
        replacements["y_direction_short_score_bps"] = short_util.astype(np.float32, copy=False)
    out = table
    for name, values in replacements.items():
        out = _replace_column(out, name, values)
    invariant_stats = _validate_repair_invariants(
        y_direction=repaired_direction,
        y_trade=repaired_trade,
        y_side=repaired_y_side,
        y_side_mask=repaired_side_mask,
        anti_short=anti_short,
        anti_long=anti_long,
        support_continue=support_continue,
        resistance_continue=resistance_continue,
        split_source=Path("<row-group>"),
    )
    stats = {
        "rows": int(table.num_rows),
        "short_to_long_rows": int(np.sum(side_masks["short_to_long"])),
        "short_to_flat_rows": int(np.sum(side_masks["short_to_flat"])),
        "long_to_short_rows": int(np.sum(side_masks["long_to_short"])),
        "long_to_flat_rows": int(np.sum(side_masks["long_to_flat"])),
        "conflict_to_flat_rows": int(np.sum(side_masks["conflict_to_flat"])),
        "anti_short_only_rows": int(np.sum(masks["anti_short_only"])),
        "anti_long_only_rows": int(np.sum(masks["anti_long_only"])),
        "conflict_rows": int(np.sum(masks["conflict"])),
        "short_utility_repaired_rows": int(np.sum(masks["short_utility_repaired"])),
        "long_utility_repaired_rows": int(np.sum(masks["long_utility_repaired"])),
        "conflict_utility_suppressed_rows": int(np.sum(masks["conflict_utility_suppressed"])),
        **invariant_stats,
    }
    return out, stats


def _merge_stats(total: dict[str, int], part: dict[str, int]) -> None:
    for key, value in part.items():
        total[key] = int(total.get(key, 0)) + int(value)


def _repair_parquet(src: Path, dst: Path) -> dict[str, Any]:
    pf = pq.ParquetFile(src)
    missing = [name for name in REQUIRED_COLUMNS if name not in pf.schema_arrow.names]
    if missing:
        raise RuntimeError(f"{src} lacks required repair columns: {missing}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_name(f"{dst.name}.tmp")
    tmp_dst.unlink(missing_ok=True)
    total: dict[str, int] = {}
    writer: pq.ParquetWriter | None = None
    try:
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            repaired, stats = _repair_table(table)
            _merge_stats(total, stats)
            if writer is None:
                writer = pq.ParquetWriter(tmp_dst, repaired.schema, compression="zstd")
            writer.write_table(repaired)
    finally:
        if writer is not None:
            writer.close()
    repaired_pf = pq.ParquetFile(tmp_dst)
    if int(repaired_pf.metadata.num_rows) != int(pf.metadata.num_rows):
        tmp_dst.unlink(missing_ok=True)
        raise RuntimeError(
            f"[XAU_STRUCTURAL_REPAIR_ROWCOUNT_MISMATCH] {src}: "
            f"input={pf.metadata.num_rows} output={repaired_pf.metadata.num_rows}"
        )
    tmp_dst.replace(dst)
    return {
        "source": str(src),
        "output": str(dst),
        "row_groups": int(pf.num_row_groups),
        **total,
    }


def _update_manifest(src: Path, dst: Path, output_parquet: Path, stats: dict[str, Any]) -> None:
    payload = json.loads(src.read_text(encoding="utf-8"))
    payload["output_data_path"] = str(output_parquet)
    extra = payload.setdefault("extra", {})
    extra["hierarchical_direction_targets"] = hierarchical_direction_label_contract()[
        "hierarchical_direction_targets"
    ]
    extra["structural_utility_repair_applied"] = {
        "schema_version": "xau_structural_utility_label_repair_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(src),
        "source_parquet": stats["source"],
        "output_parquet": stats["output"],
        "repaired_columns": list(REPAIRED_COLUMNS),
        "utility_margin_bps": float(XAU_STRUCTURAL_UTILITY_REPAIR_MARGIN_BPS),
        "mae_margin_bps": float(XAU_STRUCTURAL_UTILITY_REPAIR_MAE_MARGIN_BPS),
        "stats": stats,
        "runtime_rule_free": True,
    }
    dst.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _copy_build_proof(input_dir: Path, output_dir: Path, split_stats: dict[str, Any]) -> None:
    src = input_dir / "DATASET_BUILD_PROOF.json"
    payload: dict[str, Any] = {}
    if src.exists():
        payload = json.loads(src.read_text(encoding="utf-8"))
    payload["hierarchical_direction_targets"] = hierarchical_direction_label_contract()[
        "hierarchical_direction_targets"
    ]
    payload["structural_utility_repair_applied"] = {
        "schema_version": "xau_structural_utility_label_repair_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dataset_dir": str(input_dir),
        "output_dataset_dir": str(output_dir),
        "repaired_columns": list(REPAIRED_COLUMNS),
        "utility_margin_bps": float(XAU_STRUCTURAL_UTILITY_REPAIR_MARGIN_BPS),
        "mae_margin_bps": float(XAU_STRUCTURAL_UTILITY_REPAIR_MAE_MARGIN_BPS),
        "split_stats": split_stats,
        "runtime_rule_free": True,
    }
    (output_dir / "DATASET_BUILD_PROOF.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", type=str, required=True)
    parser.add_argument("--instrument", default="XAUUSD")
    args = parser.parse_args()

    if str(args.instrument).strip().upper() != "XAUUSD":
        raise RuntimeError("repair_entry_xau_structural_utility_labels_v1 is XAUUSD-only")
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.exists():
        raise RuntimeError(f"missing input dir: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output dir must be empty/nonexistent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    split_stats: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        src_parquet = input_dir / f"{args.stem}_{split}.parquet"
        src_manifest = input_dir / f"{args.stem}_{split}.manifest.json"
        dst_parquet = output_dir / src_parquet.name
        dst_manifest = output_dir / src_manifest.name
        if not src_parquet.exists() or not src_manifest.exists():
            raise RuntimeError(f"missing split artifacts for {split}: {src_parquet} {src_manifest}")
        stats = _repair_parquet(src_parquet, dst_parquet)
        _update_manifest(src_manifest, dst_manifest, dst_parquet, stats)
        split_stats[split] = stats

    for extra in input_dir.iterdir():
        if extra.name == "DATASET_BUILD_PROOF.json":
            continue
        if extra.suffix in {".parquet", ".json"}:
            continue
        target = output_dir / extra.name
        if extra.is_file():
            shutil.copy2(extra, target)
    _copy_build_proof(input_dir, output_dir, split_stats)
    print(json.dumps({"output_dir": str(output_dir), "split_stats": split_stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
