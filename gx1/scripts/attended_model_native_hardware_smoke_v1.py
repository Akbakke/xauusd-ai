#!/usr/bin/env python3
"""One bounded, non-promotable CUDA architecture smoke for the Entry model.

This is deliberately *not* a dataset or training route.  The production
normalization fit has to cover the complete immutable TRAIN population, so a
short run must not pretend that a 10k-row subset is a valid replacement.  The
module instead builds the exact production architecture, its specialist
routing, and a contract-valid synthetic normalization surface, then executes
one deterministic forward/backward/optimizer step.  It has no output path and
cannot create a bundle, candidate, TEST result, paper route, or live authority.

It is admitted only by ``gx1_capped_run.sh --attended-smoke``.  That parent
owns the hard cgroup, wall-clock, thermal and actual-power protections.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_MTF_CONTEXT_TIMEFRAMES
from gx1.contracts.entry_model_native_input_normalization_v1 import (
    CTX_CAT_DOMAINS,
    CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS,
    EXPECTED_SURFACES,
    MTF_SEMANTIC_CATEGORICAL_DOMAINS,
    build_input_normalization_contract,
    fit_ctx_cat_contract,
    fit_surface_normalization,
    share_temporal_alias_stats_from_signal,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    model_native_context_temporal_alias_policy,
    require_multi_tf_specialist_routing_v4,
)
from gx1.features.htf_features import (
    MULTI_TF_FEATURE_COUNT_V4,
    MULTI_TF_PER_BAR_FEATURES_V4,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    EntryV10CtxHybridTransformer,
)
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


HARDWARE_SMOKE_SCHEMA_VERSION = "entry_model_native_attended_hardware_smoke_v1"
HARDWARE_SMOKE_BATCH_SIZE = 8
# Must cover D1's 252-bar tensor plus every row of the fixed batch.
HARDWARE_SMOKE_NORMALIZATION_ROWS = 512
HARDWARE_SMOKE_SEED = 1337
_PER_TF_SEQ_LENS = {"M5": 16, "M15": 64, "H1": 96, "H4": 96, "D1": 252}


def _signal_names() -> list[str]:
    fields = list(
        MODEL_NATIVE_BASE_FIELDS
        + MODEL_NATIVE_MANDATORY_SELECTED_FIELDS
        + MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS
    )
    if len(fields) != MODEL_NATIVE_SIGNAL_DIM or len(fields) != len(set(fields)):
        raise RuntimeError("[ATTENDED_HARDWARE_SMOKE_SIGNAL_FIELDS_INVALID]")
    return fields


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _selection_hash(namespace: str) -> str:
    return hashlib.sha256(namespace.encode("utf-8")).hexdigest()


def _synthetic_normalization() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return validation-grade synthetic tensors with no market-data lineage.

    ``require_input_normalization_contract`` needs a complete surface contract
    even though this route has no data authority.  The synthetic lineage is
    explicit and is retained in memory only; it must never be written or used
    by a train/candidate route.
    """

    rows = HARDWARE_SMOKE_NORMALIZATION_ROWS
    row = np.arange(rows, dtype=np.float32)
    signal_names = _signal_names()
    mtf_names = list(MULTI_TF_PER_BAR_FEATURES_V4)
    if len(signal_names) != MODEL_NATIVE_SIGNAL_DIM:
        raise RuntimeError("[ATTENDED_HARDWARE_SMOKE_SIGNAL_WIDTH_INVALID]")
    if len(mtf_names) != MULTI_TF_FEATURE_COUNT_V4:
        raise RuntimeError("[ATTENDED_HARDWARE_SMOKE_MTF_WIDTH_INVALID]")

    ctx_cont = np.column_stack(
        [
            row * np.float32(0.01 + (index + 1) / 1000.0)
            + np.float32((index % 7) * 0.1)
            for index in range(MODEL_NATIVE_CTX_CONT_DIM)
        ]
    ).astype(np.float32)
    for name in CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS:
        index = list(MODEL_NATIVE_CTX_CONT_FIELDS).index(name)
        ctx_cont[:, index] = row % 5
    signal = np.column_stack(
        [
            row * np.float32(0.02 + (index + 1) / 1000.0)
            + np.float32((index % 5) * 0.2)
            for index in range(MODEL_NATIVE_SIGNAL_DIM)
        ]
    ).astype(np.float32)
    alias_policy = model_native_context_temporal_alias_policy(signal_names)
    for alias in alias_policy["aliases"]:
        signal[:, int(alias["signal_index"])] = ctx_cont[
            :, int(alias["ctx_cont_index"])
        ]

    signal_surface = fit_surface_normalization(
        signal,
        surface="signal",
        field_names=signal_names,
        semantic_categorical_domains={
            f"ctx_cont.{name}": domain
            for name, domain in CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS.items()
            if f"ctx_cont.{name}" in signal_names
        },
    )
    ctx_cont_surface = share_temporal_alias_stats_from_signal(
        fit_surface_normalization(
            ctx_cont,
            surface="ctx_cont",
            field_names=MODEL_NATIVE_CTX_CONT_FIELDS,
            semantic_categorical_domains=CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS,
        ),
        signal_surface,
        temporal_aliases=alias_policy["aliases"],
        ctx_cont_values=ctx_cont,
    )
    surfaces: dict[str, Any] = {"signal": signal_surface, "ctx_cont": ctx_cont_surface}
    mtf_values: dict[str, np.ndarray] = {}
    for tf in ("m5", "m15", "h1", "h4", "d1"):
        values = np.column_stack(
            [
                row * np.float32(0.03 + (index + 1) / 100.0)
                + np.float32((index % 3) * 0.1)
                for index in range(MULTI_TF_FEATURE_COUNT_V4)
            ]
        ).astype(np.float32)
        if "ema_stack_aligned_v2" in mtf_names:
            values[:, mtf_names.index("ema_stack_aligned_v2")] = (row % 3) - 1
        if "regime_class_id" in mtf_names:
            values[:, mtf_names.index("regime_class_id")] = row % 5
        mtf_values[tf] = values
        surfaces[f"mtf_{tf}"] = fit_surface_normalization(
            values,
            surface=f"mtf_{tf}",
            field_names=mtf_names,
            semantic_categorical_domains=(
                MTF_SEMANTIC_CATEGORICAL_DOMAINS
                if "regime_class_id" in mtf_names
                else {}
            ),
        )
    if tuple(surfaces) != EXPECTED_SURFACES:
        raise RuntimeError("[ATTENDED_HARDWARE_SMOKE_SURFACE_ORDER_INVALID]")

    ctx_cat = np.column_stack(
        [
            np.arange(rows, dtype=np.int64) % len(domain)
            for domain in CTX_CAT_DOMAINS.values()
        ]
    )
    windows = {
        tf: {
            "left_index_inclusive": 0,
            "right_index_exclusive": rows,
            "selected_unique_row_count": rows,
            "selected_row_indices_sha256": _selection_hash(f"{tf}:indices"),
            "selected_row_values_sha256": _selection_hash(f"{tf}:values"),
            "time_min_utc": "2026-01-01T00:00:00+00:00",
            "time_max_utc": "2026-01-01T10:35:00+00:00",
        }
        for tf in _PER_TF_SEQ_LENS
    }
    lineage = {
        "dataset_run_id": "ATTENDED_HARDWARE_SMOKE_NO_DATA_AUTHORITY_V1",
        "train_parquet_path": "/attended-hardware-smoke/no-data.parquet",
        "train_parquet_sha256": "0" * 64,
        "train_manifest_path": "/attended-hardware-smoke/no-data.manifest.json",
        "train_manifest_sha256": "1" * 64,
        "train_row_count": rows,
        "entry_train_decision_row_count": rows - 1,
        "exit_train_decision_row_count": 1,
        "local_fit_row_count": rows,
        "context_fit_row_count": rows,
        "val_fit_row_count": 0,
        "test_fit_row_count": 0,
        "train_time_min_utc": "2026-01-01T00:00:00+00:00",
        "train_time_max_utc": "2026-01-01T10:35:00+00:00",
        "m5_prebuilt_path": "/attended-hardware-smoke/no-data-m5.parquet",
        "m5_prebuilt_sha256": "2" * 64,
        "mtf_cache_manifest_path": "/attended-hardware-smoke/no-data-mtf.json",
        "mtf_cache_manifest_sha256": "3" * 64,
        "mtf_builder_version": HARDWARE_SMOKE_SCHEMA_VERSION,
        "mtf_feature_names_sha256": _canonical_sha256(mtf_names),
        "per_tf_seq_lens": dict(_PER_TF_SEQ_LENS),
        "per_tf_shift_seconds": {"M5": 300, "M15": 900, "H1": 3600, "H4": 14400, "D1": 86400},
        "per_tf_fit_windows": windows,
    }
    normalization = build_input_normalization_contract(
        fit_start_utc=lineage["train_time_min_utc"],
        fit_end_utc=lineage["train_time_max_utc"],
        surfaces=surfaces,
        ctx_cat=fit_ctx_cat_contract(ctx_cat, field_names=MODEL_NATIVE_CTX_CAT_FIELDS),
        lineage=lineage,
        temporal_aliases=alias_policy["aliases"],
    )
    samples = {
        "signal": signal,
        "ctx_cont": ctx_cont,
        "ctx_cat": ctx_cat,
        **{f"mtf_{tf}": value for tf, value in mtf_values.items()},
    }
    return normalization, samples


def _batch(samples: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    batch = HARDWARE_SMOKE_BATCH_SIZE
    seq_len = 96
    out = {
        "seq_x": torch.from_numpy(
            np.stack([samples["signal"][i : i + seq_len] for i in range(batch)])
        ),
        "snap_x": torch.from_numpy(samples["signal"][:batch].copy()),
        "ctx_cont": torch.from_numpy(samples["ctx_cont"][:batch].copy()),
        "ctx_cat": torch.from_numpy(samples["ctx_cat"][:batch].copy()),
    }
    for tf in ENTRY_MTF_CONTEXT_TIMEFRAMES:
        tf_l = str(tf).lower()
        width = _PER_TF_SEQ_LENS[str(tf)]
        out[f"seq_{tf_l}"] = torch.from_numpy(
            np.stack([samples[f"mtf_{tf_l}"][i : i + width] for i in range(batch)])
        )
    return out


def _build_model(*, specialist_audit_json: Path, normalization: dict[str, Any]) -> EntryV10CtxHybridTransformer:
    signal_names = _signal_names()
    specialist_indices, specialist_meta = trainer._load_specialist_fusion_contract(
        specialist_audit_json,
        expected_signal_dim=MODEL_NATIVE_SIGNAL_DIM,
        ordered_signal_names=signal_names,
        contract_mode=MODEL_NATIVE_CONTRACT_MODE,
    )
    multi_tf_specialist_indices = {
        str(name): list(indices)
        for name, indices in require_multi_tf_specialist_routing_v4(
            MULTI_TF_PER_BAR_FEATURES_V4
        ).items()
    }
    return EntryV10CtxHybridTransformer(
        seq_input_dim=MODEL_NATIVE_SIGNAL_DIM,
        snap_input_dim=MODEL_NATIVE_SIGNAL_DIM,
        seq_len=96,
        dropout=0.05,
        ctx_cont_dim=MODEL_NATIVE_CTX_CONT_DIM,
        ctx_cat_dim=MODEL_NATIVE_CTX_CAT_DIM,
        m15_seq_dim=MULTI_TF_FEATURE_COUNT_V4,
        h1_seq_dim=MULTI_TF_FEATURE_COUNT_V4,
        h4_seq_dim=MULTI_TF_FEATURE_COUNT_V4,
        d1_seq_dim=MULTI_TF_FEATURE_COUNT_V4,
        m15_seq_len=_PER_TF_SEQ_LENS["M15"],
        h1_seq_len=_PER_TF_SEQ_LENS["H1"],
        h4_seq_len=_PER_TF_SEQ_LENS["H4"],
        d1_seq_len=_PER_TF_SEQ_LENS["D1"],
        m5_seq_dim=MULTI_TF_FEATURE_COUNT_V4,
        m5_seq_len=_PER_TF_SEQ_LENS["M5"],
        multi_tf_num_layers=2,
        multi_tf_scale=0.5,
        specialist_input_indices=specialist_indices,
        specialist_ctx_cont_indices={
            str(name): list(values)
            for name, values in specialist_meta["context_routing"]["ctx_cont_indices"].items()
        },
        specialist_ctx_cont_nominal_indices={
            str(name): list(values)
            for name, values in specialist_meta["context_routing"]["ctx_cont_nominal_indices"].items()
        },
        specialist_ctx_cat_indices={
            str(name): list(values)
            for name, values in specialist_meta["context_routing"]["ctx_cat_indices"].items()
        },
        multi_tf_specialist_input_indices=multi_tf_specialist_indices,
        temporal_alias_signal_indices=list(
            specialist_meta["context_routing"]["temporal_alias_policy"]["signal_indices"]
        ),
        temporal_alias_ctx_cont_indices=list(
            specialist_meta["context_routing"]["temporal_alias_policy"]["ctx_cont_indices"]
        ),
        specialist_num_layers=1,
        specialist_fusion_scale=0.25,
        cross_family_fusion_scale=0.25,
        input_normalization=normalization,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("attended, no-data model-native CUDA hardware smoke")
    parser.add_argument("--attended-hardware-smoke", action="store_true", required=True)
    parser.add_argument("--device", choices=("cuda",), required=True)
    parser.add_argument("--specialist-audit-json", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("[ATTENDED_HARDWARE_SMOKE_CUDA_UNAVAILABLE]")
    audit_path = args.specialist_audit_json.expanduser().resolve(strict=True)
    if not audit_path.is_file() or audit_path.is_symlink():
        raise RuntimeError("[ATTENDED_HARDWARE_SMOKE_SPECIALIST_AUDIT_INVALID]")

    torch.manual_seed(HARDWARE_SMOKE_SEED)
    np.random.seed(HARDWARE_SMOKE_SEED)
    torch.cuda.empty_cache()
    normalization, samples = _synthetic_normalization()
    model = _build_model(specialist_audit_json=audit_path, normalization=normalization).cuda()
    model.require_input_normalization_state()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    batch = {name: value.cuda(non_blocking=False) for name, value in _batch(samples).items()}
    torch.cuda.synchronize()
    start = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    out = model(
        batch["seq_x"],
        batch["snap_x"],
        ctx_cat=batch["ctx_cat"],
        ctx_cont=batch["ctx_cont"],
        seq_m15=batch["seq_m15"],
        seq_h1=batch["seq_h1"],
        seq_h4=batch["seq_h4"],
        seq_d1=batch["seq_d1"],
    )
    loss = out["entry_action_q_bps"].float().square().mean()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("[ATTENDED_HARDWARE_SMOKE_NONFINITE_LOSS]")
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    peak_mib = int(torch.cuda.max_memory_allocated() // (1024 * 1024))
    print(
        "[ATTENDED_HARDWARE_SMOKE_PASS] "
        f"schema={HARDWARE_SMOKE_SCHEMA_VERSION} batch_size={HARDWARE_SMOKE_BATCH_SIZE} "
        f"elapsed_seconds={elapsed:.3f} peak_cuda_memory_mib={peak_mib} "
        "authority=none data_authority=none candidate=false test=false promotion=false live=false"
    )
    del batch, optimizer, model, samples, normalization
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
