#!/usr/bin/env python3
"""Capped synthetic runnability benchmark for production-shape Exit episodes."""

from __future__ import annotations

import argparse
import copy
import json
import resource
import time

import torch

from gx1.contracts.entry_exit_feature_base_v1 import (
    EXIT_FEATURE_SEQUENCE_BARS,
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT,
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    EntryV10CtxHybridTransformer,
)
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_MAX_PATH_BARS,
    UNIFIED_EXIT_PATH_FEATURE_DIM,
)
from tests.model_native_input_normalization_support import (
    input_normalization_fixture,
)


TF_LENGTHS = {"M5": 16, "M15": 64, "H1": 96, "H4": 96, "D1": 252}
TF_EPISODE_CLOSURES = {"M5": 103, "M15": 35, "H1": 10, "H4": 4, "D1": 2}


def _round_robin(width: int) -> dict[str, list[int]]:
    names = tuple(MODEL_NATIVE_TRAINING_SPECIALISTS)
    return {
        name: list(range(position, width, len(names)))
        for position, name in enumerate(names)
    }


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    batch = int(args.batch)
    if batch < 1:
        raise RuntimeError("BENCHMARK_BATCH_INVALID")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("BENCHMARK_CUDA_UNAVAILABLE")
    torch.manual_seed(20260814)
    tf_names = list(MULTI_TF_PER_BAR_FEATURES_V4)
    tf_width = len(tf_names)
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=MODEL_NATIVE_SIGNAL_DIM,
        snap_input_dim=MODEL_NATIVE_SIGNAL_DIM,
        seq_len=96,
        dropout=0.0,
        multi_tf_num_layers=1,
        multi_tf_scale=0.5,
        specialist_num_layers=1,
        specialist_fusion_scale=0.25,
        cross_family_fusion_scale=0.25,
        ctx_cont_dim=MODEL_NATIVE_CTX_CONT_DIM,
        ctx_cat_dim=MODEL_NATIVE_CTX_CAT_DIM,
        m5_seq_dim=tf_width,
        m15_seq_dim=tf_width,
        h1_seq_dim=tf_width,
        h4_seq_dim=tf_width,
        d1_seq_dim=tf_width,
        m5_seq_len=TF_LENGTHS["M5"],
        m15_seq_len=TF_LENGTHS["M15"],
        h1_seq_len=TF_LENGTHS["H1"],
        h4_seq_len=TF_LENGTHS["H4"],
        d1_seq_len=TF_LENGTHS["D1"],
        specialist_input_indices=_round_robin(MODEL_NATIVE_SIGNAL_DIM),
        specialist_ctx_cont_indices={
            str(name): list(values)
            for name, values in MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT[
                "ctx_cont_indices"
            ].items()
        },
        specialist_ctx_cont_nominal_indices={
            str(name): list(values)
            for name, values in MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT[
                "ctx_cont_nominal_indices"
            ].items()
        },
        specialist_ctx_cat_indices={
            str(name): list(values)
            for name, values in MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT[
                "ctx_cat_indices"
            ].items()
        },
        multi_tf_specialist_input_indices=_round_robin(tf_width),
        temporal_alias_signal_indices=[],
        temporal_alias_ctx_cont_indices=[],
        input_normalization=input_normalization_fixture(
            signal_names=[
                f"production_signal_{index}"
                for index in range(MODEL_NATIVE_SIGNAL_DIM)
            ],
            mtf_names=tf_names,
            per_tf_seq_lens=TF_LENGTHS,
        ),
    )
    model = model.to(device)
    target_model = copy.deepcopy(model).eval()
    target_model.requires_grad_(False)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    materialize_start = time.perf_counter()
    local_rows = EXIT_FEATURE_SEQUENCE_BARS - 1 + UNIFIED_EXIT_MAX_PATH_BARS
    inputs: dict[str, object] = {
        "entry_decision_representation": torch.randn(batch, 128, device=device),
        "exit_local_history_x": torch.randn(
            batch, local_rows, MODEL_NATIVE_SIGNAL_DIM, device=device
        ),
        "exit_state_ctx_cat": torch.zeros(
            batch,
            UNIFIED_EXIT_MAX_PATH_BARS,
            MODEL_NATIVE_CTX_CAT_DIM,
            dtype=torch.long,
            device=device,
        ),
        "exit_state_ctx_cont": torch.zeros(
            batch,
            UNIFIED_EXIT_MAX_PATH_BARS,
            MODEL_NATIVE_CTX_CONT_DIM,
            device=device,
        ),
        "exit_path_x": torch.randn(
            batch,
            2,
            UNIFIED_EXIT_MAX_PATH_BARS,
            UNIFIED_EXIT_PATH_FEATURE_DIM,
            device=device,
        ),
        "exit_mtf_histories": {},
        "exit_mtf_gathers": {},
        "exit_mtf_history_lengths": {},
    }
    for tf in EXIT_MTF_CONTEXT_TIMEFRAMES:
        history_rows = TF_LENGTHS[tf] + TF_EPISODE_CLOSURES[tf] - 1
        history = torch.zeros(batch, history_rows, tf_width, device=device)
        gather = torch.div(
            torch.arange(UNIFIED_EXIT_MAX_PATH_BARS),
            max(1, UNIFIED_EXIT_MAX_PATH_BARS // TF_EPISODE_CLOSURES[tf]),
            rounding_mode="floor",
        ).clamp(max=TF_EPISODE_CLOSURES[tf] - 1).to(device) + TF_LENGTHS[tf] - 1
        inputs["exit_mtf_histories"][tf.lower()] = history
        inputs["exit_mtf_gathers"][tf.lower()] = gather.view(1, -1).expand(
            batch, -1
        ).clone()
        inputs["exit_mtf_history_lengths"][tf.lower()] = torch.full(
            (batch,), history_rows, dtype=torch.long, device=device
        )
    materialize_seconds = time.perf_counter() - materialize_start
    input_bytes = sum(
        int(value.numel() * value.element_size())
        for value in (
            inputs["entry_decision_representation"],
            inputs["exit_local_history_x"],
            inputs["exit_state_ctx_cat"],
            inputs["exit_state_ctx_cont"],
            inputs["exit_path_x"],
            *inputs["exit_mtf_histories"].values(),
            *inputs["exit_mtf_gathers"].values(),
            *inputs["exit_mtf_history_lengths"].values(),
        )
    )
    target_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        target_start = time.perf_counter()
    with torch.no_grad():
        target_output = target_model.forward_exit_episode(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    target_seconds = time.perf_counter() - target_start
    online_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    output = model.forward_exit_episode(**inputs)
    loss = output["exit_action_q_bps"].square().mean()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    online_forward_backward_seconds = time.perf_counter() - online_start
    optimizer_start = time.perf_counter()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    optimizer_step_seconds = time.perf_counter() - optimizer_start
    if target_output["exit_action_q_bps"].shape != output["exit_action_q_bps"].shape:
        raise RuntimeError("BENCHMARK_OUTPUT_SHAPE_SPLIT_BRAIN")
    print(
        json.dumps(
            {
                "schema_version": "gx1_unified_exit_episode_benchmark_v1",
                "synthetic_production_shape": True,
                "batch": batch,
                "device": str(device),
                "states_per_side": UNIFIED_EXIT_MAX_PATH_BARS,
                "sides": 2,
                "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
                "mtf_dim": tf_width,
                "per_tf_sequence_lengths": TF_LENGTHS,
                "materialize_seconds": materialize_seconds,
                "target_forward_seconds": target_seconds,
                "online_forward_backward_seconds": online_forward_backward_seconds,
                "optimizer_step_seconds": optimizer_step_seconds,
                "total_seconds": materialize_seconds
                + target_seconds
                + online_forward_backward_seconds
                + optimizer_step_seconds,
                "input_bytes": input_bytes,
                "peak_rss_bytes": _rss_bytes(),
                "peak_cuda_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated())
                    if device.type == "cuda"
                    else 0
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
