#!/usr/bin/env python3
"""Capped synthetic MTF cost probe for current versus proposed capacities."""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

from gx1.contracts.entry_mtf_temporal_receptive_field_policy_v1 import (
    PROPOSED_MINIMUM_WINDOW_BARS,
    TIMEFRAME_ORDER,
    canonical_json_sha256,
    current_window_observation,
    temporal_receptive_field_policy,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    require_multi_tf_specialist_routing_v4,
)
from gx1.features.htf_features import (
    MULTI_TF_FEATURE_COUNT_V4,
    MULTI_TF_PER_BAR_FEATURES_V4,
)


SCHEMA_VERSION = "gx1_entry_mtf_temporal_synthetic_cost_benchmark_v1"
DECISION = "DIAGNOSTIC_COST_MEASUREMENT_NO_MODEL_OR_TRADING_AUTHORITY"


def _current_rss_kib() -> int:
    with open("/proc/self/status", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    raise RuntimeError("TEMPORAL_BENCHMARK_RSS_UNAVAILABLE")


class _SyntheticMtfFamilyEncoder(nn.Module):
    """Shape-faithful family/TF Transformer cost surface, not a candidate."""

    def __init__(
        self,
        *,
        routing: Mapping[str, tuple[int, ...]],
        d_model: int,
        n_heads: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.family_order = tuple(routing)
        self.projections = nn.ModuleDict(
            {
                family: nn.Linear(len(indices), d_model)
                for family, indices in routing.items()
            }
        )
        self.encoders = nn.ModuleDict()
        for family in self.family_order:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoders[family] = nn.TransformerEncoder(
                layer,
                num_layers=layers,
                enable_nested_tensor=False,
            )
            self.register_buffer(
                f"indices_{family}",
                torch.tensor(routing[family], dtype=torch.long),
                persistent=False,
            )
        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 3),
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tokens: list[torch.Tensor] = []
        for timeframe in TIMEFRAME_ORDER:
            source = inputs[timeframe]
            for family in self.family_order:
                indices = getattr(self, f"indices_{family}")
                encoded = self.encoders[family](
                    self.projections[family](source.index_select(-1, indices))
                )
                tokens.append(encoded[:, -1, :])
        return self.readout(torch.stack(tokens, dim=1).mean(dim=1))


def benchmark_temporal_profile(
    *,
    profile: str,
    phase: str,
    batch_size: int,
    warmup_iterations: int,
    measured_iterations: int,
    d_model: int,
    n_heads: int,
    layers: int,
) -> dict[str, Any]:
    if profile not in {"current", "proposed"} or phase not in {"forward", "training"}:
        raise RuntimeError("TEMPORAL_BENCHMARK_PROFILE_OR_PHASE_INVALID")
    if (
        batch_size < 1
        or warmup_iterations < 0
        or measured_iterations < 1
        or d_model < 4
        or n_heads < 1
        or d_model % n_heads != 0
        or layers < 1
    ):
        raise RuntimeError("TEMPORAL_BENCHMARK_CONFIG_INVALID")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.manual_seed(20260814)
    np.random.seed(20260814)
    windows = (
        current_window_observation()
        if profile == "current"
        else dict(PROPOSED_MINIMUM_WINDOW_BARS)
    )
    routing = {
        family: tuple(int(index) for index in indices)
        for family, indices in require_multi_tf_specialist_routing_v4(
            MULTI_TF_PER_BAR_FEATURES_V4
        ).items()
    }
    if sum(len(indices) for indices in routing.values()) != MULTI_TF_FEATURE_COUNT_V4:
        raise RuntimeError("TEMPORAL_BENCHMARK_ROUTING_PARTITION_INVALID")
    model = _SyntheticMtfFamilyEncoder(
        routing=routing,
        d_model=d_model,
        n_heads=n_heads,
        layers=layers,
    )
    inputs = {
        timeframe: torch.randn(
            batch_size,
            windows[timeframe],
            MULTI_TF_FEATURE_COUNT_V4,
            dtype=torch.float32,
        )
        for timeframe in TIMEFRAME_ORDER
    }
    target = torch.zeros((batch_size, 3), dtype=torch.float32)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    baseline_rss_kib = _current_rss_kib()

    def one_iteration() -> float:
        if phase == "forward":
            model.eval()
            with torch.inference_mode():
                output = model(inputs)
            return float(output.square().mean().item())
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        return float(loss.detach().item())

    for _ in range(warmup_iterations):
        one_iteration()
    durations: list[float] = []
    terminal_value = 0.0
    for _ in range(measured_iterations):
        started = time.perf_counter()
        terminal_value = one_iteration()
        durations.append(time.perf_counter() - started)
    peak_rss_kib = max(
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        _current_rss_kib(),
        baseline_rss_kib,
    )
    input_bytes = sum(tensor.numel() * tensor.element_size() for tensor in inputs.values())
    family_count = len(routing)
    token_cells = batch_size * family_count * sum(windows.values())
    attention_score_cells = (
        batch_size
        * family_count
        * n_heads
        * layers
        * sum(length * length for length in windows.values())
    )
    policy = temporal_receptive_field_policy()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "policy_contract_sha256": policy["contract_sha256"],
        "profile": profile,
        "phase": phase,
        "device": "cpu",
        "windows": windows,
        "config": {
            "batch_size": batch_size,
            "warmup_iterations": warmup_iterations,
            "measured_iterations": measured_iterations,
            "input_width": MULTI_TF_FEATURE_COUNT_V4,
            "family_count": family_count,
            "d_model": d_model,
            "n_heads": n_heads,
            "layers": layers,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "thread_count": torch.get_num_threads(),
        },
        "workload": {
            "input_bytes": input_bytes,
            "family_token_cells_per_iteration": token_cells,
            "attention_score_cells_per_iteration": attention_score_cells,
        },
        "measurement": {
            "wall_seconds": durations,
            "wall_seconds_mean": math.fsum(durations) / len(durations),
            "wall_seconds_min": min(durations),
            "wall_seconds_max": max(durations),
            "baseline_rss_kib": baseline_rss_kib,
            "peak_rss_kib": peak_rss_kib,
            "incremental_peak_rss_kib": max(peak_rss_kib - baseline_rss_kib, 0),
            "terminal_scalar": terminal_value,
        },
        "authority": {
            "synthetic_cost_diagnostic_only": True,
            "candidate_quality_claim": False,
            "architecture_selection_authority": False,
            "trading_decision_authority": False,
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("current", "proposed"), required=True)
    parser.add_argument("--phase", choices=("forward", "training"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measured-iterations", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    args = parser.parse_args()
    report = benchmark_temporal_profile(
        profile=args.profile,
        phase=args.phase,
        batch_size=args.batch_size,
        warmup_iterations=args.warmup_iterations,
        measured_iterations=args.measured_iterations,
        d_model=args.d_model,
        n_heads=args.n_heads,
        layers=args.layers,
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = (
    "DECISION",
    "SCHEMA_VERSION",
    "benchmark_temporal_profile",
)
