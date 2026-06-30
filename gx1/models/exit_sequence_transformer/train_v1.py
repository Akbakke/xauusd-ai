#!/usr/bin/env python3
"""Preflight-only active Exit sequence Transformer trainer core.

This module defines the active Exit Transformer model contract and a small
preflight forward-pass path. It deliberately does not run optimizer steps or
write train bundles until a later train-execution gate enables that path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


READY_TRAINING_PLAN_DECISION = "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READY_FOR_VEDTAK_REVIEW"
EXPECTED_HEADS = (
    "exit_now_logit",
    "hold_value_bps",
    "exit_now_reward_bps",
    "giveback_risk_bps",
    "mfe_capture_ratio",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


class ExitSequenceTransformerV1(nn.Module):
    """Causal masked sequence Transformer with exact active Exit heads."""

    def __init__(self, training_plan: dict[str, Any], normalization: dict[str, Any]) -> None:
        super().__init__()
        architecture = training_plan.get("architecture") if isinstance(training_plan.get("architecture"), dict) else {}
        encoder = architecture.get("encoder") if isinstance(architecture.get("encoder"), dict) else {}
        self.numeric_features = list(architecture.get("numeric_state_features") or [])
        self.categorical_features = list(architecture.get("categorical_state_features") or [])
        self.output_heads = list(architecture.get("output_heads") or [])
        if self.output_heads != list(EXPECTED_HEADS):
            raise ValueError(f"unexpected output heads: {self.output_heads}")
        self.max_sequence_length = int(architecture.get("planned_max_sequence_length") or 0)
        if self.max_sequence_length <= 0:
            raise ValueError("planned_max_sequence_length must be positive")
        d_model = int(encoder.get("d_model") or 128)
        n_heads = int(encoder.get("n_heads") or 4)
        n_layers = int(encoder.get("num_layers") or 3)
        dim_feedforward = int(encoder.get("dim_feedforward") or 256)
        dropout = float(encoder.get("dropout") or 0.10)
        if d_model % n_heads != 0:
            raise ValueError("d_model must divide n_heads")
        if encoder.get("causal_mask_required") is not True:
            raise ValueError("causal_mask_required must be true")

        categorical_vocab = normalization.get("categorical_vocab") if isinstance(normalization.get("categorical_vocab"), dict) else {}
        input_dim = len(self.numeric_features)
        self.categorical_embeddings = nn.ModuleDict()
        self.categorical_vocab_sizes: dict[str, int] = {}
        for field in self.categorical_features:
            vocab = categorical_vocab.get(field) if isinstance(categorical_vocab.get(field), list) else []
            vocab_size = len(vocab) + 1
            emb_dim = int(min(8, max(2, vocab_size)))
            self.categorical_vocab_sizes[field] = vocab_size
            self.categorical_embeddings[field] = nn.Embedding(vocab_size, emb_dim)
            input_dim += emb_dim
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = nn.Embedding(self.max_sequence_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.heads = nn.ModuleDict({head: nn.Linear(d_model, 1) for head in self.output_heads})

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: dict[str, torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, seq_len, _ = numeric.shape
        pieces = [numeric.float()]
        for field in self.categorical_features:
            ids = categorical[field].long().clamp(min=0, max=self.categorical_vocab_sizes[field] - 1)
            pieces.append(self.categorical_embeddings[field](ids))
        x = torch.cat(pieces, dim=-1)
        positions = torch.arange(seq_len, device=x.device).view(1, seq_len).expand(batch, seq_len)
        x = self.input_projection(x) + self.position(positions)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        encoded = self.encoder(x, mask=causal_mask, src_key_padding_mask=~valid_mask.bool())
        return {name: head(encoded).squeeze(-1) for name, head in self.heads.items()}


def load_training_plan_report(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _read_json(path)
    if report.get("decision") != READY_TRAINING_PLAN_DECISION:
        raise ValueError(f"training plan decision is not ready: {report.get('decision')}")
    plan = report.get("training_plan") if isinstance(report.get("training_plan"), dict) else {}
    if plan.get("model_family") != "exit_sequence_transformer_v1":
        raise ValueError(f"unexpected model family: {plan.get('model_family')}")
    return report, plan


def load_normalization(training_plan: dict[str, Any]) -> dict[str, Any]:
    dataset = training_plan.get("dataset") if isinstance(training_plan.get("dataset"), dict) else {}
    path = Path(str(dataset.get("normalization_json") or "")).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"normalization_json missing: {path}")
    normalization = _read_json(path)
    if "train_split_only" not in str(normalization.get("normalization_policy") or ""):
        raise ValueError("normalization policy must be train split only")
    return normalization


def _split_path(training_plan: dict[str, Any], split: str) -> Path:
    dataset = training_plan.get("dataset") if isinstance(training_plan.get("dataset"), dict) else {}
    shards = dataset.get("shards") if isinstance(dataset.get("shards"), dict) else {}
    path = Path(str(shards.get(split) or "")).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"model dataset shard missing for {split}: {path}")
    return path


def _normalise_numeric(frame: pd.DataFrame, fields: list[str], normalization: dict[str, Any]) -> np.ndarray:
    numeric_norm = normalization.get("numeric") if isinstance(normalization.get("numeric"), dict) else {}
    columns: list[np.ndarray] = []
    for field in fields:
        stats = numeric_norm.get(field) if isinstance(numeric_norm.get(field), dict) else {}
        mean = float(stats.get("mean") or 0.0)
        std = float(stats.get("std") or 1.0)
        if not np.isfinite(std) or abs(std) < 1e-9:
            std = 1.0
        values = pd.to_numeric(frame[field], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(mean)
        columns.append(((values.to_numpy(dtype=np.float32) - mean) / std).astype(np.float32))
    return np.stack(columns, axis=1) if columns else np.zeros((len(frame), 0), dtype=np.float32)


def _categorical_ids(frame: pd.DataFrame, fields: list[str], normalization: dict[str, Any]) -> dict[str, np.ndarray]:
    categorical_vocab = normalization.get("categorical_vocab") if isinstance(normalization.get("categorical_vocab"), dict) else {}
    out: dict[str, np.ndarray] = {}
    for field in fields:
        vocab = [str(value) for value in categorical_vocab.get(field, [])]
        mapping = {value: idx for idx, value in enumerate(vocab)}
        oov = len(vocab)
        out[field] = frame[field].astype(str).map(lambda value: mapping.get(value, oov)).to_numpy(dtype=np.int64)
    return out


def prepare_preflight_batch(
    training_plan: dict[str, Any],
    normalization: dict[str, Any],
    *,
    split: str = "train",
    max_episodes: int = 4,
) -> dict[str, Any]:
    architecture = training_plan.get("architecture") if isinstance(training_plan.get("architecture"), dict) else {}
    numeric_features = list(architecture.get("numeric_state_features") or [])
    categorical_features = list(architecture.get("categorical_state_features") or [])
    max_len = int(architecture.get("planned_max_sequence_length") or 0)
    shard = _split_path(training_plan, split)
    frame = pd.read_csv(shard, low_memory=False)
    frame = frame.sort_values(["exit_episode_id", "exit_timestep"], kind="mergesort")
    episode_ids = list(dict.fromkeys(frame["exit_episode_id"].astype(str).tolist()))[:max_episodes]
    if not episode_ids:
        raise ValueError(f"no episodes found in {shard}")
    numeric = np.zeros((len(episode_ids), max_len, len(numeric_features)), dtype=np.float32)
    categorical = {
        field: np.zeros((len(episode_ids), max_len), dtype=np.int64)
        for field in categorical_features
    }
    valid = np.zeros((len(episode_ids), max_len), dtype=bool)
    lengths: list[int] = []
    for row_idx, episode_id in enumerate(episode_ids):
        group = frame[frame["exit_episode_id"].astype(str) == episode_id].head(max_len)
        lengths.append(int(len(group)))
        norm_numeric = _normalise_numeric(group, numeric_features, normalization)
        cat_ids = _categorical_ids(group, categorical_features, normalization)
        seq_len = len(group)
        numeric[row_idx, :seq_len, :] = norm_numeric
        for field in categorical_features:
            categorical[field][row_idx, :seq_len] = cat_ids[field]
        valid[row_idx, :seq_len] = True
    return {
        "split": split,
        "shard": str(shard),
        "episode_ids": episode_ids,
        "lengths": lengths,
        "numeric": torch.from_numpy(numeric),
        "categorical": {field: torch.from_numpy(values) for field, values in categorical.items()},
        "valid_mask": torch.from_numpy(valid),
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(1)
    training_plan_json = Path(args.training_plan_json).expanduser().resolve()
    report, training_plan = load_training_plan_report(training_plan_json)
    normalization = load_normalization(training_plan)
    device = torch.device(args.device)
    batch = prepare_preflight_batch(
        training_plan,
        normalization,
        split=str(args.split),
        max_episodes=int(args.max_episodes),
    )
    model = ExitSequenceTransformerV1(training_plan, normalization).to(device)
    model.eval()
    numeric = batch["numeric"].to(device)
    categorical = {field: values.to(device) for field, values in batch["categorical"].items()}
    valid_mask = batch["valid_mask"].to(device)
    with torch.no_grad():
        outputs = model(numeric, categorical, valid_mask)
    finite_by_head = {
        head: bool(torch.isfinite(value[valid_mask]).all().item())
        for head, value in outputs.items()
    }
    output_shapes = {head: list(value.shape) for head, value in outputs.items()}
    parameter_count = int(sum(param.numel() for param in model.parameters()))
    ready = (
        list(outputs) == list(EXPECTED_HEADS)
        and all(finite_by_head.values())
        and parameter_count > 0
        and bool(valid_mask.any().item())
    )
    manifest = {
        "schema_version": "entry_exit_transformer_pretrain_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS" if ready else "FAIL",
        "training_plan_json": str(training_plan_json),
        "training_plan_json_sha256": _sha256_file(training_plan_json),
        "model_family": training_plan.get("model_family"),
        "output_heads": list(outputs),
        "expected_output_heads": list(EXPECTED_HEADS),
        "finite_by_head": finite_by_head,
        "output_shapes": output_shapes,
        "parameter_count": parameter_count,
        "preflight_batch": {
            "split": batch["split"],
            "shard": batch["shard"],
            "episode_count": len(batch["episode_ids"]),
            "lengths": batch["lengths"],
            "numeric_shape": list(numeric.shape),
            "valid_token_count": int(valid_mask.sum().item()),
        },
        "resource_guardrails": (training_plan.get("resource_guardrails") if isinstance(training_plan.get("resource_guardrails"), dict) else {}),
        "trainer_started": False,
        "optimizer_steps": 0,
        "replay_started": False,
        "iql_distillation_started": False,
        "promotion_shadow_live_allowed": False,
        "exit_training_allowed": False,
    }
    out = str(args.out_manifest_json or "").strip()
    if out:
        out_path = Path(out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-plan-json", required=True)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--out-manifest-json", default="")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-episodes", type=int, default=4)
    ap.add_argument("--device", default="cpu", choices=("cpu",))
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if not args.preflight_only:
        raise SystemExit("FATAL: active Exit Transformer training is not enabled; use --preflight-only for audit.")
    manifest = run_preflight(args)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default))
    return 0 if manifest.get("decision") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
