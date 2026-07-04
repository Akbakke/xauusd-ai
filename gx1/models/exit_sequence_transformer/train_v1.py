#!/usr/bin/env python3
"""Active Exit sequence Transformer trainer core.

This module defines the active Exit Transformer model contract, a preflight
forward-pass path and the supervised train loop used after explicit
train-execution enablement. The CLI remains fail-closed unless
``--enable-training`` and a valid ``ENTRY_EXIT_TRANSFORMER_TRAIN_`` vedtak are
provided. It never replays, distills IQL, promotes, shadows or touches live
paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


READY_TRAINING_PLAN_DECISION = "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READY_FOR_VEDTAK_REVIEW"
READY_TRAIN_EXECUTION_REVIEW_DECISION = "ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_READY_FOR_EXPLICIT_VEDTAK_PACKAGE"
READY_POST_TRAIN_CONTRACT_DECISION = "ENTRY_EXIT_TRANSFORMER_POST_TRAIN_AUDIT_CONTRACT_READY"
READY_FEATURE_ALIGNMENT_DECISION = "ENTRY_EXIT_FEATURE_ALIGNMENT_READY_FOR_EXIT_TRANSFORMER_TRAINING_REVIEW"
TRAINING_COMPLETE_DECISION = "ENTRY_EXIT_TRANSFORMER_TRAINING_RUN_COMPLETE"
VEDTAK_PREFIX = "ENTRY_EXIT_TRANSFORMER_TRAIN_"
EXPECTED_HEADS = (
    "exit_now_logit",
    "hold_value_bps",
    "exit_now_reward_bps",
    "giveback_risk_bps",
    "mfe_capture_ratio",
)
TARGET_COLUMNS_BY_HEAD = {
    "exit_now_logit": "exit_now_label",
    "hold_value_bps": "future_best_exit_lift_bps",
    "exit_now_reward_bps": "exit_now_reward_bps",
    "giveback_risk_bps": "future_giveback_from_peak_bps",
    "mfe_capture_ratio": "exit_now_mfe_capture_ratio_reward",
}
# ---------------------------------------------------------------------------
# Exit-parity tuning knobs (2026-07-04, exit-parity wave — mirrors the entry
# smart520 recipe surface). ALL defaults equal the previous hardcoded values,
# so behaviour is byte-identical until a knob is explicitly set. Every value
# used lands in the training summary for the pretrain-manifest/audit chain.
#   EXIT_LOSS_W_*            : multi-objective reward mix (was hardcoded
#                              1.0 / 0.30 / 0.30 / 0.20 / 0.20)
#   EXIT_POS_WEIGHT_MIN/MAX  : exit-now BCE class-balance clamp (was 1 / 10)
#   EXIT_PRED_BALANCE_ALPHA  : exit-now prediction-rate balance penalty
#                              (entry FLAT-repair analogue for the binary exit
#                              head; 0.0 = off)
#   EXIT_REWARD_RANK_WEIGHT  : pairwise margin rank loss on exit_now_reward —
#                              learn to RANK bars by exit-favourability so the
#                              policy exits at tops, not just regress
#                              magnitudes (entry path-quality-rank analogue;
#                              0.0 = off)
#   EXIT_REWARD_RANK_MARGIN  : margin for the rank loss (scaled units)
#   EXIT_CKPT_MONITOR        : best-checkpoint metric: val_loss (default) |
#                              exit_now_accuracy (load-bearing analogue of
#                              entry's dir_acc monitor)
import os as _os

def _env_float(name: str, default: float) -> float:
    raw = str(_os.environ.get(name, "") or "").strip()
    return float(raw) if raw else float(default)

EXIT_LOSS_W_EXIT_NOW_BCE = _env_float("EXIT_LOSS_W_EXIT_NOW_BCE", 1.0)
EXIT_LOSS_W_HOLD_VALUE = _env_float("EXIT_LOSS_W_HOLD_VALUE", 0.30)
EXIT_LOSS_W_EXIT_NOW_REWARD = _env_float("EXIT_LOSS_W_EXIT_NOW_REWARD", 0.30)
EXIT_LOSS_W_GIVEBACK = _env_float("EXIT_LOSS_W_GIVEBACK", 0.20)
EXIT_LOSS_W_MFE_CAPTURE = _env_float("EXIT_LOSS_W_MFE_CAPTURE", 0.20)
EXIT_POS_WEIGHT_MIN = _env_float("EXIT_POS_WEIGHT_MIN", 1.0)
EXIT_POS_WEIGHT_MAX = _env_float("EXIT_POS_WEIGHT_MAX", 10.0)
EXIT_PRED_BALANCE_ALPHA = _env_float("EXIT_PRED_BALANCE_ALPHA", 0.0)
EXIT_REWARD_RANK_WEIGHT = _env_float("EXIT_REWARD_RANK_WEIGHT", 0.0)
EXIT_REWARD_RANK_MARGIN = _env_float("EXIT_REWARD_RANK_MARGIN", 0.10)
EXIT_CKPT_MONITOR = str(_os.environ.get("EXIT_CKPT_MONITOR", "val_loss") or "val_loss").strip()
EXIT_PARITY_KNOBS = {
    "EXIT_LOSS_W_EXIT_NOW_BCE": EXIT_LOSS_W_EXIT_NOW_BCE,
    "EXIT_LOSS_W_HOLD_VALUE": EXIT_LOSS_W_HOLD_VALUE,
    "EXIT_LOSS_W_EXIT_NOW_REWARD": EXIT_LOSS_W_EXIT_NOW_REWARD,
    "EXIT_LOSS_W_GIVEBACK": EXIT_LOSS_W_GIVEBACK,
    "EXIT_LOSS_W_MFE_CAPTURE": EXIT_LOSS_W_MFE_CAPTURE,
    "EXIT_POS_WEIGHT_MIN": EXIT_POS_WEIGHT_MIN,
    "EXIT_POS_WEIGHT_MAX": EXIT_POS_WEIGHT_MAX,
    "EXIT_PRED_BALANCE_ALPHA": EXIT_PRED_BALANCE_ALPHA,
    "EXIT_REWARD_RANK_WEIGHT": EXIT_REWARD_RANK_WEIGHT,
    "EXIT_REWARD_RANK_MARGIN": EXIT_REWARD_RANK_MARGIN,
    "EXIT_CKPT_MONITOR": EXIT_CKPT_MONITOR,
}

LOSS_TARGET_SCALE = {
    "hold_value_bps": 100.0,
    "exit_now_reward_bps": 100.0,
    "giveback_risk_bps": 100.0,
    "mfe_capture_ratio": 1.0,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _read_json(path)


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


def _resolve_device(raw: str) -> torch.device:
    requested = str(raw or "cpu").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def _require_ready_report(path: Path, required_decision: str, label: str) -> dict[str, Any]:
    if not str(path):
        raise ValueError(f"{label} path is required")
    report = _read_json_or_empty(path)
    if report.get("decision") != required_decision:
        raise ValueError(f"{label} decision is not ready: observed={report.get('decision')} required={required_decision}")
    return report


def _validate_training_enablement(args: argparse.Namespace) -> dict[str, Any]:
    vedtak = str(getattr(args, "vedtak", "") or "")
    if not bool(getattr(args, "enable_training", False)):
        raise ValueError("active Exit Transformer training is disabled; pass --enable-training only after explicit train enablement")
    if not vedtak.startswith(VEDTAK_PREFIX):
        raise ValueError(f"--vedtak must start with {VEDTAK_PREFIX}")
    if int(getattr(args, "num_workers", 0) or 0) != 0:
        raise ValueError("active Exit Transformer training requires --num-workers 0")
    train_execution_review_json = Path(str(getattr(args, "train_execution_review_json", "") or "")).expanduser().resolve()
    post_train_contract_json = Path(str(getattr(args, "post_train_contract_json", "") or "")).expanduser().resolve()
    feature_alignment_json = Path(str(getattr(args, "feature_alignment_json", "") or "")).expanduser().resolve()
    train_execution_review = _require_ready_report(
        train_execution_review_json,
        READY_TRAIN_EXECUTION_REVIEW_DECISION,
        "train-execution review",
    )
    post_train_contract = _require_ready_report(
        post_train_contract_json,
        READY_POST_TRAIN_CONTRACT_DECISION,
        "post-train audit contract",
    )
    feature_alignment = _require_ready_report(
        feature_alignment_json,
        READY_FEATURE_ALIGNMENT_DECISION,
        "Entry-to-Exit feature alignment",
    )
    for label, report in (
        ("train-execution review", train_execution_review),
        ("post-train audit contract", post_train_contract),
        ("Entry-to-Exit feature alignment", feature_alignment),
    ):
        if report.get("promotion_shadow_live_allowed") not in (False, None):
            raise ValueError(f"{label} must not allow shadow/live promotion")
    return {
        "vedtak": vedtak,
        "train_execution_review_json": str(train_execution_review_json),
        "train_execution_review_json_sha256": _sha256_file(train_execution_review_json),
        "post_train_contract_json": str(post_train_contract_json),
        "post_train_contract_json_sha256": _sha256_file(post_train_contract_json),
        "feature_alignment_json": str(feature_alignment_json),
        "feature_alignment_json_sha256": _sha256_file(feature_alignment_json),
    }


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


def _target_values(frame: pd.DataFrame, field: str, *, default: float = 0.0) -> np.ndarray:
    if field not in frame.columns:
        return np.full(len(frame), float(default), dtype=np.float32)
    if field == "exit_now_label":
        values = frame[field].astype(str).str.lower().isin({"true", "1", "yes"}).astype(np.float32)
        return values.to_numpy(dtype=np.float32)
    values = pd.to_numeric(frame[field], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return values.to_numpy(dtype=np.float32)


def prepare_supervised_batch(
    training_plan: dict[str, Any],
    normalization: dict[str, Any],
    *,
    split: str,
    max_episodes: int = 0,
) -> dict[str, Any]:
    architecture = training_plan.get("architecture") if isinstance(training_plan.get("architecture"), dict) else {}
    numeric_features = list(architecture.get("numeric_state_features") or [])
    categorical_features = list(architecture.get("categorical_state_features") or [])
    max_len = int(architecture.get("planned_max_sequence_length") or 0)
    shard = _split_path(training_plan, split)
    frame = pd.read_csv(shard, low_memory=False)
    frame = frame.sort_values(["exit_episode_id", "exit_timestep"], kind="mergesort")
    episode_ids = list(dict.fromkeys(frame["exit_episode_id"].astype(str).tolist()))
    if max_episodes and max_episodes > 0:
        episode_ids = episode_ids[: int(max_episodes)]
    if not episode_ids:
        raise ValueError(f"no episodes found in {shard}")
    numeric = np.zeros((len(episode_ids), max_len, len(numeric_features)), dtype=np.float32)
    categorical = {
        field: np.zeros((len(episode_ids), max_len), dtype=np.int64)
        for field in categorical_features
    }
    targets = {
        head: np.zeros((len(episode_ids), max_len), dtype=np.float32)
        for head in EXPECTED_HEADS
    }
    valid = np.zeros((len(episode_ids), max_len), dtype=bool)
    lengths: list[int] = []
    row_counts: list[int] = []
    for row_idx, episode_id in enumerate(episode_ids):
        group = frame[frame["exit_episode_id"].astype(str) == episode_id].head(max_len)
        lengths.append(int(len(group)))
        row_counts.append(int(len(group)))
        norm_numeric = _normalise_numeric(group, numeric_features, normalization)
        cat_ids = _categorical_ids(group, categorical_features, normalization)
        seq_len = len(group)
        numeric[row_idx, :seq_len, :] = norm_numeric
        for field in categorical_features:
            categorical[field][row_idx, :seq_len] = cat_ids[field]
        for head, column in TARGET_COLUMNS_BY_HEAD.items():
            targets[head][row_idx, :seq_len] = _target_values(group, column)
        valid[row_idx, :seq_len] = True
    return {
        "split": split,
        "shard": str(shard),
        "episode_ids": episode_ids,
        "lengths": lengths,
        "row_count": int(sum(row_counts)),
        "numeric": torch.from_numpy(numeric),
        "categorical": {field: torch.from_numpy(values) for field, values in categorical.items()},
        "targets": {head: torch.from_numpy(values) for head, values in targets.items()},
        "valid_mask": torch.from_numpy(valid),
    }


def _slice_batch(batch: dict[str, Any], indices: np.ndarray, device: torch.device) -> dict[str, Any]:
    idx = torch.as_tensor(indices, dtype=torch.long)
    return {
        "numeric": batch["numeric"][idx].to(device),
        "categorical": {field: values[idx].to(device) for field, values in batch["categorical"].items()},
        "targets": {head: values[idx].to(device) for head, values in batch["targets"].items()},
        "valid_mask": batch["valid_mask"][idx].to(device),
    }


def _supervised_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], valid_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    mask = valid_mask.bool()
    labels = targets["exit_now_logit"][mask].float()
    logits = outputs["exit_now_logit"][mask]
    pos = torch.clamp(labels.sum(), min=1.0)
    neg = torch.clamp(torch.tensor(float(labels.numel()), device=labels.device) - labels.sum(), min=1.0)
    pos_weight = torch.clamp(neg / pos, min=float(EXIT_POS_WEIGHT_MIN), max=float(EXIT_POS_WEIGHT_MAX))
    loss_terms: dict[str, torch.Tensor] = {
        "exit_now_logit_bce": F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    }
    for head in ("hold_value_bps", "exit_now_reward_bps", "giveback_risk_bps", "mfe_capture_ratio"):
        scale = float(LOSS_TARGET_SCALE[head])
        loss_terms[f"{head}_huber"] = F.smooth_l1_loss(outputs[head][mask] / scale, targets[head][mask] / scale)
    # Exit-now prediction-rate balance penalty (entry FLAT-repair analogue for
    # the binary exit head; 0.0 = off = prior behaviour).
    if EXIT_PRED_BALANCE_ALPHA > 0.0:
        pred_rate = torch.sigmoid(logits).mean()
        label_rate = labels.mean()
        loss_terms["exit_now_pred_balance"] = (pred_rate - label_rate).abs()
    # Pairwise margin rank loss on exit_now_reward: learn to RANK bars by
    # exit-favourability (exit at tops), not just regress magnitudes.
    if EXIT_REWARD_RANK_WEIGHT > 0.0:
        scale = float(LOSS_TARGET_SCALE["exit_now_reward_bps"])
        pred_r = outputs["exit_now_reward_bps"][mask] / scale
        true_r = targets["exit_now_reward_bps"][mask] / scale
        n = pred_r.numel()
        if n >= 2:
            # DETERMINISTIC pairing (reverse-index), not torch.randperm: this loss
            # term is also evaluated inside _evaluate_model, so a random pairing
            # would make the val-loss-monitored best-checkpoint selection
            # non-reproducible. Reverse pairs each sample with a distant partner;
            # training-batch shuffling still varies the pairs across epochs.
            perm = torch.arange(n - 1, -1, -1, device=pred_r.device)
            a, b = pred_r, pred_r[perm]
            ta, tb = true_r, true_r[perm]
            sign = torch.sign(ta - tb)
            valid_pairs = sign != 0
            if bool(valid_pairs.any()):
                rank = torch.clamp(
                    float(EXIT_REWARD_RANK_MARGIN) - sign[valid_pairs] * (a - b)[valid_pairs],
                    min=0.0,
                ).mean()
                loss_terms["exit_now_reward_rank"] = rank
    total = (
        EXIT_LOSS_W_EXIT_NOW_BCE * loss_terms["exit_now_logit_bce"]
        + EXIT_LOSS_W_HOLD_VALUE * loss_terms["hold_value_bps_huber"]
        + EXIT_LOSS_W_EXIT_NOW_REWARD * loss_terms["exit_now_reward_bps_huber"]
        + EXIT_LOSS_W_GIVEBACK * loss_terms["giveback_risk_bps_huber"]
        + EXIT_LOSS_W_MFE_CAPTURE * loss_terms["mfe_capture_ratio_huber"]
    )
    if "exit_now_pred_balance" in loss_terms:
        total = total + EXIT_PRED_BALANCE_ALPHA * loss_terms["exit_now_pred_balance"]
    if "exit_now_reward_rank" in loss_terms:
        total = total + EXIT_REWARD_RANK_WEIGHT * loss_terms["exit_now_reward_rank"]
    return total, {name: float(value.detach().cpu().item()) for name, value in loss_terms.items()}


def _evaluate_model(model: ExitSequenceTransformerV1, batch: dict[str, Any], *, batch_size: int, device: torch.device) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    term_sums: dict[str, float] = {}
    token_count = 0
    exit_probs: list[np.ndarray] = []
    exit_labels: list[np.ndarray] = []
    mae_sums = {head: 0.0 for head in ("hold_value_bps", "exit_now_reward_bps", "giveback_risk_bps", "mfe_capture_ratio")}
    finite_by_head = {head: True for head in EXPECTED_HEADS}
    with torch.no_grad():
        indices = np.arange(len(batch["episode_ids"]))
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            item = _slice_batch(batch, chunk, device)
            outputs = model(item["numeric"], item["categorical"], item["valid_mask"])
            loss, terms = _supervised_loss(outputs, item["targets"], item["valid_mask"])
            mask = item["valid_mask"].bool()
            count = int(mask.sum().item())
            token_count += count
            losses.append(float(loss.detach().cpu().item()) * count)
            for name, value in terms.items():
                term_sums[name] = term_sums.get(name, 0.0) + value * count
            probs = torch.sigmoid(outputs["exit_now_logit"][mask]).detach().cpu().numpy()
            labels = item["targets"]["exit_now_logit"][mask].detach().cpu().numpy()
            exit_probs.append(probs)
            exit_labels.append(labels)
            for head in mae_sums:
                pred = outputs[head][mask]
                target = item["targets"][head][mask]
                mae_sums[head] += float(torch.abs(pred - target).sum().detach().cpu().item())
            for head, value in outputs.items():
                finite_by_head[head] = bool(finite_by_head[head] and torch.isfinite(value[mask]).all().item())
    probs_all = np.concatenate(exit_probs) if exit_probs else np.zeros(0, dtype=np.float32)
    labels_all = np.concatenate(exit_labels) if exit_labels else np.zeros(0, dtype=np.float32)
    pred_labels = probs_all >= 0.5
    accuracy = float((pred_labels == (labels_all >= 0.5)).mean()) if len(labels_all) else 0.0
    return {
        "loss": float(sum(losses) / max(token_count, 1)),
        "loss_terms": {name: float(value / max(token_count, 1)) for name, value in term_sums.items()},
        "token_count": int(token_count),
        "episode_count": int(len(batch["episode_ids"])),
        "exit_now_accuracy": accuracy,
        "exit_now_label_rate": float(labels_all.mean()) if len(labels_all) else 0.0,
        "exit_now_pred_rate": float(pred_labels.mean()) if len(pred_labels) else 0.0,
        "exit_now_prob_mean": float(probs_all.mean()) if len(probs_all) else 0.0,
        "exit_now_prob_std": float(probs_all.std()) if len(probs_all) else 0.0,
        "mae": {head: float(value / max(token_count, 1)) for head, value in mae_sums.items()},
        "finite_by_head": finite_by_head,
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


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    enablement = _validate_training_enablement(args)
    seed = int(getattr(args, "seed", 1337) or 1337)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    training_plan_json = Path(args.training_plan_json).expanduser().resolve()
    report, training_plan = load_training_plan_report(training_plan_json)
    normalization = load_normalization(training_plan)
    device = _resolve_device(str(args.device))
    epochs = int(getattr(args, "epochs", 1) or 1)
    batch_size = int(getattr(args, "batch_size", 32) or 32)
    if epochs <= 0:
        raise ValueError("--epochs must be positive")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    max_train_episodes = int(getattr(args, "max_train_episodes", 0) or 0)
    max_eval_episodes = int(getattr(args, "max_eval_episodes", 0) or 0)
    train_batch = prepare_supervised_batch(
        training_plan,
        normalization,
        split="train",
        max_episodes=max_train_episodes,
    )
    val_batch = prepare_supervised_batch(
        training_plan,
        normalization,
        split="val",
        max_episodes=max_eval_episodes,
    )
    test_batch = prepare_supervised_batch(
        training_plan,
        normalization,
        split="test",
        max_episodes=max_eval_episodes,
    )
    model = ExitSequenceTransformerV1(training_plan, normalization).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(getattr(args, "lr", 3e-4) or 3e-4),
        weight_decay=float(getattr(args, "weight_decay", 0.01) or 0.01),
    )
    clip_grad_norm = float(getattr(args, "clip_grad_norm", 1.0) or 1.0)
    optimizer_steps = 0
    epoch_metrics: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    rng = np.random.default_rng(seed)
    train_indices = np.arange(len(train_batch["episode_ids"]))
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(train_indices)
        train_loss_sum = 0.0
        train_token_count = 0
        train_term_sums: dict[str, float] = {}
        for start in range(0, len(train_indices), batch_size):
            chunk = train_indices[start : start + batch_size]
            item = _slice_batch(train_batch, chunk, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(item["numeric"], item["categorical"], item["valid_mask"])
            loss, terms = _supervised_loss(outputs, item["targets"], item["valid_mask"])
            loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            optimizer_steps += 1
            token_count = int(item["valid_mask"].sum().item())
            train_loss_sum += float(loss.detach().cpu().item()) * token_count
            train_token_count += token_count
            for name, value in terms.items():
                train_term_sums[name] = train_term_sums.get(name, 0.0) + value * token_count
        train_eval = _evaluate_model(model, train_batch, batch_size=batch_size, device=device)
        val_eval = _evaluate_model(model, val_batch, batch_size=batch_size, device=device)
        epoch_row = {
            "epoch": int(epoch),
            "optimizer_steps": int(optimizer_steps),
            "train_loss_step_weighted": float(train_loss_sum / max(train_token_count, 1)),
            "train_step_loss_terms": {name: float(value / max(train_token_count, 1)) for name, value in train_term_sums.items()},
            "train": train_eval,
            "val": val_eval,
        }
        epoch_metrics.append(epoch_row)
        # Load-bearing checkpoint monitor (EXIT_CKPT_MONITOR): val_loss keeps
        # prior behaviour; exit_now_accuracy selects on the decision metric
        # (entry dir_acc-monitor analogue). Score is stored negated for accuracy
        # so 'lower is better' stays uniform.
        if EXIT_CKPT_MONITOR == "exit_now_accuracy":
            _ckpt_score = -float(val_eval.get("exit_now_accuracy") or 0.0)
        else:
            _ckpt_score = float(val_eval["loss"])
        if _ckpt_score < best_val_loss:
            best_val_loss = _ckpt_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = {
        "train": _evaluate_model(model, train_batch, batch_size=batch_size, device=device),
        "val": _evaluate_model(model, val_batch, batch_size=batch_size, device=device),
        "test": _evaluate_model(model, test_batch, batch_size=batch_size, device=device),
    }
    finite_forward = all(
        all(split_metrics["finite_by_head"].values())
        for split_metrics in final_metrics.values()
    )
    noncollapsed_exit_now = float(final_metrics["val"]["exit_now_prob_std"]) > 1e-6
    out_bundle_dir = Path(str(args.out_bundle_dir or "")).expanduser().resolve()
    if not str(args.out_bundle_dir or "").strip():
        raise ValueError("--out-bundle-dir is required for training")
    out_bundle_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_bundle_dir / "model_state_dict.pt"
    metadata_path = out_bundle_dir / "bundle_metadata.json"
    metrics_path = out_bundle_dir / "training_metrics.json"
    dataset = training_plan.get("dataset") if isinstance(training_plan.get("dataset"), dict) else {}
    feature_schema_json = Path(str(dataset.get("feature_schema_json") or "")).expanduser()
    normalization_json = Path(str(dataset.get("normalization_json") or "")).expanduser()
    metadata = {
        "schema_version": "entry_exit_transformer_bundle_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": TRAINING_COMPLETE_DECISION if finite_forward and optimizer_steps > 0 else "ENTRY_EXIT_TRANSFORMER_TRAINING_RUN_FAILED",
        "model_family": "exit_sequence_transformer_v1",
        "vedtak": enablement["vedtak"],
        "training_plan_json": str(training_plan_json),
        "training_plan_json_sha256": _sha256_file(training_plan_json),
        **enablement,
        "feature_schema_json": str(feature_schema_json),
        "feature_schema_json_sha256": _sha256_file(feature_schema_json) if feature_schema_json.is_file() else "",
        "normalization_json": str(normalization_json),
        "normalization_json_sha256": _sha256_file(normalization_json) if normalization_json.is_file() else "",
        "dataset_shards": dataset.get("shards") if isinstance(dataset.get("shards"), dict) else {},
        "target_contract": {
            "target_columns_by_head": TARGET_COLUMNS_BY_HEAD,
            "loss_target_scale": LOSS_TARGET_SCALE,
            "state_timing": "AS_OF_CLOSED_M5_BAR_T_WITH_ENTRY_SNAPSHOT",
            "reward_source": "active Entry-bound Exit state/reward contract",
        },
        "output_heads": list(EXPECTED_HEADS),
        "parameter_count": int(sum(param.numel() for param in model.parameters())),
        "training_config": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": float(getattr(args, "lr", 3e-4) or 3e-4),
            "weight_decay": float(getattr(args, "weight_decay", 0.01) or 0.01),
            "clip_grad_norm": clip_grad_norm,
            "device": str(device),
            "num_workers": int(getattr(args, "num_workers", 0) or 0),
            # Rule-4 provenance: the effective exit-parity recipe (all knobs,
            # incl. defaults) — the audit/pretrain-manifest chain reads this.
            "exit_parity_knobs": dict(EXIT_PARITY_KNOBS),
            "max_train_episodes": max_train_episodes,
            "max_eval_episodes": max_eval_episodes,
        },
        "episode_counts": {
            "train": len(train_batch["episode_ids"]),
            "val": len(val_batch["episode_ids"]),
            "test": len(test_batch["episode_ids"]),
        },
        "row_counts": {
            "train": train_batch["row_count"],
            "val": val_batch["row_count"],
            "test": test_batch["row_count"],
        },
        "optimizer_steps": int(optimizer_steps),
        "finite_forward": bool(finite_forward),
        "noncollapsed_exit_now": bool(noncollapsed_exit_now),
        "final_metrics": final_metrics,
        "epoch_metrics": epoch_metrics,
        "trainer_started": True,
        "replay_started": False,
        "iql_distillation_started": False,
        "exit_iql_allowed": False,
        "promotion_shadow_live_allowed": False,
        "shadow_started": False,
        "live_started": False,
    }
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "metadata": metadata,
        },
        model_path,
    )
    metadata["model_state_dict_path"] = str(model_path)
    metadata["model_state_dict_sha256"] = _sha256_file(model_path)
    metadata["bundle_dir"] = str(out_bundle_dir)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps({"epoch_metrics": epoch_metrics, "final_metrics": final_metrics}, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-plan-json", required=True)
    ap.add_argument("--train-execution-review-json", default="")
    ap.add_argument("--post-train-contract-json", default="")
    ap.add_argument("--feature-alignment-json", default="")
    ap.add_argument("--vedtak", default="")
    ap.add_argument("--enable-training", action="store_true")
    ap.add_argument("--out-bundle-dir", default="")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--out-manifest-json", default="")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-episodes", type=int, default=4)
    ap.add_argument("--max-train-episodes", type=int, default=0)
    ap.add_argument("--max-eval-episodes", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip-grad-norm", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.preflight_only:
        manifest = run_preflight(args)
        print(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default))
        return 0 if manifest.get("decision") == "PASS" else 2
    try:
        report = run_training(args)
    except Exception as exc:
        raise SystemExit(f"FATAL: active Exit Transformer training blocked: {exc}") from exc
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return 0 if report.get("decision") == TRAINING_COMPLETE_DECISION else 2


if __name__ == "__main__":
    raise SystemExit(main())
