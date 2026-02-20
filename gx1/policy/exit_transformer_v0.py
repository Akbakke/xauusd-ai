"""
Exit Transformer V0 – imitation of MASTER_EXIT_V1 score_v1 (HOLD/EXIT).

Small transformer encoder on sequence of exit features; output head for EXIT prob.
Deterministic inference; training from exits_*.jsonl with BCE.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gx1.contracts.exit_io import (
    EXIT_IO_FEATURE_COUNT,
    EXIT_IO_VERSION,
    config_hash_v3,
    validate_window_v3,
)

log = logging.getLogger(__name__)

# Optional torch (inference may run without it if decider not used)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    TORCH_AVAILABLE = False

DEFAULT_WINDOW_LEN = 64
DEFAULT_D_MODEL = 64
DEFAULT_N_LAYERS = 2
DEFAULT_N_HEADS = 4
DEFAULT_THRESHOLD = 0.5

LAST_GO_FILENAME = "LAST_GO.txt"
TRUTH_E2E_REPORTS = "reports/truth_e2e_sanity"


def get_last_go_exits_dataset(gx1_data: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolve SSoT training dataset from GX1_DATA/reports/truth_e2e_sanity/LAST_GO.txt.
    Returns dict with: go_run_dir (Path), go_run_id (str), exits_jsonl_path (Path).
    Raises FileNotFoundError if LAST_GO or exits jsonl missing.
    """
    base = Path(gx1_data or os.environ.get("GX1_DATA", "")).expanduser().resolve()
    if not base or not base.is_dir():
        raise FileNotFoundError("GX1_DATA not set or not a directory")
    last_go_path = base / TRUTH_E2E_REPORTS / LAST_GO_FILENAME
    if not last_go_path.exists():
        raise FileNotFoundError(f"LAST_GO pointer missing: {last_go_path}")
    go_run_dir = Path(last_go_path.read_text().strip()).expanduser().resolve()
    if not go_run_dir.is_dir():
        raise FileNotFoundError(f"LAST_GO run dir does not exist: {go_run_dir}")
    go_run_id = go_run_dir.name
    chunk_footer = go_run_dir / "replay" / "chunk_0" / "chunk_footer.json"
    if chunk_footer.exists():
        try:
            with open(chunk_footer, "r", encoding="utf-8") as f:
                footer_id = json.load(f).get("run_id") or ""
                if footer_id:
                    go_run_id = footer_id
        except (json.JSONDecodeError, OSError):
            pass
    exits_dir = go_run_dir / "replay" / "chunk_0" / "logs" / "exits"
    exits_jsonl_path = exits_dir / f"exits_{go_run_id}.jsonl"
    if not exits_jsonl_path.exists():
        # Fallback: runner may write exits_YYYYMMDD.jsonl; use any single exits_*.jsonl in dir
        candidates = sorted(exits_dir.glob("exits_*.jsonl")) if exits_dir.exists() else []
        if len(candidates) == 1:
            exits_jsonl_path = candidates[0]
        elif not candidates:
            raise FileNotFoundError(f"Exits jsonl not found: {exits_jsonl_path}")
        else:
            raise FileNotFoundError(
                f"Exits jsonl ambiguous: expected {exits_jsonl_path} or single exits_*.jsonl, found {[p.name for p in candidates]}"
            )
    return {
        "go_run_dir": go_run_dir,
        "go_run_id": go_run_id,
        "exits_jsonl_path": exits_jsonl_path,
    }


def _make_deterministic() -> None:
    if not TORCH_AVAILABLE:
        return
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore


if TORCH_AVAILABLE:

    class ExitTransformerV0(nn.Module):
        """Small transformer encoder: input (B, T, D), output (B,) EXIT probability (sigmoid)."""

        def __init__(
            self,
            input_dim: int = EXIT_IO_FEATURE_COUNT,
            window_len: int = DEFAULT_WINDOW_LEN,
            d_model: int = DEFAULT_D_MODEL,
            n_heads: int = DEFAULT_N_HEADS,
            n_layers: int = DEFAULT_N_LAYERS,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.window_len = window_len
            self.d_model = d_model
            self.proj = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, T, D)
            h = self.proj(x)
            h = self.transformer(h)
            last = h[:, -1, :]
            logit = self.head(last).squeeze(-1)
            return torch.sigmoid(logit)

else:
    ExitTransformerV0 = None  # type: ignore


def save_exit_transformer_artifacts(
    model: Any,
    out_dir: Path,
    window_len: int,
    d_model: int,
    n_layers: int,
    feature_names_hash: str = "",
    input_dim: Optional[int] = None,
) -> Tuple[Path, Path, str]:
    """
    Save exit_transformer_v0.pt, exit_transformer_config.json, exit_transformer_sha256.txt under out_dir.
    Returns (model_path, config_path, model_sha).     input_dim defaults to EXIT_IO_FEATURE_COUNT when None.
    """
    if not TORCH_AVAILABLE or model is None:
        raise RuntimeError("PyTorch and model required to save")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "exit_transformer_v0.pt"
    torch.save(model.state_dict(), model_path)
    with open(model_path, "rb") as f:
        model_sha = hashlib.sha256(f.read()).hexdigest()
    in_dim = input_dim if input_dim is not None else EXIT_IO_FEATURE_COUNT
    config = {
        "window_len": window_len,
        "d_model": d_model,
        "n_layers": n_layers,
        "input_dim": in_dim,
        "exit_ml_io_version": EXIT_IO_VERSION,
        "feature_names_hash": feature_names_hash or config_hash_v3(window_len, d_model, n_layers),
    }
    config_path = out_dir / "exit_transformer_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    sha_path = out_dir / "exit_transformer_sha256.txt"
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(model_sha)
    return model_path, config_path, model_sha


class ExitTransformerDecider:
    """
    Holds loaded model + config; builds window from feature history; returns (exit_prob, decision, reason).
    Deterministic; no randomness at inference.
    """

    def __init__(
        self,
        model_path: Path,
        config: Dict[str, Any],
        threshold: float = DEFAULT_THRESHOLD,
        model_sha: Optional[str] = None,
    ):
        if not TORCH_AVAILABLE or ExitTransformerV0 is None:
            raise RuntimeError("PyTorch required for ExitTransformerDecider")
        self.model_path = Path(model_path)
        self.config = config
        self.threshold = float(threshold)
        self.window_len = int(config.get("window_len", DEFAULT_WINDOW_LEN))
        self.d_model = int(config.get("d_model", DEFAULT_D_MODEL))
        self.n_layers = int(config.get("n_layers", DEFAULT_N_LAYERS))
        self.input_dim = int(config.get("input_dim", EXIT_IO_FEATURE_COUNT))
        _make_deterministic()
        self.model = ExitTransformerV0(
            input_dim=self.input_dim,
            window_len=self.window_len,
            d_model=self.d_model,
            n_layers=self.n_layers,
        )
        state = torch.load(self.model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self._model_sha = model_sha or ""

    def set_model_sha(self, sha: str) -> None:
        self._model_sha = sha

    @property
    def model_sha(self) -> str:
        return self._model_sha

    def predict(self, window: np.ndarray) -> Tuple[float, str, str]:
        """
        window: (window_len, input_dim). Returns (exit_prob, decision, reason).
        """
        if self.input_dim != EXIT_IO_FEATURE_COUNT:
            raise RuntimeError(
                f"legacy exit IO disabled; only 35-dim (IOV3_CLEAN) supported. Got input_dim={self.input_dim}"
            )
        validate_window_v3(window, self.window_len, self.input_dim, context="ExitTransformerDecider.predict")
        _make_deterministic()
        with torch.no_grad():
            x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
            prob = self.model(x).item()
        decision = "EXIT" if prob > self.threshold else "HOLD"
        reason = "ML_EXIT_TRANSFORMER" if decision == "EXIT" else ""
        return (float(prob), decision, reason)


def load_exit_transformer_decider(
    model_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,
) -> ExitTransformerDecider:
    """Load model + config + sha from GX1_DATA-style dir; return ExitTransformerDecider."""
    model_dir = Path(model_dir)
    model_path = model_dir / "exit_transformer_v0.pt"
    config_path = model_dir / "exit_transformer_config.json"
    sha_path = model_dir / "exit_transformer_sha256.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Exit transformer model not found: {model_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    model_sha = ""
    if sha_path.exists():
        model_sha = sha_path.read_text(encoding="utf-8").strip()
    return ExitTransformerDecider(
        model_path=model_path,
        config=config,
        threshold=threshold,
        model_sha=model_sha,
    )


def _git_sha_safe() -> str:
    """Return git rev-parse HEAD or empty string."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[2],
        )
        return (r.stdout or "").strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def train_from_exits_jsonl(
    exits_jsonl_path: Path,
    out_dir: Optional[Path] = None,
    window_len: int = DEFAULT_WINDOW_LEN,
    d_model: int = DEFAULT_D_MODEL,
    n_layers: int = DEFAULT_N_LAYERS,
    n_heads: int = DEFAULT_N_HEADS,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_fold: int = 9,
    seed: int = 42,
    stride: int = 1,
    source_run_id: Optional[str] = None,
    source_run_dir: Optional[str] = None,
    gx1_data: Optional[str] = None,
    use_pos_weight: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Load dataset from exits jsonl, train transformer, save artifacts.
    When out_dir is None, writes to GX1_DATA/models/exit_transformer_v0/<dataset_sha256>/.
    Deterministic (seed, CPU). Writes TRAIN_REPORT.json. Returns dict with model_path, config_path,
    model_sha256, dataset_sha256, train_report_path.
    window_len: default 64 (full retrain); callers may pass e.g. 8 for sanity speed.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for training")
    from gx1.datasets.exit_transformer_dataset import (
        load_exits_jsonl,
        load_dataset_from_exits_jsonl,
        train_val_split,
    )
    exits_jsonl_path = Path(exits_jsonl_path).resolve()
    if not exits_jsonl_path.exists():
        raise FileNotFoundError(f"Exits jsonl not found: {exits_jsonl_path}")

    dataset_sha256 = hashlib.sha256(exits_jsonl_path.read_bytes()).hexdigest()
    records = load_exits_jsonl(exits_jsonl_path)
    n_lines_total = len(records)
    by_trade: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        tid = r.get("trade_id") or ""
        if tid not in by_trade:
            by_trade[tid] = []
        by_trade[tid].append(r)
    n_trades = sum(1 for bars in by_trade.values() if len(bars) >= window_len)
    n_lines_used = sum(len(bars) for bars in by_trade.values() if len(bars) >= window_len)

    _make_deterministic()
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, y, meta, io_version, feature_dim = load_dataset_from_exits_jsonl(
        exits_jsonl_path,
        window_len=window_len,
        stride=stride,
    )
    if X.shape[0] == 0:
        raise ValueError(
            f"No samples from {exits_jsonl_path}; need at least one trade with >= {window_len} bars"
        )
    n_windows = int(X.shape[0])
    exit_rate = float(np.mean(y))
    pos_weight_val: Optional[float] = None
    if use_pos_weight and 0 < exit_rate < 1:
        pos_weight_val = (1.0 - exit_rate) / max(exit_rate, 1e-6)
    X_train, y_train, X_val, y_val, meta_train, meta_val = train_val_split(X, y, meta, val_fold=val_fold)
    if X_train.shape[0] == 0:
        X_train, y_train = X, y
        X_val = np.zeros((0,) + X.shape[1:], dtype=X.dtype)
        y_val = np.zeros(0, dtype=y.dtype)
        meta_val = []

    base = Path(gx1_data or os.environ.get("GX1_DATA", "")).expanduser().resolve()
    if out_dir is None:
        if not base or not base.is_dir():
            raise ValueError("out_dir is None and GX1_DATA not set or not a directory")
        out_dir = base / "models" / "exit_transformer_v0" / dataset_sha256
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = ExitTransformerV0(
        input_dim=feature_dim,
        window_len=window_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction="none")  # type: ignore[union-attr]
    dataset_train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1),
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, generator=gen
    )
    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx).unsqueeze(1)
            loss_per = criterion(out, by)
            if pos_weight_val is not None:
                w = torch.where(by > 0.5, torch.full_like(by, pos_weight_val), torch.ones_like(by))
                loss_per = loss_per * w
            loss = loss_per.mean()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
        y_t = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1).to(device)
        lt = criterion(model(X_t).unsqueeze(1), y_t)
        final_train_loss = float(lt.mean().item())
        if X_val.shape[0] > 0:
            X_v = torch.from_numpy(X_val.astype(np.float32)).to(device)
            y_v = torch.from_numpy(y_val.astype(np.float32)).unsqueeze(1).to(device)
            lv = criterion(model(X_v).unsqueeze(1), y_v)
            final_val_loss = float(lv.mean().item())
        else:
            final_val_loss = float("nan")
    feature_names_hash = config_hash_v3(window_len, d_model, n_layers)
    model_path, config_path, model_sha = save_exit_transformer_artifacts(
        model, out_dir, window_len=window_len, d_model=d_model, n_layers=n_layers,
        feature_names_hash=feature_names_hash,
        input_dim=feature_dim,
    )
    run_id = source_run_id or ""
    run_dir = source_run_dir or ""
    train_report = {
        "source_exits_jsonl_path": str(exits_jsonl_path.resolve()),
        "source_run_id": run_id,
        "source_run_dir": run_dir,
        "dataset_sha256": dataset_sha256,
        "n_lines_total": n_lines_total,
        "n_lines_used": n_lines_used,
        "n_trades": n_trades,
        "n_windows": n_windows,
        "window_len": window_len,
        "stride": stride,
        "io_version": io_version,
        "exit_ml_io_version": EXIT_IO_VERSION,
        "context_required": False,
        "feature_count": feature_dim,
        "feature_names_hash": feature_names_hash,
        "exit_rate": exit_rate,
        "pos_weight": pos_weight_val,
        "seed": seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha_safe(),
    }
    train_report_path = out_dir / "TRAIN_REPORT.json"
    with open(train_report_path, "w", encoding="utf-8") as f:
        json.dump(train_report, f, indent=2)

    # Run verify after training; hard GO/NO-GO (raise on fail).
    verify_result = verify_exit_transformer_artifacts(out_dir)
    if not verify_result.get("passed", False):
        failures = verify_result.get("failures", [])
        raise RuntimeError(
            "Exit transformer verify failed (GO/NO-GO): " + "; ".join(failures)
        )

    return {
        "model_path": model_path,
        "config_path": config_path,
        "model_sha256": model_sha,
        "dataset_sha256": dataset_sha256,
        "train_report_path": train_report_path,
    }


def verify_exit_transformer_artifacts(out_dir: Path) -> Dict[str, Any]:
    """
    TRUTH-grade verify: hashes, config vs report, and smoke inference.
    Reads TRAIN_REPORT.json, config, sha256; recomputes hashes; runs smoke forward pass.
    Returns dict with passed, failures, recomputed hashes, smoke_stats; writes VERIFY_REPORT.json.
    """
    out_dir = Path(out_dir)
    failures: List[str] = []
    report_path = out_dir / "TRAIN_REPORT.json"
    config_path = out_dir / "exit_transformer_config.json"
    sha_path = out_dir / "exit_transformer_sha256.txt"
    model_path = out_dir / "exit_transformer_v0.pt"
    recomputed: Dict[str, str] = {}

    if not report_path.exists():
        failures.append(f"TRAIN_REPORT.json missing: {report_path}")
        result = {"passed": False, "failures": failures, "recomputed": recomputed}
        verify_path = out_dir / "VERIFY_REPORT.json"
        with open(verify_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    source_exits = Path(report.get("source_exits_jsonl_path") or "")
    dataset_sha256_report = report.get("dataset_sha256") or ""
    window_len_report = report.get("window_len")
    feature_names_hash_report = report.get("feature_names_hash") or ""
    n_lines_used = report.get("n_lines_used", 0)
    n_windows = report.get("n_windows", 0)
    final_train_loss = report.get("final_train_loss")
    final_val_loss = report.get("final_val_loss")

    if source_exits.exists():
        recomputed["dataset_sha256"] = hashlib.sha256(source_exits.read_bytes()).hexdigest()
    else:
        recomputed["dataset_sha256"] = ""
        failures.append(f"exits jsonl not found for recompute: {source_exits}")
    if recomputed.get("dataset_sha256") and recomputed["dataset_sha256"] != dataset_sha256_report:
        failures.append("dataset_sha256 does not match TRAIN_REPORT")

    if model_path.exists():
        recomputed["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    else:
        recomputed["model_sha256"] = ""
        failures.append(f"model file missing: {model_path}")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        window_len_config = config.get("window_len")
        config_feature_hash = config.get("feature_names_hash") or ""
        if window_len_report is not None and window_len_config != window_len_report:
            failures.append(f"config.window_len {window_len_config} != report.window_len {window_len_report}")
        if feature_names_hash_report and config_feature_hash != feature_names_hash_report:
            failures.append("config.feature_names_hash != report.feature_names_hash")
        exit_ml_io = config.get("exit_ml_io_version") or report.get("exit_ml_io_version")
        if exit_ml_io != EXIT_IO_VERSION:
            failures.append(
                f"config/report exit_ml_io_version must be {EXIT_IO_VERSION!r}, got {exit_ml_io!r}"
            )
        recomputed["feature_names_hash"] = config_hash_v3(
            int(config.get("window_len", 0)),
            int(config.get("d_model", 0)),
            int(config.get("n_layers", 0)),
        )
    else:
        failures.append(f"config missing: {config_path}")
        config = {}
        recomputed["feature_names_hash"] = ""

    if sha_path.exists():
        file_sha = sha_path.read_text(encoding="utf-8").strip()
        if recomputed.get("model_sha256") and file_sha != recomputed["model_sha256"]:
            failures.append("exit_transformer_sha256.txt does not match recomputed model_sha256")
    else:
        failures.append(f"sha256 file missing: {sha_path}")

    if n_lines_used <= 0 or n_windows <= 0:
        failures.append("report.n_lines_used or n_windows <= 0")
    if final_train_loss is None or not (isinstance(final_train_loss, (int, float)) and abs(final_train_loss) != float("inf")):
        failures.append("final_train_loss missing or not finite")
    if final_val_loss is not None and isinstance(final_val_loss, (int, float)):
        if not math.isnan(final_val_loss) and not math.isfinite(final_val_loss):
            failures.append("final_val_loss not finite")

    io_version_report = report.get("io_version") or report.get("exit_ml_io_version") or EXIT_IO_VERSION
    feature_count_report = report.get("feature_count")
    if feature_count_report is not None and feature_count_report != EXIT_IO_FEATURE_COUNT:
        failures.append(
            f"report.feature_count must be {EXIT_IO_FEATURE_COUNT} (IOV3_CLEAN), got {feature_count_report}"
        )
    if config.get("input_dim") != EXIT_IO_FEATURE_COUNT:
        failures.append(
            f"config.input_dim must be {EXIT_IO_FEATURE_COUNT}, got {config.get('input_dim')}"
        )

    smoke_stats: Dict[str, Any] = {}
    if TORCH_AVAILABLE and ExitTransformerV0 is not None and model_path.exists() and config_path.exists() and source_exits.exists():
        try:
            from gx1.datasets.exit_transformer_dataset import load_dataset_from_exits_jsonl
            window_len = int(config.get("window_len", DEFAULT_WINDOW_LEN))
            X, _, _, _, _ = load_dataset_from_exits_jsonl(
                source_exits, window_len=window_len, stride=1,
            )
            N = min(32, X.shape[0])
            if N > 0:
                decider = load_exit_transformer_decider(out_dir, threshold=0.5)
                probs: List[float] = []
                for i in range(N):
                    w = X[i]
                    p, _, _ = decider.predict(w)
                    probs.append(p)
                probs_arr = np.array(probs)
                smoke_stats["n_samples"] = N
                smoke_stats["exit_prob_min"] = float(np.min(probs_arr))
                smoke_stats["exit_prob_max"] = float(np.max(probs_arr))
                smoke_stats["exit_prob_mean"] = float(np.mean(probs_arr))
                smoke_stats["exit_prob_std"] = float(np.std(probs_arr))
                if not (0 <= smoke_stats["exit_prob_min"] <= 1 and 0 <= smoke_stats["exit_prob_max"] <= 1):
                    failures.append("smoke: exit_prob outside [0,1]")
                if smoke_stats["exit_prob_min"] == smoke_stats["exit_prob_max"] and abs(smoke_stats["exit_prob_mean"] - 0.5) < 1e-5:
                    failures.append("smoke: all exit_prob ~0.5 (model may be dead)")
            else:
                failures.append("smoke: no windows for inference")
        except Exception as e:
            failures.append(f"smoke inference: {e!s}")
            smoke_stats = {"error": str(e)}

    passed = len(failures) == 0
    result = {
        "passed": passed,
        "failures": failures,
        "recomputed": recomputed,
        "smoke_stats": smoke_stats,
    }
    verify_path = out_dir / "VERIFY_REPORT.json"
    with open(verify_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
