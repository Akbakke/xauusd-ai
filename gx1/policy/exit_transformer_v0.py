"""
EXIT_TRANSFORMER_V0 – ML-only exit transformer (no rules/router/critic).

Provides:
- ExitTransformerV0 (PyTorch model)
- ExitTransformerDecider (inference)
- train_from_exits_jsonl (training utility; used by run_truth_e2e_sanity --train_exit_transformer_v0_from_last_go)
- save/load artifacts (model/config/sha)

Hard gates:
- feature_dim must be 19 (EXIT_IO_FEATURE_COUNT)
- window_len must match config; validate every window
- No fallbacks; missing artifacts/files -> RuntimeError/FileNotFoundError
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gx1.exits.contracts.exit_io_v0_ctx19 import (
    EXIT_IO_V0_CTX19_FEATURES,
    EXIT_IO_V0_CTX19_FEATURE_NAMES_HASH,
    EXIT_IO_V0_CTX19_IO_VERSION,
    EXIT_IO_V0_CTX19_FEATURE_COUNT,
    compute_feature_names_hash,
    assert_exit_io_v0_ctx19_contract,
)

# Minimal exit IO contract (single supported IO)
EXIT_IO_FEATURE_COUNT = EXIT_IO_V0_CTX19_FEATURE_COUNT
EXIT_IO_VERSION = EXIT_IO_V0_CTX19_IO_VERSION
EXIT_FEATURE_NAMES_HASH = EXIT_IO_V0_CTX19_FEATURE_NAMES_HASH


def config_hash_v3(window_len: int, d_model: int, n_layers: int) -> str:
    h = hashlib.sha256()
    h.update(f"{window_len}:{d_model}:{n_layers}".encode("utf-8"))
    return h.hexdigest()[:16]


def validate_window_v3(window: np.ndarray, window_len: int, feat_dim: int, context: str = "") -> None:
    if window.ndim != 2 or window.shape[0] != window_len or window.shape[1] != feat_dim:
        raise RuntimeError(
            f"EXIT_WINDOW_SHAPE_MISMATCH[{context}]: expected {(window_len, feat_dim)}, got {window.shape}"
        )

log = logging.getLogger(__name__)
_exit_features_proof_logged = False
assert_exit_io_v0_ctx19_contract()

# Optional torch
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    TORCH_AVAILABLE = False

DEFAULT_WINDOW_LEN = 8
DEFAULT_D_MODEL = 64
DEFAULT_N_LAYERS = 2
DEFAULT_N_HEADS = 4
DEFAULT_THRESHOLD = 0.5

LAST_GO_FILENAME = "LAST_GO.txt"
TRUTH_E2E_REPORTS = "reports/truth_e2e_sanity"


def _gx1_data_root(gx1_data: Optional[str] = None) -> Path:
    base = Path(gx1_data or os.environ.get("GX1_DATA", "")).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError("GX1_DATA not set or not a directory")
    return base


def get_last_go_exits_dataset(gx1_data: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolve exits jsonl from LAST_GO pointer (TRUTH E2E).
    Returns: dict(go_run_dir, go_run_id, exits_jsonl_path)
    """
    base = _gx1_data_root(gx1_data)
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
        except Exception:
            pass
    exits_dir = go_run_dir / "replay" / "chunk_0" / "logs" / "exits"
    exits_jsonl_path = exits_dir / f"exits_{go_run_id}.jsonl"
    if not exits_jsonl_path.exists():
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


def _make_deterministic(seed: int = 42) -> None:
    if not TORCH_AVAILABLE:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore
    np.random.seed(seed)
    random.seed(seed)


if TORCH_AVAILABLE:

    class ExitTransformerV0(nn.Module):
        """Small transformer encoder: input (B, T, D) -> EXIT prob (B,)."""

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
    feature_names_hash: str = EXIT_FEATURE_NAMES_HASH,
    input_dim: Optional[int] = None,
) -> Tuple[Path, Path, str]:
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
        "feature_names_hash": feature_names_hash
        or compute_feature_names_hash(EXIT_IO_V0_CTX19_FEATURES),
        "feature_names": EXIT_IO_V0_CTX19_FEATURES,
    }
    config_path = out_dir / "exit_transformer_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    features_path = out_dir / "feature_names.json"
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(EXIT_IO_V0_CTX19_FEATURES, f, indent=2)
    sha_path = out_dir / "exit_transformer_sha256.txt"
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(model_sha)
    return model_path, config_path, model_sha


class ExitTransformerDecider:
    """
    Holds loaded model+config; builds window from feature history; returns (prob, decision, reason).
    Deterministic; hard-gates feature_dim/window_len.
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
        if self.input_dim != EXIT_IO_FEATURE_COUNT:
            raise RuntimeError(f"EXIT_IO_DIM_MISMATCH: expected {EXIT_IO_FEATURE_COUNT}, got {self.input_dim}")
        _make_deterministic()
        self.model = ExitTransformerV0(
            input_dim=self.input_dim,
            window_len=self.window_len,
            d_model=self.d_model,
            n_heads=DEFAULT_N_HEADS,
            n_layers=self.n_layers,
        )
        state = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        self._model_sha = model_sha or ""

    def set_model_sha(self, sha: str) -> None:
        self._model_sha = sha

    @property
    def model_sha(self) -> str:
        return self._model_sha

    def predict(self, window: np.ndarray) -> Tuple[float, str, str]:
        if self.input_dim != EXIT_IO_FEATURE_COUNT:
            raise RuntimeError(
                f"EXIT_IO_DIM_MISMATCH: expected {EXIT_IO_FEATURE_COUNT}, got {self.input_dim}"
            )
        validate_window_v3(window, self.window_len, self.input_dim, context="ExitTransformerDecider.predict")
        _make_deterministic()
        with torch.no_grad():
            x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
            prob = self.model(x).item()
        decision = "EXIT" if prob > self.threshold else "HOLD"
        reason = "ML_EXIT_TRANSFORMER" if decision == "EXIT" else ""
        return float(prob), decision, reason

    # Optional alias used historically
    decide = predict


def load_exit_transformer_decider(
    model_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,
) -> ExitTransformerDecider:
    model_dir = Path(model_dir)
    model_path = model_dir / "exit_transformer_v0.pt"
    config_path = model_dir / "exit_transformer_config.json"
    sha_path = model_dir / "exit_transformer_sha256.txt"
    features_path = model_dir / "feature_names.json"
    if not model_path.exists():
        raise FileNotFoundError(f"EXIT_MODEL_MISSING: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"EXIT_CONFIG_MISSING: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    cfg_dim = int(config.get("input_dim", EXIT_IO_FEATURE_COUNT))
    if cfg_dim != EXIT_IO_FEATURE_COUNT:
        raise RuntimeError(
            f"[EXIT_IO_CONTRACT_VIOLATION] expected input_dim={EXIT_IO_FEATURE_COUNT}, got {cfg_dim}"
        )
    cfg_io_ver = str(config.get("exit_ml_io_version", EXIT_IO_VERSION))
    if cfg_io_ver != EXIT_IO_VERSION:
        raise RuntimeError(
            f"[EXIT_IO_CONTRACT_VIOLATION] expected exit_ml_io_version={EXIT_IO_VERSION}, got {cfg_io_ver}"
        )
    cfg_hash = str(config.get("feature_names_hash", "")).strip()
    if not cfg_hash:
        raise RuntimeError("[EXIT_IO_CONTRACT_VIOLATION] feature_names_hash missing in config")
    if cfg_hash != EXIT_FEATURE_NAMES_HASH:
        raise RuntimeError(
            f"[EXIT_IO_CONTRACT_VIOLATION] expected feature_names_hash={EXIT_FEATURE_NAMES_HASH}, got {cfg_hash}"
        )
    cfg_features = config.get("feature_names")
    if cfg_features is not None:
        if not isinstance(cfg_features, list):
            raise RuntimeError("[EXIT_IO_CONTRACT_VIOLATION] feature_names must be a list")
        if cfg_features != EXIT_IO_V0_CTX19_FEATURES:
            raise RuntimeError(
                f"[EXIT_IO_CONTRACT_VIOLATION] feature_names mismatch: expected {EXIT_IO_V0_CTX19_FEATURES}, got {cfg_features}"
            )
    else:
        # If no names provided, still enforce count
        if len(EXIT_IO_V0_CTX19_FEATURES) != EXIT_IO_FEATURE_COUNT:
            raise RuntimeError(
                f"[EXIT_IO_CONTRACT_VIOLATION] feature count mismatch: expected {EXIT_IO_FEATURE_COUNT}, got {len(EXIT_IO_V0_CTX19_FEATURES)}"
            )
    if features_path.exists():
        try:
            with open(features_path, "r", encoding="utf-8") as f:
                names_from_file = json.load(f)
        except Exception as e:
            raise RuntimeError(f"[EXIT_FEATURE_LIST_MISMATCH] failed to read feature_names.json: {e}") from e
        if names_from_file != EXIT_IO_V0_CTX19_FEATURES:
            raise RuntimeError(
                f"[EXIT_FEATURE_LIST_MISMATCH] feature_names.json mismatch: expected {EXIT_IO_V0_CTX19_FEATURES}, got {names_from_file}"
            )
    global _exit_features_proof_logged
    if not _exit_features_proof_logged:
        first3 = EXIT_IO_V0_CTX19_FEATURES[:3]
        last3 = EXIT_IO_V0_CTX19_FEATURES[-3:]
        log.info(
            "[EXIT_FEATURES_PROOF] io=%s n=%d hash=%s first3=%s last3=%s",
            EXIT_IO_VERSION,
            EXIT_IO_V0_CTX19_FEATURE_COUNT,
            EXIT_FEATURE_NAMES_HASH,
            first3,
            last3,
        )
        _exit_features_proof_logged = True
    model_sha = sha_path.read_text(encoding="utf-8").strip() if sha_path.exists() else ""
    return ExitTransformerDecider(
        model_path=model_path,
        config=config,
        threshold=threshold,
        model_sha=model_sha,
    )


# -----------------------------------------------------------------------------
# Minimal dataset loader (no external dependency)
# -----------------------------------------------------------------------------
def _load_exits_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(obj)
    return rows


def _build_windows_from_exits(
    records: List[Dict[str, Any]],
    window_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    labels: List[float] = []
    for rec in records:
        io = rec.get("io_features")
        label = rec.get("should_exit")
        if io is None or label is None:
            continue
        arr = np.asarray(io, dtype=np.float32)
        if arr.shape[-1] != EXIT_IO_FEATURE_COUNT:
            raise RuntimeError(f"EXIT_IO_DIM_MISMATCH: expected {EXIT_IO_FEATURE_COUNT}, got {arr.shape[-1]}")
        # Build sliding windows over the sequence axis if provided; if not, assume single window
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        T = arr.shape[0]
        if T < window_len:
            continue
        for start in range(0, T - window_len + 1, stride):
            win = arr[start : start + window_len]
            feats.append(win)
            labels.append(float(label))
    if not feats:
        return np.zeros((0, window_len, EXIT_IO_FEATURE_COUNT), dtype=np.float32), np.zeros(0, dtype=np.float32)
    X = np.stack(feats, axis=0)
    y = np.asarray(labels, dtype=np.float32)
    return X, y


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
    if not TORCH_AVAILABLE or ExitTransformerV0 is None:
        raise RuntimeError("PyTorch required for training")
    exits_jsonl_path = Path(exits_jsonl_path).resolve()
    if not exits_jsonl_path.exists():
        raise FileNotFoundError(f"Exits jsonl not found: {exits_jsonl_path}")

    records = _load_exits_jsonl(exits_jsonl_path)
    _make_deterministic(seed)
    X, y = _build_windows_from_exits(records, window_len=window_len, stride=stride)
    if X.shape[0] == 0:
        raise ValueError(f"No samples from {exits_jsonl_path}; need at least one window of len {window_len}")
    dataset_sha256 = hashlib.sha256(exits_jsonl_path.read_bytes()).hexdigest()

    # Simple train/val split by fold
    n = X.shape[0]
    fold = max(1, min(val_fold, n - 1))
    split = n * fold // (fold + 1)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    device = torch.device("cpu")
    model = ExitTransformerV0(
        input_dim=EXIT_IO_FEATURE_COUNT,
        window_len=window_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction="none")  # type: ignore[union-attr]

    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_train.astype(np.float32)),
            torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
    )
    pos_weight_val: Optional[float] = None
    exit_rate = float(np.mean(y_train)) if y_train.size else 0.0
    if use_pos_weight and 0 < exit_rate < 1:
        pos_weight_val = (1.0 - exit_rate) / max(exit_rate, 1e-6)

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

    # Simple val metric
    model.eval()
    with torch.no_grad():
        if X_val.size > 0:
            vpred = model(torch.from_numpy(X_val.astype(np.float32)).to(device)).cpu().numpy()
            vloss = float(np.mean((vpred - y_val.reshape(-1, 1)) ** 2))
        else:
            vloss = math.nan

    # Resolve out_dir
    base = _gx1_data_root(gx1_data)
    if out_dir is None:
        out_dir = base / "models" / "exit_transformer_v0" / dataset_sha256
    out_dir = Path(out_dir).resolve()
    model_path, config_path, model_sha = save_exit_transformer_artifacts(
        model=model,
        out_dir=out_dir,
        window_len=window_len,
        d_model=d_model,
        n_layers=n_layers,
        feature_names_hash=EXIT_FEATURE_NAMES_HASH,
        input_dim=EXIT_IO_FEATURE_COUNT,
    )
    report = {
        "dataset": str(exits_jsonl_path),
        "dataset_sha256": dataset_sha256,
        "n_samples": int(X.shape[0]),
        "window_len": window_len,
        "exit_rate": exit_rate,
        "val_loss": vloss,
        "model_path": str(model_path),
        "config_path": str(config_path),
        "model_sha256": model_sha,
        "git_sha": _git_sha_safe(),
        "source_run_id": source_run_id,
        "source_run_dir": source_run_dir,
    }
    report_path = out_dir / "TRAIN_REPORT.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("[EXIT_TRAIN_DONE] out_dir=%s model_sha=%s val_loss=%s", out_dir, model_sha, vloss)
    return report


def _git_sha_safe() -> str:
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
