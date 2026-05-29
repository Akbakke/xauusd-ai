"""ThinRecordDataset — PyTorch Dataset for the V5 thin-record exit-transformer
training dataset produced by `materialize_build_v3_training_dataset_v1.py`.

Background
----------
The V5 dataset stores io_features ONCE as a shared M1 feature matrix
(2.2M × 89 float32) plus thin per-(trade, bar) records that reference it.
At training time we reconstruct the (window_len, 89) io_features tensor by:

    io = matrix[m1_idx_now-(W-1) : m1_idx_now+1].copy()          # (W, 89)
    overlay_rows = overlays[overlay_offset + overlay_start_row :
                            overlay_offset + overlay_start_row + n_in_trade_bars]
    io[in_trade_start_in_win : in_trade_start_in_win + n_in_trade_bars,
       trade_state_indices] = overlay_rows                       # in-trade fill

The dataset is laid out as:

    <dataset_dir>/
    ├── m1_feature_matrix.npy            (2.2M × 89 float32, mmap-friendly)
    ├── m1_time_ns.npy                   (2.2M int64)
    ├── trade_state_overlays.f32         (raw float32, shape (N, 19))
    ├── overlay_index.parquet            (trade_uid → overlay_offset, overlay_length)
    ├── records.jsonl                    (one record per (trade, bar))
    └── manifest.json                    (io_version, input_dim, window_len, ...)

Each record (one JSON object per line) carries:
    ts, run_id, trade_uid, trade_id, side, m1_idx_now, in_trade_start_in_win,
    n_in_trade_bars, overlay_start_row, scalars, teacher_final_pnl_bps,
    teacher_final_mfe_bps, teacher_final_mae_bps, teacher_duration_bars

Usage
-----
    from gx1.exits.training.thin_record_dataset import ThinRecordDataset

    ds = ThinRecordDataset("/path/to/exit_v3_v7_training_2020_2026_canonical_v3")
    sample = ds[0]
    # sample["io_features"]: (window_len, input_dim) float32 tensor
    # sample["scalars"]: dict[str, float]
    # sample["teacher"]: dict[str, float]
    # sample["meta"]: dict[str, str]

The dataset can be wrapped with a label-attaching transform to produce
training-ready (x, y) pairs — see `_attach_labels_to_thin_records` below.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ThinRecordDataset(Dataset):
    """Memory-efficient PyTorch dataset over the V5 thin-record format."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        memmap_matrix: bool = True,
        memmap_overlays: bool = True,
        records_offset: int = 0,
        records_limit: Optional[int] = None,
        load_records_eagerly: bool = True,
        build_offset_index: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.manifest = self._load_manifest()
        self.records_path: Optional[Path] = None
        self.uid_to_offsets: Optional[Dict[str, List[Tuple[int, int]]]] = None

        # Load shared tensors (mmap'd by default)
        matrix_path = self.dataset_dir / self.manifest["files"]["m1_feature_matrix"]
        overlays_path = self.dataset_dir / self.manifest["files"]["trade_state_overlays"]
        time_ns_path = self.dataset_dir / self.manifest["files"]["m1_time_ns"]
        index_path = self.dataset_dir / self.manifest["files"]["overlay_index"]
        records_path = self.dataset_dir / self.manifest["files"]["records"]

        self.matrix = np.load(str(matrix_path), mmap_mode="r" if memmap_matrix else None)
        self.m1_time_ns = np.load(str(time_ns_path), mmap_mode="r" if memmap_matrix else None)

        # overlays.f32 is a raw binary file: shape (N, 19) float32
        n_overlay_cols = int(self.manifest["files"].get("trade_state_overlays_cols", 19))
        if memmap_overlays:
            self.overlays = np.memmap(
                overlays_path, dtype=np.float32, mode="r"
            ).reshape(-1, n_overlay_cols)
        else:
            self.overlays = np.fromfile(overlays_path, dtype=np.float32).reshape(-1, n_overlay_cols)

        # overlay_index: trade_uid → (overlay_offset, overlay_length)
        oi = pd.read_parquet(index_path)
        self.overlay_index: Dict[str, Tuple[int, int]] = {
            row["trade_uid"]: (int(row["overlay_offset"]), int(row["overlay_length"]))
            for _, row in oi.iterrows()
        }

        # Records: load JSONL into memory or stream
        self.input_dim = int(self.manifest["input_dim"])
        self.window_len = int(self.manifest["window_len"])
        self.trade_state_indices = list(self.manifest["trade_state_feature_indices"])

        self.records_path = records_path
        if load_records_eagerly:
            if build_offset_index:
                self.records, self.uid_to_offsets = self._load_records_jsonl_with_offsets(
                    records_path, offset=records_offset, limit=records_limit,
                )
            else:
                self.records = self._load_records_jsonl(
                    records_path, offset=records_offset, limit=records_limit,
                )
        else:
            self.records = []
            if build_offset_index:
                self.uid_to_offsets = self._build_offset_index(records_path)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _load_manifest(self) -> Dict[str, Any]:
        mp = self.dataset_dir / "manifest.json"
        if not mp.exists():
            raise FileNotFoundError(f"[THIN_RECORD_DATASET] manifest.json not found at {mp}")
        return json.loads(mp.read_text())

    def _load_records_jsonl(
        self, path: Path, *, offset: int = 0, limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < offset:
                    continue
                if limit is not None and (i - offset) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    def _load_records_jsonl_with_offsets(
        self, path: Path, *, offset: int = 0, limit: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Tuple[int, int]]]]:
        """Load records AND build a per-trade byte-offset index in a single pass.

        The index maps trade_uid -> list of (byte_offset, byte_length) tuples
        so workers can later seek+read the JSONL slice for a specific trade
        without parent-process inheritance of the records dict.
        """
        out: List[Dict[str, Any]] = []
        from collections import defaultdict
        uid_to_offsets: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        i = 0
        with path.open("rb") as f:
            byte_off = 0
            while True:
                line = f.readline()
                if not line:
                    break
                ll = len(line)
                if i >= offset and (limit is None or (i - offset) < limit):
                    stripped = line.strip()
                    if stripped:
                        rec = json.loads(stripped)
                        out.append(rec)
                        uid = rec.get("trade_uid") or rec.get("trade_id")
                        if uid:
                            uid_to_offsets[uid].append((byte_off, ll))
                byte_off += ll
                i += 1
                if limit is not None and (i - offset) >= limit:
                    break
        return out, dict(uid_to_offsets)

    def _build_offset_index(self, path: Path) -> Dict[str, List[Tuple[int, int]]]:
        """Build trade_uid -> [(byte_offset, byte_length)] index without storing records.

        Used for spawn-worker labeling: parent stays small (just the index, ~30 MB
        for 2.5M records); workers seek+read their slice on demand.
        """
        from collections import defaultdict
        uid_to_offsets: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        with path.open("rb") as f:
            byte_off = 0
            while True:
                line = f.readline()
                if not line:
                    break
                ll = len(line)
                stripped = line.strip()
                if stripped:
                    rec = json.loads(stripped)
                    uid = rec.get("trade_uid") or rec.get("trade_id")
                    if uid:
                        uid_to_offsets[uid].append((byte_off, ll))
                byte_off += ll
        return dict(uid_to_offsets)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        io = self._reconstruct_io_features(rec)
        return {
            "io_features": torch.from_numpy(io),  # (W, 89) float32
            "scalars": dict(rec.get("scalars", {})),
            "teacher": {
                "final_pnl_bps": float(rec.get("teacher_final_pnl_bps", 0.0) or 0.0),
                "final_mfe_bps": float(rec.get("teacher_final_mfe_bps", 0.0) or 0.0),
                "final_mae_bps": float(rec.get("teacher_final_mae_bps", 0.0) or 0.0),
                "duration_bars": int(rec.get("teacher_duration_bars", 0) or 0),
            },
            "meta": {
                "ts": str(rec.get("ts", "")),
                "trade_uid": str(rec.get("trade_uid", "")),
                "trade_id": str(rec.get("trade_id", "")),
                "side": str(rec.get("side", "")),
                "run_id": str(rec.get("run_id", "")),
            },
        }

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_io_features(self, rec: Dict[str, Any]) -> np.ndarray:
        """Reconstruct the (window_len, input_dim) io_features tensor for a record."""
        m1_idx_now = int(rec["m1_idx_now"])
        win_start = m1_idx_now - self.window_len + 1
        win_end = m1_idx_now + 1
        if win_start < 0 or win_end > int(self.matrix.shape[0]):
            raise IndexError(
                f"[THIN_RECORD_DATASET] m1_idx_now={m1_idx_now} window out of bounds "
                f"(matrix len={self.matrix.shape[0]}, window_len={self.window_len})"
            )
        io = np.array(self.matrix[win_start:win_end], dtype=np.float32, copy=True)

        # Apply trade-state overlay over the in-trade portion of the window
        n_in_trade = int(rec.get("n_in_trade_bars", 0) or 0)
        if n_in_trade > 0:
            in_trade_start = int(rec.get("in_trade_start_in_win", 0))
            overlay_start_row = int(rec.get("overlay_start_row", 0))
            trade_uid = rec.get("trade_uid")
            if trade_uid not in self.overlay_index:
                raise KeyError(f"[THIN_RECORD_DATASET] trade_uid {trade_uid} not in overlay_index")
            ovl_off, ovl_len = self.overlay_index[trade_uid]
            slice_start = ovl_off + overlay_start_row
            slice_end = slice_start + n_in_trade
            if slice_end > ovl_off + ovl_len:
                raise IndexError(
                    f"[THIN_RECORD_DATASET] overlay slice out of range for trade {trade_uid}: "
                    f"slice_end={slice_end} bound={ovl_off + ovl_len}"
                )
            overlay_rows = np.array(self.overlays[slice_start:slice_end], dtype=np.float32, copy=False)
            io[in_trade_start: in_trade_start + n_in_trade, self.trade_state_indices] = overlay_rows
        return io


def attach_labels_to_thin_records(
    records: List[Dict[str, Any]],
    *,
    derive_should_exit_fn=None,
) -> None:
    """Attach training labels to thin records in-place using teacher hindsight.

    The V3 trainer expects each record to carry:
      - should_exit              (binary main label)
      - profit_protect_should_exit  (binary)
      - profit_protect_train_mask   (binary)
      - sample_weight (optional, float)

    These are derived from teacher_final_pnl_bps / teacher_final_mfe_bps /
    teacher_final_mae_bps / teacher_duration_bars + per-bar scalars.

    The actual derivation logic should be imported from
    `gx1.policy.exit_transformer_v0._attach_labels_to_exit_records` once the
    trainer-side adapter is integrated. This stub provides the hook point.
    """
    if derive_should_exit_fn is None:
        # Import lazily — avoids circular import + keeps this module standalone.
        from gx1.policy import exit_transformer_v0
        derive_should_exit_fn = exit_transformer_v0._attach_labels_to_exit_records  # type: ignore[attr-defined]
    derive_should_exit_fn(records)


__all__ = ["ThinRecordDataset", "attach_labels_to_thin_records"]
