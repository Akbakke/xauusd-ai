"""Disk-spilling labeled dataset for the V6+ exit-transformer trainers.

Background
----------
The V5 / V6-in-memory pipeline did:
  1. Load full 2.5M-record JSONL into parent RAM (~6-8 GB).
  2. Build 3 split-lists referencing the same records (timestamp split).
  3. Run spawn-labeler that returns an `out` dict (~5-8 GB more in parent).
  4. Hold all of the above for the full training run.

That breaks 16 GB hosts: parent RSS ends up 12+ GB before training starts.
The kernel OOM-kills whichever process happens to have the highest oom_score
(observed: a spawn worker hit 12.5 GB anon-rss during the V6 attempt).

This module replaces that pipeline with a disk-only flow:

  1. Build trade-uid -> [(byte_off, byte_len)] index over records.jsonl.
     Parent retains only the index (~30 MB for 53k uids).
  2. Split uids by timestamp in one pass over records.jsonl. No record
     storage; only uid -> first_ts is held.
  3. For each split, spawn-label trades and stream labeled records to a
     per-split labeled.jsonl. Parent collects only uid -> [byte ranges
     in labeled.jsonl].
  4. DiskLabeledThinDataset reads ONE labeled record per __getitem__ via
     seek + JSON parse.

Result: parent RSS during V6 training is dominated by model weights + the
mmap'd matrix, not by record storage.

Spawn worker code is reused from `train_exit_v5_thin_records` (the
`_spawn_init_worker` + `_spawn_label_one_trade` pair already opens
ThinRecordDataset with `load_records_eagerly=False`).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from gx1.exits.training.thin_record_dataset import ThinRecordDataset


# ---------------------------------------------------------------------------
# UID-time split (no records held in RAM)
# ---------------------------------------------------------------------------


def _parse_ts(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    try:
        ts = pd.Timestamp(s)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts
    except Exception:
        return None


def split_uids_by_time(
    records_path: Path,
    *,
    train_cutoff: str,
    val_cutoff: str,
) -> Tuple[
    Dict[str, List[Tuple[int, int]]],
    Dict[str, List[Tuple[int, int]]],
    Dict[str, List[Tuple[int, int]]],
]:
    """One-pass split over records.jsonl by trade-uid first-record timestamp.

    Returns (train_uid_to_offsets, val_uid_to_offsets, test_uid_to_offsets).
    Each value is the list of (byte_off, byte_len) ranges for that uid in
    records.jsonl, in JSONL order.
    """
    train_ts = pd.Timestamp(train_cutoff, tz="UTC")
    val_ts = pd.Timestamp(val_cutoff, tz="UTC")

    uid_first_ts: Dict[str, pd.Timestamp] = {}
    uid_offsets: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    with records_path.open("rb") as f:
        byte_off = 0
        for line in f:
            ll = len(line)
            stripped = line.strip()
            if stripped:
                rec = json.loads(stripped)
                uid = rec.get("trade_uid") or rec.get("trade_id")
                if uid:
                    uid_offsets[uid].append((byte_off, ll))
                    ts = _parse_ts(rec.get("ts"))
                    if ts is not None:
                        prev = uid_first_ts.get(uid)
                        if prev is None or ts < prev:
                            uid_first_ts[uid] = ts
            byte_off += ll

    train: Dict[str, List[Tuple[int, int]]] = {}
    val: Dict[str, List[Tuple[int, int]]] = {}
    test: Dict[str, List[Tuple[int, int]]] = {}
    for uid, offsets in uid_offsets.items():
        first_ts = uid_first_ts.get(uid)
        if first_ts is None or first_ts < train_ts:
            train[uid] = offsets
        elif first_ts < val_ts:
            val[uid] = offsets
        else:
            test[uid] = offsets
    return train, val, test


# ---------------------------------------------------------------------------
# Side-weights without loading records (peeks first record per uid)
# ---------------------------------------------------------------------------


def compute_side_weights_from_offsets(
    records_path: Path,
    uid_to_offsets: Dict[str, List[Tuple[int, int]]],
) -> Tuple[Dict[str, float], int, int]:
    """Compute long/short class-balance weights from per-uid first-record side.

    Each trade contributes its full record count (so weights balance records,
    not trades — matches the v5/v6 in-memory implementation that mutated
    `r['sample_weight']` per record).

    Returns ({"long": long_w, "short": short_w}, n_long_records, n_short_records).
    """
    n_long_recs = 0
    n_short_recs = 0
    with open(records_path, "rb") as f:
        for byte_ranges in uid_to_offsets.values():
            if not byte_ranges:
                continue
            byte_off, byte_len = byte_ranges[0]
            f.seek(byte_off)
            line = f.read(byte_len).strip()
            if not line:
                continue
            rec = json.loads(line)
            side = rec.get("side", "")
            n_recs = len(byte_ranges)
            if side == "long":
                n_long_recs += n_recs
            elif side == "short":
                n_short_recs += n_recs
    n_total = n_long_recs + n_short_recs
    return (
        {
            "long": n_total / max(2 * n_long_recs, 1),
            "short": n_total / max(2 * n_short_recs, 1),
        },
        n_long_recs,
        n_short_recs,
    )


# ---------------------------------------------------------------------------
# Sequential disk-spill labeler (used when EXIT_LABEL_WORKERS=1 or as fallback)
# ---------------------------------------------------------------------------


def _label_to_disk_sequential(
    dataset_dir: Path,
    uid_to_offsets: Dict[str, List[Tuple[int, int]]],
    out_jsonl: Path,
    *,
    progress_every: int = 5000,
) -> Dict[str, List[Tuple[int, int]]]:
    from gx1.policy.exit_transformer_v0 import _attach_labels_to_exit_records  # type: ignore

    base = ThinRecordDataset(
        dataset_dir, load_records_eagerly=False, build_offset_index=False,
    )
    io_version = str(base.manifest.get("io_version", ""))
    n_trades = len(uid_to_offsets)
    out_offsets: Dict[str, List[Tuple[int, int]]] = {}
    t0 = time.time()

    with open(base.records_path, "rb") as records_fh, out_jsonl.open("wb") as out_fh:
        byte_off_out = 0
        for ti, (uid, byte_ranges) in enumerate(uid_to_offsets.items()):
            trade_recs: List[Dict[str, Any]] = []
            for byte_o, byte_l in byte_ranges:
                records_fh.seek(byte_o)
                ln = records_fh.read(byte_l)
                stripped = ln.strip()
                if stripped:
                    trade_recs.append(json.loads(stripped))
            for rec in trade_recs:
                sc = rec.get("scalars") or {}
                for k, v in sc.items():
                    rec.setdefault(k, v)
                rec.setdefault("exit_ml_io_version", io_version)
                rec["io_features"] = base._reconstruct_io_features(rec).tolist()
            _attach_labels_to_exit_records(trade_recs)
            for rec in trade_recs:
                rec.pop("io_features", None)
            ranges: List[Tuple[int, int]] = []
            for rec in trade_recs:
                line = (json.dumps(rec, default=float) + "\n").encode("utf-8")
                out_fh.write(line)
                ranges.append((byte_off_out, len(line)))
                byte_off_out += len(line)
            out_offsets[uid] = ranges
            if (ti + 1) % progress_every == 0:
                print(
                    f"   {ti+1:,}/{n_trades:,} trades labeled "
                    f"({time.time()-t0:.1f}s)",
                    flush=True,
                )
    return out_offsets


# ---------------------------------------------------------------------------
# Spawn disk-spill labeler
# ---------------------------------------------------------------------------


def label_uids_to_disk(
    dataset_dir: Path,
    uid_to_offsets: Dict[str, List[Tuple[int, int]]],
    *,
    out_jsonl: Path,
    n_workers: int = 1,
    chunksize: int = 4,
    progress_every: int = 5000,
) -> Dict[str, List[Tuple[int, int]]]:
    """Label trades and stream labeled records to disk.

    Parent retains only `uid -> [(byte_off, byte_len) in out_jsonl]`. The
    labeled records themselves never live in parent RAM beyond a single chunk.

    `chunksize` defaults to 4 (was 16 in v6); smaller chunks reduce pickle
    buffer pressure on the worker side and avoid the spike we suspect drove
    the OOM crash earlier.
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if n_workers <= 1:
        return _label_to_disk_sequential(
            dataset_dir, uid_to_offsets, out_jsonl, progress_every=progress_every,
        )

    from gx1.scripts.train_exit_v5_thin_records import (
        _spawn_init_worker,
        _spawn_label_one_trade,
    )

    out_offsets: Dict[str, List[Tuple[int, int]]] = {}
    n_trades = len(uid_to_offsets)
    jobs = list(uid_to_offsets.items())
    t0 = time.time()
    ctx = mp.get_context("spawn")

    with out_jsonl.open("wb") as fh, ctx.Pool(
        processes=n_workers,
        initializer=_spawn_init_worker,
        initargs=(str(dataset_dir),),
    ) as pool:
        byte_off = 0
        for ti, (uid, labeled) in enumerate(
            pool.imap_unordered(_spawn_label_one_trade, jobs, chunksize=chunksize)
        ):
            ranges: List[Tuple[int, int]] = []
            for rec in labeled:
                line = (json.dumps(rec, default=float) + "\n").encode("utf-8")
                fh.write(line)
                ranges.append((byte_off, len(line)))
                byte_off += len(line)
            out_offsets[uid] = ranges
            if (ti + 1) % progress_every == 0:
                print(
                    f"   {ti+1:,}/{n_trades:,} trades labeled "
                    f"({time.time()-t0:.1f}s)",
                    flush=True,
                )
    return out_offsets


# ---------------------------------------------------------------------------
# Disk-backed Dataset
# ---------------------------------------------------------------------------


def flatten_uid_offsets(
    uid_to_offsets: Dict[str, List[Tuple[int, int]]],
    uid_order: Optional[List[str]] = None,
) -> List[Tuple[int, int]]:
    """Flatten {uid: [(byte_off, byte_len), ...]} into a single list in uid_order.

    If `uid_order` is None, iterates dict insertion order.
    """
    flat: List[Tuple[int, int]] = []
    keys = uid_order if uid_order is not None else list(uid_to_offsets.keys())
    for uid in keys:
        flat.extend(uid_to_offsets[uid])
    return flat


class DiskLabeledThinDataset(Dataset):
    """Reads labeled records from a JSONL on disk via byte offsets.

    Parent retains only the offset list (~16 bytes/record) and the model;
    each __getitem__ does one disk seek + JSON parse + numpy reconstruction.

    Side weights are applied at read time so the labeled JSONL doesn't need
    to be re-written when class-balance settings change.

    Note: file handle is opened lazily and reused across __getitem__ calls.
    DataLoader num_workers must stay 0 — file handles are not pickleable.
    """

    def __init__(
        self,
        base: ThinRecordDataset,
        *,
        labeled_jsonl: Path,
        record_offsets: List[Tuple[int, int]],
        side_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.base = base
        self.labeled_jsonl = Path(labeled_jsonl)
        self.record_offsets = record_offsets
        self.side_weights: Dict[str, float] = side_weights or {}
        self._fh: Optional[Any] = None
        self._fh_pid: Optional[int] = None

    def __len__(self) -> int:
        return len(self.record_offsets)

    # ── DataLoader-worker safety (2026-05-26) ─────────────────────────────────
    # The raw file handle must NOT be shared across processes: on fork the
    # inherited fd causes seek races (silent read corruption); on spawn it isn't
    # picklable. Fix: open a PER-PROCESS handle (reopen when the pid changes) and
    # drop the handle from pickled state. This makes num_workers>0 safe → the
    # data loader can prefetch in parallel (was the real V3-train bottleneck).
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_fh_pid"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._fh = None
        self._fh_pid = None

    def _ensure_fh(self) -> None:
        import os as _os
        pid = _os.getpid()
        if self._fh is None or self._fh_pid != pid:
            # reopen in this process (handles both fork-inherited fd and spawn)
            self._fh = open(self.labeled_jsonl, "rb")
            self._fh_pid = pid

    def _read_record(self, idx: int) -> Dict[str, Any]:
        self._ensure_fh()
        byte_off, byte_len = self.record_offsets[idx]
        self._fh.seek(byte_off)
        return json.loads(self._fh.read(byte_len))

    def _sample_weight(self, rec: Dict[str, Any]) -> float:
        base_w = float(rec.get("sample_weight", 1.0) or 1.0)
        if self.side_weights:
            return base_w * self.side_weights.get(rec.get("side", ""), 1.0)
        return base_w

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = self._read_record(idx)
        io = self.base._reconstruct_io_features(rec)
        x = torch.from_numpy(io)
        y = torch.tensor(float(rec.get("should_exit", 0.0) or 0.0), dtype=torch.float32)
        w = torch.tensor(self._sample_weight(rec), dtype=torch.float32)
        return x, y, w


class DiskMultiTaskThinDataset(DiskLabeledThinDataset):
    """V6 multi-task variant — returns (main, profit, family) labels.

    Family-class mapping is sourced from `gx1.policy.exit_transformer_v0`
    (model `family_head_dim` matches `len(EXIT_AUX_FAMILY_NAMES)`). Records
    whose `exit_aux_family` is not one of the canonical family names get
    `EXIT_AUX_FAMILY_IGNORE_INDEX` (-100) so torch's cross_entropy skips them
    via its default `ignore_index`.

    V12.2 multi-TF: when `multi_tf_feats` is provided, the y_dict gets
    additional keys `seq_m5, seq_m15, seq_h1, seq_h4, seq_d1` — each a
    (multi_tf_seq_len, n_features) tensor sliced at-or-before the record's
    timestamp. Training helpers extract these and pass to model.forward.
    """

    def __init__(self, *args,
                 multi_tf_feats: Optional[Dict[str, "pd.DataFrame"]] = None,
                 multi_tf_shift: Optional[Dict[str, "pd.Timedelta"]] = None,
                 multi_tf_seq_len: int = 96,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_tf_feats = multi_tf_feats
        self.multi_tf_shift = multi_tf_shift
        self.multi_tf_seq_len = int(multi_tf_seq_len)

    def __getitem__(
        self, idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        from gx1.policy.exit_transformer_v0 import (
            EXIT_AUX_FAMILY_TO_ID, EXIT_AUX_FAMILY_IGNORE_INDEX,
        )
        rec = self._read_record(idx)
        io = self.base._reconstruct_io_features(rec)
        x = torch.from_numpy(io)
        y_main = torch.tensor(
            float(rec.get("should_exit", 0.0) or 0.0), dtype=torch.float32,
        )
        y_profit = torch.tensor(
            float(rec.get("profit_protect_should_exit", 0.0) or 0.0),
            dtype=torch.float32,
        )
        family_str = str(rec.get("exit_aux_family") or "").strip().upper()
        family_idx = EXIT_AUX_FAMILY_TO_ID.get(family_str, EXIT_AUX_FAMILY_IGNORE_INDEX)
        y_family = torch.tensor(family_idx, dtype=torch.long)
        w = torch.tensor(self._sample_weight(rec), dtype=torch.float32)
        out_dict: Dict[str, torch.Tensor] = {
            "main": y_main, "profit": y_profit, "family": y_family,
        }
        # V3 dip-head recovery targets (gated by enable_dip_head at train time).
        for _Kdip in (12, 48, 144):
            out_dict[f"dip_recovery_K{_Kdip}"] = torch.tensor(
                float(rec.get(f"y_dip_recovery_K{_Kdip}", 0.0) or 0.0), dtype=torch.float32,
            )
        # V3 NEW-head targets (2026-05-26) — exit-oriented, forward-from-in-trade.
        # Gated by enable_*_head at train time (default 0.0 if labeler didn't write them).
        for _K in (12, 48, 144):
            for _nm in ("forecast", "tail_mae", "time_to_recovery_frac",
                        "time_to_worst_frac", "vol_fwd"):
                out_dict[f"v3_{_nm}_K{_K}"] = torch.tensor(
                    float(rec.get(f"y_v3_{_nm}_K{_K}", 0.0) or 0.0), dtype=torch.float32,
                )
        # V12.2: slice multi-TF windows at record timestamp
        if self.multi_tf_feats is not None:
            import pandas as pd
            from gx1.features.htf_features import get_last_n_at_or_before
            ts = pd.Timestamp(rec.get("ts"))
            for tf, feats in self.multi_tf_feats.items():
                arr = get_last_n_at_or_before(
                    feats, ts, n=self.multi_tf_seq_len,
                    tf_shift=self.multi_tf_shift[tf],
                )
                out_dict[f"seq_{tf.lower()}"] = torch.from_numpy(arr)
        return x, out_dict, w


# ---------------------------------------------------------------------------
# Pos-rate scan over labeled JSONL (for BCE pos_weight, logging)
# ---------------------------------------------------------------------------


def scan_labeled_pos_rate(
    labeled_jsonl: Path, record_offsets: List[Tuple[int, int]],
) -> float:
    """Compute should_exit pos rate over a labeled-JSONL slice without loading all."""
    if not record_offsets:
        return 0.0
    n_pos = 0
    n_total = 0
    with open(labeled_jsonl, "rb") as f:
        for byte_off, byte_len in record_offsets:
            f.seek(byte_off)
            line = f.read(byte_len).strip()
            if not line:
                continue
            rec = json.loads(line)
            n_total += 1
            if float(rec.get("should_exit", 0.0) or 0.0) > 0.5:
                n_pos += 1
    return n_pos / max(n_total, 1)


__all__ = [
    "split_uids_by_time",
    "compute_side_weights_from_offsets",
    "label_uids_to_disk",
    "flatten_uid_offsets",
    "DiskLabeledThinDataset",
    "DiskMultiTaskThinDataset",
    "scan_labeled_pos_rate",
]
