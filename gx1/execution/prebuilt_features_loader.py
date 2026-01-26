"""
Prebuilt Features Loader - Isolated module for loading and accessing prebuilt features.

FASE 1: This module is completely isolated from feature-building code.
It does NOT import:
- gx1.features.basic_v1
- gx1.execution.live_features
- gx1.features.runtime_v10_ctx
- gx1.features.runtime_sniper_core

It ONLY:
- Loads parquet files
- Validates SHA256
- Provides .loc[timestamp] access to prebuilt features
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd

log = logging.getLogger(__name__)


class PrebuiltFeaturesLoader:
    """
    Isolated loader for prebuilt features.
    
    FASE 1: This class has ZERO dependencies on feature-building code.
    """
    
    def __init__(self, prebuilt_path: Path):
        # Initialize lookup counters (thread-safe via runner-level counters)
        # These are tracked at runner level, but we store metadata here
        self.prebuilt_index_aligned = False  # Set to True after alignment validation
        self.subset_first_ts = None
        self.subset_last_ts = None
        self.subset_rows = 0
        """
        Initialize prebuilt features loader.
        
        Args:
            prebuilt_path: Path to prebuilt features parquet file
            
        Raises:
            FileNotFoundError: If prebuilt file or manifest does not exist
            RuntimeError: If SHA256 validation fails
        """
        self.prebuilt_path = Path(prebuilt_path)
        self.manifest_path = self.prebuilt_path.parent / f"{self.prebuilt_path.stem}.manifest.json"
        
        # Validate files exist
        if not self.prebuilt_path.exists():
            raise FileNotFoundError(
                f"[PREBUILT_FAIL] Prebuilt features file not found: {self.prebuilt_path}"
            )
        
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"[PREBUILT_FAIL] Prebuilt features manifest not found: {self.manifest_path}"
            )
        
        # Load manifest
        with open(self.manifest_path, "r") as f:
            self.manifest = json.load(f)
        
        # Load prebuilt features DataFrame
        log.info(f"[PREBUILT_LOADER] Loading prebuilt features from {self.prebuilt_path}")
        self.df = pd.read_parquet(self.prebuilt_path)
        log.info(f"[PREBUILT_LOADER] Loaded {len(self.df):,} rows, {len(self.df.columns)} columns")
        
        # DEL 6: Hardening tripwire - hard-fail if CLOSE or other reserved columns found
        from gx1.runtime.column_collision_guard import RESERVED_CANDLE_COLUMNS
        reserved_found = []
        for col in self.df.columns:
            if col.lower() in RESERVED_CANDLE_COLUMNS:
                reserved_found.append(col)
        
        if reserved_found:
            raise RuntimeError(
                f"[PREBUILT_LOADER_FAIL] Prebuilt schema contains reserved candle columns: {reserved_found}. "
                f"Reserved columns (case-insensitive): {sorted(RESERVED_CANDLE_COLUMNS)}. "
                f"Prebuilt schema is contaminated - CLOSE and other reserved columns must be dropped before writing parquet. "
                f"Check that build_fullyear_features_parquet.py sanitize_feature_columns() is working correctly. "
                f"Prebuilt file: {self.prebuilt_path}"
            )
        
        # Validate SHA256
        self.sha256 = self._compute_sha256()
        expected_sha256 = self.manifest.get("sha256")
        if expected_sha256 and self.sha256 != expected_sha256:
            raise RuntimeError(
                f"[PREBUILT_FAIL] SHA256 mismatch: expected={expected_sha256[:16]}..., "
                f"computed={self.sha256[:16]}...\n"
                f"Instructions: Rebuild prebuilt features file."
            )
        
        log.info(f"[PREBUILT_LOADER] SHA256 validated: {self.sha256[:16]}...")
        
        # Store metadata
        self.schema_version = self.manifest.get("schema_version", "unknown")
        self.prebuilt_path_resolved = str(self.prebuilt_path.resolve())
        
        # Ensure index is DatetimeIndex
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if "ts" in self.df.columns:
                self.df["ts"] = pd.to_datetime(self.df["ts"])
                self.df = self.df.set_index("ts")
            else:
                raise ValueError(
                    f"[PREBUILT_FAIL] Prebuilt features DataFrame must have DatetimeIndex or 'ts' column. "
                    f"Got index type: {type(self.df.index)}"
                )
        
        # Sort by index
        self.df = self.df.sort_index()
        
        # Initialize lookup metadata (set after alignment validation)
        self.prebuilt_index_aligned = False  # Set to True after alignment validation
        self.subset_first_ts = None
        self.subset_last_ts = None
        self.subset_rows = 0
        
        # ATOMIC LOOKUP ACCOUNTING: Track lookup attempts/hits/misses atomically
        # These counters are maintained by PrebuiltFeaturesLoader to ensure atomicity
        self._lookup_attempts = 0
        self._lookup_hits = 0
        self._lookup_misses = 0
        self._lookup_miss_details = []  # Store first 3 miss details for debugging
    
    def get_lookup_accounting(self) -> Dict[str, Any]:
        """
        Get lookup accounting statistics.
        
        Returns:
            Dict with attempts, hits, misses, miss_details, and invariant_holds
        """
        # Invariant: hits + misses == attempts (atomic accounting)
        invariant_holds = (self._lookup_hits + self._lookup_misses) == self._lookup_attempts
        return {
            "attempts": self._lookup_attempts,
            "hits": self._lookup_hits,
            "misses": self._lookup_misses,
            "miss_details": self._lookup_miss_details[:3],  # First 3 only
            "invariant_holds": invariant_holds,
        }
    
    def _compute_sha256(self) -> str:
        """Compute SHA256 hash of prebuilt features file."""
        sha256_hash = hashlib.sha256()
        with open(self.prebuilt_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def get_features_for_timestamp(
        self, 
        timestamp: pd.Timestamp,
        required_columns: Optional[list] = None
    ) -> pd.Series:
        """
        Get features for a specific timestamp using .loc[timestamp].
        
        FASE 1: This is the ONLY way to access prebuilt features.
        No feature-building code is involved.
        
        Args:
            timestamp: Timestamp to look up
            required_columns: Optional list of required column names
            
        Returns:
            pd.Series with features for the timestamp
            
        Raises:
            KeyError: If timestamp is not found in prebuilt features
            RuntimeError: If required columns are missing
        """
        try:
            features = self.df.loc[timestamp]
        except KeyError:
            # Find nearest timestamp for better error message
            nearest_before = self.df.index[self.df.index <= timestamp]
            nearest_after = self.df.index[self.df.index > timestamp]
            
            nearest_ts = None
            diff_sec = None
            if len(nearest_before) > 0:
                nearest_ts = nearest_before[-1]
                diff_sec = (timestamp - nearest_ts).total_seconds()
            elif len(nearest_after) > 0:
                nearest_ts = nearest_after[0]
                diff_sec = (nearest_ts - timestamp).total_seconds()
            
            error_msg = f"[PREBUILT_FAIL] Timestamp {timestamp} not found in prebuilt features."
            if nearest_ts:
                error_msg += f" Nearest: {nearest_ts} (diff={diff_sec:.1f}s)"
            raise KeyError(error_msg)
        
        # Validate required columns
        if required_columns:
            missing_cols = set(required_columns) - set(features.index if isinstance(features, pd.Series) else features.columns)
            if missing_cols:
                raise RuntimeError(
                    f"[PREBUILT_FAIL] Missing required columns: {sorted(missing_cols)}"
                )
        
        return features
    
    def get_features_for_timestamps(
        self,
        timestamps: pd.DatetimeIndex,
        required_columns: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Get features for multiple timestamps using .loc[timestamps].
        
        FASE 1: This is the ONLY way to access prebuilt features in batch.
        
        Args:
            timestamps: DatetimeIndex of timestamps to look up
            required_columns: Optional list of required column names
            
        Returns:
            pd.DataFrame with features for the timestamps
            
        Raises:
            KeyError: If any timestamp is not found
            RuntimeError: If required columns are missing
        """
        # Use .loc with indexer to get multiple rows
        try:
            features_df = self.df.loc[timestamps]
        except KeyError as e:
            # Find which timestamps are missing
            missing = []
            for ts in timestamps:
                if ts not in self.df.index:
                    missing.append(ts)
            
            raise KeyError(
                f"[PREBUILT_FAIL] {len(missing)} timestamps not found in prebuilt features. "
                f"First missing: {missing[0] if missing else 'N/A'}"
            ) from e
        
        # Validate required columns
        if required_columns:
            missing_cols = set(required_columns) - set(features_df.columns)
            if missing_cols:
                raise RuntimeError(
                    f"[PREBUILT_FAIL] Missing required columns: {sorted(missing_cols)}"
                )
        
        return features_df
    
    def validate_timestamp_alignment(
        self,
        raw_timestamps: pd.DatetimeIndex,
        sample_size: int = 1000,
        random_mid_check: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that prebuilt features align with raw data timestamps.
        
        Args:
            raw_timestamps: DatetimeIndex from raw data
            sample_size: Number of timestamps to check (first N)
            random_mid_check: If True, also check a random mid-point
            
        Returns:
            (is_valid, error_message)
        """
        # Check first N timestamps
        check_timestamps = raw_timestamps[:sample_size]
        missing = []
        for ts in check_timestamps:
            if ts not in self.df.index:
                missing.append(ts)
        
        if missing:
            return False, f"First {sample_size} check: {len(missing)} timestamps missing. First: {missing[0]}"
        
        # Random mid-point check
        if random_mid_check and len(raw_timestamps) > sample_size * 2:
            import random
            mid_idx = random.randint(sample_size, len(raw_timestamps) - sample_size)
            mid_ts = raw_timestamps[mid_idx]
            if mid_ts not in self.df.index:
                return False, f"Mid-point check (idx={mid_idx}): timestamp {mid_ts} missing"
        
        return True, None
    
    def get_common_index(self, other_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Get common timestamps between prebuilt features and another index.
        
        FASE 1: This ensures deterministic alignment before chunking.
        
        Args:
            other_index: DatetimeIndex to intersect with
            
        Returns:
            DatetimeIndex with common timestamps
        """
        common = self.df.index.intersection(other_index)
        common_sorted = common.sort_values()
        
        # Update metadata after alignment
        if len(common_sorted) > 0:
            self.prebuilt_index_aligned = True
            self.subset_first_ts = common_sorted[0]
            self.subset_last_ts = common_sorted[-1]
            self.subset_rows = len(common_sorted)
        
        return common_sorted
    
    def validate_feature_schema(
        self,
        feature_meta_path: Path,
        is_replay: bool = True,
    ) -> None:
        """
        Validate prebuilt feature schema/dims against model expectations (feature_meta).
        
        DEL C3: Hard-fail on schema/dims mismatch with detailed diff.
        
        Args:
            feature_meta_path: Path to feature_meta.json (model expectations)
            is_replay: If True, hard-fail on mismatch. If False, log warning.
            
        Raises:
            RuntimeError: If schema/dims mismatch (hard-fail in replay mode)
        """
        if not feature_meta_path.exists():
            error_msg = f"[PREBUILT_FAIL] Feature meta file not found: {feature_meta_path}"
            if is_replay:
                raise FileNotFoundError(error_msg)
            else:
                log.warning(f"{error_msg} (live mode: continuing)")
                return
        
        # Load feature_meta
        with open(feature_meta_path, "r") as f:
            feature_meta = json.load(f)
        
        # Extract expected features
        seq_features = feature_meta.get("seq_features", [])
        snap_features = feature_meta.get("snap_features", [])
        
        if not seq_features or not snap_features:
            error_msg = (
                f"[PREBUILT_FAIL] Feature meta missing seq_features or snap_features. "
                f"seq_features: {len(seq_features) if seq_features else 0}, "
                f"snap_features: {len(snap_features) if snap_features else 0}"
            )
            if is_replay:
                raise ValueError(error_msg)
            else:
                log.warning(f"{error_msg} (live mode: continuing)")
                return
        
        # Get actual prebuilt columns
        actual_cols = set(self.df.columns)
        expected_seq = set(seq_features)
        expected_snap = set(snap_features)
        expected_all = expected_seq | expected_snap
        
        # Build diff report
        missing_seq = expected_seq - actual_cols
        missing_snap = expected_snap - actual_cols
        missing_all = expected_all - actual_cols
        extra_cols = actual_cols - expected_all
        
        # Check for mismatches
        has_mismatch = bool(missing_all or (extra_cols and is_replay))
        
        if has_mismatch:
            # Build detailed diff message
            diff_lines = []
            diff_lines.append("[PREBUILT_FAIL] Feature schema/dims mismatch:")
            diff_lines.append("")
            
            if missing_seq:
                diff_lines.append(f"Missing seq features ({len(missing_seq)}):")
                for col in sorted(missing_seq)[:10]:  # First 10
                    diff_lines.append(f"  - {col}")
                if len(missing_seq) > 10:
                    diff_lines.append(f"  ... and {len(missing_seq) - 10} more")
                diff_lines.append("")
            
            if missing_snap:
                diff_lines.append(f"Missing snap features ({len(missing_snap)}):")
                for col in sorted(missing_snap)[:10]:  # First 10
                    diff_lines.append(f"  - {col}")
                if len(missing_snap) > 10:
                    diff_lines.append(f"  ... and {len(missing_snap) - 10} more")
                diff_lines.append("")
            
            if extra_cols and is_replay:
                diff_lines.append(f"Extra columns in prebuilt ({len(extra_cols)}):")
                for col in sorted(extra_cols)[:10]:  # First 10
                    diff_lines.append(f"  + {col}")
                if len(extra_cols) > 10:
                    diff_lines.append(f"  ... and {len(extra_cols) - 10} more")
                diff_lines.append("")
            
            diff_lines.append(f"Expected seq dim: {len(expected_seq)}")
            diff_lines.append(f"Expected snap dim: {len(expected_snap)}")
            diff_lines.append(f"Actual prebuilt cols: {len(actual_cols)}")
            diff_lines.append("")
            diff_lines.append("Instructions: Rebuild prebuilt features with correct feature_meta.")
            
            error_msg = "\n".join(diff_lines)
            
            if is_replay:
                raise RuntimeError(error_msg)
            else:
                log.warning(f"{error_msg} (live mode: continuing)")
        
        log.info(
            f"[PREBUILT_LOADER] Feature schema validated: "
            f"seq={len(seq_features)}, snap={len(snap_features)}, "
            f"all_match={not has_mismatch}"
        )