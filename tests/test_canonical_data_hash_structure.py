#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRUTH Canonical Data Hash Structure Tests

Tests that canonical_data_hash is sensitive to all structural changes:
- Row count
- Column count
- Column order
- Column names (including whitespace)
- Schema/dtypes
- Cell values

And stable when nothing changes.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_canonical_data_hash(df: pd.DataFrame) -> str:
    """
    Compute canonical data hash (extracted from merge_artifacts logic).
    
    This function replicates the exact logic from replay_eval_gated_parallel.py
    to ensure test consistency.
    """
    import json as json_lib
    import numpy as np
    
    # TRUTH: Validate column names (no hidden whitespace)
    for col in df.columns:
        if col != col.strip():
            raise RuntimeError(
                f"[TRUTH_CANONICAL_HASH_FAIL] Column name '{col}' contains whitespace. "
                f"Canonical hash requires normalized column names (name == name.strip())."
            )
    
    # TRUTH: Assert no NaNs in hashed columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Use is_datetime64_any_dtype to catch all datetime variants
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    hashed_cols = list(numeric_cols) + datetime_cols
    
    for col in hashed_cols:
        if col in df.columns:
            # Use isna() which works for both numeric NaNs and datetime NaT
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                raise RuntimeError(
                    f"[TRUTH_CANONICAL_HASH_FAIL] Column '{col}' has {null_count} NaNs. "
                    f"Canonical hash requires no NaNs in hashed columns for determinism."
                )
    
    # Build canonical schema JSON (sorted keys, canonical dtype names)
    schema_dict = {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            schema_dict[str(col)] = "datetime64[ns, UTC]"
        elif pd.api.types.is_integer_dtype(df[col]):
            schema_dict[str(col)] = "int64"
        elif pd.api.types.is_float_dtype(df[col]):
            schema_dict[str(col)] = "float64"
        else:
            schema_dict[str(col)] = str(df[col].dtype)
    canonical_schema_json = json_lib.dumps(schema_dict, sort_keys=True)
    
    # Build normalized data bytes (dtype/endianness normalized)
    data_bytes = b""
    for col in df.columns:
        col_data = df[col]
        
        # Determine canonical dtype
        if col_data.dtype == 'object':
            # String columns: convert to bytes with explicit encoding
            col_str = col_data.astype(str).str.cat(sep='|')
            col_bytes = col_str.encode('utf-8')
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            # Datetime: convert to int64 nanoseconds (UTC, deterministic)
            col_array = col_data.values.astype('datetime64[ns]')
            # Ensure contiguous and native endianness
            col_array = np.ascontiguousarray(col_array.view('int64'))
            if col_array.dtype.byteorder not in ('=', '|'):  # Not native or no byte order
                col_array = col_array.byteswap(False).newbyteorder('=')  # Force native
            col_bytes = col_array.tobytes()
        elif pd.api.types.is_numeric_dtype(col_data):
            # Numeric: normalize to float64 (or int64 for integers)
            if pd.api.types.is_integer_dtype(col_data):
                canonical_dtype = 'int64'
            else:
                canonical_dtype = 'float64'
            
            # Hard-cast to canonical dtype
            col_array = col_data.values.astype(canonical_dtype)
            # Ensure contiguous and native endianness
            col_array = np.ascontiguousarray(col_array)
            if col_array.dtype.byteorder not in ('=', '|'):  # Not native or no byte order
                col_array = col_array.byteswap(False).newbyteorder('=')  # Force native
            col_bytes = col_array.tobytes()
        else:
            # Other types: convert to string then bytes
            col_bytes = col_data.astype(str).str.cat(sep='|').encode('utf-8')
        
        data_bytes += col_bytes
    
    # Build explicit header structure
    n_rows = len(df)
    n_cols = len(df.columns)
    column_order = "|".join(df.columns)
    
    # Construct canonical hash input with explicit header
    header_lines = [
        f"ROWS={n_rows}",
        f"COLS={n_cols}",
        f"COLUMN_ORDER={column_order}",
        f"SCHEMA_JSON={canonical_schema_json}",
        "DATA_BYTES=",
    ]
    header_bytes = "\n".join(header_lines).encode('utf-8')
    
    # Combine header + data bytes
    canonical_hash_input = header_bytes + data_bytes
    
    # Compute hash
    return hashlib.sha256(canonical_hash_input).hexdigest()


def create_baseline_df() -> pd.DataFrame:
    """Create a small deterministic DataFrame for testing."""
    df = pd.DataFrame({
        "trade_uid": ["uid1", "uid2", "uid3", "uid4", "uid5"],
        "pnl_bps": [10.5, -5.2, 3.7, -2.1, 8.9],
        "entry_time": pd.to_datetime([
            "2025-01-01T00:00:00Z",
            "2025-01-02T00:00:00Z",
            "2025-01-03T00:00:00Z",
            "2025-01-04T00:00:00Z",
            "2025-01-05T00:00:00Z",
        ], utc=True),
        "trade_count": [1, 2, 3, 4, 5],
    })
    # Ensure datetime column is datetime64[ns] (not object)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    return df


class TestCanonicalDataHashStructure:
    """Test suite for canonical data hash structure sensitivity."""
    
    def test_a_baseline_stability(self):
        """A. Baseline stability: hash must match when computed twice."""
        df = create_baseline_df()
        hash1 = compute_canonical_data_hash(df)
        hash2 = compute_canonical_data_hash(df)
        assert hash1 == hash2, "Hash must be stable for identical input"
    
    def test_b_row_count_sensitivity(self):
        """B. Row count sensitivity: hash must change when rows change."""
        df = create_baseline_df()
        baseline_hash = compute_canonical_data_hash(df)
        
        # Add one row
        df_added = df.copy()
        df_added.loc[len(df_added)] = ["uid6", 1.0, pd.to_datetime("2025-01-06T00:00:00Z", utc=True), 6]
        hash_added = compute_canonical_data_hash(df_added)
        assert hash_added != baseline_hash, "Hash must change when row is added"
        
        # Remove one row
        df_removed = df.iloc[:-1].copy()
        hash_removed = compute_canonical_data_hash(df_removed)
        assert hash_removed != baseline_hash, "Hash must change when row is removed"
    
    def test_c_column_count_sensitivity(self):
        """C. Column count sensitivity: hash must change when columns change."""
        df = create_baseline_df()
        baseline_hash = compute_canonical_data_hash(df)
        
        # Add a new column
        df_added = df.copy()
        df_added["new_col"] = [1.0, 2.0, 3.0, 4.0, 5.0]
        hash_added = compute_canonical_data_hash(df_added)
        assert hash_added != baseline_hash, "Hash must change when column is added"
        
        # Drop a column
        df_dropped = df.drop(columns=["trade_count"])
        hash_dropped = compute_canonical_data_hash(df_dropped)
        assert hash_dropped != baseline_hash, "Hash must change when column is dropped"
    
    def test_d_column_order_sensitivity(self):
        """D. Column order sensitivity: hash must change when column order changes."""
        df = create_baseline_df()
        baseline_hash = compute_canonical_data_hash(df)
        
        # Reorder columns (same data)
        df_reordered = df[["trade_count", "pnl_bps", "entry_time", "trade_uid"]]
        hash_reordered = compute_canonical_data_hash(df_reordered)
        assert hash_reordered != baseline_hash, "Hash must change when column order changes"
    
    def test_e_column_name_sensitivity(self):
        """E. Column name sensitivity: hash must change when column name changes."""
        df = create_baseline_df()
        baseline_hash = compute_canonical_data_hash(df)
        
        # Rename a column (same data)
        df_renamed = df.rename(columns={"pnl_bps": "pnl_bps_renamed"})
        hash_renamed = compute_canonical_data_hash(df_renamed)
        assert hash_renamed != baseline_hash, "Hash must change when column name changes"
        
        # Test whitespace in column name (must hard-fail)
        df_whitespace = df.copy()
        df_whitespace.columns = [col if col != "pnl_bps" else "pnl_bps " for col in df_whitespace.columns]
        try:
            compute_canonical_data_hash(df_whitespace)
            assert False, "Should have raised RuntimeError for whitespace in column name"
        except RuntimeError as e:
            assert "contains whitespace" in str(e), f"Expected whitespace error, got: {e}"
    
    def test_f_schema_sensitivity(self):
        """F. Schema sensitivity: hash must change when dtype changes."""
        df = create_baseline_df()
        baseline_hash = compute_canonical_data_hash(df)
        
        # Change dtype: int64 → float64 (same numeric values)
        df_float = df.copy()
        df_float["trade_count"] = df_float["trade_count"].astype("float64")
        hash_float = compute_canonical_data_hash(df_float)
        assert hash_float != baseline_hash, "Hash must change when dtype changes (int64 → float64)"
    
    def test_g_cell_value_sensitivity(self):
        """G. Cell-level sensitivity: hash must change when any cell value changes."""
        df = create_baseline_df()
        baseline_hash = compute_canonical_data_hash(df)
        
        # Change a single float value by tiny amount
        df_modified = df.copy()
        df_modified.loc[0, "pnl_bps"] = df_modified.loc[0, "pnl_bps"] + 1e-10
        hash_modified = compute_canonical_data_hash(df_modified)
        assert hash_modified != baseline_hash, "Hash must change when cell value changes (even by 1e-10)"
        
        # Change a string value
        df_modified2 = df.copy()
        df_modified2.loc[0, "trade_uid"] = "uid1_modified"
        hash_modified2 = compute_canonical_data_hash(df_modified2)
        assert hash_modified2 != baseline_hash, "Hash must change when string cell value changes"
    
    def test_h_determinism_guardrails(self):
        """H. Determinism guardrails: verify header structure."""
        df = create_baseline_df()
        
        # Recompute hash manually to verify structure
        import json as json_lib
        import numpy as np
        
        # Build schema
        schema_dict = {}
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                schema_dict[str(col)] = "datetime64[ns, UTC]"
            elif pd.api.types.is_integer_dtype(df[col]):
                schema_dict[str(col)] = "int64"
            elif pd.api.types.is_float_dtype(df[col]):
                schema_dict[str(col)] = "float64"
            else:
                schema_dict[str(col)] = str(df[col].dtype)
        canonical_schema_json = json_lib.dumps(schema_dict, sort_keys=True)
        
        # Build data bytes (simplified for test)
        data_bytes = b""
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_array = df[col].values.astype('float64' if pd.api.types.is_float_dtype(df[col]) else 'int64')
                col_array = np.ascontiguousarray(col_array)
                if col_array.dtype.byteorder not in ('=', '|'):
                    col_array = col_array.byteswap(False).newbyteorder('=')
                data_bytes += col_array.tobytes()
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_array = df[col].values.astype('datetime64[ns]')
                col_array = np.ascontiguousarray(col_array.view('int64'))
                if col_array.dtype.byteorder not in ('=', '|'):
                    col_array = col_array.byteswap(False).newbyteorder('=')
                data_bytes += col_array.tobytes()
            else:
                col_bytes = df[col].astype(str).str.cat(sep='|').encode('utf-8')
                data_bytes += col_bytes
        
        # Build header
        n_rows = len(df)
        n_cols = len(df.columns)
        column_order = "|".join(df.columns)
        header_lines = [
            f"ROWS={n_rows}",
            f"COLS={n_cols}",
            f"COLUMN_ORDER={column_order}",
            f"SCHEMA_JSON={canonical_schema_json}",
            "DATA_BYTES=",
        ]
        header_bytes = "\n".join(header_lines).encode('utf-8')
        canonical_hash_input = header_bytes + data_bytes
        
        # Verify header structure
        hash_input_str = canonical_hash_input.decode('utf-8', errors='ignore')
        assert hash_input_str.startswith("ROWS="), "Hash input must start with ROWS="
        assert "COLS=" in hash_input_str, "Hash input must contain COLS="
        assert "COLUMN_ORDER=" in hash_input_str, "Hash input must contain COLUMN_ORDER="
        assert "SCHEMA_JSON=" in hash_input_str, "Hash input must contain SCHEMA_JSON="
        assert "DATA_BYTES=" in hash_input_str, "Hash input must contain DATA_BYTES="
        
        # Verify ROWS/COLS match df shape
        assert f"ROWS={n_rows}" in hash_input_str, f"ROWS must match df shape: {n_rows}"
        assert f"COLS={n_cols}" in hash_input_str, f"COLS must match df shape: {n_cols}"
        
        # Verify hash matches
        computed_hash = hashlib.sha256(canonical_hash_input).hexdigest()
        function_hash = compute_canonical_data_hash(df)
        assert computed_hash == function_hash, "Manual hash computation must match function"
    
    def test_nan_assertion(self):
        """Test that NaNs in hashed columns cause hard-fail."""
        df = create_baseline_df()
        
        # Add NaN to numeric column
        df_nan = df.copy()
        df_nan.loc[0, "pnl_bps"] = np.nan
        try:
            compute_canonical_data_hash(df_nan)
            assert False, "Should have raised RuntimeError for NaNs in numeric column"
        except RuntimeError as e:
            assert "has" in str(e) and "NaNs" in str(e), f"Expected NaN error, got: {e}"
        
        # Add NaN to datetime column
        df_nan_dt = df.copy()
        df_nan_dt.loc[0, "entry_time"] = pd.NaT
        # Verify NaT is detected
        assert df_nan_dt["entry_time"].isna().sum() > 0, "NaT should be detected as NaN"
        try:
            compute_canonical_data_hash(df_nan_dt)
            assert False, "Should have raised RuntimeError for NaNs in datetime column"
        except RuntimeError as e:
            assert "has" in str(e) and ("NaNs" in str(e) or "null" in str(e).lower()), f"Expected NaN error, got: {e}"


if __name__ == "__main__":
    # Run all tests
    test_suite = TestCanonicalDataHashStructure()
    tests = [
        ("Baseline stability", test_suite.test_a_baseline_stability),
        ("Row count sensitivity", test_suite.test_b_row_count_sensitivity),
        ("Column count sensitivity", test_suite.test_c_column_count_sensitivity),
        ("Column order sensitivity", test_suite.test_d_column_order_sensitivity),
        ("Column name sensitivity", test_suite.test_e_column_name_sensitivity),
        ("Schema sensitivity", test_suite.test_f_schema_sensitivity),
        ("Cell value sensitivity", test_suite.test_g_cell_value_sensitivity),
        ("Determinism guardrails", test_suite.test_h_determinism_guardrails),
        ("NaN assertion", test_suite.test_nan_assertion),
    ]
    
    passed = 0
    failed = 0
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
