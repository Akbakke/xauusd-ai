#!/bin/bash
# Verification script for MINI_JSONL_TEST_3
# Checks all requirements and performs targeted audit if needed

set -euo pipefail

OUTPUT_DIR="${1:-}"
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR=$(find gx1/wf_runs -type d -name "MINI_JSONL_TEST_*" | sort | tail -1)
fi

if [ -z "$OUTPUT_DIR" ] || [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory not found: $OUTPUT_DIR"
    exit 1
fi

echo "=================================================================================="
echo "MINI_JSONL_TEST_3 VERIFICATION"
echo "=================================================================================="
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo ""

JSONL_FILE="$OUTPUT_DIR/trade_journal/merged_trade_index.jsonl"
if [ ! -f "$JSONL_FILE" ]; then
    echo "ERROR: JSONL file not found: $JSONL_FILE"
    exit 1
fi

echo "=== 1. JSONL VERIFICATION ==="
python3 << PYTHON_EOF
import json
import sys
import glob
from pathlib import Path

sys.path.insert(0, '.')
from gx1.scripts.merge_trade_journals import verify_merged_index_jsonl

output_dir = "$OUTPUT_DIR"
jsonl_file = f"{output_dir}/trade_journal/merged_trade_index.jsonl"
PYTHON_EOF

result = verify_merged_index_jsonl(jsonl_file)

print(f"Total records: {result['total_records']}")
print(f"Collisions: {result['collisions_count']}")
print()

missing_entry_time = result.get('missing_fields_by_type', {}).get('missing_entry_time', {}).get('count', 0)
missing_side = result.get('missing_fields_by_type', {}).get('missing_side', {}).get('count', 0)

print(f"missing_entry_time: {missing_entry_time}")
print(f"missing_side: {missing_side}")
print()

# Check schema and trade_uid
with open(jsonl_file, 'r') as f:
    first_line = f.readline()
    if first_line:
        first_record = json.loads(first_line)
        schema_ok = first_record.get('schema_version') == 'trade_index_v1'
    else:
        schema_ok = False

missing_trade_uid = result.get('missing_fields_by_type', {}).get('missing_trade_uid', {}).get('count', 0)

print("=== STATUS ===")
if schema_ok:
    print("✅ Schema version: trade_index_v1")
else:
    print("❌ Schema version: inconsistent")
if missing_trade_uid == 0:
    print("✅ Trade UID: all present")
else:
    print(f"❌ Trade UID: {missing_trade_uid} missing")
if result['collisions_count'] == 0:
    print("✅ Collisions: 0")
else:
    print(f"❌ Collisions: {result['collisions_count']}")
if missing_entry_time == 0 and missing_side == 0:
    print("✅ Core fields: all present")
else:
    print(f"❌ Core fields: {missing_entry_time} missing entry_time, {missing_side} missing side")
print()

# Overall
all_pass = (schema_ok and missing_trade_uid == 0 and result['collisions_count'] == 0 and missing_entry_time == 0 and missing_side == 0)

if all_pass:
    print("=== ✅ ALL JSONL CHECKS PASSED ===")
    sys.exit(0)
else:
    print("=== ❌ SOME JSONL CHECKS FAILED ===")
    print()
    
    # TARGETED AUDIT
    if missing_entry_time > 0 or missing_side > 0:
        print("=== TARGETED AUDIT ===")
        missing_paths = result.get('missing_fields_by_type', {}).get('missing_entry_time', {}).get('sample_paths', [])
        
        # Read JSONL to find all missing records
        missing_trade_uids = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if not record.get('entry_time') or not record.get('side'):
                        missing_trade_uids.append({
                            'trade_uid': record.get('trade_uid'),
                            'trade_id': record.get('trade_id'),
                            'source_path': record.get('source_path'),
                            'trade_key': record.get('trade_key'),
                        })
        
        print(f"Found {len(missing_trade_uids)} records with missing entry_time or side")
        print()
        
        for i, missing in enumerate(missing_trade_uids[:10], 1):
            print(f"--- Missing Record {i} ---")
            print(f"trade_uid: {missing['trade_uid']}")
            print(f"trade_id: {missing['trade_id']}")
            print(f"trade_key: {missing['trade_key']}")
            print(f"source_path: {missing['source_path']}")
            print()
            
            # Find all journal files matching trade_id or trade_uid
            trade_id = missing['trade_id']
            trade_uid = missing['trade_uid']
            
            # Search in all chunks
            chunks_dir = Path(output_dir) / "parallel_chunks"
            matching_files = []
            
            if chunks_dir.exists():
                for chunk_dir in chunks_dir.glob("chunk_*"):
                    trades_dir = chunk_dir / "trade_journal" / "trades"
                    if trades_dir.exists():
                        # Search by trade_id
                        if trade_id:
                            for pattern in [f"{trade_id}.json", f"*{trade_id}*"]:
                                matching_files.extend(trades_dir.glob(pattern))
                        # Search by trade_uid
                        if trade_uid:
                            for pattern in [f"{trade_uid}.json", f"*{trade_uid}*"]:
                                matching_files.extend(trades_dir.glob(pattern))
            
            # Remove duplicates
            matching_files = list(set(matching_files))
            
            if matching_files:
                print(f"  Found {len(matching_files)} matching journal file(s):")
                for file_path in matching_files:
                    try:
                        with open(file_path, 'r') as f:
                            trade_data = json.load(f)
                        print(f"    {file_path.name}:")
                        print(f"      trade_key: {trade_data.get('trade_key')}")
                        print(f"      has entry_snapshot: {trade_data.get('entry_snapshot') is not None}")
                        print(f"      has exit_summary: {trade_data.get('exit_summary') is not None}")
                        if trade_data.get('entry_snapshot') is None and trade_data.get('exit_summary') is not None:
                            print(f"      ⚠️  ORPHAN: exit_summary exists but entry_snapshot is None")
                    except Exception as e:
                        print(f"      Error reading {file_path}: {e}")
            else:
                print(f"  No matching journal files found")
            print()
    
    # Check for exit-only journals
    print("=== CHECKING FOR EXIT-ONLY JOURNALS ===")
    chunks_dir = Path(output_dir) / "parallel_chunks"
    exit_only_count = 0
    
    if chunks_dir.exists():
        for chunk_dir in chunks_dir.glob("chunk_*"):
            trades_dir = chunk_dir / "trade_journal" / "trades"
            if trades_dir.exists():
                for json_file in trades_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r') as f:
                            trade_data = json.load(f)
                        if trade_data.get('exit_summary') and not trade_data.get('entry_snapshot'):
                            exit_only_count += 1
                            if exit_only_count <= 5:
                                print(f"  ORPHAN: {json_file.name}")
                                print(f"    trade_uid: {trade_data.get('trade_uid')}")
                                print(f"    trade_id: {trade_data.get('trade_id')}")
                                print(f"    trade_key: {trade_data.get('trade_key')}")
                    except Exception:
                        pass
    
    if exit_only_count > 0:
        print(f"  Found {exit_only_count} exit-only journals (exit_summary exists but entry_snapshot is None)")
    else:
        print("  ✅ No exit-only journals found")
    print()
    
    sys.exit(1)

PYTHON_EOF
JSONL_STATUS=$?

echo ""
echo "=== 2. PERFORMANCE SUMMARY ==="
if [ -f "$OUTPUT_DIR/REPLAY_PERF_SUMMARY.json" ]; then
    python3 << PYTHON_EOF
import json
import sys

output_dir = "$OUTPUT_DIR"
perf_file = f"{output_dir}/REPLAY_PERF_SUMMARY.json"
PYTHON_EOF

d = json.load(open(perf_file))
ec = d.get('entry_counters', {})

print('Entry Counters:')
print(f'  n_trades_created: {ec.get("n_trades_created", "N/A")}')
print(f'  n_entry_snapshots_written: {ec.get("n_entry_snapshots_written", "N/A")}')
print(f'  n_entry_snapshots_failed: {ec.get("n_entry_snapshots_failed", "N/A")}')
print()

n_trades = ec.get('n_trades_created')
n_written = ec.get('n_entry_snapshots_written')
n_failed = ec.get('n_entry_snapshots_failed')

if n_written == n_trades and n_failed == 0:
    print('✅ PASS: Entry snapshot atomicity invariants')
    print(f'  n_entry_snapshots_written ({n_written}) == n_trades_created ({n_trades})')
    print(f'  n_entry_snapshots_failed ({n_failed}) == 0')
    sys.exit(0)
else:
    print('❌ FAIL: Entry snapshot atomicity invariants')
    print(f'  Expected: n_entry_snapshots_written={n_trades}, n_entry_snapshots_failed=0')
    print(f'  Actual: n_entry_snapshots_written={n_written}, n_entry_snapshots_failed={n_failed}')
    sys.exit(1)
PYTHON_EOF
    PERF_STATUS=$?
else
    echo "⚠️  Merged perf summary ikke funnet - kjører merge..."
    python3 scripts/merge_perf_summaries.py "$OUTPUT_DIR" 2>&1 | tail -5
    PERF_STATUS=1
fi

echo ""
echo "=================================================================================="
if [ $JSONL_STATUS -eq 0 ] && [ $PERF_STATUS -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED"
    exit 0
else
    echo "❌ SOME CHECKS FAILED"
    exit 1
fi

