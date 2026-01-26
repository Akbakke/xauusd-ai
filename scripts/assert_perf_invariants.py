#!/usr/bin/env python3
"""
Assert performance summary invariants (chunk or merged).

Usage:
    python3 scripts/assert_perf_invariants.py chunk <path_to_chunk_REPLAY_PERF_SUMMARY.json>
    python3 scripts/assert_perf_invariants.py merged <path_to_merged_REPLAY_PERF_SUMMARY.json> [<chunk_dir1> <chunk_dir2> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def assert_chunk_invariants(data: Dict[str, Any], chunk_dir: Path) -> List[str]:
    """Assert chunk-level invariants. Returns list of failure messages."""
    failures = []
    
    # A. chunk_id must match directory name chunk_(\d+)
    chunk_id = data.get("chunk_id")
    expected_chunk_id = None
    if chunk_dir:
        import re
        # Match chunk_(\d+) anywhere in path
        match = re.search(r"chunk_(\d+)", str(chunk_dir))
        if match:
            chunk_idx = int(match.group(1))
            expected_chunk_id = f"chunk_{chunk_idx:03d}"
    if chunk_id and expected_chunk_id:
        if chunk_id != expected_chunk_id:
            failures.append(f"A: chunk_id mismatch: {chunk_id} != {expected_chunk_id}")
    
    # B. window_start and window_end must exist and be parseable; window_start < window_end
    window_start = data.get("window_start")
    window_end = data.get("window_end")
    if window_start is None:
        failures.append("B: window_start missing")
    if window_end is None:
        failures.append("B: window_end missing")
    if window_start and window_end:
        try:
            from datetime import datetime
            start_dt = datetime.fromisoformat(window_start.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(window_end.replace('Z', '+00:00'))
            if start_dt >= end_dt:
                failures.append(f"B: window_start ({window_start}) >= window_end ({window_end})")
        except Exception as e:
            failures.append(f"B: Cannot parse window timestamps: {e}")
    
    # C. duration_sec >= 0, feat_time_sec >= 0
    duration_sec = data.get("duration_sec", 0.0)
    if duration_sec < 0:
        failures.append(f"C: duration_sec < 0: {duration_sec}")
    
    runner_perf = data.get("runner_perf_metrics", {})
    feat_time_sec = runner_perf.get("feat_time_sec", 0.0)
    if feat_time_sec < 0:
        failures.append(f"C: feat_time_sec < 0: {feat_time_sec}")
    
    # D. bars_total > 0
    bars_total = data.get("bars_total", 0)
    if bars_total <= 0:
        failures.append(f"D: bars_total <= 0: {bars_total}")
    
    # E. 0 <= bars_processed <= bars_total
    bars_processed = data.get("bars_processed", 0)
    if bars_processed < 0:
        failures.append(f"E: bars_processed < 0: {bars_processed}")
    if bars_processed > bars_total:
        failures.append(f"E: bars_processed ({bars_processed}) > bars_total ({bars_total})")
    
    # F. trades_total >= 0
    trades_total = data.get("trades_total", 0)
    if trades_total < 0:
        failures.append(f"F: trades_total < 0: {trades_total}")
    
    # G. feature_top_blocks is list; each element has name (non-empty), total_sec >= 0, count >= 0 (int)
    feature_top_blocks = data.get("feature_top_blocks", [])
    if not isinstance(feature_top_blocks, list):
        failures.append("G: feature_top_blocks is not a list")
    else:
        for i, block in enumerate(feature_top_blocks):
            if not isinstance(block, dict):
                failures.append(f"G[{i}]: block is not a dict")
                continue
            name = block.get("name")
            if not name or not isinstance(name, str):
                failures.append(f"G[{i}]: name missing or not string: {name}")
            total_sec = block.get("total_sec", -1.0)
            if total_sec < 0:
                failures.append(f"G[{i}]: total_sec < 0: {total_sec}")
            count = block.get("count", -1)
            if not isinstance(count, int) or count < 0:
                failures.append(f"G[{i}]: count missing or not non-negative int: {count}")
    
    # H. top_pandas_ops is list; each element has name (non-empty), total_sec >= 0, count >= 0 (int)
    top_pandas_ops = data.get("top_pandas_ops", [])
    if not isinstance(top_pandas_ops, list):
        failures.append("H: top_pandas_ops is not a list")
    else:
        for i, op in enumerate(top_pandas_ops):
            if not isinstance(op, dict):
                failures.append(f"H[{i}]: op is not a dict")
                continue
            name = op.get("name")
            if not name or not isinstance(name, str):
                failures.append(f"H[{i}]: name missing or not string: {name}")
            total_sec = op.get("total_sec", -1.0)
            if total_sec < 0:
                failures.append(f"H[{i}]: total_sec < 0: {total_sec}")
            count = op.get("count", -1)
            if not isinstance(count, int) or count < 0:
                failures.append(f"H[{i}]: count missing or not non-negative int: {count}")
    
    # I. status exists and is "complete" in normal smoke-run
    # Note: This is checked in scenario-specific tests, not here
    
    # TELEMETRY INVARIANTS (SNIPER telemetry contract)
    counters = data.get("entry_counters", {})
    if counters:
        n_cycles = counters.get("n_cycles", 0)
        n_precheck_pass = counters.get("n_precheck_pass", 0)
        n_candidates = counters.get("n_entry_candidates", counters.get("n_candidates", 0))
        n_candidate_pass = counters.get("n_candidate_pass", 0)
        n_trades_created = counters.get("n_trades_created", counters.get("n_entry_accepted", 0))
        trades_total = data.get("trades_total", 0)
        
        # TELEMETRY_INV_1: 0 <= n_precheck_pass <= n_cycles
        if n_precheck_pass < 0:
            failures.append(f"TELEMETRY_INV_1: n_precheck_pass < 0: {n_precheck_pass}")
        if n_precheck_pass > n_cycles:
            failures.append(f"TELEMETRY_INV_1: n_precheck_pass ({n_precheck_pass}) > n_cycles ({n_cycles})")
        
        # TELEMETRY_INV_2: 0 <= n_candidates <= n_precheck_pass
        if n_candidates < 0:
            failures.append(f"TELEMETRY_INV_2: n_candidates < 0: {n_candidates}")
        if n_candidates > n_precheck_pass:
            failures.append(f"TELEMETRY_INV_2: n_candidates ({n_candidates}) > n_precheck_pass ({n_precheck_pass})")
        
        # TELEMETRY_INV_3: 0 <= n_candidate_pass <= n_candidates
        if n_candidate_pass < 0:
            failures.append(f"TELEMETRY_INV_3: n_candidate_pass < 0: {n_candidate_pass}")
        if n_candidate_pass > n_candidates:
            failures.append(f"TELEMETRY_INV_3: n_candidate_pass ({n_candidate_pass}) > n_candidates ({n_candidates})")
        
        # TELEMETRY_INV_4: 0 <= n_trades_created <= n_candidate_pass
        if n_trades_created < 0:
            failures.append(f"TELEMETRY_INV_4: n_trades_created < 0: {n_trades_created}")
        if n_trades_created > n_candidate_pass:
            failures.append(f"TELEMETRY_INV_4: n_trades_created ({n_trades_created}) > n_candidate_pass ({n_candidate_pass})")
        
        # TELEMETRY_INV_5: For replay: n_trades_created == trades_total (if trades_total available)
        if trades_total > 0 and n_trades_created != trades_total:
            # Only warn for chunks (merged check is more strict)
            pass  # Warning already logged in run_mini_replay_perf.py
        
        # TELEMETRY_INV_6: All veto_pre_* sums <= (n_cycles - n_precheck_pass)
        veto_pre_keys = ["veto_pre_session", "veto_pre_regime", "veto_pre_atr", "veto_pre_warmup",
                         "veto_pre_spread", "veto_pre_killswitch", "veto_pre_model_missing", "veto_pre_nan_features"]
        veto_pre_sum = sum(counters.get(k, 0) for k in veto_pre_keys)
        blocked_at_stage0 = n_cycles - n_precheck_pass
        if veto_pre_sum > blocked_at_stage0:
            failures.append(f"TELEMETRY_INV_6: veto_pre sum ({veto_pre_sum}) > blocked_at_stage0 ({blocked_at_stage0})")
        
        # TELEMETRY_INV_7: All veto_cand_* sums <= (n_candidates - n_candidate_pass)
        veto_cand_keys = ["veto_cand_threshold", "veto_cand_risk_guard", "veto_cand_max_trades", "veto_cand_big_brain"]
        veto_cand_sum = sum(counters.get(k, 0) for k in veto_cand_keys)
        blocked_at_stage1 = n_candidates - n_candidate_pass
        if veto_cand_sum > blocked_at_stage1:
            failures.append(f"TELEMETRY_INV_7: veto_cand sum ({veto_cand_sum}) > blocked_at_stage1 ({blocked_at_stage1})")
        
        # TELEMETRY_INV_8: Stage-1 attribution invariant - all blocked candidates must have a veto reason
        # missing = (n_candidates - n_candidate_pass) - sum(veto_cand_*) must be 0
        missing = blocked_at_stage1 - veto_cand_sum
        if missing != 0:
            # Build breakdown string for error message
            veto_breakdown = ", ".join([f"{k}={counters.get(k, 0)}" for k in veto_cand_keys])
            failures.append(
                f"TELEMETRY_INV_8: Stage-1 attribution mismatch: missing={missing} "
                f"(n_candidates={n_candidates}, n_candidate_pass={n_candidate_pass}, blocked={blocked_at_stage1}, "
                f"veto_cand_sum={veto_cand_sum}, breakdown: {veto_breakdown})"
            )
        
        # OPPGAVE 2: Hard/Soft eligibility invariants
        n_eligible_hard = counters.get("n_eligible_hard", 0)
        n_eligible_cycles = counters.get("n_eligible_cycles", 0)
        n_context_built = counters.get("n_context_built", 0)
        
        # ELIGIBILITY_INV_1: 0 <= n_eligible_hard <= n_cycles
        if n_eligible_hard < 0:
            failures.append(f"ELIGIBILITY_INV_1: n_eligible_hard < 0: {n_eligible_hard}")
        if n_eligible_hard > n_cycles:
            failures.append(f"ELIGIBILITY_INV_1: n_eligible_hard ({n_eligible_hard}) > n_cycles ({n_cycles})")
        
        # ELIGIBILITY_INV_2: 0 <= n_eligible_cycles <= n_eligible_hard
        if n_eligible_cycles < 0:
            failures.append(f"ELIGIBILITY_INV_2: n_eligible_cycles < 0: {n_eligible_cycles}")
        if n_eligible_cycles > n_eligible_hard:
            failures.append(f"ELIGIBILITY_INV_2: n_eligible_cycles ({n_eligible_cycles}) > n_eligible_hard ({n_eligible_hard})")
        
        # ELIGIBILITY_INV_3: n_context_built <= n_eligible_cycles (if context features enabled)
        import os
        context_features_enabled = os.getenv("ENTRY_CONTEXT_FEATURES_ENABLED", "false").lower() == "true"
        if context_features_enabled:
            if n_context_built < 0:
                failures.append(f"ELIGIBILITY_INV_3: n_context_built < 0: {n_context_built}")
            if n_context_built > n_eligible_cycles:
                failures.append(f"ELIGIBILITY_INV_3: n_context_built ({n_context_built}) > n_eligible_cycles ({n_eligible_cycles})")
        
        # ELIGIBILITY_INV_4: n_candidates <= n_context_built (if context features enabled) or n_eligible_cycles (if disabled)
        if context_features_enabled:
            if n_candidates > n_context_built:
                failures.append(f"ELIGIBILITY_INV_4: n_candidates ({n_candidates}) > n_context_built ({n_context_built})")
        else:
            if n_candidates > n_eligible_cycles:
                failures.append(f"ELIGIBILITY_INV_4: n_candidates ({n_candidates}) > n_eligible_cycles ({n_eligible_cycles})")
        
        # DEL 4: Ctx model telemetry invariants (replay only, when ctx-modell is active)
        n_ctx_model_calls = counters.get("n_ctx_model_calls", 0)
        n_v10_calls = counters.get("n_v10_calls", 0)
        n_context_missing_or_invalid = counters.get("n_context_missing_or_invalid", 0)
        n_context_built = counters.get("n_context_built", 0)
        
        # CTX_INV_0: if ctx_expected, n_ctx_model_calls > 0 (fail-fast in replay)
        ctx_expected = counters.get("ctx_expected", False)
        if ctx_expected:
            if n_ctx_model_calls == 0 and n_v10_calls > 0:
                # Hard fail in replay mode (we assume replay if ctx_expected is True)
                v10_none_reason_counts = counters.get('v10_none_reason_counts', {}) or counters.get('v10_none_reason_counts_full', {})
                v10_exception_stacks = counters.get('v10_exception_stacks', {})
                dominant_reason = max(v10_none_reason_counts.items(), key=lambda x: x[1]) if v10_none_reason_counts else ("UNKNOWN", 0)
                
                # Get stack excerpt for dominant reason
                stack_excerpt = ""
                if dominant_reason[0] in v10_exception_stacks:
                    stack_excerpt = f"\nStack excerpt for {dominant_reason[0]}:\n{v10_exception_stacks[dominant_reason[0]]}"
                elif "FEATURE_BUILD_TIMEOUT" in dominant_reason[0]:
                    stack_excerpt = "\n(Feature building timeout - check pandas operations in build_v9_runtime_features)"
                
                failures.append(
                    f"CTX_INV_0 (FAIL-FAST): ctx_expected=True but n_ctx_model_calls=0. "
                    f"This indicates ctx was built but not consumed. "
                    f"n_context_built={n_context_built}, n_v10_calls={n_v10_calls}, "
                    f"dominant_reason={dominant_reason[0]} (count={dominant_reason[1]}), "
                    f"v10_none_reason_counts={v10_none_reason_counts}{stack_excerpt}"
                )
        
        # Check if ctx-modell is active (n_ctx_model_calls > 0 or n_context_built > 0)
        ctx_model_active = n_ctx_model_calls > 0 or (context_features_enabled and n_context_built > 0)
        
        if ctx_model_active:
            # CTX_INV_1: n_ctx_model_calls == n_v10_calls (when ctx-modell is active)
            if n_ctx_model_calls != n_v10_calls:
                failures.append(
                    f"CTX_INV_1: n_ctx_model_calls ({n_ctx_model_calls}) != n_v10_calls ({n_v10_calls}). "
                    "All V10 calls should use ctx-modell when ctx is active."
                )
            
            # CTX_INV_2: n_context_missing_or_invalid == 0 (replay only)
            # Note: In live mode, this may be > 0 (degraded mode), so we only check in replay
            # We can't easily detect replay mode here, so we'll just log a warning if > 0
            if n_context_missing_or_invalid > 0:
                # This is a warning, not a hard failure (live mode may have degraded context)
                # But in replay, it should be 0
                failures.append(
                    f"CTX_INV_2: n_context_missing_or_invalid ({n_context_missing_or_invalid}) > 0. "
                    "Context build failures detected (may be acceptable in live mode, but not in replay)."
                )
            
            # CTX_INV_3: n_context_built == n_v10_calls (when ctx-modell is active)
            if n_context_built != n_v10_calls:
                failures.append(
                    f"CTX_INV_3: n_context_built ({n_context_built}) != n_v10_calls ({n_v10_calls}). "
                    "Context should be built for all V10 calls when ctx-modell is active."
                )
            
            # CTX_INV_4: ctx_proof_fail_count == 0 if proof is enabled (replay only)
            ctx_proof_enabled = counters.get("ctx_proof_enabled", False)
            ctx_proof_fail_count = counters.get("ctx_proof_fail_count", 0)
            if ctx_proof_enabled and ctx_proof_fail_count > 0:
                # In replay: hard fail if any proof check fails
                failures.append(
                    f"CTX_INV_4: ctx_proof_fail_count ({ctx_proof_fail_count}) > 0. "
                    f"CTX consumption proof failed {ctx_proof_fail_count} time(s). "
                    f"This indicates ctx is not being consumed by the model. "
                    f"ctx_proof_pass_count={counters.get('ctx_proof_pass_count', 0)}"
                )
        
        # When ctx_expected == False: don't require ctx invariants, but log what happened
        if not ctx_expected:
            if n_ctx_model_calls > 0 or n_context_built > 0:
                # Log that ctx was used even though not expected (may be intentional for testing)
                pass  # Not a failure, just informational
    
    return failures


def assert_merged_invariants(data: Dict[str, Any], chunk_data_list: List[Dict[str, Any]]) -> List[str]:
    """Assert merged-level invariants. Returns list of failure messages."""
    failures = []
    
    if not chunk_data_list:
        failures.append("MERGED: No chunk data provided for validation")
        return failures
    
    # J. bars_total == SUM(chunk bars_total)
    merged_bars_total = data.get("bars_total", 0)
    chunk_bars_total_sum = sum(s.get("bars_total", 0) for s in chunk_data_list)
    if merged_bars_total != chunk_bars_total_sum:
        failures.append(f"J: bars_total mismatch: merged={merged_bars_total}, sum(chunks)={chunk_bars_total_sum}")
    
    # K. bars_processed == SUM(chunk bars_processed)
    merged_bars_processed = data.get("bars_processed", 0)
    chunk_bars_processed_sum = sum(s.get("bars_processed", 0) for s in chunk_data_list)
    if merged_bars_processed != chunk_bars_processed_sum:
        failures.append(f"K: bars_processed mismatch: merged={merged_bars_processed}, sum(chunks)={chunk_bars_processed_sum}")
    
    # L. trades_total == SUM(chunk trades_total)
    merged_trades_total = data.get("trades_total", 0)
    chunk_trades_total_sum = sum(s.get("trades_total", 0) for s in chunk_data_list)
    if merged_trades_total != chunk_trades_total_sum:
        failures.append(f"L: trades_total mismatch: merged={merged_trades_total}, sum(chunks)={chunk_trades_total_sum}")
    
    # M. feat_time_sec == SUM(chunk feat_time_sec) (tolerance 1e-6)
    merged_feat_time = data.get("runner_perf_metrics", {}).get("feat_time_sec", 0.0)
    chunk_feat_time_sum = sum(
        s.get("runner_perf_metrics", {}).get("feat_time_sec", 0.0) for s in chunk_data_list
    )
    if abs(merged_feat_time - chunk_feat_time_sum) > 1e-6:
        failures.append(f"M: feat_time_sec mismatch: merged={merged_feat_time:.6f}, sum(chunks)={chunk_feat_time_sum:.6f}")
    
    # N. duration_sec == MAX(chunk duration_sec)
    merged_duration = data.get("duration_sec", 0.0)
    chunk_durations = [s.get("duration_sec", 0.0) for s in chunk_data_list]
    chunk_duration_max = max(chunk_durations) if chunk_durations else 0.0
    if abs(merged_duration - chunk_duration_max) > 1.0:  # 1 second tolerance for parallel execution
        failures.append(f"N: duration_sec mismatch: merged={merged_duration:.2f}, max(chunks)={chunk_duration_max:.2f}")
    
    # O. Feature blocks merged by name: total_sec == SUM, count == SUM
    merged_blocks = {b.get("name"): b for b in data.get("feature_top_blocks", [])}
    chunk_blocks_by_name: Dict[str, Dict[str, float]] = {}
    for chunk_data in chunk_data_list:
        for block in chunk_data.get("feature_top_blocks", []):
            name = block.get("name")
            if name:
                if name not in chunk_blocks_by_name:
                    chunk_blocks_by_name[name] = {"total_sec": 0.0, "count": 0}
                chunk_blocks_by_name[name]["total_sec"] += block.get("total_sec", 0.0)
                chunk_blocks_by_name[name]["count"] += block.get("count", 0)
    
    for block_name, merged_block in merged_blocks.items():
        if block_name not in chunk_blocks_by_name:
            continue  # Block might not appear in all chunks
        chunk_total_sec = chunk_blocks_by_name[block_name]["total_sec"]
        chunk_count = chunk_blocks_by_name[block_name]["count"]
        merged_total_sec = merged_block.get("total_sec", 0.0)
        merged_count = merged_block.get("count", 0)
        if abs(merged_total_sec - chunk_total_sec) > 1e-6:
            failures.append(f"O: block '{block_name}' total_sec mismatch: merged={merged_total_sec:.6f}, sum(chunks)={chunk_total_sec:.6f}")
        if merged_count != chunk_count:
            failures.append(f"O: block '{block_name}' count mismatch: merged={merged_count}, sum(chunks)={chunk_count}")
    
    # P. merged feature_top_blocks sorted desc on total_sec, limited to top 15
    merged_blocks_list = data.get("feature_top_blocks", [])
    if len(merged_blocks_list) > 15:
        failures.append(f"P: feature_top_blocks has {len(merged_blocks_list)} entries, expected <= 15")
    prev_sec = float('inf')
    for i, block in enumerate(merged_blocks_list):
        total_sec = block.get("total_sec", 0.0)
        if total_sec > prev_sec:
            failures.append(f"P: feature_top_blocks not sorted desc at index {i}: {prev_sec:.6f} > {total_sec:.6f}")
        prev_sec = total_sec
    
    # Q. merged top_pandas_ops sorted desc on total_sec, limited to top 10
    merged_pandas_ops = data.get("top_pandas_ops", [])
    if len(merged_pandas_ops) > 10:
        failures.append(f"Q: top_pandas_ops has {len(merged_pandas_ops)} entries, expected <= 10")
    prev_sec = float('inf')
    for i, op in enumerate(merged_pandas_ops):
        total_sec = op.get("total_sec", 0.0)
        if total_sec > prev_sec:
            failures.append(f"Q: top_pandas_ops not sorted desc at index {i}: {prev_sec:.6f} > {total_sec:.6f}")
        prev_sec = total_sec
    
    # R. chunk_completion table must list ALL chunk_ids and status per chunk
    chunk_completion = data.get("chunk_completion", [])
    chunk_completion_ids = {chunk.get("chunk_id") for chunk in chunk_completion}
    chunk_data_ids = {s.get("chunk_id") for s in chunk_data_list}
    if chunk_completion_ids != chunk_data_ids:
        missing = chunk_data_ids - chunk_completion_ids
        extra = chunk_completion_ids - chunk_data_ids
        if missing:
            failures.append(f"R: chunk_completion missing chunks: {missing}")
        if extra:
            failures.append(f"R: chunk_completion has extra chunks: {extra}")
    
    # TELEMETRY MERGED INVARIANTS
    counters = data.get("entry_counters", {})
    if counters:
        n_trades_created = counters.get("n_trades_created", counters.get("n_entry_accepted", 0))
        trades_total = data.get("trades_total", 0)
        
        # TELEMETRY_MERGED_INV_1: n_trades_created == trades_total (replay invariant)
        if trades_total >= 0 and n_trades_created != trades_total:
            failures.append(f"TELEMETRY_MERGED_INV_1: n_trades_created ({n_trades_created}) != trades_total ({trades_total})")
        
        # TELEMETRY_MERGED_INV_2: Merged counters match sum of chunk counters
        chunk_n_cycles_sum = sum(s.get("entry_counters", {}).get("n_cycles", 0) for s in chunk_data_list)
        merged_n_cycles = counters.get("n_cycles", 0)
        if merged_n_cycles != chunk_n_cycles_sum:
            failures.append(f"TELEMETRY_MERGED_INV_2: n_cycles mismatch: merged={merged_n_cycles}, sum(chunks)={chunk_n_cycles_sum}")
        
        chunk_n_trades_sum = sum(s.get("entry_counters", {}).get("n_trades_created", s.get("entry_counters", {}).get("n_entry_accepted", 0)) for s in chunk_data_list)
        if n_trades_created != chunk_n_trades_sum:
            failures.append(f"TELEMETRY_MERGED_INV_2: n_trades_created mismatch: merged={n_trades_created}, sum(chunks)={chunk_n_trades_sum}")
    
    return failures


def assert_journal_invariants(
    run_dir: Path,
    trades_total: Optional[int] = None,
    is_replay: bool = True,
) -> List[str]:
    """
    Assert trade journal invariants (COMMIT D).
    
    Parameters
    ----------
    run_dir : Path
        Root directory of the replay run
    trades_total : int, optional
        Expected total number of trades (from perf summary)
    is_replay : bool
        Whether this is a replay run (collisions must be 0)
    
    Returns
    -------
    List[str]
        List of failure messages (empty if all invariants pass)
    """
    failures = []
    
    # Find merged trade_journal_index.csv
    merged_index_csv = run_dir / "trade_journal" / "trade_journal_index.csv"
    
    if not merged_index_csv.exists():
        failures.append(
            f"JOURNAL_INV_0: Merged trade_journal_index.csv not found: {merged_index_csv}"
        )
        return failures
    
    # Count collisions using helper script
    import subprocess
    try:
        result = subprocess.run(
            ["python3", "scripts/count_trade_key_collisions.py", str(merged_index_csv)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            failures.append(
                f"JOURNAL_INV_0: Failed to count collisions: {result.stderr}"
            )
            return failures
        
        # Parse output
        collisions_count = 0
        unique_trade_keys = 0
        for line in result.stdout.split('\n'):
            if line.startswith('collisions_count='):
                collisions_count = int(line.split('=')[1])
            elif line.startswith('unique_duplicate_keys='):
                unique_trade_keys = int(line.split('=')[1])
        
        # JOURNAL_INV_1: collisions_count == 0 (for replay)
        if is_replay:
            if collisions_count != 0:
                failures.append(
                    f"JOURNAL_INV_1: trade_key collisions detected: collisions_count={collisions_count} "
                    f"(must be 0 for replay runs). Index CSV: {merged_index_csv}"
                )
        
        # JOURNAL_INV_2: trades_total == unique(trade_key) (when trades_total is available)
        if trades_total is not None:
            # Count unique trade_keys from CSV
            import csv
            trade_keys = set()
            with open(merged_index_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trade_key = row.get('trade_key', '').strip()
                    if trade_key:
                        trade_keys.add(trade_key)
            
            unique_count = len(trade_keys)
            if trades_total != unique_count:
                failures.append(
                    f"JOURNAL_INV_2: trades_total ({trades_total}) != unique(trade_key) ({unique_count}). "
                    f"Index CSV: {merged_index_csv}"
                )
        
    except Exception as e:
        failures.append(
            f"JOURNAL_INV_0: Exception while checking journal invariants: {e}"
        )
    
    return failures


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/assert_perf_invariants.py chunk <json_path>")
        print("       python3 scripts/assert_perf_invariants.py merged <json_path> <chunk_dir1> <chunk_dir2> ...")
        sys.exit(1)
    
    mode = sys.argv[1]
    json_path = Path(sys.argv[2])
    
    if not json_path.exists():
        print(f"FAIL: JSON file not found: {json_path}")
        sys.exit(1)
    
    with open(json_path) as f:
        data = json.load(f)
    
    failures = []
    
    if mode == "chunk":
        chunk_dir = json_path.parent
        failures = assert_chunk_invariants(data, chunk_dir)
    elif mode == "merged":
        chunk_dirs = [Path(d) for d in sys.argv[3:]] if len(sys.argv) > 3 else []
        chunk_data_list = []
        for chunk_dir in chunk_dirs:
            chunk_json = chunk_dir / "REPLAY_PERF_SUMMARY.json"
            if chunk_json.exists():
                with open(chunk_json) as f:
                    chunk_data_list.append(json.load(f))
            else:
                print(f"WARNING: Chunk summary not found: {chunk_json}")
        failures = assert_merged_invariants(data, chunk_data_list)
    else:
        print(f"FAIL: Unknown mode: {mode}")
        sys.exit(1)
    
    if failures:
        print("FAIL: Invariant violations:")
        for failure in failures:
            print(f"  {failure}")
        sys.exit(1)
    else:
        print("PASS: All invariants satisfied")
        sys.exit(0)


if __name__ == "__main__":
    main()

