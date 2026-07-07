#!/usr/bin/env python3
"""
Verify Entry Flow Gap - Deterministic Verification Tool

Reads replay outputs and answers deterministically what happened in the critical gap:
AFTER_SOFT_ELIGIBILITY_PASSED -> BEFORE_STAGE0_CHECK

Terminal states:
1) EXCEPTION_CAUGHT
2) STAGE0_REACHED
3) UNKNOWN_EXIT (structural / impossible path)
4) INSUFFICIENT_EVIDENCE

Usage:
    python3 gx1/scripts/verify_entry_flow_gap.py <path>
    python3 gx1/scripts/verify_entry_flow_gap.py <path> --write-json <output_path>
    python3 gx1/scripts/verify_entry_flow_gap.py <path> --verbose

Path can be:
- Run root directory (searches for ENTRY_FEATURES_USED_MASTER.json)
- Chunk directory (searches for ENTRY_FEATURES_USED.json)
- Direct JSON file path
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

def load_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file safely."""
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        return None
    return None

def find_telemetry_files(path: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """
    Find telemetry files in priority order.
    
    Returns:
        (master_json, chunk_json, chunk_footer, fail_capsule)
    """
    master_json = None
    chunk_json = None
    chunk_footer = None
    fail_capsule = None
    
    if path.is_file():
        # Direct file path
        if path.name == "ENTRY_FEATURES_USED_MASTER.json":
            master_json = path
        elif path.name == "ENTRY_FEATURES_USED.json":
            chunk_json = path
        elif path.name == "chunk_footer.json":
            chunk_footer = path
        elif path.name == "CHUNK_FAIL_CAPSULE.json":
            fail_capsule = path
        return (master_json, chunk_json, chunk_footer, fail_capsule)
    
    # Directory - search for files
    if (path / "ENTRY_FEATURES_USED_MASTER.json").exists():
        master_json = path / "ENTRY_FEATURES_USED_MASTER.json"
    
    if (path / "ENTRY_FEATURES_USED.json").exists():
        chunk_json = path / "ENTRY_FEATURES_USED.json"
    
    if (path / "chunk_footer.json").exists():
        chunk_footer = path / "chunk_footer.json"
    
    if (path / "CHUNK_FAIL_CAPSULE.json").exists():
        fail_capsule = path / "CHUNK_FAIL_CAPSULE.json"
    
    # Also check chunk_* subdirectories if in run root
    if not master_json and not chunk_json:
        for chunk_dir in sorted(path.glob("chunk_*")):
            if chunk_dir.is_dir():
                if (chunk_dir / "ENTRY_FEATURES_USED.json").exists():
                    chunk_json = chunk_dir / "ENTRY_FEATURES_USED.json"
                    if (chunk_dir / "chunk_footer.json").exists():
                        chunk_footer = chunk_dir / "chunk_footer.json"
                    if (chunk_dir / "CHUNK_FAIL_CAPSULE.json").exists():
                        fail_capsule = chunk_dir / "CHUNK_FAIL_CAPSULE.json"
                    break
    
    return (master_json, chunk_json, chunk_footer, fail_capsule)

def extract_fields(
    master_json: Optional[Path],
    chunk_json: Optional[Path],
    chunk_footer: Optional[Path],
    fail_capsule: Optional[Path],
) -> Dict[str, Any]:
    """
    Extract fields from telemetry files in priority order.
    
    Returns normalized structure with all fields.
    """
    payload = {
        "control_flow_counts": {},
        "exception_gap": None,
        "entry_routing_aggregate": {},
        "v10_callsite": {},
        "model_entry": {},
        "transformer_forward_calls": None,
        "paths_used": [],
    }
    
    # Priority 1: ENTRY_FEATURES_USED_MASTER.json
    if master_json:
        data = load_json_safe(master_json)
        if data:
            payload["paths_used"].append(str(master_json))
            payload["control_flow_counts"] = data.get("control_flow", {}).get("counts", {})
            payload["exception_gap"] = data.get("exception_gap")
            payload["entry_routing_aggregate"] = data.get("entry_routing_aggregate", {})
            payload["v10_callsite"] = data.get("v10_callsite", {})
            payload["model_entry"] = data.get("model_entry", {})
            payload["transformer_forward_calls"] = data.get("transformer_forward_calls")
            return payload
    
    # Priority 2: ENTRY_FEATURES_USED.json
    if chunk_json:
        data = load_json_safe(chunk_json)
        if data:
            payload["paths_used"].append(str(chunk_json))
            payload["control_flow_counts"] = data.get("control_flow", {}).get("counts", {})
            payload["exception_gap"] = data.get("exception_gap")
            payload["entry_routing_aggregate"] = data.get("entry_routing_aggregate", {})
            payload["v10_callsite"] = data.get("v10_callsite", {})
            payload["model_entry"] = data.get("model_entry", {})
            payload["transformer_forward_calls"] = data.get("transformer_forward_calls")
    
    # Priority 3: chunk_footer.json (fallback for control_flow)
    if chunk_footer:
        data = load_json_safe(chunk_footer)
        if data:
            if str(chunk_footer) not in payload["paths_used"]:
                payload["paths_used"].append(str(chunk_footer))
            if not payload["control_flow_counts"]:
                payload["control_flow_counts"] = data.get("control_flow", {}).get("counts", {})
            if not payload["exception_gap"]:
                payload["exception_gap"] = data.get("exception_gap")
            if not payload["v10_callsite"]:
                payload["v10_callsite"] = data.get("v10_callsite", {})
    
    # Priority 4: CHUNK_FAIL_CAPSULE.json (SSoT for exception_gap)
    if fail_capsule:
        data = load_json_safe(fail_capsule)
        if data:
            if str(fail_capsule) not in payload["paths_used"]:
                payload["paths_used"].append(str(fail_capsule))
            # Fail capsule is SSoT for exception_gap
            if data.get("exception_gap"):
                payload["exception_gap"] = data.get("exception_gap")
    
    return payload

def decide_status(payload: Dict[str, Any]) -> Tuple[str, int, Dict[str, Any]]:
    """
    Decide terminal state based on extracted fields.
    
    Returns:
        (status, exit_code, report_data)
    """
    control_flow_counts = payload.get("control_flow_counts", {})
    exception_gap = payload.get("exception_gap")
    
    soft_passed_count = control_flow_counts.get("AFTER_SOFT_ELIGIBILITY_PASSED", 0)
    stage0_count = control_flow_counts.get("BEFORE_STAGE0_CHECK", 0)
    gap_exception_count = control_flow_counts.get("EXCEPTION_IN_SOFT_TO_STAGE0_GAP", 0)
    early_return_count = control_flow_counts.get("EARLY_RETURN_IN_GAP", 0)
    
    soft_passed = soft_passed_count > 0
    stage0_reached = stage0_count > 0
    gap_exception = gap_exception_count > 0 or exception_gap is not None
    
    transformer_forward_calls = payload.get("transformer_forward_calls")
    transformer_called = transformer_forward_calls is not None and transformer_forward_calls > 0
    
    entry_routing_aggregate = payload.get("entry_routing_aggregate", {})
    selected_model_counts = entry_routing_aggregate.get("selected_model_counts", {})
    routed_v10_count = selected_model_counts.get("v10_hybrid", 0)
    routed_v10 = routed_v10_count > 0
    
    v10_callsite = payload.get("v10_callsite", {})
    callsite_entered = v10_callsite.get("entered", 0)
    callsite_returned = v10_callsite.get("returned", 0)
    callsite_exception = v10_callsite.get("exception", 0)
    
    model_entry = payload.get("model_entry", {})
    model_attempt_calls = model_entry.get("model_attempt_calls", {})
    model_forward_calls = model_entry.get("model_forward_calls", {})
    
    # Sum model attempts/forwards (handle both dict and int)
    model_attempts = 0
    if isinstance(model_attempt_calls, dict):
        model_attempts = sum(model_attempt_calls.values())
    elif isinstance(model_attempt_calls, (int, float)):
        model_attempts = int(model_attempt_calls)
    
    model_forwards = 0
    if isinstance(model_forward_calls, dict):
        model_forwards = sum(model_forward_calls.values())
    elif isinstance(model_forward_calls, (int, float)):
        model_forwards = int(model_forward_calls)
    
    report_data = {
        "soft_passed_count": soft_passed_count,
        "stage0_count": stage0_count,
        "gap_exception_count": gap_exception_count,
        "early_return_count": early_return_count,
        "exception_present": gap_exception,
        "exception_gap": exception_gap,
        "transformer_forward_calls": transformer_forward_calls,
        "routing_v10_count": routed_v10_count,
        "callsite_entered": callsite_entered,
        "callsite_returned": callsite_returned,
        "callsite_exception": callsite_exception,
        "model_attempts": model_attempts,
        "model_forwards": model_forwards,
        "hints": [],
    }
    
    # Decision logic
    if gap_exception:
        # Case A: EXCEPTION_CAUGHT
        status = "EXCEPTION_CAUGHT"
        exit_code = 10
        if exception_gap:
            report_data["hints"].append(f"See CHUNK_FAIL_CAPSULE.json for full context")
        return (status, exit_code, report_data)
    
    elif stage0_reached:
        # Case B: STAGE0_REACHED
        status = "STAGE0_REACHED"
        
        # Check if fully verified
        required_checks = []
        if not transformer_called:
            required_checks.append("transformer_forward_calls == 0")
            report_data["hints"].append("Transformer not called - check Stage-0 filter or routing")
        
        routing_known = bool(entry_routing_aggregate)
        if routing_known and not routed_v10:
            required_checks.append("routing != v10_hybrid")
            report_data["hints"].append("Routing did not select v10_hybrid - check V10 enable state")
        elif not routing_known:
            report_data["hints"].append("WARNING: Routing data not available")
        
        if required_checks:
            status = "STAGE0_REACHED_BUT_NOT_FULLY_VERIFIED"
            exit_code = 20
            report_data["hints"].append(f"Missing: {', '.join(required_checks)}")
        else:
            exit_code = 0
            report_data["hints"].append("All verification criteria met")
        
        return (status, exit_code, report_data)
    
    elif soft_passed and not stage0_reached and not gap_exception:
        # Case C: UNKNOWN_EXIT
        status = "UNKNOWN_EXIT"
        exit_code = 30
        report_data["hints"].append("Possible structural exits: sys.exit/os._exit/signal/multiprocessing abort")
        return (status, exit_code, report_data)
    
    else:
        # Case D: INSUFFICIENT_EVIDENCE
        status = "INSUFFICIENT_EVIDENCE"
        exit_code = 40
        if not soft_passed:
            report_data["hints"].append("Soft eligibility never passed - check eligibility gates")
        else:
            report_data["hints"].append("No telemetry found or incomplete data")
        return (status, exit_code, report_data)

def print_summary(status: str, exit_code: int, report_data: Dict[str, Any], paths_used: List[str], verbose: bool = False) -> None:
    """Print human-readable summary."""
    print("=" * 80)
    print("ENTRY FLOW GAP VERIFICATION")
    print("=" * 80)
    print()
    
    print(f"Status: {status}")
    print(f"Exit Code: {exit_code}")
    print()
    
    if paths_used:
        print("Primary Sources:")
        for path in paths_used:
            print(f"  - {path}")
        print()
    
    print("Control Flow Summary:")
    print(f"  AFTER_SOFT_ELIGIBILITY_PASSED: {report_data['soft_passed_count']}")
    print(f"  BEFORE_STAGE0_CHECK: {report_data['stage0_count']}")
    print(f"  EXCEPTION_IN_SOFT_TO_STAGE0_GAP: {report_data['gap_exception_count']}")
    print(f"  EARLY_RETURN_IN_GAP: {report_data['early_return_count']}")
    print()
    
    if status == "EXCEPTION_CAUGHT":
        print("EXCEPTION CAUGHT:")
        exception_gap = report_data.get("exception_gap")
        if exception_gap:
            print(f"  Type: {exception_gap.get('exc_type', 'UNKNOWN')}")
            print(f"  Message: {exception_gap.get('exc_msg', 'UNKNOWN')}")
            print(f"  Line: {exception_gap.get('line', 'UNKNOWN')}")
            print(f"  Timestamp: {exception_gap.get('ts', 'UNKNOWN')}")
        print()
        print("→ ACTION: Review exception and fix root cause")
        print("→ See CHUNK_FAIL_CAPSULE.json for full context")
        print()
    
    elif status == "STAGE0_REACHED" or status == "STAGE0_REACHED_BUT_NOT_FULLY_VERIFIED":
        print("✓ STAGE-0 REACHED")
        print()
        
        if report_data['soft_passed_count'] > 0:
            print("✓ SOFT ELIGIBILITY PASSED")
        
        routing_known = report_data.get('routing_v10_count') is not None
        if routing_known:
            if report_data['routing_v10_count'] > 0:
                print("✓ ROUTING OK (v10_hybrid)")
            else:
                print("✗ ROUTING: Not v10_hybrid")
        else:
            print("⚠ ROUTING: Unknown (data not available)")
        
        if report_data.get('transformer_forward_calls', 0) > 0:
            print(f"✓ TRANSFORMER CALLED ({report_data['transformer_forward_calls']} calls)")
        else:
            print("✗ TRANSFORMER NOT CALLED")
        
        print()
        
        if verbose:
            print("Detailed Counters:")
            print(f"  V10 Callsite Entered: {report_data['callsite_entered']}")
            print(f"  V10 Callsite Returned: {report_data['callsite_returned']}")
            print(f"  V10 Callsite Exception: {report_data['callsite_exception']}")
            print(f"  Model Attempts: {report_data['model_attempts']}")
            print(f"  Model Forwards: {report_data['model_forwards']}")
            print()
        
        if status == "STAGE0_REACHED_BUT_NOT_FULLY_VERIFIED":
            print("⚠ PARTIAL VERIFICATION")
            print("Stage-0 reached but some criteria missing:")
            for hint in report_data.get('hints', []):
                if 'Missing:' in hint:
                    print(f"  {hint}")
            print()
            print("→ ACTION: Investigate missing criteria")
        else:
            print("=" * 80)
            print("✓ ENTRY-FLOW VERIFIED")
            print("=" * 80)
            print()
            print("All criteria met:")
            print("  ✓ BEFORE_STAGE0_CHECK > 0")
            print("  ✓ transformer_forward_calls > 0")
            if routing_known:
                print("  ✓ entry_routing.selected_model == 'v10_hybrid'")
            print()
            print("→ READY FOR A/B TESTS (XGB ↔ Transformer)")
        print()
    
    elif status == "UNKNOWN_EXIT":
        print("✗ UNKNOWN EXIT (SOFT->STAGE0)")
        print()
        print("AFTER_SOFT_ELIGIBILITY_PASSED > 0 but BEFORE_STAGE0_CHECK == 0")
        print("and no EXCEPTION_IN_SOFT_TO_STAGE0_GAP or EARLY_RETURN_IN_GAP recorded")
        print()
        print("→ ACTION: Investigate structural control-flow issue:")
        print("  - sys.exit() / os._exit() calls")
        print("  - multiprocessing abort")
        print("  - C-extension side-effects")
        print("  - signal handlers")
        print()
        if report_data.get('hints'):
            for hint in report_data['hints']:
                print(f"  Hint: {hint}")
        print()
    
    elif status == "INSUFFICIENT_EVIDENCE":
        print("✗ INSUFFICIENT EVIDENCE")
        print()
        if report_data.get('hints'):
            for hint in report_data['hints']:
                print(f"  {hint}")
        print()
        print("→ ACTION: Check telemetry collection and file paths")
        print()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify entry flow gap - deterministic verification tool"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to run root directory, chunk directory, or JSON file"
    )
    parser.add_argument(
        "--write-json",
        type=Path,
        default=None,
        help="Write JSON report to this path (default: verify_entry_flow_gap_report.json in script dir)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed counters"
    )
    
    args = parser.parse_args()
    
    if not args.path.exists():
        print(f"ERROR: Path not found: {args.path}")
        sys.exit(1)
    
    # Find telemetry files
    master_json, chunk_json, chunk_footer, fail_capsule = find_telemetry_files(args.path)
    
    if not master_json and not chunk_json and not chunk_footer:
        print(f"ERROR: No telemetry files found in {args.path}")
        print("Expected one of:")
        print("  - ENTRY_FEATURES_USED_MASTER.json")
        print("  - ENTRY_FEATURES_USED.json")
        print("  - chunk_footer.json")
        sys.exit(1)
    
    # Extract fields
    payload = extract_fields(master_json, chunk_json, chunk_footer, fail_capsule)
    
    # Decide status
    status, exit_code, report_data = decide_status(payload)
    
    # Add paths to report
    report_data["paths_used"] = payload["paths_used"]
    report_data["status"] = status
    report_data["exit_code"] = exit_code
    
    # Print summary
    print_summary(status, exit_code, report_data, payload["paths_used"], args.verbose)
    
    # Write JSON report
    json_output_path = args.write_json
    if json_output_path is None:
        # Default: write to script directory
        script_dir = Path(__file__).parent
        json_output_path = script_dir / "verify_entry_flow_gap_report.json"
    
    try:
        with open(json_output_path, "w") as f:
            json.dump(report_data, f, indent=2)
        if args.verbose:
            print(f"JSON report written to: {json_output_path}")
    except Exception as e:
        print(f"WARNING: Failed to write JSON report: {e}")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
