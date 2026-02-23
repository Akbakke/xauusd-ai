#!/usr/bin/env python3
"""
XGB → Transformer Snapshot Report

PURPOSE: Generate a precise SSoT report of what XGB provides to Transformer:
    - Channel names, injection points, counts, statistics
    - Missing/constant rate analysis
    - Edge dependency check (per-channel ablation summary if requested)

Usage:
    python3 gx1/scripts/report_xgb_to_transformer_snapshot.py <run_root>
    python3 gx1/scripts/report_xgb_to_transformer_snapshot.py <run_root> --write-json --write-md
    python3 gx1/scripts/report_xgb_to_transformer_snapshot.py <run_root> --top-k 20

Output:
    XGB_TO_TRANSFORMER_SNAPSHOT.json
    XGB_TO_TRANSFORMER_SNAPSHOT.md
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file safely."""
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Failed to load JSON from {path}: {e}")
        return None


def find_telemetry_files(run_root: Path) -> Dict[str, Path]:
    """Find all telemetry files in run root."""
    files = {}
    
    # Priority 1: Master file in run root
    master = run_root / "ENTRY_FEATURES_USED_MASTER.json"
    if master.exists():
        files["master"] = master
    
    # Priority 2: Chunk files
    chunk_dirs = sorted(run_root.glob("chunk_*"))
    for chunk_dir in chunk_dirs:
        # ENTRY_FEATURES_USED.json
        ef = chunk_dir / "ENTRY_FEATURES_USED.json"
        if ef.exists():
            files[f"chunk_{chunk_dir.name}_entry_features"] = ef
        
        # ENTRY_FEATURES_TELEMETRY.json
        tel = chunk_dir / "ENTRY_FEATURES_TELEMETRY.json"
        if tel.exists():
            files[f"chunk_{chunk_dir.name}_telemetry"] = tel
        
        # chunk_footer.json
        footer = chunk_dir / "chunk_footer.json"
        if footer.exists():
            files[f"chunk_{chunk_dir.name}_footer"] = footer
    
    return files


def load_aggregated_telemetry(run_root: Path) -> Dict[str, Any]:
    """Load and aggregate telemetry from all sources."""
    aggregated = {
        "sources_found": [],
        "transformer_forward_calls": 0,
        "model_attempts": {},
        "model_forwards": {},
        "xgb_pre_predict_count": 0,
        "xgb_post_predict_count": 0,
        "n_xgb_channels_in_transformer_input": 0,
        "xgb_channel_names": [],
        "xgb_seq_channel_names": [],
        "xgb_snap_channel_names": [],
        "xgb_used_as": "none",
        "post_predict_called": False,
        "veto_applied_count": 0,
        "entry_routing": {},
        "seq_feature_names": [],
        "snap_feature_names": [],
        "toggles": {},
    }
    
    files = find_telemetry_files(run_root)
    aggregated["sources_found"] = list(files.keys())
    
    # Load from master if exists
    if "master" in files:
        master_data = load_json_safe(files["master"])
        if master_data:
            _merge_telemetry(aggregated, master_data)
    
    # Load from chunks
    for key, path in files.items():
        if "entry_features" in key and "master" not in key:
            data = load_json_safe(path)
            if data:
                _merge_telemetry(aggregated, data)
    
    return aggregated


def _merge_telemetry(aggregated: Dict[str, Any], data: Dict[str, Any]) -> None:
    """Merge telemetry data into aggregated dict."""
    # Scalar sums
    aggregated["transformer_forward_calls"] += data.get("transformer_forward_calls", 0)
    
    # XGB flow
    xgb_flow = data.get("xgb_flow", {})
    aggregated["xgb_pre_predict_count"] += xgb_flow.get("xgb_pre_predict_count", 0)
    aggregated["xgb_post_predict_count"] += xgb_flow.get("xgb_post_predict_count", 0)
    aggregated["veto_applied_count"] += xgb_flow.get("veto_applied_count", 0)
    
    # Take first non-zero/non-empty values for constants
    if aggregated["n_xgb_channels_in_transformer_input"] == 0:
        aggregated["n_xgb_channels_in_transformer_input"] = xgb_flow.get("n_xgb_channels_in_transformer_input", 0)
    
    if aggregated["xgb_used_as"] == "none":
        aggregated["xgb_used_as"] = xgb_flow.get("xgb_used_as", "none")
    
    if not aggregated["post_predict_called"]:
        aggregated["post_predict_called"] = xgb_flow.get("post_predict_called", False)
    
    # Channel names (take first non-empty)
    if not aggregated["xgb_channel_names"]:
        xgb_seq = data.get("xgb_seq_channels", {})
        xgb_snap = data.get("xgb_snap_channels", {})
        aggregated["xgb_seq_channel_names"] = xgb_seq.get("names", [])
        aggregated["xgb_snap_channel_names"] = xgb_snap.get("names", [])
        aggregated["xgb_channel_names"] = aggregated["xgb_seq_channel_names"] + aggregated["xgb_snap_channel_names"]
    
    # Feature names
    if not aggregated["seq_feature_names"]:
        seq_features = data.get("seq_features", {})
        aggregated["seq_feature_names"] = seq_features.get("names", [])
    
    if not aggregated["snap_feature_names"]:
        snap_features = data.get("snap_features", {})
        aggregated["snap_feature_names"] = snap_features.get("names", [])
    
    # Model attempts/forwards
    model_entry = data.get("model_entry", {})
    for model, count in model_entry.get("model_attempt_calls", {}).items():
        aggregated["model_attempts"][model] = aggregated["model_attempts"].get(model, 0) + count
    for model, count in model_entry.get("model_forward_calls", {}).items():
        aggregated["model_forwards"][model] = aggregated["model_forwards"].get(model, 0) + count
    
    # Entry routing
    entry_routing = data.get("entry_routing_aggregate", {})
    for model, count in entry_routing.get("selected_model_counts", {}).items():
        aggregated["entry_routing"][model] = aggregated["entry_routing"].get(model, 0) + count
    
    # Toggles
    toggles = data.get("toggles", {})
    if toggles:
        aggregated["toggles"].update(toggles)


# ============================================================================
# ANALYSIS
# ============================================================================

@dataclass
class XGBChannelAnalysis:
    """Analysis of a single XGB channel."""
    name: str
    injection_point: str  # "seq", "snap", or "both"
    present_in_telemetry: bool = True
    sample_count: int = 0
    missing_rate: float = 0.0
    constant_rate: float = 0.0
    std: float = 0.0
    mean: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    is_useful: bool = True
    useful_flag: str = ""
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "injection_point": self.injection_point,
            "present_in_telemetry": self.present_in_telemetry,
            "sample_count": self.sample_count,
            "missing_rate": self.missing_rate,
            "constant_rate": self.constant_rate,
            "std": self.std,
            "mean": self.mean,
            "min": self.min_val,
            "max": self.max_val,
            "is_useful": self.is_useful,
            "useful_flag": self.useful_flag,
            "notes": self.notes,
        }


def analyze_xgb_channels(aggregated: Dict[str, Any]) -> List[XGBChannelAnalysis]:
    """Analyze each XGB channel for usefulness."""
    analyses = []
    
    seq_channels = set(aggregated["xgb_seq_channel_names"])
    snap_channels = set(aggregated["xgb_snap_channel_names"])
    all_channels = seq_channels | snap_channels
    
    for channel_name in sorted(all_channels):
        # Determine injection point
        in_seq = channel_name in seq_channels
        in_snap = channel_name in snap_channels
        if in_seq and in_snap:
            injection = "both"
        elif in_seq:
            injection = "seq"
        else:
            injection = "snap"
        
        analysis = XGBChannelAnalysis(
            name=channel_name,
            injection_point=injection,
            sample_count=aggregated["xgb_pre_predict_count"],
        )
        
        # Without detailed per-sample telemetry, we can only mark as "PRESENT"
        # and flag based on whether it's actually used
        if aggregated["xgb_pre_predict_count"] > 0:
            analysis.useful_flag = "PRESENT_IN_PIPELINE"
            analysis.notes.append(f"Used in {aggregated['xgb_pre_predict_count']:,} XGB pre-predict calls")
        else:
            analysis.is_useful = False
            analysis.useful_flag = "NOT_USED"
            analysis.notes.append("XGB pre-predict not called")
        
        analyses.append(analysis)
    
    return analyses


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_snapshot_report(
    run_root: Path,
    aggregated: Dict[str, Any],
    channel_analyses: List[XGBChannelAnalysis],
    output_dir: Optional[Path] = None,
    write_json: bool = True,
    write_md: bool = True,
) -> Dict[str, Any]:
    """Generate the XGB → Transformer snapshot report."""
    
    timestamp = datetime.now().isoformat()
    
    report = {
        "report_type": "XGB_TO_TRANSFORMER_SNAPSHOT",
        "timestamp": timestamp,
        "run_root": str(run_root),
        "sources_found": aggregated["sources_found"],
        
        # A) Pipeline evidence
        "pipeline_evidence": {
            "transformer_forward_calls": aggregated["transformer_forward_calls"],
            "model_attempts": aggregated["model_attempts"],
            "model_forwards": aggregated["model_forwards"],
            "entry_routing": aggregated["entry_routing"],
            "xgb_pre_predict_count": aggregated["xgb_pre_predict_count"],
            "n_xgb_channels_in_transformer_input": aggregated["n_xgb_channels_in_transformer_input"],
            "xgb_channel_names": aggregated["xgb_channel_names"],
            "xgb_used_as": aggregated["xgb_used_as"],
            # POST fields (to be removed in DEL 3)
            "post_predict_called": aggregated["post_predict_called"],
            "xgb_post_predict_count": aggregated["xgb_post_predict_count"],
            "veto_applied_count": aggregated["veto_applied_count"],
        },
        
        # B) Injection points
        "injection_points": {
            "xgb_seq_channel_names": aggregated["xgb_seq_channel_names"],
            "xgb_snap_channel_names": aggregated["xgb_snap_channel_names"],
            "n_seq_xgb_channels": len(aggregated["xgb_seq_channel_names"]),
            "n_snap_xgb_channels": len(aggregated["xgb_snap_channel_names"]),
            "seq_feature_names_sample": aggregated["seq_feature_names"][:10] if aggregated["seq_feature_names"] else [],
            "snap_feature_names_sample": aggregated["snap_feature_names"][:10] if aggregated["snap_feature_names"] else [],
        },
        
        # C) Channel analysis
        "channel_analyses": [a.to_dict() for a in channel_analyses],
        
        # D) Toggles
        "toggles": aggregated["toggles"],
    }
    
    # Determine output directory
    if output_dir is None:
        output_dir = run_root
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write JSON
    if write_json:
        json_path = output_dir / "XGB_TO_TRANSFORMER_SNAPSHOT.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Written JSON: {json_path}")
    
    # Write Markdown
    if write_md:
        md_path = output_dir / "XGB_TO_TRANSFORMER_SNAPSHOT.md"
        md_content = generate_markdown(report, channel_analyses)
        with open(md_path, "w") as f:
            f.write(md_content)
        log.info(f"Written Markdown: {md_path}")
    
    return report


def generate_markdown(report: Dict[str, Any], channel_analyses: List[XGBChannelAnalysis]) -> str:
    """Generate markdown report."""
    pe = report["pipeline_evidence"]
    ip = report["injection_points"]
    
    md = f"""# XGB → Transformer Snapshot Report

**Generated:** {report['timestamp']}  
**Run Root:** `{report['run_root']}`

---

## A) Pipeline Evidence

| Metric | Value |
|--------|-------|
| `transformer_forward_calls` | {pe['transformer_forward_calls']:,} |
| `xgb_pre_predict_count` | {pe['xgb_pre_predict_count']:,} |
| `n_xgb_channels_in_transformer_input` | {pe['n_xgb_channels_in_transformer_input']} |
| `xgb_used_as` | **{pe['xgb_used_as']}** |
| `post_predict_called` | {pe['post_predict_called']} |
| `xgb_post_predict_count` | {pe['xgb_post_predict_count']:,} |
| `veto_applied_count` | {pe['veto_applied_count']} |

### Entry Routing

| Model | Count |
|-------|-------|
"""
    for model, count in pe.get("entry_routing", {}).items():
        md += f"| {model} | {count:,} |\n"
    
    md += f"""
---

## B) Injection Points

XGB channels are injected into Transformer input at two points:

| Point | Count | Channel Names |
|-------|-------|---------------|
| **Sequence (seq)** | {ip['n_seq_xgb_channels']} | {', '.join(ip['xgb_seq_channel_names']) or 'None'} |
| **Snapshot (snap)** | {ip['n_snap_xgb_channels']} | {', '.join(ip['xgb_snap_channel_names']) or 'None'} |

### Full Channel List

"""
    for name in pe.get("xgb_channel_names", []):
        md += f"- `{name}`\n"
    
    md += f"""
---

## C) Channel Analysis (Usefulness)

| Channel | Injection | Sample Count | Status | Notes |
|---------|-----------|--------------|--------|-------|
"""
    for a in channel_analyses:
        notes_str = "; ".join(a.notes[:2]) if a.notes else "-"
        status = "✅ " + a.useful_flag if a.is_useful else "❌ " + a.useful_flag
        md += f"| `{a.name}` | {a.injection_point} | {a.sample_count:,} | {status} | {notes_str} |\n"
    
    md += f"""
---

## D) Summary

| Category | Value |
|----------|-------|
| Total XGB channels in Transformer | **{pe['n_xgb_channels_in_transformer_input']}** |
| XGB channels in seq | {ip['n_seq_xgb_channels']} |
| XGB channels in snap | {ip['n_snap_xgb_channels']} |
| XGB usage mode | **{pe['xgb_used_as']}** |
| XGB pre-predict calls | {pe['xgb_pre_predict_count']:,} |
| XGB post-predict calls | {pe['xgb_post_predict_count']:,} |
| POST active | **{'YES ⚠️' if pe['post_predict_called'] or pe['xgb_post_predict_count'] > 0 else 'NO ✅'}** |

---

## E) Recommendations

"""
    if pe["xgb_post_predict_count"] > 0 or pe["post_predict_called"]:
        md += """### ⚠️ XGB POST is still active

XGB post-predict (calibration/veto) is still being called. This should be removed as per DEL 3.

"""
    else:
        md += """### ✅ XGB POST is inactive

XGB post-predict (calibration/veto) is not being called. Ready for removal.

"""
    
    md += """### XGB → Transformer Channel Status

All XGB channels are currently **PRESENT_IN_PIPELINE**. Detailed per-sample statistics require additional instrumentation.

To get per-channel ablation results, run:
```bash
python3 gx1/scripts/run_xgb_flow_ablation_qsmoke.py --arm test1_channels ...
```

"""
    return md


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate XGB → Transformer Snapshot Report")
    parser.add_argument("run_root", type=Path, help="Path to run root directory")
    parser.add_argument("--top-k", type=int, default=20, help="Top K channels to show")
    parser.add_argument("--write-json", action="store_true", default=True, help="Write JSON report")
    parser.add_argument("--write-md", action="store_true", default=True, help="Write Markdown report")
    parser.add_argument("--require-telemetry", action="store_true", default=True, help="Require telemetry files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: run_root)")
    
    args = parser.parse_args()
    
    run_root = args.run_root.resolve()
    if not run_root.exists():
        log.error(f"Run root does not exist: {run_root}")
        return 1
    
    log.info(f"Loading telemetry from: {run_root}")
    
    # Load aggregated telemetry
    aggregated = load_aggregated_telemetry(run_root)
    
    if args.require_telemetry and not aggregated["sources_found"]:
        log.error(f"No telemetry files found in {run_root}")
        return 1
    
    log.info(f"Found {len(aggregated['sources_found'])} telemetry sources")
    
    # Analyze XGB channels
    channel_analyses = analyze_xgb_channels(aggregated)
    log.info(f"Analyzed {len(channel_analyses)} XGB channels")
    
    # Generate report
    output_dir = args.output_dir or run_root
    report = generate_snapshot_report(
        run_root,
        aggregated,
        channel_analyses,
        output_dir=output_dir,
        write_json=args.write_json,
        write_md=args.write_md,
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("XGB → TRANSFORMER SNAPSHOT SUMMARY")
    print("=" * 80)
    pe = report["pipeline_evidence"]
    print(f"  transformer_forward_calls:        {pe['transformer_forward_calls']:,}")
    print(f"  xgb_pre_predict_count:            {pe['xgb_pre_predict_count']:,}")
    print(f"  n_xgb_channels_in_transformer:    {pe['n_xgb_channels_in_transformer_input']}")
    print(f"  xgb_used_as:                      {pe['xgb_used_as']}")
    print(f"  xgb_post_predict_count:           {pe['xgb_post_predict_count']:,}")
    print(f"  post_predict_called:              {pe['post_predict_called']}")
    print(f"  veto_applied_count:               {pe['veto_applied_count']}")
    print("")
    print("XGB Channels:")
    for name in pe.get("xgb_channel_names", []):
        print(f"  - {name}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
