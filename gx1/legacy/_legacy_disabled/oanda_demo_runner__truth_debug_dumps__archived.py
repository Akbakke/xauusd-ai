# ARCHIVED: truth/debug dump helpers extracted from gx1/execution/oanda_demo_runner.py
"""
Archived copies of legacy truth/debug dump helpers that were removed from
`gx1.execution.oanda_demo_runner.GX1DemoRunner`.

These functions are preserved for reference only and are not imported by the
current runner. They retain their original behavior and signatures.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import json as _json
import json as jsonlib

log = logging.getLogger(__name__)


def _write_xgb_input_truth_dump_if_ready(self, force: bool = False) -> None:
    """
    Archived: Write XGB input truth dump once per run (TRUTH/PREBUILT only, chunk_0 only).

    - If force=False: writes only when captured rows >= target (default 2000)
    - If force=True: writes with whatever rows were captured (if any) at end of replay

    Hard truth check (TRUTH): if >50% of features are constant in the sampled matrix,
    write XGB_INPUT_DEGENERATE_FATAL.json and raise RuntimeError.
    """
    Path_local = Path

    is_truth_run = os.getenv("GX1_RUN_MODE", "").upper() == "TRUTH" or os.getenv("GX1_TRUTH_MODE", "0") == "1"
    is_prebuilt = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1" and os.getenv("GX1_FEATURE_BUILD_DISABLED", "0") == "1"
    chunk_id_env = os.getenv("GX1_CHUNK_ID", "")
    if not (is_truth_run and is_prebuilt and str(chunk_id_env) == "0"):
        return

    if not hasattr(self, "_xgb_truth_dump_done"):
        self._xgb_truth_dump_done = False
    if self._xgb_truth_dump_done:
        return

    rows = getattr(self, "_xgb_truth_dump_rows", None)
    feat_names = getattr(self, "_xgb_truth_dump_feature_names", None)
    ssot_details = getattr(self, "_xgb_feature_names_ssot_details", None)
    n_target = int(getattr(self, "_xgb_truth_dump_n_target", 2000))
    if not rows or not feat_names:
        return
    if not force and len(rows) < n_target:
        return

    # Resolve run_root from explicit_output_dir if present (chunk dir -> parent)
    run_root = None
    if getattr(self, "explicit_output_dir", None):
        try:
            p = Path_local(str(self.explicit_output_dir))
            run_root = p.parent if p.name.startswith("chunk_") else p
        except Exception:
            run_root = None
    if run_root is None:
        raise RuntimeError("[TRUTH_FAIL] XGB_INPUT_TRUTH_DUMP cannot resolve run_root for writing artifacts")

    dump_json_path = run_root / "XGB_INPUT_TRUTH_DUMP.json"
    dump_md_path = run_root / "XGB_INPUT_TRUTH_DUMP.md"
    fatal_path = run_root / "XGB_INPUT_DEGENERATE_FATAL.json"

    if dump_json_path.exists():
        self._xgb_truth_dump_done = True
        return

    M = np.stack(rows, axis=0)
    feat_names = list(feat_names)
    n_rows_s, n_feat_s = int(M.shape[0]), int(M.shape[1])
    if n_feat_s != len(feat_names):
        raise RuntimeError(
            "[TRUTH_FAIL] XGB_INPUT_TRUTH_DUMP feature_names length mismatch: "
            f"n_feat={n_feat_s} names_len={len(feat_names)}"
        )

    per_col = []
    per_feature_stats_by_name: Dict[str, Any] = {}
    constant_names = []
    all_zero_names = []
    std_pairs = []
    for j, name in enumerate(feat_names):
        col = M[:, j]
        min_v = float(np.min(col))
        max_v = float(np.max(col))
        mean_v = float(np.mean(col))
        std_v = float(np.std(col))
        uniq = np.unique(col)
        unique_count = int(uniq.shape[0])
        pct_zero = float(np.mean(col == 0.0))
        is_constant = unique_count == 1
        const_value = float(uniq[0]) if is_constant else None
        if is_constant:
            constant_names.append(name)
        if pct_zero >= 1.0:
            all_zero_names.append(name)
        std_pairs.append((std_v, name))
        per_col.append(
            {
                "feature": name,
                "min": min_v,
                "max": max_v,
                "mean": mean_v,
                "std": std_v,
                "unique_count": unique_count,
                "percent_zero": pct_zero,
                "is_constant": is_constant,
                "const_value": const_value,
            }
        )
        per_feature_stats_by_name[name] = {
            "min": min_v,
            "max": max_v,
            "mean": mean_v,
            "std": std_v,
            "unique_count": unique_count,
            "percent_zero": pct_zero,
            "is_constant": is_constant,
            "const_value": const_value,
        }

    std_pairs_sorted = sorted(std_pairs, key=lambda x: (-x[0], x[1]))
    top20_by_std = [{"feature": n, "std": float(s)} for (s, n) in std_pairs_sorted[:20]]

    n_constant = int(len(constant_names))
    n_all_zero = int(len(all_zero_names))
    constant_frac = float(n_constant) / float(n_feat_s) if n_feat_s else 0.0
    varying_names = [n for n in feat_names if n not in constant_names]
    first5 = np.round(M[:5, :], 6).tolist()
    first5_rows_by_name = []
    for row in first5:
        first5_rows_by_name.append({name: row[i] for i, name in enumerate(feat_names)})

    payload = {
        "run_id": os.getenv("GX1_RUN_ID"),
        "chunk_id": os.getenv("GX1_CHUNK_ID"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows_sampled": n_rows_s,
        "n_features": n_feat_s,
        "X_shape": [n_rows_s, n_feat_s],
        "feature_names_ordered": feat_names,
        "feature_names_ssot": ssot_details,
        "degeneracy_summary": {
            "n_constant_features": n_constant,
            "constant_fraction": constant_frac,
            "n_all_zero_features": n_all_zero,
            "constant_feature_names": constant_names,
            "all_zero_feature_names": all_zero_names,
            "varying_feature_names": varying_names,
        },
        "per_feature_stats_by_name": per_feature_stats_by_name,
        "per_feature_stats_rows": per_col,
        "top20_features_by_std": [
            {
                "feature": r["feature"],
                "std": float(r["std"]),
                "min": per_feature_stats_by_name.get(r["feature"], {}).get("min"),
                "max": per_feature_stats_by_name.get(r["feature"], {}).get("max"),
                "percent_zero": per_feature_stats_by_name.get(r["feature"], {}).get("percent_zero"),
            }
            for r in top20_by_std
        ],
        "first5_rows_preview_by_name": first5_rows_by_name,
        "notes": [
            "Captured from the exact XGB input vector right before predict_proba().",
            "TRUTH/PREBUILT only; chunk_0 only; first N rows only (or force finalize at end).",
        ],
    }

    tmp_json = dump_json_path.with_suffix(dump_json_path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_json, "w", encoding="utf-8") as f:
        _json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_json, dump_json_path)

    md_lines = []
    md_lines.append("## XGB INPUT TRUTH DUMP")
    md_lines.append("")
    md_lines.append(f"- **run_id**: `{payload.get('run_id')}`")
    md_lines.append(f"- **chunk_id**: `{payload.get('chunk_id')}`")
    md_lines.append(f"- **generated_utc**: `{payload.get('generated_utc')}`")
    md_lines.append("")
    md_lines.append("## Section 1 — Shape")
    md_lines.append(f"- **n_rows_sampled**: `{n_rows_s}`")
    md_lines.append(f"- **n_features**: `{n_feat_s}`")
    md_lines.append("")
    md_lines.append("## Section 2 — Degeneracy Summary")
    md_lines.append(f"- **n_constant_features**: `{n_constant}` (fraction=`{constant_frac:.4f}`)")
    md_lines.append(f"- **n_all_zero_features**: `{n_all_zero}`")
    md_lines.append("")
    md_lines.append("## Ordered feature list (n=28)")
    for n in feat_names:
        md_lines.append(f"- `{n}`")
    md_lines.append("")
    md_lines.append(f"## Varying features (n={len(varying_names)})")
    for n in varying_names:
        md_lines.append(f"- `{n}`")
    md_lines.append("")
    md_lines.append("### Constant features (names)")
    for n in constant_names[:120]:
        md_lines.append(f"- `{n}`")
    md_lines.append("")
    md_lines.append("## Section 3 — Top 20 Features by Std")
    for r in payload.get("top20_features_by_std") or []:
        md_lines.append(
            f"- `{r['feature']}` std={float(r['std']):.8f} min={r.get('min')} max={r.get('max')} percent_zero={r.get('percent_zero')}"
        )
    md_lines.append("")
    md_lines.append("## Section 4 — First 5 rows (matrix preview)")
    md_lines.append("```")
    for row in first5_rows_by_name:
        md_lines.append(str(row))
    md_lines.append("```")
    md_text = "\n".join(md_lines) + "\n"
    tmp_md = dump_md_path.with_suffix(dump_md_path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_md, "w", encoding="utf-8") as f:
        f.write(md_text)
    os.replace(tmp_md, dump_md_path)

    if constant_frac > 0.50:
        fatal = {
            "error_type": "XGB_INPUT_DEGENERATE_FATAL",
            "run_id": payload.get("run_id"),
            "chunk_id": payload.get("chunk_id"),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "n_rows_sampled": n_rows_s,
            "n_features": n_feat_s,
            "n_constant_features": n_constant,
            "constant_fraction": constant_frac,
            "threshold": 0.50,
            "dump_json_path": str(dump_json_path),
            "constant_feature_names": constant_names,
            "all_zero_feature_names": all_zero_names,
            "varying_feature_names": varying_names,
        }
        tmp_f = fatal_path.with_suffix(fatal_path.suffix + f".tmp.{os.getpid()}")
        with open(tmp_f, "w", encoding="utf-8") as f:
            _json.dump(fatal, f, indent=2)
        os.replace(tmp_f, fatal_path)
        raise RuntimeError(
            "XGB input matrix degenerate: "
            f"constant_fraction={constant_frac:.4f} "
            f"all_zero_first20={all_zero_names[:20]} varying_first20={varying_names[:20]}"
        )

    self._xgb_truth_dump_done = True
    try:
        self._xgb_truth_dump_rows = []  # type: ignore[attr-defined]
    except Exception:
        pass


def _log_xgb_input_debug(
    self,
    session: str,
    timestamp: Optional[str],
    feature_list: List[str],
    X: np.ndarray,
    missing_features: List[str],
) -> None:
    """Archived: Log per-feature debug stats for XGB input (sampled, TRUTH/SMOKE only)."""
    if not self.xgb_input_debug_enabled or not self.xgb_input_debug_jsonl_path:
        return

    session_upper = session.upper()
    if session_upper not in ["EU", "OVERLAP"]:
        return

    self.xgb_input_debug_call_counts[session_upper] += 1
    call_count = self.xgb_input_debug_call_counts[session_upper]
    logged_count = self.xgb_input_debug_logged_counts[session_upper]

    if logged_count >= self.xgb_input_debug_max_per_session:
        return
    if call_count % self.xgb_input_debug_sample_n != 0:
        return

    # Extract vector
    vec = X[0] if X.shape[0] == 1 else X.flatten()
    finite_mask = np.isfinite(vec)
    n_nonfinite = int(np.sum(~finite_mask))
    n_zero = int(np.sum(vec == 0.0))

    # Per-feature stats (single value -> min=max=value, std=0.0)
    feature_stats = {}
    for idx, feat_name in enumerate(feature_list):
        if idx >= len(vec):
            break
        val = vec[idx]
        if not np.isfinite(val):
            feature_stats[feat_name] = {"min": None, "max": None, "std": None}
        else:
            feature_stats[feat_name] = {"min": float(val), "max": float(val), "std": 0.0}

    # Update flat-input counter (std == 0)
    std_val = float(np.std(vec[finite_mask])) if np.any(finite_mask) else 0.0
    if std_val == 0.0:
        self.xgb_input_debug_flat_counts[session_upper] += 1
    else:
        self.xgb_input_debug_flat_counts[session_upper] = 0

    # Optional debug-assert (TRUTH/SMOKE): K consecutive flat inputs
    run_mode = os.getenv("GX1_RUN_MODE", "").upper()
    is_truth = run_mode in ["TRUTH", "SMOKE"] or os.getenv("GX1_TRUTH_TELEMETRY", "0") == "1"
    if is_truth and self.xgb_input_debug_flat_counts[session_upper] >= self.xgb_input_debug_flat_k:
        raise RuntimeError(
            f"[XGB_INPUT_DEBUG_FLAT] Session={session_upper} input std==0 for "
            f"{self.xgb_input_debug_flat_counts[session_upper]} sampled calls (K={self.xgb_input_debug_flat_k}). "
            "Input appears flat at XGB input build step."
        )

    log_entry = {
        "ts": timestamp,
        "session": session_upper,
        "n_features": len(feature_list),
        "n_nonfinite": n_nonfinite,
        "n_zero": n_zero,
        "missing_features_count": len(missing_features),
        "missing_features_sample": missing_features[:10],
        "feature_stats": feature_stats,
    }

    try:
        with open(self.xgb_input_debug_jsonl_path, "a", encoding="utf-8") as f:
            f.write(jsonlib.dumps(log_entry, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.xgb_input_debug_logged_counts[session_upper] += 1
    except Exception as e:
        log.warning(f"[XGB_INPUT_DEBUG] Failed to write debug log: {e}")


def write_xgb_fingerprint_summary_static(
    output_dir: Path,
    run_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
    min_logged: int = 50,
) -> None:
    """Archived: Static method to write XGB fingerprint summary."""
    # Glob all per-process JSONL files (include chunk subdirs)
    jsonl_files = list(output_dir.rglob("XGB_INPUT_FINGERPRINT.*.jsonl"))

    if not jsonl_files:
        log.warning("[XGB_FINGERPRINT] No JSONL files found, skipping summary")
        return

    # Read all logged entries from all per-process files
    entries: List[Dict[str, Any]] = []
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(jsonlib.loads(line))
        except Exception as e:
            log.warning(f"[XGB_FINGERPRINT] Failed to read {jsonl_file}: {e}")
            continue

    if not entries:
        log.warning("[XGB_FINGERPRINT] No entries found in JSONL files")
        return

    # Analyze per session
    summary: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "output_dir": str(output_dir),
        "run_id": run_id,
        "chunk_id": chunk_id,
        "sessions": {},
    }

    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        sess = entry.get("session", "UNKNOWN")
        by_session[sess].append(entry)

    run_mode = os.getenv("GX1_RUN_MODE", "").upper()
    is_truth = run_mode in ["TRUTH", "SMOKE"] or os.getenv("GX1_TRUTH_TELEMETRY", "0") == "1"

    for session, sess_entries in by_session.items():
        n_logged = len(sess_entries)
        fingerprints = [e.get("fingerprint") for e in sess_entries if e.get("fingerprint")]
        n_unique_fingerprints = len(set(fingerprints))
        unique_ratio = n_unique_fingerprints / n_logged if n_logged > 0 else 0.0

        # Compute input/output stats
        input_stds = [e.get("input_std") for e in sess_entries if e.get("input_std") is not None]
        input_mins = [e.get("input_min") for e in sess_entries if e.get("input_min") is not None]
        input_maxs = [e.get("input_max") for e in sess_entries if e.get("input_max") is not None]
        p_long_raws = [e.get("p_long_raw") for e in sess_entries if e.get("p_long_raw") is not None]
        uncertainties = [e.get("uncertainty") for e in sess_entries if e.get("uncertainty") is not None]

        sess_summary: Dict[str, Any] = {
            "n_logged": n_logged,
            "n_unique_fingerprints": n_unique_fingerprints,
            "unique_ratio": unique_ratio,
            "inputs_std_min": float(np.min(input_stds)) if input_stds else None,
            "inputs_std_max": float(np.max(input_stds)) if input_stds else None,
            "outputs_min": float(np.min(p_long_raws)) if p_long_raws else None,
            "outputs_max": float(np.max(p_long_raws)) if p_long_raws else None,
            "outputs_std": float(np.std(p_long_raws)) if p_long_raws else None,
            "uncertainty_min": float(np.min(uncertainties)) if uncertainties else None,
            "uncertainty_max": float(np.max(uncertainties)) if uncertainties else None,
            "uncertainty_std": float(np.std(uncertainties)) if uncertainties else None,
        }

        # Determine verdict
        if n_logged >= min_logged:
            if n_unique_fingerprints == 1:
                sess_summary["verdict"] = "IDENTICAL_INPUTS"
                # TRUTH invariant: FATAL if EU/OVERLAP have identical inputs
                if is_truth and session in ["EU", "OVERLAP"]:
                    ts_range = {
                        "first": sess_entries[0].get("ts"),
                        "last": sess_entries[-1].get("ts"),
                    }
                    capsule = {
                        "generated_at": datetime.utcnow().isoformat() + "Z",
                        "session": session,
                        "n_logged": n_logged,
                        "n_unique_fingerprints": n_unique_fingerprints,
                        "fingerprint": fingerprints[0] if fingerprints else None,
                        "ts_range": ts_range,
                        "input_stats": {
                            "std_min": sess_summary["inputs_std_min"],
                            "std_max": sess_summary["inputs_std_max"],
                        },
                        "output_stats": {
                            "min": sess_summary["outputs_min"],
                            "max": sess_summary["outputs_max"],
                            "std": sess_summary["outputs_std"],
                        },
                        "run_id": run_id,
                        "chunk_id": chunk_id,
                    }
                    capsule_path = output_dir / "XGB_INPUT_IDENTICAL_FATAL.json"
                    try:
                        with open(capsule_path, "w", encoding="utf-8") as f:
                            jsonlib.dump(capsule, f, indent=2, sort_keys=True)
                        log.error(f"[XGB_FINGERPRINT] FATAL: {session} has identical inputs (n_logged={n_logged}, fingerprint={fingerprints[0][:16] if fingerprints else 'N/A'}...)")
                        raise RuntimeError(
                            f"[XGB_INPUT_IDENTICAL_FATAL] Session {session} has identical XGB inputs across {n_logged} logged calls. "
                            f"This indicates a feature feed/sanitizer/prebuilt mapping bug. "
                            f"Capsule written to: {capsule_path}"
                        )
                    except RuntimeError:
                        raise
                    except Exception as e:
                        log.error(f"[XGB_FINGERPRINT] Failed to write FATAL capsule: {e}")
            else:
                sess_summary["verdict"] = "VARIABLE_INPUTS"
        else:
            sess_summary["verdict"] = "INSUFFICIENT_SAMPLES"

        summary["sessions"][session] = sess_summary

    # Write summary JSON
    summary_path = output_dir / "XGB_INPUT_FINGERPRINT_SUMMARY.json"
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            jsonlib.dump(summary, f, indent=2, sort_keys=True)
        log.info(f"[XGB_FINGERPRINT] Wrote summary to {summary_path}")
    except Exception as e:
        log.error(f"[XGB_FINGERPRINT] Failed to write summary: {e}")

    # Write summary MD
    md_path = output_dir / "XGB_INPUT_FINGERPRINT_SUMMARY.md"
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# XGB Input Fingerprint Summary\n\n")
            f.write(f"- Generated at: `{summary['generated_at']}`\n")
            f.write(f"- Output dir: `{summary['output_dir']}`\n")
            f.write(f"- Run ID: `{summary['run_id']}`\n")
            f.write(f"- Chunk ID: `{summary['chunk_id']}`\n\n")
            for session, sess_summary in sorted(summary["sessions"].items()):
                f.write(f"## Session {session}\n\n")
                f.write(f"- n_logged: {sess_summary['n_logged']}\n")
                f.write(f"- n_unique_fingerprints: {sess_summary['n_unique_fingerprints']}\n")
                f.write(f"- unique_ratio: {sess_summary['unique_ratio']:.4f}\n")
                f.write(f"- verdict: **{sess_summary['verdict']}**\n")
                if sess_summary.get('inputs_std_min') is not None:
                    f.write(f"- inputs_std: min={sess_summary['inputs_std_min']:.6f}, max={sess_summary['inputs_std_max']:.6f}\n")
                if sess_summary.get('outputs_std') is not None:
                    f.write(f"- outputs_std: {sess_summary['outputs_std']:.6f}\n")
                f.write("\n")
        log.info(f"[XGB_FINGERPRINT] Wrote summary MD to {md_path}")
    except Exception as e:
        log.error(f"[XGB_FINGERPRINT] Failed to write summary MD: {e}")


def _write_xgb_fingerprint_summary(self) -> None:
    """Archived: Instance method wrapper for fingerprint summary."""
    if not self.xgb_fingerprint_enabled or not self.explicit_output_dir:
        return
    write_xgb_fingerprint_summary_static(
        output_dir=self.explicit_output_dir,
        run_id=getattr(self, "run_id", None),
        chunk_id=getattr(self, "chunk_id", None),
        min_logged=self.xgb_fingerprint_min_logged,
    )


if __name__ == "__main__":
    # Smoke test: ensure module imports and functions are callable.
    print("Archived truth/debug dump helpers loaded:")
    print(
        [
            "_write_xgb_input_truth_dump_if_ready",
            "_log_xgb_input_debug",
            "write_xgb_fingerprint_summary_static",
            "_write_xgb_fingerprint_summary",
        ]
    )
