"""
Sweep EXIT environment overrides against TRUTH e2e and summarize results.

Runs a grid over exit threshold, hysteresis, arb hysteresis, and min_pnl gates
by invoking the canonical TRUTH replay with deterministic settings. Produces a
CSV and a short markdown summary of the top configurations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


RUN_ID_RE = re.compile(r"E2E_SANITY_\d{8}_\d{6}")
REPO_ROOT = Path(__file__).resolve().parents[2]
GX1_DATA = Path("/home/andre2/GX1_DATA")
TRUTH_FILE = REPO_ROOT / "gx1" / "configs" / "canonical_truth_signal_only.json"
TRUTH_CMD = [
    "/home/andre2/venvs/gx1/bin/python",
    "-m",
    "gx1.scripts.run_truth_e2e_sanity",
    "--truth-file",
    str(TRUTH_FILE),
]


@dataclass
class RunResult:
    status: str
    run_id: str
    exit_code: int
    thr: float
    k: int
    arb_hyst: int
    min_pnl: float
    fail_reason: str = ""
    log_tail: str = ""
    exit_eval_long: int = 0
    exit_eval_short: int = 0
    exit_close_long: int = 0
    exit_close_short: int = 0
    entry_accept_long: int = 0
    entry_accept_short: int = 0
    n_probs: int = 0
    ge_thr: int = 0
    min_prob: float = 0.0
    mean_prob: float = 0.0
    max_prob: float = 0.0
    mean_pnl_bps_eval: float = 0.0
    mean_bars_held_eval: float = 0.0

    @property
    def exit_close_total(self) -> int:
        return int(self.exit_close_long) + int(self.exit_close_short)

    @property
    def exit_eval_total(self) -> int:
        return int(self.exit_eval_long) + int(self.exit_eval_short)

    @property
    def close_rate(self) -> float:
        denom = max(self.exit_eval_total, 1)
        return float(self.exit_close_total) / float(denom)

    @property
    def balanced_score(self) -> float:
        return self.close_rate * float(self.mean_prob)


def compile_repo() -> None:
    proc = subprocess.run(
        ["/home/andre2/venvs/gx1/bin/python", "-m", "compileall", "-q", "gx1"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"compileall failed: {proc.stdout}\\n{proc.stderr}")


def build_grid() -> List[Tuple[float, int, int, float]]:
    thresholds = [0.005, 0.01, 0.02, 0.05]
    ks = [1, 2, 3]
    arb_hyst_values = [1, 2]
    min_pnls = [-999.0, -10.0, 0.0]
    grid: List[Tuple[float, int, int, float]] = []
    for thr in thresholds:
        for k in ks:
            for arb_hyst in arb_hyst_values:
                for min_pnl in min_pnls:
                    grid.append((thr, k, arb_hyst, min_pnl))
    return grid


def parse_run_id(text: str) -> Optional[str]:
    m = RUN_ID_RE.search(text)
    return m.group(0) if m else None


def classify_fail_reason(text: str) -> str:
    if "[EXIT_CONTRACT]" in text:
        return "EXIT_CONTRACT"
    if "[EXIT_IO_CONTRACT_VIOLATION]" in text:
        return "EXIT_IO_CONTRACT_VIOLATION"
    if "[EXIT_INPUT_AUDIT_ASSERT]" in text:
        return "EXIT_INPUT_AUDIT_ASSERT"
    if "RuntimeError" in text:
        return "RUNTIME_ERROR"
    return "TRUTH_FAIL"


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def parse_exits_jsonl(path: Path, thr: float) -> Tuple[int, int, float, float, float, float, float]:
    if not path.exists():
        return 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0
    n_probs = 0
    ge_thr = 0
    probs: List[float] = []
    pnls: List[float] = []
    bars: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            computed = rec.get("computed") or {}
            prob = computed.get("prob_close")
            if prob is None:
                continue
            try:
                prob_f = float(prob)
            except Exception:
                continue
            n_probs += 1
            probs.append(prob_f)
            if prob_f >= thr:
                ge_thr += 1
            scalars = rec.get("scalars") or {}
            pnl_bps_now = scalars.get("pnl_bps_now")
            bars_held = scalars.get("bars_held")
            try:
                if pnl_bps_now is not None:
                    pnls.append(float(pnl_bps_now))
                if bars_held is not None:
                    bars.append(float(bars_held))
            except Exception:
                pass
    if not probs:
        return 0, 0, 0.0, 0.0, 0.0, safe_mean(pnls), safe_mean(bars)
    return (
        n_probs,
        ge_thr,
        float(min(probs)),
        safe_mean(probs),
        float(max(probs)),
        safe_mean(pnls),
        safe_mean(bars),
    )


def load_footer_metrics(run_dir: Path) -> Dict[str, int]:
    footer = read_json(run_dir / "replay" / "chunk_0" / "chunk_footer.json") or {}
    return {
        "exit_eval_long": int(footer.get("exit_eval_long") or 0),
        "exit_eval_short": int(footer.get("exit_eval_short") or 0),
        "exit_close_long": int(footer.get("exit_close_long") or 0),
        "exit_close_short": int(footer.get("exit_close_short") or 0),
        "entry_accept_long": int(footer.get("entry_accept_long") or 0),
        "entry_accept_short": int(footer.get("entry_accept_short") or 0),
    }


def run_truth(
    thr: float,
    k: int,
    arb_hyst: int,
    min_pnl: float,
) -> Tuple[str, int, str]:
    env = os.environ.copy()
    env.update(
        {
            "GX1_EXIT_AUDIT_STRICT": "1",
            "GX1_EXIT_THRESHOLD": str(thr),
            "GX1_EXIT_REQUIRE_CONSECUTIVE": str(k),
            "GX1_ARB_MIN_EXIT_PROB": str(thr),
            "GX1_ARB_EXIT_PROB_HYSTERESIS": str(arb_hyst),
            "GX1_ARB_MIN_PNL_BPS": str(min_pnl),
            "GX1_DATA": str(GX1_DATA),
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    proc = subprocess.run(
        TRUTH_CMD,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    combined = (proc.stdout or "") + "\\n" + (proc.stderr or "")
    run_id = parse_run_id(combined) or f"RUN_FAIL_{int(time.time())}"
    return run_id, proc.returncode, combined


def write_csv(out_path: Path, rows: List[RunResult]) -> None:
    base_fields = list(asdict(rows[0]).keys())
    extra_fields = ["exit_close_total", "exit_eval_total", "close_rate", "balanced_score"]
    fieldnames = base_fields + extra_fields
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = asdict(r)
            row.update(
                {
                    "exit_close_total": r.exit_close_total,
                    "exit_eval_total": r.exit_eval_total,
                    "close_rate": r.close_rate,
                    "balanced_score": r.balanced_score,
                }
            )
            writer.writerow(row)


def write_markdown(out_path: Path, rows: List[RunResult]) -> None:
    if not rows:
        out_path.write_text("# Sweep results\\n\\nNo runs.\\n", encoding="utf-8")
        return
    by_close = sorted(rows, key=lambda r: r.exit_close_total, reverse=True)[:10]
    by_rate = sorted(rows, key=lambda r: r.close_rate, reverse=True)[:10]
    by_bal = sorted(rows, key=lambda r: r.balanced_score, reverse=True)[:10]
    lines: List[str] = ["# EXIT sweep top-10\\n"]
    lines.append("## Most closes\\n")
    lines.extend(
        f"- {r.run_id}: close_total={r.exit_close_total} close_rate={r.close_rate:.4f} mean_prob={r.mean_prob:.4f}"
        for r in by_close
    )
    lines.append("\\n## Best close_rate\\n")
    lines.extend(
        f"- {r.run_id}: close_rate={r.close_rate:.4f} closes={r.exit_close_total} mean_prob={r.mean_prob:.4f}"
        for r in by_rate
    )
    lines.append("\\n## Balanced (close_rate * mean_prob)\\n")
    lines.extend(
        f"- {r.run_id}: score={r.balanced_score:.6f} close_rate={r.close_rate:.4f} mean_prob={r.mean_prob:.4f} closes={r.exit_close_total}"
        for r in by_bal
    )
    out_path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")


def run_sweep(
    max_runs: int,
    dry_run: bool,
    mode: str,
    pilot_runs: int,
    zoom_top: int,
    zoom_per_top: int,
) -> Tuple[List[RunResult], Path]:
    grid_full = build_grid()

    def pick_pilot_indices(n: int, total: int) -> List[int]:
        if n <= 1:
            return [0]
        idxs = set()
        for i in range(n):
            idx = round(i * (total - 1) / max(n - 1, 1))
            idxs.add(int(idx))
        return sorted(idxs)

    selected: List[Tuple[float, int, int, float]] = []
    if mode == "grid":
        selected = grid_full[:max_runs]
    else:
        pilot_idx = pick_pilot_indices(min(pilot_runs, max_runs), len(grid_full))
        pilot = [grid_full[i] for i in pilot_idx]
        planned = set(pilot)
        selected.extend(pilot)
        remaining_slots = max_runs - len(selected)
        if remaining_slots > 0 and not dry_run:
            # Pilot runs first, then derive zoom set after pilot results are known.
            pass  # placeholder, zoom handled after pilot execution below
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("/home/andre2/GX1_DATA/reports/exit_sweeps") / f"SWEEP_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"MODE={mode}")
        for thr, k, arb_hyst, min_pnl in selected:
            print(f"DRY-RUN would run thr={thr} k={k} arb_hyst={arb_hyst} min_pnl={min_pnl}")
        return [], out_dir

    results: List[RunResult] = []

    csv_path = out_dir / "sweep_results.csv"
    md_path = out_dir / "top10.md"

    def write_outputs():
        if results:
            write_csv(csv_path, results)
            write_markdown(md_path, results)

    def run_combo(thr: float, k: int, arb_hyst: int, min_pnl: float, idx: int, total: int) -> None:
        start_ts = time.time()
        print(
            f"[SWEEP] START {idx}/{total} thr={thr} k={k} arb_hyst={arb_hyst} min_pnl={min_pnl}",
            flush=True,
        )
        run_id, exit_code, combined = run_truth(thr, k, arb_hyst, min_pnl)
        run_dir = GX1_DATA / "reports" / "truth_e2e_sanity" / run_id
        log_tail = "\n".join(combined.strip().splitlines()[-30:])
        if exit_code == 0 and not run_dir.exists():
            exit_code = 1
            combined += f"\n[MISSING_RUN_DIR] {run_dir} not found"
            log_tail = "\n".join(combined.strip().splitlines()[-30:])
            fail_reason = "RUN_DIR_MISSING"
        else:
            fail_reason = classify_fail_reason(combined) if exit_code != 0 else ""
        if exit_code != 0:
            print(f"Run {run_id} failed exit_code={exit_code}", flush=True)
            results.append(
                RunResult(
                    status="FAIL",
                    run_id=run_id,
                    exit_code=exit_code,
                    fail_reason=fail_reason,
                    log_tail=log_tail,
                    thr=thr,
                    k=k,
                    arb_hyst=arb_hyst,
                    min_pnl=min_pnl,
                )
            )
            elapsed = int(time.time() - start_ts)
            print(
                f"[SWEEP] DONE {idx}/{total} run_id={run_id} status=FAIL reason={fail_reason} elapsed={elapsed}s",
                flush=True,
            )
            write_outputs()
            return
        metrics = load_footer_metrics(run_dir)
        exits_path = run_dir / "replay" / "chunk_0" / "logs" / "exits" / f"exits_{run_id}.jsonl"
        (
            n_probs,
            ge_thr,
            min_prob,
            mean_prob,
            max_prob,
            mean_pnl_eval,
            mean_bars_eval,
        ) = parse_exits_jsonl(exits_path, thr)
        res = RunResult(
            status="OK",
            run_id=run_id,
            exit_code=exit_code,
            thr=thr,
            k=k,
            arb_hyst=arb_hyst,
            min_pnl=min_pnl,
            exit_eval_long=metrics["exit_eval_long"],
            exit_eval_short=metrics["exit_eval_short"],
            exit_close_long=metrics["exit_close_long"],
            exit_close_short=metrics["exit_close_short"],
            entry_accept_long=metrics["entry_accept_long"],
            entry_accept_short=metrics["entry_accept_short"],
            n_probs=n_probs,
            ge_thr=ge_thr,
            min_prob=min_prob,
            mean_prob=mean_prob,
            max_prob=max_prob,
            mean_pnl_bps_eval=mean_pnl_eval,
            mean_bars_held_eval=mean_bars_eval,
        )
        results.append(res)
        elapsed = int(time.time() - start_ts)
        print(
            f"[SWEEP] DONE {idx}/{total} run_id={run_id} status=OK closes={res.exit_close_total} "
            f"evals={res.exit_eval_total} elapsed={elapsed}s",
            flush=True,
        )
        write_outputs()

    total_planned = len(selected)
    # Run pilot (or all if grid)
    for idx, combo in enumerate(selected, start=1):
        run_combo(*combo, idx=idx, total=max(len(selected), 1))

    if mode == "adaptive":
        ok_runs = [r for r in results if r.status == "OK"]
        if ok_runs:
            ok_sorted = sorted(
                ok_runs,
                key=lambda r: (-r.exit_close_total, -r.close_rate, -r.mean_prob),
            )
            zoom_candidates: List[Tuple[float, int, int, float]] = []
            planned_set = set(selected)

            def clamp_thr(x: float) -> float:
                return max(0.0025, min(0.10, x))

            def clamp_k(x: int) -> int:
                return max(1, min(4, x))

            for top in ok_sorted[: zoom_top]:
                thr_opts = {clamp_thr(top.thr / 2.0), clamp_thr(top.thr), clamp_thr(top.thr * 2.0)}
                k_opts = {clamp_k(top.k - 1), clamp_k(top.k), clamp_k(top.k + 1)}
                arb_opts = {1, 2}
                min_pnl_opts = {-999.0, -10.0, 0.0}
                neighbors: List[Tuple[float, int, int, float]] = []
                for t in sorted(thr_opts):
                    for kk in sorted(k_opts):
                        for ah in sorted(arb_opts):
                            for mp in sorted(min_pnl_opts):
                                if (t, kk, ah, mp) in planned_set:
                                    continue
                                neighbors.append((t, kk, ah, mp))
                for nb in neighbors:
                    if len(zoom_candidates) >= zoom_per_top:
                        break
                    zoom_candidates.append(nb)
                    planned_set.add(nb)
                if len(zoom_candidates) >= zoom_per_top * zoom_top:
                    break
            remaining_slots = max_runs - len(selected)
            zoom_selected = zoom_candidates[:remaining_slots]
            total_planned = len(selected) + len(zoom_selected)
            start_idx = len(selected) + 1
            for offset, combo in enumerate(zoom_selected, start=start_idx):
                run_combo(*combo, idx=offset, total=total_planned)

    if results:
        write_outputs()
    return results, out_dir


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep EXIT env overrides over TRUTH e2e.")
    parser.add_argument("--max-runs", type=int, default=72, help="Max runs from the grid (default 72).")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing.")
    parser.add_argument(
        "--mode",
        choices=["grid", "adaptive"],
        default="grid",
        help="grid = full grid; adaptive = pilot then zoom.",
    )
    parser.add_argument("--pilot-runs", type=int, default=12, help="Adaptive: number of pilot runs.")
    parser.add_argument("--zoom-top", type=int, default=3, help="Adaptive: top configs to zoom.")
    parser.add_argument("--zoom-per-top", type=int, default=8, help="Adaptive: zoom samples per top config.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    compile_repo()
    results, out_dir = run_sweep(
        max_runs=args.max_runs,
        dry_run=args.dry_run,
        mode=args.mode,
        pilot_runs=args.pilot_runs,
        zoom_top=args.zoom_top,
        zoom_per_top=args.zoom_per_top,
    )
    if args.dry_run:
        print("Dry-run complete.")
        return
    if results:
        print(f"Completed {len(results)} runs. Output dir: {out_dir}")
    else:
        print(f"No runs executed. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
