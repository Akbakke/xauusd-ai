"""
V9 Optuna parameter search for Entry-IQL.

Searches over (skip_weight_multiplier, skip_oversample_factor, balance_actions)
on a subsampled forward-outcome dataset for fast iteration. Returns composite
objective: phase7_pnl_bps - degeneracy_penalty (penalize SKIP < 3% or
TAKE_LONG/TAKE_SHORT < 10%).

Parallel: launch N instances pointing at the same SQLite study.

Usage:
  # First instance (creates study):
  python -m gx1.scripts.v9_optuna_entry_iql_search \\
      --study-name v9_entry_search_v1 --n-trials 30 --sample-n-rows 10000

  # Additional parallel workers (join existing study):
  python -m gx1.scripts.v9_optuna_entry_iql_search \\
      --study-name v9_entry_search_v1 --n-trials 30 --sample-n-rows 10000

After done, query best trial:
  python -c "import optuna; s=optuna.load_study(study_name='v9_entry_search_v1', \\
    storage='sqlite:////tmp/v9_optuna.db'); print(s.best_trial.params, s.best_value)"
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import optuna

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/andre2/venvs/gx1/bin/python3"
DEFAULT_FORWARD_OUTCOME = "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/CANDIDATE_FORWARD_OUTCOME_RUN/v1_full"
DEFAULT_EXIT_IQL_ROOT = "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/EXIT_IQL_V4_R_NET_REAL_20260505T165807Z"
DEFAULT_STORAGE = "sqlite:////tmp/v9_optuna.db"


def _run_entry_iql(
    *, out_root: Path, sample_n_rows: int, balance_actions: bool,
    skip_weight_multiplier: float, skip_oversample_factor: int,
    forward_outcome_dir: str,
) -> dict:
    cmd = [
        PYTHON, "-u", "-m", "gx1.scripts.materialize_build_entry_iql_v2",
        "--budget", "fast",
        "--sample-n-rows", str(sample_n_rows),
        "--forward-outcome-dir", forward_outcome_dir,
        "--out-root", str(out_root),
        "--skip-weight-multiplier", str(skip_weight_multiplier),
        "--skip-oversample-factor", str(skip_oversample_factor),
    ]
    if balance_actions:
        cmd.append("--balance-actions")
    env = os.environ.copy()
    env["GX1_SIGNAL_BRIDGE_VERSION"] = "3"
    env["PYTHONPATH"] = str(REPO_ROOT)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        raise RuntimeError(f"entry_iql failed rc={p.returncode}\nstderr={p.stderr[-2000:]}")
    return {"stdout": p.stdout, "out_root": str(out_root)}


def _run_phase7(*, entry_iql_root: Path, exit_iql_root: str, forward_outcome_dir: str,
                out_root: Path) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-u", "-m", "gx1.scripts.materialize_joint_entry_exit_iql_validation_gate_v2",
        "--forward-outcome-dir", forward_outcome_dir,
        "--entry-iql-root", str(entry_iql_root),
        "--exit-iql-root", exit_iql_root,
        "--out-root", str(out_root),
    ]
    env = os.environ.copy()
    env["GX1_SIGNAL_BRIDGE_VERSION"] = "3"
    env["PYTHONPATH"] = str(REPO_ROOT)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise RuntimeError(f"phase7 failed rc={p.returncode}\nstderr={p.stderr[-2000:]}")

    summary_path = out_root / "summary_v1.json"
    if not summary_path.exists():
        raise RuntimeError(f"phase7 did not write summary_v1.json at {summary_path}\n"
                           f"stdout tail:\n{p.stdout[-2000:]}")
    with open(summary_path) as f:
        return json.load(f)


def _objective_score(phase7_summary: dict) -> tuple[float, dict]:
    """Composite score: joint_mean_pnl_bps minus degeneracy penalty.

    Phase 7 summary structure: top-level has 'headline_v1' sub-dict containing
    joint_mean_pnl_bps_v1, entry_action_distribution_v1, exit_iql_active_fraction_v1.
    """
    metrics = phase7_summary.get("headline_v1") or phase7_summary
    pnl = float(metrics.get("joint_mean_pnl_bps_v1", 0.0))
    dist = metrics.get("entry_action_distribution_v1", {})
    skip = float(dist.get("SKIP", 0.0))
    take_long = float(dist.get("TAKE_LONG_NOW", 0.0))
    take_short = float(dist.get("TAKE_SHORT_NOW", 0.0))

    # Penalty: SKIP < 3% OR any take < 10% (degenerate distribution)
    skip_under = max(0.0, 0.03 - skip)            # 0..0.03
    take_under = max(0.0, 0.10 - min(take_long, take_short))  # 0..0.10
    penalty = (skip_under * 200.0) + (take_under * 100.0)

    score = pnl - penalty
    diag = {
        "pnl_bps": pnl, "skip": skip, "take_long": take_long, "take_short": take_short,
        "skip_under_3pct": skip_under, "take_under_10pct": take_under, "penalty": penalty,
        "final_score": score,
    }
    return score, diag


def make_objective(args: argparse.Namespace):
    def objective(trial: optuna.Trial) -> float:
        balance_actions = trial.suggest_categorical("balance_actions", [True, False])
        skip_weight = trial.suggest_float("skip_weight_multiplier", 1.0, 50.0, log=True) if balance_actions else 1.0
        oversample = trial.suggest_int("skip_oversample_factor", 1, 30)

        with tempfile.TemporaryDirectory(prefix="v9_optuna_trial_", dir="/tmp") as td:
            entry_root = Path(td) / f"trial_{trial.number}_entry"
            phase7_root = Path(td) / f"trial_{trial.number}_phase7"
            entry_root.mkdir(parents=True, exist_ok=True)
            try:
                _run_entry_iql(
                    out_root=entry_root,
                    sample_n_rows=args.sample_n_rows,
                    balance_actions=balance_actions,
                    skip_weight_multiplier=skip_weight,
                    skip_oversample_factor=oversample,
                    forward_outcome_dir=args.forward_outcome_dir,
                )
                phase7 = _run_phase7(
                    entry_iql_root=entry_root,
                    exit_iql_root=args.exit_iql_root,
                    forward_outcome_dir=args.forward_outcome_dir,
                    out_root=phase7_root,
                )
                score, diag = _objective_score(phase7)
                for k, v in diag.items():
                    trial.set_user_attr(k, v)
                print(f"[trial {trial.number}] params={trial.params} score={score:.2f} diag={diag}",
                      flush=True)
                return score
            except Exception as e:
                print(f"[trial {trial.number}] FAILED: {e}", flush=True)
                raise

    return objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", default="v9_entry_search_v1")
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--sample-n-rows", type=int, default=10000)
    parser.add_argument("--forward-outcome-dir", default=DEFAULT_FORWARD_OUTCOME)
    parser.add_argument("--exit-iql-root", default=DEFAULT_EXIT_IQL_ROOT)
    parser.add_argument("--seed", type=int, default=20260506)
    args = parser.parse_args()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )
    print(f"[v9-optuna] study='{args.study_name}' storage='{args.storage}' "
          f"existing trials={len(study.trials)}, target={args.n_trials}", flush=True)

    study.optimize(make_objective(args), n_trials=args.n_trials, gc_after_trial=True)

    print("\n=== BEST TRIAL ===", flush=True)
    print(f"params={study.best_params}", flush=True)
    print(f"score={study.best_value:.2f}", flush=True)
    print(f"user_attrs={study.best_trial.user_attrs}", flush=True)


if __name__ == "__main__":
    main()
