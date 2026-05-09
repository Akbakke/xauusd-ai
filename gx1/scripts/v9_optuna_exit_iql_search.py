"""
V9 Optuna parameter search for Exit-IQL.

Searches over (k_primary, exit_reward_multiplier) on a subsampled per-bar M1
dataset. Returns composite objective: phase7_pnl_bps + activity_bonus
(reward higher EXIT_IQL_SIGNAL fraction since validation showed those trades
average +158 bps mean).

Validation findings (2026-05-07):
- EXIT_IQL_SIGNAL gives +158 bps mean (5x forced terminal +32 bps)
- Loss rate is 3.3% for exit-iql vs 24.3% for forced
- Even worst-quartile exit-iql trade is +109 bps (better than forced median +26)
- So pushing EXIT_IQL_SIGNAL fraction from 3.8% → 8-10% should add big PnL

Usage (parallel):
  for i in 1 2 3 4; do
    python -m gx1.scripts.v9_optuna_exit_iql_search \\
        --study-name v9_exit_search --n-trials 20 --sample-n-rows 50000 \\
        --seed $((20260507+i)) > /tmp/v9_exit_optuna_w$i.log 2>&1 &
  done
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import optuna

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/andre2/venvs/gx1/bin/python3"
DEFAULT_PER_BAR_DIR = "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/EXIT_IQL_PER_BAR_DATASET_V2_M1"
DEFAULT_FORWARD_OUTCOME = "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/CANDIDATE_FORWARD_OUTCOME_RUN/v1_full"
DEFAULT_ENTRY_IQL_ROOT = "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/BUILD_ENTRY_IQL_V2_20260506T132525Z_LOCK"
DEFAULT_STORAGE = "sqlite:////tmp/v9_exit_optuna.db"

# K_HORIZONS in M1 bars: [5, 20, 60, 120, 240, 480]
ALLOWED_K_PRIMARY = [60, 120, 240, 480]


def _run_exit_iql(
    *, out_root: Path, sample_n_rows: int, k_primary: int,
    exit_reward_multiplier: float, per_bar_dir: str,
) -> dict:
    cmd = [
        PYTHON, "-u", "-m", "gx1.scripts.materialize_build_exit_iql_v3_m1",
        "--per-bar-dir", per_bar_dir,
        "--budget", "fast",
        "--variants", "R_NET_REAL",
        "--sample-n-rows", str(sample_n_rows),
        "--out-root", str(out_root),
        "--k-primary", str(k_primary),
        "--exit-reward-multiplier", str(exit_reward_multiplier),
    ]
    env = os.environ.copy()
    env["GX1_SIGNAL_BRIDGE_VERSION"] = "3"
    env["PYTHONPATH"] = str(REPO_ROOT)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        raise RuntimeError(f"exit_iql failed rc={p.returncode}\nstderr={p.stderr[-2000:]}")
    return {"out_root": str(out_root)}


def _run_phase7(*, entry_iql_root: str, exit_iql_root: Path, forward_outcome_dir: str,
                out_root: Path) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-u", "-m", "gx1.scripts.materialize_joint_entry_exit_iql_validation_gate_v2",
        "--forward-outcome-dir", forward_outcome_dir,
        "--entry-iql-root", entry_iql_root,
        "--exit-iql-root", str(exit_iql_root),
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
        raise RuntimeError(f"phase7 did not write summary_v1.json:\n{p.stdout[-2000:]}")
    with open(summary_path) as f:
        return json.load(f)


def _objective_score(phase7_summary: dict) -> tuple[float, dict]:
    """Composite: PnL + activity bonus (reward higher EXIT_IQL_SIGNAL fraction).

    Validation showed EXIT_IQL_SIGNAL trades give +158 bps mean. Pushing
    activity from 3.8% → 8% should add ~10 bps/candidate. We reward this
    explicitly so the optimizer doesn't degenerate to "fire less, take fewer
    risks" mode.
    """
    metrics = phase7_summary.get("headline_v1") or phase7_summary
    pnl = float(metrics.get("joint_mean_pnl_bps_v1", 0.0))
    active_frac = float(metrics.get("exit_iql_active_fraction_v1", 0.0))

    # Bonus: reward activity fraction up to 15% (validation showed Exit-IQL is
    # genuinely good, more fires = more PnL up to a point)
    activity_target = 0.10
    activity_bonus = 0.0
    if active_frac < activity_target:
        # Penalize being below 8% (current 3.8% baseline)
        activity_bonus = -max(0.0, 0.08 - active_frac) * 200.0
    elif active_frac > 0.20:
        # Penalize over-firing (would lose forced-terminal winners)
        activity_bonus = -(active_frac - 0.20) * 100.0

    score = pnl + activity_bonus
    diag = {
        "pnl_bps": pnl, "exit_iql_active_frac": active_frac,
        "activity_bonus": activity_bonus, "final_score": score,
    }
    return score, diag


def make_objective(args: argparse.Namespace):
    def objective(trial: optuna.Trial) -> float:
        k_primary = trial.suggest_categorical("k_primary", ALLOWED_K_PRIMARY)
        exit_mult = trial.suggest_float("exit_reward_multiplier", 0.8, 2.0)

        with tempfile.TemporaryDirectory(prefix="v9_exit_optuna_trial_", dir="/tmp") as td:
            exit_root = Path(td) / f"trial_{trial.number}_exit"
            phase7_root = Path(td) / f"trial_{trial.number}_phase7"
            exit_root.mkdir(parents=True, exist_ok=True)
            try:
                _run_exit_iql(
                    out_root=exit_root,
                    sample_n_rows=args.sample_n_rows,
                    k_primary=k_primary,
                    exit_reward_multiplier=exit_mult,
                    per_bar_dir=args.per_bar_dir,
                )
                phase7 = _run_phase7(
                    entry_iql_root=args.entry_iql_root,
                    exit_iql_root=exit_root,
                    forward_outcome_dir=args.forward_outcome_dir,
                    out_root=phase7_root,
                )
                score, diag = _objective_score(phase7)
                for k, v in diag.items():
                    trial.set_user_attr(k, v)
                print(f"[exit-trial {trial.number}] k_primary={k_primary} "
                      f"exit_mult={exit_mult:.2f} score={score:.2f} diag={diag}",
                      flush=True)
                return score
            except Exception as e:
                print(f"[exit-trial {trial.number}] FAILED: {e}", flush=True)
                raise

    return objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", default="v9_exit_search")
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--sample-n-rows", type=int, default=50000,
                        help="M1 bars sample size (per-bar dataset is much larger than entry)")
    parser.add_argument("--per-bar-dir", default=DEFAULT_PER_BAR_DIR)
    parser.add_argument("--entry-iql-root", default=DEFAULT_ENTRY_IQL_ROOT)
    parser.add_argument("--forward-outcome-dir", default=DEFAULT_FORWARD_OUTCOME)
    parser.add_argument("--seed", type=int, default=20260507)
    args = parser.parse_args()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )
    print(f"[v9-exit-optuna] study='{args.study_name}' storage='{args.storage}' "
          f"existing trials={len(study.trials)}, target={args.n_trials}", flush=True)

    study.optimize(make_objective(args), n_trials=args.n_trials, gc_after_trial=True)

    print("\n=== BEST EXIT TRIAL ===", flush=True)
    print(f"params={study.best_params}", flush=True)
    print(f"score={study.best_value:.2f}", flush=True)
    print(f"user_attrs={study.best_trial.user_attrs}", flush=True)


if __name__ == "__main__":
    main()
