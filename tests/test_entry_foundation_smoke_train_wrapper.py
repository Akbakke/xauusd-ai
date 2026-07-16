import subprocess
from pathlib import Path


REPO = Path("/home/andre2/src/GX1_ENGINE")
WRAPPER = REPO / "scripts/run_entry_foundation_seq146_smoke_train.sh"
CONTROL = REPO / "scripts/entry_next_edge_control.sh"
EXPECTED_AUX_FLAGS = (
    "--enable-tf-agreement-head",
    "--enable-path-quality-variance-head",
    "--enable-position-size-head",
    "--enable-dip-head",
    "--enable-forecast-head",
    "--enable-timing-head",
    "--enable-tail-risk-head",
    "--enable-vol-forecast-head",
    "--enable-mtf-direction-head",
)


def _run_wrapper(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(WRAPPER), *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def test_smoke_train_dry_run_prints_post_smoke_audit_command() -> None:
    result = _run_wrapper("--vedtak", "PYTEST_DRY_RUN", "--dry-run")

    assert "Smoke train command:" in result.stdout
    assert "Real-train preflight command: scripts/entry_next_edge_control.sh verify --quiet" in result.stdout
    assert "Real-train preflight command: scripts/entry_next_edge_control.sh foundation-guardrails --quiet" in result.stdout
    assert "Real-train preflight command: scripts/entry_next_edge_control.sh train-readiness --quiet" in result.stdout
    assert "Pre-train run manifest path:" in result.stdout
    assert "Smoke resource cap: mem=22G swap=2G runner=scripts/gx1_capped_run.sh num_workers=0" in result.stdout
    assert "Capped smoke train command:" in result.stdout
    assert "scripts/gx1_capped_run.sh --mem 22G --swap 2G --" in result.stdout
    assert "ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST" in result.stdout
    assert "GX1_ENTRY_ALLOW_TRAIN_ENV_OVERRIDES=1" in result.stdout
    assert "ENTRY_AUX_BAD_PATH_WEIGHT=1.00" in result.stdout
    assert "ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT=2.00" in result.stdout
    assert "ENTRY_BAD_PATH_QUALITY_RANK_MARGIN=0.25" in result.stdout
    assert "ENTRY_BAD_PATH_QUALITY_RANK_QUANTILE=0.25" in result.stdout
    assert "ENTRY_BAD_PATH_PROB_PENALTY=0.24" in result.stdout
    assert "ENTRY_PRED_BALANCE_ALPHA=0.05" in result.stdout
    assert "ENTRY_PRED_BALANCE_TARGET=label" in result.stdout
    assert "ENTRY_PRED_BALANCE_CLASS_WEIGHTS=1.0\\,1.0\\,1.0" in result.stdout
    assert "ENTRY_DIRECTION_CE_SCALE=1.30" in result.stdout
    assert "GX1_V10_CKPT_MONITOR=dir_acc" in result.stdout
    assert "ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT=0.0" in result.stdout
    assert "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL=0.0" in result.stdout
    assert "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE=0.0" in result.stdout
    assert "ENTRY_CKPT_DIRECTION_SLICE_GUARD=0" in result.stdout
    assert "ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT=0.0" in result.stdout
    assert "ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION=0.0" in result.stdout
    assert "ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR=0.0" in result.stdout
    assert "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE=1.0" in result.stdout
    assert "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT=0.0" in result.stdout
    assert "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE=0.02" in result.stdout
    assert "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT=0.0" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_RECALL_LOSS_WEIGHT=0.0" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT=0.0" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS=8" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT=0.0" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_TRUE_MARGIN=0.10" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS=8" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT=0.0" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN=0.02" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS=8" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT=0.0" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE=0.02" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS=8" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION=mean" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER=0" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS=8" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE=0" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS=6" in result.stdout
    assert "ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT=0.0" in result.stdout
    assert "ENTRY_DIRECTION_VS_FLAT_MARGIN=0.0" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT=0.0" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS=15.0" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN=0.10" in result.stdout
    assert "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT=0.0" in result.stdout
    assert "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS=15.0" in result.stdout
    assert "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN=0.10" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT=0.0" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS=15.0" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS=0.0" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH=0.50" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN=0.10" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT=0.0" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS=15.0" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS=0.0" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH=0.50" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP=4.0" in result.stdout
    assert "ENTRY_DIRECTION_HIERARCHICAL_COMPOSITION=0" in result.stdout
    assert "ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP=0.0" in result.stdout
    assert "ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL=0" in result.stdout
    assert "ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE=0" in result.stdout
    assert "ENTRY_HIER_CTX_PRIOR_ADAPTER=0" in result.stdout
    assert "ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE=0.0" in result.stdout
    assert "ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT=0.0" in result.stdout
    assert "ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS=8" in result.stdout
    assert "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION=0.50" in result.stdout
    assert "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR=0.10" in result.stdout
    assert "ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN=0.10" in result.stdout
    assert "ENTRY_HIER_LEGACY_CE_MULT=0.35" in result.stdout
    assert "ENTRY_HIER_SIDE_VALIDITY_WEIGHT=0.0" in result.stdout
    assert "ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS=10.0" in result.stdout
    assert "ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP=20.0" in result.stdout
    assert "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT=0.0" in result.stdout
    assert "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE=0.02" in result.stdout
    assert "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT=0.0" in result.stdout
    assert "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE=0.02" in result.stdout
    assert "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS=8" in result.stdout
    assert "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT=0.0" in result.stdout
    assert "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN=0.02" in result.stdout
    assert "ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT=0.0" in result.stdout
    assert "ENTRY_HIER_FLAT_LOGIT_MARGIN=0.10" in result.stdout
    assert "ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT=0.0" in result.stdout
    assert "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS=8" in result.stdout
    assert "ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT=0.0" in result.stdout
    assert "ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT=0.0" in result.stdout
    assert "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS=8" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_CE_WEIGHT=0.0" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT=0.0" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT=0.0" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN=0.02" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_MIN_ROWS=8" in result.stdout
    assert "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT=0.0" in result.stdout
    assert "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE=0.02" in result.stdout
    assert "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT=0.0" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE=0.02" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS=8" in result.stdout
    assert "ENTRY_HIER_POCKET_ABSTAIN_WEIGHT=0.0" in result.stdout
    assert "ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT=0.0" in result.stdout
    assert "ENTRY_HIER_POCKET_UTILITY_MARGIN_BPS=10.0" in result.stdout
    assert "ENTRY_TRENDLINE_RAIL_AUX_WEIGHT=0.0" in result.stdout
    assert "ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT=0.0" in result.stdout
    assert "ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT=0.0" in result.stdout
    assert "ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT=0.0" in result.stdout
    assert "ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT=0.0" in result.stdout
    assert "ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT=0.0" in result.stdout
    assert "ENTRY_SYMMETRIC_NEGATIVES=1" in result.stdout
    assert "ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT=0.05" in result.stdout
    assert "ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT=0.25" in result.stdout
    assert "ENTRY_SPECIALIST_GATE_MIN_MEAN=0.02" in result.stdout
    assert "--enable-specialist-fusion" in result.stdout
    for flag in EXPECTED_AUX_FLAGS:
        assert flag in result.stdout
    assert "--enable-hold-horizon-head" not in result.stdout
    assert "Post-smoke audit command:" in result.stdout
    assert "audit-smoke-bundle" in result.stdout
    assert "--bundle-dir" in result.stdout
    assert "--dataset-dir" in result.stdout
    assert "--out-dir" in result.stdout
    assert "entry_foundation_smoke_bundle_audit_20260628_v1" in result.stdout
    assert "--require-head-contract" in result.stdout
    assert "--pretrain-manifest-json" in result.stdout
    assert "--require-edge" in result.stdout


def test_smart_smoke_dry_run_uses_xau_direction_repair_recipe() -> None:
    result = _run_wrapper(
        "--smart-seq520",
        "--vedtak",
        "SMART_SEQ520_XAU_DIRECTION_REPAIR_PYTEST",
        "--dry-run",
    )

    assert "Smoke train command:" in result.stdout
    assert "smart_seq520_candidate" in result.stdout
    assert "Real-train preflight command: scripts/entry_next_edge_control.sh smart-smoke-readiness --quiet" in result.stdout
    assert (
        "Real-train preflight command: scripts/entry_next_edge_control.sh smart-trainability-readiness --quiet"
        in result.stdout
    )
    assert "Real-train preflight command: scripts/entry_next_edge_control.sh verify --quiet" not in result.stdout
    assert (
        "Real-train preflight command: scripts/entry_next_edge_control.sh foundation-guardrails --quiet"
        not in result.stdout
    )
    assert (
        "Real-train preflight command: scripts/entry_next_edge_control.sh train-readiness --quiet"
        not in result.stdout
    )
    assert "ENTRY_PRED_BALANCE_ALPHA=0.50" in result.stdout
    assert "ENTRY_PRED_BALANCE_CLASS_WEIGHTS=1.0\\,1.0\\,4.0" in result.stdout
    assert "ENTRY_DIRECTION_CE_SCALE=4.00" in result.stdout
    assert "ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT=0.50" in result.stdout
    assert "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL=0.35" in result.stdout
    assert "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE=0.05" in result.stdout
    assert "ENTRY_CKPT_DIRECTION_SLICE_GUARD=1" in result.stdout
    assert "ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT=12.00" in result.stdout
    assert "ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION=0.50" in result.stdout
    assert "ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR=0.05" in result.stdout
    assert "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE=0.05" in result.stdout
    assert "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT=8.00" in result.stdout
    assert "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE=0.02" in result.stdout
    assert "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT=8.00" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION=0.50" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR=0.05" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_RECALL_LOSS_WEIGHT=4.00" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT=2.00" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS=8" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT=2.00" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_TRUE_MARGIN=0.10" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS=8" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT=4.00" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN=0.02" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS=8" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT=3.00" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE=0.02" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS=8" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION=mean_max" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER=1" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS=8" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE=3" in result.stdout
    assert "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS=6" in result.stdout
    assert "ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT=4.00" in result.stdout
    assert "ENTRY_DIRECTION_VS_FLAT_MARGIN=0.10" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT=4.00" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS=15.0" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN=0.10" in result.stdout
    assert "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT=6.00" in result.stdout
    assert "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS=15.0" in result.stdout
    assert "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN=0.10" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT=8.00" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS=15.0" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS=0.0" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH=0.50" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN=0.10" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT=8.00" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS=15.0" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS=0.0" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH=0.50" in result.stdout
    assert "ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP=4.0" in result.stdout
    assert "ENTRY_DIRECTION_HIERARCHICAL_COMPOSITION=1" in result.stdout
    assert "ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP=0.18" in result.stdout
    assert "ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL=1" in result.stdout
    assert "ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE=1" in result.stdout
    assert "ENTRY_HIER_CTX_PRIOR_ADAPTER=1" in result.stdout
    assert "ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE=0.50" in result.stdout
    assert "ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT=8.00" in result.stdout
    assert "ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS=8" in result.stdout
    assert "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION=0.50" in result.stdout
    assert "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR=0.10" in result.stdout
    assert "ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN=0.10" in result.stdout
    assert "ENTRY_HIER_LEGACY_CE_MULT=1.00" in result.stdout
    assert "ENTRY_BAD_PATH_PROB_PENALTY=0.0" in result.stdout
    assert "ENTRY_HIER_TRADE_WEIGHT=2.00" in result.stdout
    assert "ENTRY_HIER_SIDE_WEIGHT=1.75" in result.stdout
    assert "ENTRY_HIER_UTILITY_WEIGHT=1.00" in result.stdout
    assert "--lr 3e-4" in result.stdout
    assert "--grad-clip-norm 1.0" in result.stdout
    assert "--weight-decay 1e-5" in result.stdout
    assert "ENTRY_HIER_BAD_PATH_WEIGHT=1.25" in result.stdout
    assert "ENTRY_HIER_SIDE_VALIDITY_WEIGHT=1.50" in result.stdout
    assert "ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS=15.0" in result.stdout
    assert "ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP=8.0" in result.stdout
    assert "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT=4.00" in result.stdout
    assert "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE=0.02" in result.stdout
    assert "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT=4.00" in result.stdout
    assert "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE=0.02" in result.stdout
    assert "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS=8" in result.stdout
    assert "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT=4.00" in result.stdout
    assert "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN=0.02" in result.stdout
    assert "ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT=8.00" in result.stdout
    assert "ENTRY_HIER_FLAT_LOGIT_MARGIN=0.10" in result.stdout
    assert "ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT=8.00" in result.stdout
    assert "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS=8" in result.stdout
    assert "ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT=4.00" in result.stdout
    assert "ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT=4.00" in result.stdout
    assert "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS=8" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_CE_WEIGHT=4.00" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT=3.00" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT=4.00" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN=0.02" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_MIN_ROWS=8" in result.stdout
    assert "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT=4.00" in result.stdout
    assert "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE=0.02" in result.stdout
    assert "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT=4.00" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE=0.02" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE=0.10" in result.stdout
    assert "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS=8" in result.stdout
    assert "ENTRY_HIER_POCKET_ABSTAIN_WEIGHT=5.00" in result.stdout
    assert "ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT=3.00" in result.stdout
    assert "ENTRY_HIER_POCKET_UTILITY_MARGIN_BPS=30.0" in result.stdout
    assert "ENTRY_TRENDLINE_RAIL_AUX_WEIGHT=1.00" in result.stdout
    assert "ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT=1.50" in result.stdout
    assert "ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT=1.50" in result.stdout
    assert "ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT=1.75" in result.stdout
    assert "ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT=5.00" in result.stdout
    assert "ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT=4.00" in result.stdout
    assert "ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT=3.00" in result.stdout
    assert "ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT=5.00" in result.stdout
    assert "ENTRY_TRENDLINE_RAIL_MARGIN=1.00" in result.stdout
    assert "ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_BPS=30.0" in result.stdout
    assert "ENTRY_FLAT_CLASS_WEIGHT_FLOOR=1.00" in result.stdout
    assert "--enable-xau-direction-repair-heads" in result.stdout
    assert "--anchor-gate-init 0.0" in result.stdout


def test_smoke_train_dry_run_can_explicitly_skip_post_smoke_audit() -> None:
    result = _run_wrapper(
        "--vedtak",
        "PYTEST_DRY_RUN",
        "--dry-run",
        "--no-require-edge-audit",
        "--skip-smoke-audit",
    )

    assert "Smoke train command:" in result.stdout
    assert "Post-smoke audit command: skipped by --skip-smoke-audit" in result.stdout


def test_smoke_train_rejects_require_edge_without_audit() -> None:
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--vedtak",
            "PYTEST_DRY_RUN",
            "--dry-run",
            "--skip-smoke-audit",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "--require-edge-audit cannot be combined with --skip-smoke-audit" in result.stderr


def test_seq215_manifest_requires_seq215_vedtak() -> None:
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--challenger-seq215",
            "--vedtak",
            "ENTRY_FOUNDATION_SMOKE_TRAIN_20260629_SEQ146_V1",
            "--manifest-only",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "requires an explicit SEQ215 vedtak id" in result.stderr


def test_seq215_smoke_dry_run_requires_seq215_vedtak() -> None:
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--challenger-seq215",
            "--vedtak",
            "ENTRY_FOUNDATION_SMOKE_TRAIN_20260630_SEQ146_V1",
            "--dry-run",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "requires an explicit SEQ215 vedtak id" in result.stderr
    assert "Smoke train command:" not in result.stdout


def test_smoke_train_wrapper_enforces_train_readiness_for_real_train() -> None:
    text = WRAPPER.read_text(encoding="utf-8")

    assert "entry_next_edge_control.sh verify --quiet" in text
    assert "entry_next_edge_control.sh foundation-guardrails --quiet" in text
    assert "entry_next_edge_control.sh train-readiness --quiet" in text
    assert "ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST" in text
    assert "trainer_started_by_manifest_writer" in text
    assert "smoke_recipe_env" in text
    assert "command_env_value" in text
    assert "ENTRY_AUX_BAD_PATH_WEIGHT" in text
    assert "ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT" in text
    assert "ENTRY_BAD_PATH_QUALITY_RANK_MARGIN" in text
    assert "ENTRY_BAD_PATH_QUALITY_RANK_QUANTILE" in text
    assert "ENTRY_BAD_PATH_PROB_PENALTY" in text
    assert "ENTRY_PRED_BALANCE_ALPHA" in text
    assert "ENTRY_PRED_BALANCE_TARGET" in text
    assert "ENTRY_PRED_BALANCE_CLASS_WEIGHTS" in text
    assert "ENTRY_DIRECTION_CE_SCALE" in text
    assert "GX1_V10_CKPT_MONITOR" in text
    assert "ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT" in text
    assert "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL" in text
    assert "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE" in text
    assert "ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT" in text
    assert "ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION" in text
    assert "ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR" in text
    assert "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE" in text
    assert "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT" in text
    assert "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE" in text
    assert "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE" in text
    assert "ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT" in text
    assert "ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE" in text
    assert "ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS" in text
    assert "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT" in text
    assert "ENTRY_DIRECTION_SLICE_TRUE_MARGIN" in text
    assert "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE" in text
    assert "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS" in text
    assert "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT" in text
    assert "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN" in text
    assert "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE" in text
    assert "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS" in text
    assert "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT" in text
    assert "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE" in text
    assert "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE" in text
    assert "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS" in text
    assert "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION" in text
    assert "ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER" in text
    assert "ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS" in text
    assert "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE" in text
    assert "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS" in text
    assert "ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT" in text
    assert "ENTRY_DIRECTION_VS_FLAT_MARGIN" in text
    assert "ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT" in text
    assert "ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS" in text
    assert "ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN" in text
    assert "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT" in text
    assert "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS" in text
    assert "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN" in text
    assert "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT" in text
    assert "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS" in text
    assert "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS" in text
    assert "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH" in text
    assert "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN" in text
    assert "ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT" in text
    assert "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS" in text
    assert "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS" in text
    assert "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH" in text
    assert "ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP" in text
    assert "ENTRY_DIRECTION_HIERARCHICAL_COMPOSITION" in text
    assert "ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT" in text
    assert "ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE" in text
    assert "ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS" in text
    assert "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION" in text
    assert "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR" in text
    assert "ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN" in text
    assert "ENTRY_FOUNDATION_SMOKE_DIRECTION_UTILITY_MARGIN_WEIGHT" in text
    assert "ENTRY_FOUNDATION_SMOKE_DIRECTION_UTILITY_MIN_GAP_BPS" in text
    assert "ENTRY_FOUNDATION_SMOKE_DIRECTION_UTILITY_LOGIT_MARGIN" in text
    assert "ENTRY_FOUNDATION_SMOKE_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT" in text
    assert "ENTRY_FOUNDATION_SMOKE_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS" in text
    assert "ENTRY_FOUNDATION_SMOKE_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN" in text
    assert "ENTRY_FOUNDATION_SMOKE_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT" in text
    assert "ENTRY_FOUNDATION_SMOKE_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS" in text
    assert "ENTRY_FOUNDATION_SMOKE_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS" in text
    assert "ENTRY_FOUNDATION_SMOKE_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH" in text
    assert "ENTRY_FOUNDATION_SMOKE_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN" in text
    assert "ENTRY_FOUNDATION_SMOKE_HIER_COMPOSE_RESIDUAL_LOGIT_CAP" in text
    assert "ENTRY_FOUNDATION_SMOKE_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL" in text
    assert "ENTRY_FOUNDATION_SMOKE_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE" in text
    assert "ENTRY_FOUNDATION_SMOKE_HIER_CTX_PRIOR_ADAPTER" in text
    assert "ENTRY_FOUNDATION_SMOKE_HIER_CTX_PRIOR_ADAPTER_SCALE" in text
    assert "ENTRY_FOUNDATION_SMOKE_DIRECTION_FLAT_STARVATION_WEIGHT" in text
    assert "ENTRY_FOUNDATION_SMOKE_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE" in text
    assert "ENTRY_FOUNDATION_SMOKE_DIRECTION_FLAT_STARVATION_MIN_ROWS" in text
    assert "ENTRY_FOUNDATION_SMOKE_DIRECTION_FLAT_STARVATION_PRED_FRACTION" in text
    assert "ENTRY_FOUNDATION_SMOKE_DIRECTION_FLAT_STARVATION_PRED_FLOOR" in text
    assert "ENTRY_FOUNDATION_SMOKE_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN" in text
    assert "ENTRY_FOUNDATION_SMOKE_RESIDUAL_SCALE" in text
    assert "ENTRY_FOUNDATION_SMOKE_ANCHOR_EPS" in text
    assert 'ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP="$SMOKE_HIER_COMPOSE_RESIDUAL_LOGIT_CAP"' in text
    assert 'ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL="$SMOKE_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL"' in text
    assert (
        'ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE="$SMOKE_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE"'
        in text
    )
    assert 'ENTRY_HIER_CTX_PRIOR_ADAPTER="$SMOKE_HIER_CTX_PRIOR_ADAPTER"' in text
    assert 'ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE="$SMOKE_HIER_CTX_PRIOR_ADAPTER_SCALE"' in text
    assert 'ENTRY_RESIDUAL_SCALE="$SMOKE_RESIDUAL_SCALE"' in text
    assert 'ENTRY_ANCHOR_EPS="$SMOKE_ANCHOR_EPS"' in text
    assert "ENTRY_RESIDUAL_SCALE" in text
    assert "ENTRY_ANCHOR_EPS" in text
    assert "ENTRY_HIER_LEGACY_CE_MULT" in text
    assert "ENTRY_HIER_SIDE_VALIDITY_WEIGHT" in text
    assert "ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS" in text
    assert "ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP" in text
    assert "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT" in text
    assert "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT" in text
    assert "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT" in text
    assert "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN" in text
    assert "ENTRY_HIER_SLICE_SIDE_CE_WEIGHT" in text
    assert "ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT" in text
    assert "ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN" in text
    assert "ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT" in text
    assert "ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN" in text
    assert "ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE" in text
    assert "ENTRY_HIER_SLICE_SIDE_MIN_ROWS" in text
    assert "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT" in text
    assert "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT" in text
    assert "ENTRY_HIER_POCKET_ABSTAIN_WEIGHT" in text
    assert "ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT" in text
    assert "ENTRY_HIER_POCKET_UTILITY_MARGIN_BPS" in text
    assert "ENTRY_TRENDLINE_RAIL_AUX_WEIGHT" in text
    assert "ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT" in text
    assert "ENTRY_SYMMETRIC_NEGATIVES" in text
    assert "ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT" in text
    assert "ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT" in text
    assert "ENTRY_SPECIALIST_GATE_MIN_MEAN" in text
    assert "artifact_sha256" in text
    assert "artifact_provenance_decision" in text
    assert "artifact_fingerprints" in text
    assert "def artifact_fingerprint" in text
    assert "def run_artifact_fingerprints" in text
    assert 'if run_flavor != "foundation_seq146"' in text
    assert "FATAL: non-foundation manifest artifact fingerprint missing" in text
    assert "FATAL: non-foundation manifest artifact fingerprint hash mismatch" in text
    assert '"artifact_fingerprints": readiness_artifact_fingerprints' in text
    assert 'gate_decision(readiness, "artifact_provenance")' in text
    assert "--pretrain-manifest-json" in text
    assert "preflight_contracts" in text
    assert "feature_contract_summary" in text
    assert "foundation_objective_coverage_all_present" in text
    assert "foundation_objective_liveness_all_live" in text
    assert "foundation_source_field_liveness_all_live" in text
    assert "foundation_source_fields_by_split" in text
    assert "specialist_contract_summary" in text
    assert "required_training_specialists" in text
    assert "trainable_specialists" in text
    assert "excluded_specialist_groups" in text
    assert "specialist_model_contract_valid" in text
    assert "specialist_model_contract_failures" in text
    assert "specialist_model_contract" in text
    assert "_load_specialist_fusion_contract" in text
    assert "SPECIALIST_CONTRACT_MODE=foundation_seq146" in text
    assert "--specialist-contract-mode \"$SPECIALIST_CONTRACT_MODE\"" in text
    assert 'contract_mode = os.environ.get("SPECIALIST_CONTRACT_MODE", "foundation_seq146")' in text
    assert "--challenger-seq215" in text
    assert "SPECIALIST_CONTRACT_MODE=challenger_seq215" in text
    assert "--smart-seq520" in text
    assert "SPECIALIST_CONTRACT_MODE=smart_seq520_candidate" in text
    assert "smart-smoke-readiness --quiet" in text
    assert "smart-trainability-readiness --quiet" in text
    assert 'PREFLIGHT_GUARDRAILS_JSON="$DATA/reports/entry_smart_seq520_smoke_readiness_20260630_v1/ENTRY_SMART_SEQ520_SMOKE_READINESS_latest.json"' in text
    assert 'PREFLIGHT_READINESS_JSON="$DATA/reports/entry_smart_seq520_trainability_readiness_20260630_v1/ENTRY_SMART_SEQ520_TRAINABILITY_READINESS_latest.json"' in text
    assert 'readiness_artifact_key = "smart_trainability_readiness" if run_flavor == "smart_seq520" else "training_readiness"' in text
    assert 'guardrails_artifact_key = "smart_smoke_readiness" if run_flavor == "smart_seq520" else "foundation_guardrails"' in text
    assert "SMOKE_ENABLE_XAU_DIRECTION_REPAIR_HEADS=1" in text
    assert "GX1_PERTF_CLOSED_BAR=1" in text
    assert "SMOKE_PRED_BALANCE_ALPHA=0.50" in text
    assert "SMOKE_DIRECTION_CE_SCALE=4.00" in text
    assert "SMOKE_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT=12.00" in text
    assert "SMOKE_DIRECTION_MIN_PRED_RATE_FRACTION=0.50" in text
    assert "SMOKE_DIRECTION_MIN_PRED_RATE_FLOOR=0.05" in text
    assert "SMOKE_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE=0.05" in text
    assert "SMOKE_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT=8.00" in text
    assert "SMOKE_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE=0.02" in text
    assert "SMOKE_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE=0.10" in text
    assert "SMOKE_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT=8.00" in text
    assert "SMOKE_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION=0.50" in text
    assert "SMOKE_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR=0.05" in text
    assert "SMOKE_DIRECTION_SLICE_RECALL_LOSS_WEIGHT=4.00" in text
    assert "SMOKE_DIRECTION_SLICE_BALANCED_CE_WEIGHT=2.00" in text
    assert "SMOKE_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE=0.10" in text
    assert "SMOKE_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS=8" in text
    assert "SMOKE_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT=2.00" in text
    assert "SMOKE_DIRECTION_SLICE_TRUE_MARGIN=0.10" in text
    assert "SMOKE_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE=0.10" in text
    assert "SMOKE_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS=8" in text
    assert "SMOKE_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT=4.00" in text
    assert "SMOKE_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN=0.02" in text
    assert "SMOKE_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE=0.10" in text
    assert "SMOKE_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS=8" in text
    assert "SMOKE_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT=3.00" in text
    assert "SMOKE_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE=0.02" in text
    assert "SMOKE_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE=0.10" in text
    assert "SMOKE_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS=8" in text
    assert "SMOKE_DIRECTION_SLICE_LOSS_AGGREGATION=mean_max" in text
    assert "SMOKE_DIRECTION_SLICE_BALANCED_SAMPLER=1" in text
    assert "SMOKE_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS=8" in text
    assert "SMOKE_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE=3" in text
    assert "SMOKE_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS=6" in text
    assert "SMOKE_DIRECTION_VS_FLAT_MARGIN_WEIGHT=4.00" in text
    assert "SMOKE_DIRECTION_VS_FLAT_MARGIN=0.10" in text
    assert "SMOKE_DIRECTION_UTILITY_MARGIN_WEIGHT=4.00" in text
    assert "SMOKE_DIRECTION_UTILITY_MIN_GAP_BPS=15.0" in text
    assert "SMOKE_DIRECTION_UTILITY_LOGIT_MARGIN=0.10" in text
    assert "SMOKE_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT=6.00" in text
    assert "SMOKE_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS=15.0" in text
    assert "SMOKE_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN=0.10" in text
    assert "SMOKE_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT=8.00" in text
    assert "SMOKE_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS=15.0" in text
    assert "SMOKE_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS=0.0" in text
    assert "SMOKE_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH=0.50" in text
    assert "SMOKE_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN=0.10" in text
    assert "SMOKE_DIRECTION_FLAT_STARVATION_WEIGHT=8.00" in text
    assert "SMOKE_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE=0.10" in text
    assert "SMOKE_DIRECTION_FLAT_STARVATION_MIN_ROWS=8" in text
    assert "SMOKE_DIRECTION_FLAT_STARVATION_PRED_FRACTION=0.50" in text
    assert "SMOKE_DIRECTION_FLAT_STARVATION_PRED_FLOOR=0.10" in text
    assert "SMOKE_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN=0.10" in text
    assert "SMOKE_HIER_LEGACY_CE_MULT=1.00" in text
    assert "SMOKE_HIER_TRADE_WEIGHT=2.00" in text
    assert "SMOKE_HIER_SIDE_WEIGHT=1.75" in text
    assert "SMOKE_HIER_UTILITY_WEIGHT=1.00" in text
    assert "SMOKE_HIER_BAD_PATH_WEIGHT=1.25" in text
    assert "SMOKE_HIER_CTX_PRIOR_ADAPTER=1" in text
    assert "SMOKE_HIER_CTX_PRIOR_ADAPTER_SCALE=0.50" in text
    assert "SMOKE_HIER_POCKET_ABSTAIN_WEIGHT=5.00" in text
    assert "SMOKE_HIER_POCKET_SIDE_MARGIN_WEIGHT=3.00" in text
    assert "SMOKE_HIER_POCKET_UTILITY_MARGIN_BPS=30.0" in text
    assert "SMOKE_TRENDLINE_RAIL_AUX_WEIGHT=1.00" in text
    assert "SMOKE_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT=1.50" in text
    assert "SMOKE_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT=1.50" in text
    assert "SMOKE_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT=1.75" in text
    assert "SMOKE_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT=5.00" in text
    assert "SMOKE_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT=4.00" in text
    assert "SMOKE_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT=3.00" in text
    assert "SMOKE_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT=5.00" in text
    assert "SMOKE_TRENDLINE_RAIL_MARGIN=1.00" in text
    assert "SMOKE_TRENDLINE_RAIL_UTILITY_MARGIN_BPS=30.0" in text
    assert "SMOKE_FLAT_CLASS_WEIGHT_FLOOR=1.00" in text
    assert "SMOKE_ANCHOR_GATE_INIT=0.0" in text
    assert "--enable-xau-direction-repair-heads" in text
    assert '--anchor-gate-init "$SMOKE_ANCHOR_GATE_INIT"' in text
    assert "EXPECTED_SIGNAL_DIM=215" in text
    assert "entry_foundation_seq215_smoke_dataset_v1" in text
    assert "SMOKE_BUNDLE_AUDIT_OUT" in text
    assert "entry_foundation_smoke_bundle_audit_20260628_v1/challenger_seq215_20260630" in text
    assert '--out-dir "$SMOKE_BUNDLE_AUDIT_OUT"' in text
    assert "architecture_active_heads" in text
    assert "architecture_blocked_heads" in text
    assert "foundation_objective_routing_all_present_and_expected" in text
    assert "specialist_input_liveness_all_live" in text
    assert "target_contract_summary" in text
    assert "smoke_dataset_contract_summary" in text
    assert "audit_provenance_all_artifact_hashes_present" in text
    assert "worktree_contract_summary" in text
    assert "foundation_cleanup_critical_gate_review" in text
    assert "critical_gate_path_count" in text
    assert "--manifest-only" in text
    assert "Manifest-only stop before training" in text
    assert "require_clean_git_for_real_train" in text
    assert "require_foundation_contract_ready_for_manifest_only" in text
    assert "--no-fail-on-not-ready" in text
    assert "foundation_contract_ready_for_smoke" in text
    assert "git status --short" in text
    assert "real foundation smoke train requires clean git worktree" in text
    assert 'if [[ "$DRY_RUN" != "1" ]]' in text
    preflight_block = text[text.index('if [[ "$DRY_RUN" != "1" ]]') : text.index("STAMP=")]
    real_train_branch = preflight_block.split("else", 1)[1]
    assert real_train_branch.rindex("require_clean_git_for_real_train") < real_train_branch.rindex(
        "entry_next_edge_control.sh train-readiness --quiet"
    )
    assert "REQUIRE_EDGE_AUDIT=1" in text
    assert "--no-require-edge-audit" in text
    assert 'SMOKE_CAPPED_RUNNER=scripts/gx1_capped_run.sh' in text
    assert 'Smoke resource cap: mem=$SMOKE_RUN_MEM swap=$SMOKE_RUN_SWAP runner=$SMOKE_CAPPED_RUNNER num_workers=0' in text
    assert 'Capped smoke train command:' in text
    assert "Smoke bundle export completed; post-smoke audit pending" in text
    assert "Smoke bundle accepted by post-smoke audit" in text
    assert "Post-smoke edge audit failed; removing rejected smoke bundle" in text
    assert "--edge-test-scope smoke" not in text
    assert "--edge-test-scope strict" in text
    assert 'rm -rf -- "$OUT_BUNDLE"' in text
    assert '"memory_cap": os.environ.get("SMOKE_RUN_MEM")' in text
    assert '"swap_cap": os.environ.get("SMOKE_RUN_SWAP")' in text
    assert '"cgroup_runner": "scripts/gx1_capped_run.sh"' in text
    assert '"uses_gx1_capped_run": True' in text
    assert '"num_workers": int(command_arg_value(train_cmd, "--num-workers") or -1)' in text


def test_control_surface_exposes_manifest_only_smoke_proof() -> None:
    text = CONTROL.read_text(encoding="utf-8")

    assert "scripts/entry_next_edge_control.sh smoke-manifest --vedtak <id>" in text
    assert "scripts/entry_next_edge_control.sh smoke-manifest-seq215 --vedtak <id>" in text
    assert "scripts/entry_next_edge_control.sh smoke-train-seq215 --vedtak <id> --require-edge-audit" in text
    assert "scripts/entry_next_edge_control.sh smart-smoke-train --vedtak <id> --require-edge-audit" in text
    assert "smoke-manifest)" in text
    assert "smoke-manifest-seq215)" in text
    assert "smoke-train-seq215)" in text
    assert "smart-smoke-train)" in text
    assert "--smart-seq520 --edge-test-scope smoke" not in text
    assert "--smart-seq520 --edge-test-scope strict" in text
    assert 'run_entry_foundation_seq146_smoke_train.sh" --manifest-only' in text
    assert 'run_entry_foundation_seq146_smoke_train.sh" --challenger-seq215 --manifest-only' in text
