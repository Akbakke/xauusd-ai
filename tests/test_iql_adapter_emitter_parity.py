"""Emitter-parity regression test (2026-06-04).

Guards the fail-closed coverage guard added to the IQL V2 adapters (commit
ea46cbe7). The adapters now RAISE if a REQUIRED feature (training
std >= STD_REQUIRED_THRESHOLD) is absent at serve, instead of silently 0-filling
it (the 2026-05-19 all-LONG-bias root cause). This test proves the live builders
(`v12_entry_iql_live.build_candidate` / `v12_exit_iql_live.build_bar_state`)
emit EVERY required feature when given complete inputs — so the strict guard can
never spuriously halt the cement chain.

The builders are exercised via ``object.__new__`` (no model load — build_candidate
/ build_bar_state only touch a getattr-default warn-set on ``self``) + a real
training-frame row (which carries all canonical-v3 columns) + a complete
``v10_out`` / ``TradeState`` fixture. CPU-only and CI-safe; the test SKIPS cleanly
if the cement bundles or build frames are unavailable.

Background: the COSTFIX live deploy already verified 0/192 entry features missing
end-to-end; this encodes that invariant as a regression guard against future
builder / contract drift. (It does NOT re-run augment_canonical_v3 / the model
stack — that remains the pre-relaunch live smoke.)
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import pytest

ENTRY_BUNDLE = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    "BUILD_ENTRY_IQL_V2_COSTFIX_HEADS_FAST_20260528T071646Z"
)
EXIT_BUNDLE = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    "BUILD_EXIT_IQL_V3_M1_COSTFIX_HEADS_FAST_20260528T205051Z"
)
ENTRY_FRAME_DIR = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    "CANDIDATE_FORWARD_OUTCOME_COSTFIX_HEADS_20260528T071240Z"
)
# The live exit builder (v12_exit_iql_live.decide_for_trade -> build_bar_state)
# is passed an augmented canonical_v3 row (PrebuiltStateLoader output), NOT the V3
# per-bar scored frame. The canonical_v3 prebuilt carries the 54 base _v1_* columns
# that build_bar_state aliases to _v1_*_canon_v1 — so this is the correct serve-row
# source for the exit emitter-parity check.
EXIT_CV3_PREBUILTS = (
    "/home/andre2/GX1_DATA/data/data/prebuilt/xauusd_m5_CANONICAL_V3_2020_2026.parquet",
)


def _find_cv3_prebuilt() -> str | None:
    for p in EXIT_CV3_PREBUILTS:
        if Path(p).is_file():
            return p
    hits = glob.glob(
        "/home/andre2/GX1_DATA/**/xauusd_m5_CANONICAL_V3_2020_2026.parquet", recursive=True
    )
    return hits[0] if hits else None


def _first_parquet_row(frame_dir: Path) -> pd.Series:
    files = sorted(glob.glob(str(frame_dir / "**" / "*.parquet"), recursive=True))
    if not files:
        pytest.skip(f"no parquet under {frame_dir}")
    df = pd.read_parquet(files[0])
    if df.empty:
        pytest.skip(f"empty frame: {files[0]}")
    row = df.iloc[0].copy()
    # build_candidate / build_bar_state read row.name as a decision timestamp.
    row.name = pd.Timestamp("2026-01-06 12:00:00")
    return row


def _complete_v10_out() -> dict:
    """A v10_out dict covering every key build_candidate reads."""
    return {
        "direction_probs": [0.4, 0.35, 0.25],
        "direction_logits": [0.5, 0.2, -0.3],
        "tradable_prob": 0.6,
        "mfe_first_n": 12.0,
        "path_quality": 0.55,
        "bad_path_prob": 0.2,
        "tf_agreement_pred": 0.3,
        "path_quality_log_var": -1.0,
        "path_quality_std": 0.1,
        "position_size_pred": 0.5,
        "hold_horizon_pred": 40.0,
        "dip_pred_v1": [0.0] * 18,
        "forecast_pred_v1": [0.0] * 4,
        "timing_pred_v1": [0.0] * 12,
        "tail_risk_pred_v1": [0.0] * 6,
        "vol_forecast_pred_v1": [0.0] * 3,
    }


def _load_required(bundle: Path, variant: str, fold: str, kind: str) -> tuple[set, list]:
    if not bundle.is_dir():
        pytest.skip(f"bundle missing: {bundle}")
    if kind == "entry":
        from gx1.runtime.entry_iql_v2_adapter import EntryIQLV2Adapter as Ad
    else:
        from gx1.runtime.exit_iql_v2_adapter import ExitIQLV2Adapter as Ad
    try:
        a = Ad.load(bundle, variant=variant, fold_id=fold,
                    aggregator="mean" if kind == "entry" else "max", prefer_cuda=False)
    except Exception as e:  # noqa: BLE001 — env without torch/ckpt → skip, not fail
        pytest.skip(f"cannot load {kind} adapter: {e}")
    return set(a.required_feature_names), list(a.feature_names)


def _unresolved_required(required: set, emitted: set) -> list:
    """REQUIRED features not covered by the emitted dict.

    A required one-hot ``cat__val`` (entry, double underscore) is considered
    covered if the underlying categorical column ``cat`` was emitted (the live
    builder always emits every one-hot slot once the column is present).
    """
    miss = []
    for f in sorted(required):
        if f in emitted:
            continue
        if "__" in f and f.split("__", 1)[0] in emitted:
            continue
        miss.append(f)
    return miss


def test_entry_builder_emits_all_required():
    required, _ = _load_required(ENTRY_BUNDLE, "R_WAIT_OPP_K96_LAM50", "FOLD_1", "entry")
    assert required, "entry REQUIRED set should be non-empty"
    row = _first_parquet_row(ENTRY_FRAME_DIR)
    from gx1.execution.v12_entry_iql_live import EntryIQLLiveInference
    inst = object.__new__(EntryIQLLiveInference)  # build_candidate only uses a getattr-default
    cand = EntryIQLLiveInference.build_candidate(inst, row, {}, _complete_v10_out(), None)
    missing = _unresolved_required(required, set(cand.keys()))
    assert not missing, (
        f"entry build_candidate omits {len(missing)} REQUIRED feature(s) → strict "
        f"adapter would halt live: {missing[:20]}"
    )


def test_exit_builder_emits_all_required():
    required, _ = _load_required(EXIT_BUNDLE, "R_NET_REAL", "FOLD_1", "exit")
    assert required, "exit REQUIRED set should be non-empty"
    cv3 = _find_cv3_prebuilt()
    if cv3 is None:
        pytest.skip("canonical_v3 prebuilt not found")
    df = pd.read_parquet(cv3)
    if df.empty:
        pytest.skip(f"empty cv3 prebuilt: {cv3}")
    row = df.iloc[-1].copy()  # last closed M5 bar — what the live loader serves
    row.name = pd.Timestamp("2026-01-06 12:00:00")
    from gx1.execution.v12_exit_iql_live import ExitIQLLiveInference
    from gx1.execution.v12_trade_state import TradeState
    trade = TradeState.open(
        entry_ts=pd.Timestamp("2026-01-06 11:30:00"),
        side="long",
        entry_bid=2000.0,
        entry_ask=2000.2,
        v10_snapshot=_complete_v10_out(),
    )
    inst = object.__new__(ExitIQLLiveInference)
    bs = ExitIQLLiveInference.build_bar_state(inst, trade, row, None)
    missing = _unresolved_required(required, set(bs.keys()))
    assert not missing, (
        f"exit build_bar_state omits {len(missing)} REQUIRED feature(s) → strict "
        f"adapter would halt live: {missing[:20]}"
    )
