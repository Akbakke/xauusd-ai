"""V10 v3+ Target 1: multi-TF trend-agreement score.

For each M5 row, computes how many higher-TF trend signals AGREE with
the dominant D1 trend direction. Returns a float in [0, 1] where:
  - 1.0 = all TFs point same direction as D1
  - 0.5 = mixed (half agree, half disagree)
  - 0.0 = all higher-TF contradict D1

V10 training uses this as an aux label. During training:
  loss_direction *= (1 + lambda * (1 - tf_agreement_score))
i.e., wrong-direction loss is amplified when TF-disagreement is high.

At inference: V10's predicted tf_agreement_score is exposed in
v10_snapshot so the live runner / Entry-IQL can gate entries on it.

Live observation that motivated this (2026-05-18):
  D1_dist_from_ema200_atr = +6.42 (strong D1 uptrend)
  H4_trend_sign_cat       = 0 (cat="down" or "flat")
  M5 intraday move        = sharp down-leg
  → V10 ignored M5 contradiction and fired p_long=0.99 → 11/11 lost.

Spec: V10_V3_RETRAIN_TARGETS.md target 1.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Sign-classification thresholds per feature (tuned from training distribution).
# A feature is +1 (bullish) if value > pos_threshold, -1 (bearish) if value < neg_threshold,
# else 0 (neutral). For categoricals we map specific bucket IDs.
D1_DIST_POS_THRESHOLD = 0.5    # D1_dist_from_ema200_atr units (ATR multiples)
D1_DIST_NEG_THRESHOLD = -0.5
H1_COMP_POS_THRESHOLD = 1.05   # H1_range_compression_ratio: >1.05 = expanding (bullish if up move)
H1_COMP_NEG_THRESHOLD = 0.95
M15_COMP_POS_THRESHOLD = 1.05
M15_COMP_NEG_THRESHOLD = 0.95
MICRO_MOM_POS_THRESHOLD = 0.0  # micro_momentum_3: simple sign
MICRO_MOM_NEG_THRESHOLD = 0.0

# H4_trend_sign_cat encoding (from training distribution: only 0 and 2 observed):
#   0 → bearish
#   1 → neutral (rare)
#   2 → bullish
H4_CAT_TO_SIGN = {0: -1, 1: 0, 2: 1}


def _continuous_to_sign(values: pd.Series, pos_thr: float, neg_thr: float) -> pd.Series:
    """Map continuous feature → trend sign {-1, 0, +1}."""
    s = pd.Series(0, index=values.index, dtype=np.int8)
    s[values > pos_thr] = 1
    s[values < neg_thr] = -1
    return s


def compute_tf_agreement_score(df: pd.DataFrame) -> pd.Series:
    """Compute per-row tf_agreement_score from multi-TF features.

    Required columns in df (from BASE28 + canonical_v3 join):
      - D1_dist_from_ema200_atr  (continuous, reference TF)
      - H4_trend_sign_cat        (categorical 0/1/2)
      - H1_range_compression_ratio  (continuous)
      - M15_range_compression_ratio (continuous)
      - micro_momentum_3         (continuous, M5-side momentum)

    Returns: pd.Series of float ∈ [0, 1] aligned to df.index.

    Score = fraction of non-D1 TFs whose sign agrees with D1's sign.
    If D1 sign is 0 (neutral), score is mean(|other TF signs|==0).
    """
    # D1 reference sign
    d1_sign = _continuous_to_sign(
        df["D1_dist_from_ema200_atr"],
        D1_DIST_POS_THRESHOLD, D1_DIST_NEG_THRESHOLD,
    )

    # Other TF signs
    h4_sign = df["H4_trend_sign_cat"].map(H4_CAT_TO_SIGN).fillna(0).astype(np.int8)
    h1_sign = _continuous_to_sign(
        df["H1_range_compression_ratio"],
        H1_COMP_POS_THRESHOLD, H1_COMP_NEG_THRESHOLD,
    )
    m15_sign = _continuous_to_sign(
        df["M15_range_compression_ratio"],
        M15_COMP_POS_THRESHOLD, M15_COMP_NEG_THRESHOLD,
    )
    m5_sign = _continuous_to_sign(
        df["micro_momentum_3"],
        MICRO_MOM_POS_THRESHOLD, MICRO_MOM_NEG_THRESHOLD,
    )

    other_signs = pd.DataFrame({
        "h4": h4_sign, "h1": h1_sign, "m15": m15_sign, "m5": m5_sign,
    })

    # Agreement: for each row, count how many other-TF signs equal D1 sign
    # If D1 is 0 (neutral), agreement = count where other is also 0
    matches = other_signs.apply(lambda col: col == d1_sign).sum(axis=1)
    score = matches / other_signs.shape[1]
    return score.astype(np.float32)


def summarize(df: pd.DataFrame, score: pd.Series) -> dict:
    """Return summary stats for logging/sanity-check."""
    return {
        "n_rows": int(len(score)),
        "mean": float(score.mean()),
        "std": float(score.std()),
        "p10": float(score.quantile(0.10)),
        "p50": float(score.quantile(0.50)),
        "p90": float(score.quantile(0.90)),
        "frac_full_agreement": float((score == 1.0).mean()),
        "frac_zero_agreement": float((score == 0.0).mean()),
        "d1_sign_distribution": df["D1_dist_from_ema200_atr"].apply(
            lambda v: "up" if v > D1_DIST_POS_THRESHOLD else ("down" if v < D1_DIST_NEG_THRESHOLD else "neutral")
        ).value_counts().to_dict(),
    }
