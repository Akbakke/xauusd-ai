#!/usr/bin/env python3
"""Build the local-clock canonical feature frame used by Entry and Exit.

This owner computes only features native to the supplied local clock plus the
local SMC state.  It deliberately has no resampling code.  All M5/M15/H1/H4/D1
evidence is owned and projected later by the native-M5 V4 lane.
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.features.smc_v1 import (  # noqa: E402
    SMC_FEATURE_NAMES,
    compute_smc_features,
)
from gx1.scripts import (  # noqa: E402
    materialize_build_canonical_features_v1 as v1_builder,
)


ACTION = "BUILD_CANONICAL_LOCAL_FEATURES_V2"
CANONICAL_V2_MTF_OWNER = "native_m5_v4_only"


def build_canonical_v2(
    local_bars: pd.DataFrame,
    *,
    base_bar_duration: pd.Timedelta = pd.Timedelta(minutes=5),
) -> pd.DataFrame:
    """Build local features without creating or aligning any MTF values."""

    if (
        not isinstance(base_bar_duration, pd.Timedelta)
        or base_bar_duration not in {
            pd.Timedelta(minutes=1),
            pd.Timedelta(minutes=5),
        }
    ):
        raise RuntimeError(
            "canonical_v2 requires the exact local M1 or M5 decision clock"
        )
    print(f"[{ACTION}] step 1/2 — building local feature owners...", flush=True)
    started = _time.time()
    out = v1_builder.build_canonical_features(
        local_bars,
        decision_bar_duration=base_bar_duration,
    )
    print(
        f"[{ACTION}] local owners done in {_time.time() - started:.1f}s, "
        f"shape={out.shape}",
        flush=True,
    )

    print(f"[{ACTION}] step 2/2 — computing local SMC state...", flush=True)
    started = _time.time()
    smc = compute_smc_features(out)
    if len(smc) != len(out) or not smc.index.equals(out.index):
        raise RuntimeError("canonical_v2 local SMC row axis mismatch")
    overlap = sorted(set(SMC_FEATURE_NAMES) & set(out.columns))
    if overlap:
        raise RuntimeError(f"canonical_v2 local SMC fields already exist: {overlap}")
    for column in SMC_FEATURE_NAMES:
        out[column] = smc[column].to_numpy(copy=False)
    print(
        f"[{ACTION}] local SMC done in {_time.time() - started:.1f}s, "
        f"final shape={out.shape}",
        flush=True,
    )
    return out
