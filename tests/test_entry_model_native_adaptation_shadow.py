from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gx1.contracts.entry_model_native_adaptation_shadow_v1 import (
    ModelNativeAdaptationShadowError,
    load_bound_adaptation_shadow_evidence,
)
from tests.model_native_adaptation_support import (
    write_adaptation_bundle,
    write_adaptation_shadow,
)


def test_paired_shadow_recomputes_absolute_edge_and_candidate_superiority(
    tmp_path: Path,
) -> None:
    incumbent, incumbent_identity = write_adaptation_bundle(tmp_path, "incumbent")
    candidate, candidate_identity = write_adaptation_bundle(tmp_path, "candidate")
    shadow = write_adaptation_shadow(
        tmp_path,
        incumbent_bundle=incumbent,
        incumbent_identity=incumbent_identity,
        candidate_bundle=candidate,
        candidate_identity=candidate_identity,
    )

    loaded, binding = load_bound_adaptation_shadow_evidence(
        shadow["artifact"],
        incumbent_bundle=incumbent_identity,
        candidate_bundle=candidate_identity,
        context="UNIT_PAIRED_SHADOW",
        now_utc=shadow["event"]["created_utc"],
    )

    assert binding == shadow["artifact"]
    assert loaded["decision"] == "PASS"
    assert loaded["global_metrics"]["paired_delta_mean_lower_95_bps"] > 0.0
    assert {row["field"] for row in loaded["context_metrics"]} == {
        "session",
        "vol_regime",
    }


def test_shadow_without_paired_improvement_is_terminal_fail(tmp_path: Path) -> None:
    incumbent, incumbent_identity = write_adaptation_bundle(tmp_path, "incumbent")
    candidate, candidate_identity = write_adaptation_bundle(tmp_path, "candidate")
    shadow = write_adaptation_shadow(
        tmp_path,
        incumbent_bundle=incumbent,
        incumbent_identity=incumbent_identity,
        candidate_bundle=candidate,
        candidate_identity=candidate_identity,
        superior=False,
    )

    assert shadow["event"]["decision"] == "FAIL"
    assert "candidate_paired_delta_lower_95_not_positive" in shadow["event"][
        "failures"
    ]
    with pytest.raises(ModelNativeAdaptationShadowError, match="zero-failure PASS"):
        load_bound_adaptation_shadow_evidence(
            shadow["artifact"],
            incumbent_bundle=incumbent_identity,
            candidate_bundle=candidate_identity,
            context="UNIT_NONIMPROVING_SHADOW",
            now_utc=shadow["event"]["created_utc"],
        )


def test_mutated_paired_shadow_rows_are_rejected(tmp_path: Path) -> None:
    incumbent, incumbent_identity = write_adaptation_bundle(tmp_path, "incumbent")
    candidate, candidate_identity = write_adaptation_bundle(tmp_path, "candidate")
    shadow = write_adaptation_shadow(
        tmp_path,
        incumbent_bundle=incumbent,
        incumbent_identity=incumbent_identity,
        candidate_bundle=candidate,
        candidate_identity=candidate_identity,
    )
    rows = pd.read_parquet(shadow["rows_path"])
    rows.loc[0, "candidate_realized_pnl_bps"] = 999.0
    rows.to_parquet(shadow["rows_path"], index=False)

    with pytest.raises(ModelNativeAdaptationShadowError, match="missing or changed"):
        load_bound_adaptation_shadow_evidence(
            shadow["artifact"],
            incumbent_bundle=incumbent_identity,
            candidate_bundle=candidate_identity,
            context="UNIT_MUTATED_PAIRED_SHADOW",
            now_utc=shadow["event"]["created_utc"],
        )
