from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gx1.contracts.entry_model_native_adaptation_drift_v1 import (
    MODEL_NATIVE_ADAPTATION_DRIFT_RED,
    MODEL_NATIVE_ADAPTATION_DRIFT_STABLE,
    ModelNativeAdaptationDriftError,
    load_bound_adaptation_drift_evidence,
    recompute_adaptation_drift_evidence,
)
from gx1.scripts.finalize_entry_model_native_adaptation_drift_v1 import (
    AdaptationDriftFinalizationError,
    finalize_adaptation_drift_evidence,
)
from tests.model_native_adaptation_support import (
    adaptation_rows,
    event_binding,
    write_adaptation_bundle,
)


def _write_inputs(
    root: Path,
    *,
    loss_long: bool = False,
) -> tuple[Path, dict[str, str], Path, Path, pd.Timestamp]:
    bundle, identity = write_adaptation_bundle(root)
    now = pd.Timestamp.now(tz="UTC").floor("s")
    reference = adaptation_rows(
        start=now - pd.Timedelta(days=31),
        scope="candidate_test",
        identity=identity,
    )
    observations = adaptation_rows(
        start=now - pd.Timedelta(days=10),
        scope="broker_shadow",
        identity=identity,
        loss_long=loss_long,
    )
    reference_path = root / "adaptation_reference_20260720T120000123456Z.parquet"
    observation_path = root / "adaptation_observations_20260720T120000123456Z.parquet"
    reference.to_parquet(reference_path, index=False)
    observations.to_parquet(observation_path, index=False)
    return bundle, identity, reference_path, observation_path, now


def test_stable_drift_evidence_is_row_recomputed_and_fresh(tmp_path: Path) -> None:
    bundle, _, reference, observations, now = _write_inputs(tmp_path)
    event_path, event = finalize_adaptation_drift_evidence(
        bundle_dir=bundle,
        reference_rows_path=reference,
        observation_rows_path=observations,
        output_dir=tmp_path / "events",
    )
    loaded, binding = load_bound_adaptation_drift_evidence(
        event_binding(event_path),
        context="UNIT_ADAPTATION_DRIFT_STABLE",
        now_utc=event["created_utc"],
    )

    assert binding == event_binding(event_path)
    assert loaded["decision"] == MODEL_NATIVE_ADAPTATION_DRIFT_STABLE
    assert loaded["failures"] == []
    assert loaded["coverage"]["order_submission_count"] == 0
    assert loaded["global_metrics"]["observation_long"]["decision"] == "PASS"
    assert loaded["global_metrics"]["observation_short"]["decision"] == "PASS"

    stale_now = pd.Timestamp(event["created_utc"]) + pd.Timedelta(days=2)
    with pytest.raises(ModelNativeAdaptationDriftError, match="age_seconds"):
        load_bound_adaptation_drift_evidence(
            event_binding(event_path),
            context="UNIT_ADAPTATION_DRIFT_STALE",
            now_utc=stale_now,
        )
    assert now < pd.Timestamp(event["created_utc"])


def test_degraded_long_edge_is_terminal_drift_not_a_soft_pass(tmp_path: Path) -> None:
    bundle, _, reference, observations, _ = _write_inputs(
        tmp_path, loss_long=True
    )
    event_path, event = finalize_adaptation_drift_evidence(
        bundle_dir=bundle,
        reference_rows_path=reference,
        observation_rows_path=observations,
        output_dir=tmp_path / "events",
    )
    loaded, _ = load_bound_adaptation_drift_evidence(
        event_binding(event_path),
        context="UNIT_ADAPTATION_DRIFT_RED",
        now_utc=event["created_utc"],
    )

    assert loaded["decision"] == MODEL_NATIVE_ADAPTATION_DRIFT_RED
    assert "observation_long" in loaded["failures"]
    assert loaded["global_metrics"]["observation_long"]["decision"] == "FAIL"


def test_shadow_order_or_mutated_source_never_passes(tmp_path: Path) -> None:
    bundle, identity, reference, observations, now = _write_inputs(tmp_path)
    observation_frame = pd.read_parquet(observations)
    observation_frame.loc[0, "order_submitted"] = True
    with pytest.raises(ModelNativeAdaptationDriftError, match="submit no order"):
        recompute_adaptation_drift_evidence(
            reference_rows=pd.read_parquet(reference),
            observation_rows=observation_frame,
            bundle_identity=identity,
            event_created_utc=now,
            context="UNIT_ADAPTATION_SHADOW_ORDER",
        )

    observation_frame.loc[0, "order_submitted"] = False
    observation_frame.to_parquet(observations, index=False)
    event_path, event = finalize_adaptation_drift_evidence(
        bundle_dir=bundle,
        reference_rows_path=reference,
        observation_rows_path=observations,
        output_dir=tmp_path / "events",
    )
    observations.write_bytes(observations.read_bytes() + b"tamper")
    with pytest.raises(ModelNativeAdaptationDriftError, match="hash mismatch"):
        load_bound_adaptation_drift_evidence(
            event_binding(event_path),
            context="UNIT_ADAPTATION_MUTATED_ROWS",
            now_utc=event["created_utc"],
        )


def test_adaptation_drift_rejects_tied_model_direction(tmp_path: Path) -> None:
    _, identity, reference, observations, now = _write_inputs(tmp_path)
    observation_frame = pd.read_parquet(observations)
    observation_frame.loc[0, ["p_long", "p_short", "p_flat"]] = [
        0.45,
        0.45,
        0.10,
    ]

    with pytest.raises(ModelNativeAdaptationDriftError, match="unique model argmax"):
        recompute_adaptation_drift_evidence(
            reference_rows=pd.read_parquet(reference),
            observation_rows=observation_frame,
            bundle_identity=identity,
            event_created_utc=now,
            context="UNIT_ADAPTATION_DRIFT_TIE",
        )


def test_failed_refresh_publishes_newer_terminal_block(tmp_path: Path) -> None:
    bundle, _, reference, observations, _ = _write_inputs(tmp_path)
    output_dir = tmp_path / "events"
    stable_path, stable = finalize_adaptation_drift_evidence(
        bundle_dir=bundle,
        reference_rows_path=reference,
        observation_rows_path=observations,
        output_dir=output_dir,
    )
    mutable_alias = tmp_path / "adaptation_observations_latest.parquet"
    mutable_alias.write_bytes(observations.read_bytes())

    with pytest.raises(
        AdaptationDriftFinalizationError,
        match="canonical immutable parquet",
    ):
        finalize_adaptation_drift_evidence(
            bundle_dir=bundle,
            reference_rows_path=reference,
            observation_rows_path=mutable_alias,
            output_dir=output_dir,
        )

    with pytest.raises(ModelNativeAdaptationDriftError, match="not newest"):
        load_bound_adaptation_drift_evidence(
            event_binding(stable_path),
            context="UNIT_OLDER_STABLE_INVALIDATED",
            now_utc=stable["created_utc"],
        )
