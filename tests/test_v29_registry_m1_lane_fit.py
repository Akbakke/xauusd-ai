"""Registry artifact and M1-lane fail-closed integration tests."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from gx1.contracts.entry_exit_feature_base_v1 import EXIT_FEATURE_SEQUENCE_BARS
from gx1.features.htf_features import (
    V29_REGISTRY_M1_LANE_PARAMS_SCHEMA_VERSION,
    require_v29_registry_constants,
    require_v29_registry_m1_lane_params,
)
from gx1.scripts import build_entry_exit_m1_enriched_frame_v1 as enriched_producer
from gx1.scripts import materialize_entry_exit_m1_feature_base_v1 as feature_producer
from gx1.utils.artifact_primitives_v1 import canonical_json_sha256
from tests.htf_v29_registry_test_support import (
    synthetic_v29_registry_constants,
    synthetic_v29_registry_m1_lane_params,
    write_synthetic_m1_registry_manifest,
)


def _rehash(payload: dict) -> None:
    payload["contract_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "contract_sha256"}
    )


def test_m1_lane_payload_binds_exact_hyperfit_and_lifetimes() -> None:
    payload = synthetic_v29_registry_m1_lane_params()
    assert require_v29_registry_m1_lane_params(payload) == payload
    assert payload["schema_version"] == V29_REGISTRY_M1_LANE_PARAMS_SCHEMA_VERSION
    assert payload["level_recurrence_threshold_atr"] > 0.0
    assert payload["level_expiry_bars"] > 0
    assert payload["exit_m1"]["seq_len"] == EXIT_FEATURE_SEQUENCE_BARS
    assert payload["exit_m1"]["trendline_band_atr"] > 0.0
    assert payload["exit_m1"]["trendline_expiry_bars"] > 0
    assert not any(
        "quantile" in key or "reaction_window" in key or "retest_window" in key
        for key in payload
    )
    for nested in (
        payload["provenance"]["level_recurrence_threshold"],
        payload["provenance"]["trendline_band"],
    ):
        assert nested["future_outcomes_usage"] == (
            "TRAIN_hyperparameter_fit_only_not_apply_features"
        )
        assert nested["candidate_count_total_empirical"] > 0
        assert nested["candidate_count_scoreable"] > 0


def test_m1_lane_validator_rejects_old_keys_and_nested_mutation() -> None:
    valid = synthetic_v29_registry_m1_lane_params()
    old_named = copy.deepcopy(valid)
    old_named["level_tol_atr"] = old_named.pop(
        "level_recurrence_threshold_atr"
    )
    _rehash(old_named)
    with pytest.raises(RuntimeError, match="exact keys differ"):
        require_v29_registry_m1_lane_params(old_named)
    with pytest.raises(RuntimeError, match="exact keys differ"):
        require_v29_registry_m1_lane_params({**valid, "level_tol_quantile_q": 0.5})
    with pytest.raises(RuntimeError, match="exact keys differ"):
        require_v29_registry_m1_lane_params({**valid, "reaction_window_bars": 12})

    mutated = copy.deepcopy(valid)
    source = Path(
        mutated["provenance"]["level_recurrence_threshold"]
        ["source_provenance"]["source_artifact"]
    )
    source.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="registry hyperfit provenance"):
        require_v29_registry_m1_lane_params(mutated)


def test_m5_registry_validator_rejects_resealed_population_tamper() -> None:
    valid = synthetic_v29_registry_constants()
    tampered = copy.deepcopy(valid)
    tampered["provenance"]["trendline_band"]["H4"][
        "population_configuration"
    ]["identity_expiry_bars"] += 1
    nested = tampered["provenance"]["trendline_band"]["H4"]
    _rehash(nested)
    _rehash(tampered)
    with pytest.raises(RuntimeError, match="registry hyperfit binding"):
        require_v29_registry_constants(tampered)


def test_materializer_resolves_m1_lifetime_params_fail_closed(tmp_path: Path) -> None:
    lane = synthetic_v29_registry_m1_lane_params()
    manifest = write_synthetic_m1_registry_manifest(tmp_path, params=lane)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    source = Path(manifest_payload["output_parquet"])
    resolved = feature_producer._resolve_v29_registry_layer_params(
        manifest,
        timeframe="M1",
        expected_source=source,
        expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        expected_source_manifest=manifest,
    )
    assert resolved == {
        "level_recurrence_threshold_atr": lane[
            "level_recurrence_threshold_atr"
        ],
        "level_expiry_bars": lane["level_expiry_bars"],
        "trendline_band_atr": lane["exit_m1"]["trendline_band_atr"],
        "trendline_expiry_bars": lane["exit_m1"]["trendline_expiry_bars"],
        "trendline_seq_len": lane["exit_m1"]["seq_len"],
    }


@pytest.mark.parametrize("root_flag", ["--native-m1-root", "--native-m5-root"])
def test_enriched_cli_requires_complete_registry_fit_lineage(
    monkeypatch: pytest.MonkeyPatch, root_flag: str
) -> None:
    called: list[dict[str, object]] = []
    monkeypatch.setattr(
        enriched_producer,
        "_build_enriched_frame",
        lambda **kwargs: called.append(kwargs) or {"decision": "PASS"},
    )
    base = [
        "producer", root_flag, "/tmp/native",
        "--pair-manifest", "/tmp/pair.json",
        "--multi-tf-cache-dir", "/tmp/cache",
        "--output-parquet", "/tmp/enriched.parquet",
        "--manifest-path", "/tmp/enriched.json",
        "--checkpoint-dir", "/tmp/checkpoint",
        "--dataset-run-id", "run",
        "--pair-generation-id", "b" * 64,
        "--registry-fit-train-start", "2026-01-01T00:00:00+00:00",
        "--registry-fit-train-end", "2026-01-31T00:00:00+00:00",
        "--registry-fit-tape-manifest", "/tmp/tape.json",
        "--expected-registry-fit-tape-manifest-sha256", "c" * 64,
        "--volatility-squeeze-manifest", "/tmp/squeeze.json",
        "--expected-volatility-squeeze-manifest-sha256", "d" * 64,
    ]
    monkeypatch.setattr(sys, "argv", list(base))
    with pytest.raises(SystemExit):
        enriched_producer.main()
    assert called == []

    monkeypatch.setattr(
        sys,
        "argv",
        [*base, "--registry-fit-inner-end", "2026-01-15T00:00:00+00:00"],
    )
    enriched_producer.main()
    assert len(called) == 1
    assert called[0]["registry_fit_inner_end"] == "2026-01-15T00:00:00+00:00"
    assert "level_tol_quantile_q" not in called[0]
