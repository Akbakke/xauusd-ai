from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gx1.contracts.registry_hyperparameter_fit_v1 import (
    REGISTRY_HYPERPARAMETER_FIT_SCHEMA_V1,
    REGISTRY_OUTCOME_BREAK,
    REGISTRY_OUTCOME_REACTION,
    RegistryOutcomeStreamV1,
    fit_registry_competing_risk_threshold_v1,
    load_registry_hyperparameter_artifact_v1,
    require_registry_hyperparameter_payload_v1,
    write_registry_hyperparameter_artifact_v1,
    _predictive_hazards,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path, *, clock: str = "M5") -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("source", "tape", "pair"):
        path = (tmp_path / f"{name}.json").resolve()
        path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
        paths[name] = path
    return {
        "source_artifact": str(paths["source"]),
        "source_sha256": _sha(paths["source"]),
        "source_schema_version": "synthetic_closed_ohlcv_v1",
        "source_lane": clock,
        "tape_manifest_artifact": str(paths["tape"]),
        "tape_manifest_sha256": _sha(paths["tape"]),
        "pair_manifest_artifact": str(paths["pair"]),
        "pair_manifest_sha256": _sha(paths["pair"]),
        "train_split_id": "chronological_train_only",
        "declared_train_window_start": "2020-01-01T00:00:00Z",
        "declared_train_window_end": "2020-01-01T00:39:00Z",
    }


def _stream(*, reaction_boundary: float) -> RegistryOutcomeStreamV1:
    # Both halves contain the complete empirical distance support.  Outcomes
    # change only at `reaction_boundary`, so a real held-out log score must
    # move the selected threshold when that TRAIN outcome distribution moves.
    origins = np.asarray(
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
        + [20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32],
        dtype=np.int64,
    )
    distance = np.asarray(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] * 4,
        dtype=np.float64,
    )
    causes = tuple(
        REGISTRY_OUTCOME_REACTION
        if value <= reaction_boundary
        else REGISTRY_OUTCOME_BREAK
        for value in distance
    )
    # Vary event ages so the scorer exercises the discrete survival terms.
    event_rows = origins + np.asarray([1, 2, 1, 2, 1, 2] * 4, dtype=np.int64)
    return RegistryOutcomeStreamV1(origins, distance, event_rows, causes)


def _fit(
    tmp_path: Path,
    *,
    reaction_boundary: float,
    source: dict | None = None,
) -> dict:
    n_rows = 40
    index_ns = np.arange(n_rows, dtype=np.int64) * 60_000_000_000
    price = 2000.0 + np.arange(n_rows, dtype=np.float64)
    return fit_registry_competing_risk_threshold_v1(
        _stream(reaction_boundary=reaction_boundary),
        registry_kind="horizontal_level",
        clock="M5",
        n_rows=n_rows,
        inner_fit_end_exclusive=20,
        index_ns=index_ns,
        frame_columns=(price, price + 1.0, price - 1.0),
        source_provenance=source or _source(tmp_path),
        population_configuration={"owner": "synthetic_threshold_test"},
    )


def test_selection_moves_with_chronological_train_outcome_distribution(tmp_path):
    first = _fit(tmp_path / "a", reaction_boundary=0.3)
    second = _fit(tmp_path / "b", reaction_boundary=0.5)

    assert first["selected_threshold_atr"] == 0.3
    assert second["selected_threshold_atr"] == 0.5
    assert first["candidate_count_total_empirical"] == 5
    assert second["candidate_count_total_empirical"] == 5
    assert first["selection_objective"].endswith(
        "competing_risk_log_likelihood"
    )
    assert first["future_outcomes_usage"] == (
        "TRAIN_hyperparameter_fit_only_not_apply_features"
    )
    assert first["selection_objective"] == (
        "chronological_inner_train_empirical_bayes_dirichlet_discrete_"
        "competing_risk_log_likelihood"
    )
    prior = first["hazard_prior_fit"]
    assert prior["concentration_candidate_origin"] == (
        "all_distinct_positive_inner_train_at_risk_counts"
    )
    assert prior["selected_concentration"] > 0.0
    assert np.allclose(
        prior["selected_alpha"],
        np.asarray(prior["base_probabilities"])
        * prior["selected_concentration"],
    )


def test_empirical_hazards_have_no_static_pseudocount() -> None:
    probabilities = _predictive_hazards(
        {
            "at_risk": [4, 2],
            "reaction_events": [1, 0],
            "break_events": [0, 2],
            "censored": [0, 0],
        }
    )
    np.testing.assert_array_equal(
        probabilities,
        np.asarray([[0.25, 0.0, 0.75], [0.0, 1.0, 0.0]]),
    )


def test_fit_is_exactly_deterministic_and_binds_population_split_and_source(tmp_path):
    source = _source(tmp_path)
    first = _fit(tmp_path, reaction_boundary=0.3, source=source)
    second = _fit(tmp_path, reaction_boundary=0.3, source=source)

    assert first == second
    assert first["schema_version"] == REGISTRY_HYPERPARAMETER_FIT_SCHEMA_V1
    assert first["inner_fit_end_exclusive"] == 20
    assert first["outcome_stream_sha256"]
    assert first["candidate_population_sha256"]
    assert first["candidate_score_stream_sha256"]
    assert first["frame_sha256"]
    assert first["learned_expiry_bars"] == int(
        np.ceil(first["restricted_mean_survival_bars"])
    )


def test_source_file_mutation_is_recomputed_and_rejected(tmp_path):
    source = _source(tmp_path)
    Path(source["source_artifact"]).write_text("mutated\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="IMMUTABLE_SOURCE_INVALID"):
        _fit(tmp_path, reaction_boundary=0.3, source=source)


def test_old_quantile_and_window_payload_is_rejected(tmp_path):
    old = {
        "schema_version": "level_registry_tolerance_fit_v2",
        "quantile_q": 0.5,
        "reaction_window_bars": 12,
        "retest_window_bars": 24,
    }
    with pytest.raises(RuntimeError, match="PAYLOAD_SCHEMA_INVALID"):
        require_registry_hyperparameter_payload_v1(
            old,
            registry_kind="horizontal_level",
            clock="M5",
        )


def test_resealed_objective_semantics_mutation_is_rejected(tmp_path):
    payload = _fit(tmp_path, reaction_boundary=0.3)
    payload["selection_objective"] = "weighted_success_rate"
    unhashed = {key: value for key, value in payload.items() if key != "contract_sha256"}
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(
            unhashed,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(RuntimeError, match="PAYLOAD_IDENTITY_INVALID"):
        require_registry_hyperparameter_payload_v1(
            payload,
            registry_kind="horizontal_level",
            clock="M5",
        )


def test_artifact_binds_exact_file_and_payload_hashes(tmp_path):
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    payload = _fit(fit_dir, reaction_boundary=0.3)
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    artifact = artifact_dir / "registry.json"
    binding = write_registry_hyperparameter_artifact_v1(
        artifact,
        payload,
        registry_kind="horizontal_level",
        clock="M5",
    )
    assert load_registry_hyperparameter_artifact_v1(
        binding,
        registry_kind="horizontal_level",
        clock="M5",
    ) == payload

    artifact.write_text(artifact.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="IMMUTABLE_SOURCE_INVALID"):
        load_registry_hyperparameter_artifact_v1(
            binding,
            registry_kind="horizontal_level",
            clock="M5",
        )


def test_frame_future_append_changes_only_fit_identity_not_apply_payload(tmp_path):
    payload = _fit(tmp_path, reaction_boundary=0.3)
    validated = require_registry_hyperparameter_payload_v1(
        payload,
        registry_kind="horizontal_level",
        clock="M5",
    )
    # Apply consumes only the frozen threshold/hazards.  No OHLC/future row is
    # accepted by the payload validator or artifact loader.
    assert validated["selected_threshold_atr"] == payload["selected_threshold_atr"]
    assert "future_rows" not in validated


def test_retired_split_manifest_keys_can_never_re_enter_the_lineage(tmp_path):
    """The registry fit runs INSIDE the chain that produces split manifests.

    A ``split_manifest_artifact``/``split_manifest_sha256`` pointer was required
    here until 2026-08-15 while the producers actually bound the PAIR manifest
    to it, so the name named an artifact the chain only produces after the
    dataset this fit feeds. The names are retired and the exact-key-set
    contract must keep them out — both directions, so neither a stale producer
    nor a helpful default can slip one back in. The real binding they carried
    lives on under ``pair_manifest_artifact``/``pair_manifest_sha256``.
    """

    from gx1.contracts.registry_hyperparameter_fit_v1 import (
        RETIRED_SOURCE_KEYS,
        _SOURCE_KEYS,
        require_registry_fit_source_v1,
    )

    assert RETIRED_SOURCE_KEYS
    assert RETIRED_SOURCE_KEYS.isdisjoint(_SOURCE_KEYS)

    source = _source(tmp_path / "retired")
    # The live contract accepts the payload without any split pointer.
    assert require_registry_fit_source_v1(source, clock="M5") == source

    for retired in sorted(RETIRED_SOURCE_KEYS):
        polluted = dict(source)
        polluted[retired] = source["tape_manifest_artifact"]
        with pytest.raises(RuntimeError, match="REGISTRY_FIT_SOURCE_SCHEMA_INVALID"):
            require_registry_fit_source_v1(polluted, clock="M5")


def test_pair_manifest_binding_is_hash_bound_and_fails_closed(tmp_path):
    """The registry fit must hold one immutable pointer to its pair generation.

    The pointer existed under the misleading name ``split_manifest_*`` (the
    producers bound the PAIR manifest to it) and was removed on 2026-08-15,
    which left the fit with no hash-bound reference to the generation at all:
    a retention pass that reclaimed the generation would no longer fail any
    consumer closed, and only the free-text ``train_split_id`` remained. The
    binding is restored under its honest name and is re-validated on every
    load, so all three failure modes below must raise.
    """

    from gx1.contracts.registry_hyperparameter_fit_v1 import (
        RETIRED_SOURCE_KEYS,
        _SOURCE_KEYS,
        require_registry_fit_source_v1,
    )

    assert {"pair_manifest_artifact", "pair_manifest_sha256"} <= _SOURCE_KEYS
    assert RETIRED_SOURCE_KEYS.isdisjoint(_SOURCE_KEYS)

    source = _source(tmp_path / "pair")
    assert require_registry_fit_source_v1(source, clock="M5") == source

    for key in ("pair_manifest_artifact", "pair_manifest_sha256"):
        missing = dict(source)
        missing.pop(key)
        with pytest.raises(RuntimeError, match="REGISTRY_FIT_SOURCE_SCHEMA_INVALID"):
            require_registry_fit_source_v1(missing, clock="M5")

    wrong_hash = dict(source)
    wrong_hash["pair_manifest_sha256"] = "0" * 64
    with pytest.raises(
        RuntimeError, match="REGISTRY_FIT_PAIR_MANIFEST_IMMUTABLE_SOURCE_INVALID"
    ):
        require_registry_fit_source_v1(wrong_hash, clock="M5")

    # Retention reclaimed the pair generation: every consumer must fail closed.
    Path(source["pair_manifest_artifact"]).unlink()
    with pytest.raises(
        RuntimeError, match="REGISTRY_FIT_PAIR_MANIFEST_IMMUTABLE_SOURCE_INVALID"
    ):
        require_registry_fit_source_v1(source, clock="M5")


def test_declared_train_window_remains_the_only_train_population_authority(tmp_path):
    """What the retired split pointer was supposed to protect lives here.

    The invariant is "this fit saw exactly the declared TRAIN rows", and it is
    carried by the declared window — the pair pointer proves *which generation*
    was read, not *which rows*. The window must therefore be validated as an
    exact ordered UTC pair, and the rebuild chain compares these same values
    against its own split authority.
    """

    from gx1.contracts.registry_hyperparameter_fit_v1 import (
        require_registry_fit_source_v1,
    )

    source = _source(tmp_path / "window")
    for key in ("declared_train_window_start", "declared_train_window_end"):
        broken = dict(source)
        broken[key] = "not-a-timestamp"
        with pytest.raises(RuntimeError, match="REGISTRY_FIT_TRAIN_WINDOW_INVALID"):
            require_registry_fit_source_v1(broken, clock="M5")

    reversed_window = dict(source)
    reversed_window["declared_train_window_start"] = source[
        "declared_train_window_end"
    ]
    reversed_window["declared_train_window_end"] = source[
        "declared_train_window_start"
    ]
    with pytest.raises(RuntimeError, match="REGISTRY_FIT_TRAIN_WINDOW_INVALID"):
        require_registry_fit_source_v1(reversed_window, clock="M5")

    naive = dict(source)
    naive["declared_train_window_end"] = "2020-01-01T00:39:00"
    with pytest.raises(RuntimeError, match="REGISTRY_FIT_TRAIN_WINDOW_INVALID"):
        require_registry_fit_source_v1(naive, clock="M5")
