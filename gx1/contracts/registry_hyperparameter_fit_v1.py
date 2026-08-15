"""Immutable TRAIN-only fit contract for registry decision parameters.

The horizontal-level and trendline registries need a price-distance threshold
to turn raw geometric candidates into persistent identities.  Choosing a
fixed quantile (or a hand-written bar window) makes that operator decision an
unmeasured part of the feature definition.  This owner instead selects over
the *entire empirical threshold support* on a chronological inner-TRAIN split.

For every threshold, the exact same candidate observations are divided into
near/far populations.  A discrete competing-risk model is fitted on the early
inner-TRAIN population and scored on the later inner-TRAIN population with
empirical-Bayes predictive log loss. The possible outcomes are ``reaction``,
``break`` and right ``censor``. Both the Dirichlet base distribution and its
concentration are learned on inner-TRAIN: concentration candidates are every
distinct observed at-risk count and marginal likelihood selects one. No fixed
pseudocount enters selection. Future outcomes are therefore used only inside
the declared TRAIN window; they are never inputs to serving features.

The event-time model is non-parametric.  At each observed age it stores the
number at risk, the two event counts and the censor count. Predictive hazards
use the immutable empirical-Bayes Dirichlet artifact described above.
Physical expiry is the ceiling of the censor-aware restricted mean survival
time of the selected near population. This is an explicitly declared modelling
convention; no percentile or guessed bar count is involved.

Artifacts are immutable JSON.  The payload hash binds the exact source file,
source frame, tape and pair lineage, chronological split, candidate population,
outcome stream, selected parameter and complete hazard tables.  The separate
file hash is verified on every load, so a payload copied into another file is
not silently interchangeable with the declared artifact.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import gammaln

from gx1.utils.artifact_primitives_v1 import canonical_json_sha256, sha256_file


REGISTRY_HYPERPARAMETER_FIT_SCHEMA_V1 = (
    "registry_hyperparameter_competing_risk_fit_v2_runtime_population"
)
REGISTRY_HYPERPARAMETER_FIT_OWNER_V1 = (
    "gx1.contracts.registry_hyperparameter_fit_v1"
)
REGISTRY_OUTCOME_REACTION = "reaction"
REGISTRY_OUTCOME_BREAK = "break"
REGISTRY_OUTCOME_CENSORED = "censored"
REGISTRY_OUTCOMES_V1 = (
    REGISTRY_OUTCOME_REACTION,
    REGISTRY_OUTCOME_BREAK,
    REGISTRY_OUTCOME_CENSORED,
)

_HEX = frozenset("0123456789abcdef")
# ``declared_train_window_start`` / ``declared_train_window_end`` are the
# TRAIN-population authority here, and the rebuild chain re-checks the frozen
# values against its own --train-start / --registry-fit-train-end /
# --registry-fit-inner-end by exact timestamp equality once each lane has
# published.  A ``split_manifest_*`` pointer was additionally required until
# 2026-08-15; the producers bound the *pair* manifest under that name, so the
# name was a lie while the hash check itself was real and ran on every V4 cache
# load.  Removing it left the fit with no hash-bound pointer to the pair
# generation at all: a retention pass that reclaimed the generation used to
# fail every consumer closed, and afterwards only the free-text
# ``train_split_id`` remained.  The pointer is therefore restored on
# 2026-08-15 under its honest name and validated by ``_require_immutable_file``
# exactly as before; the retired names stay retired.
_SOURCE_KEYS = frozenset(
    {
        "source_artifact",
        "source_sha256",
        "source_schema_version",
        "source_lane",
        "tape_manifest_artifact",
        "tape_manifest_sha256",
        "pair_manifest_artifact",
        "pair_manifest_sha256",
        "train_split_id",
        "declared_train_window_start",
        "declared_train_window_end",
    }
)
# Lineage keys retired on 2026-08-15. They may never re-enter the payload; the
# owner's tests assert their absence from ``_SOURCE_KEYS``.  The binding they
# mislabelled now lives under ``pair_manifest_artifact`` /
# ``pair_manifest_sha256`` above.
RETIRED_SOURCE_KEYS = frozenset(
    {"split_manifest_artifact", "split_manifest_sha256"}
)


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _HEX for char in value)
    ):
        raise RuntimeError(f"[{label}_SHA256_INVALID]")
    return value


def _require_plain_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RuntimeError(f"[{label}_INVALID]")
    return value


def _require_immutable_file(
    value: Any,
    *,
    expected_sha256: Any,
    label: str,
) -> Path:
    path = Path(_require_plain_string(value, label=f"{label}_PATH")).expanduser()
    expected = _require_sha256(expected_sha256, label=label)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
        or sha256_file(path) != expected
    ):
        raise RuntimeError(f"[{label}_IMMUTABLE_SOURCE_INVALID]")
    return path


def require_registry_fit_source_v1(
    value: Any,
    *,
    clock: str,
) -> dict[str, Any]:
    """Validate and recompute every immutable TRAIN source binding."""

    if not isinstance(value, Mapping) or set(value) != set(_SOURCE_KEYS):
        raise RuntimeError("[REGISTRY_FIT_SOURCE_SCHEMA_INVALID]")
    observed = dict(value)
    expected_clock = _require_plain_string(clock, label="REGISTRY_FIT_CLOCK")
    if observed["source_lane"] != expected_clock:
        raise RuntimeError("[REGISTRY_FIT_SOURCE_CLOCK_MISMATCH]")
    _require_plain_string(
        observed["source_schema_version"], label="REGISTRY_FIT_SOURCE_SCHEMA"
    )
    _require_plain_string(
        observed["train_split_id"], label="REGISTRY_FIT_TRAIN_SPLIT_ID"
    )
    start = _require_plain_string(
        observed["declared_train_window_start"],
        label="REGISTRY_FIT_TRAIN_START",
    )
    end = _require_plain_string(
        observed["declared_train_window_end"],
        label="REGISTRY_FIT_TRAIN_END",
    )
    try:
        start_time = datetime.fromisoformat(
            start[:-1] + "+00:00" if start.endswith("Z") else start
        )
        end_time = datetime.fromisoformat(
            end[:-1] + "+00:00" if end.endswith("Z") else end
        )
    except ValueError as exc:
        raise RuntimeError("[REGISTRY_FIT_TRAIN_WINDOW_INVALID]") from exc
    if (
        start_time.tzinfo is None
        or end_time.tzinfo is None
        or start_time.utcoffset() != timezone.utc.utcoffset(start_time)
        or end_time.utcoffset() != timezone.utc.utcoffset(end_time)
        or start not in {start_time.isoformat(), start_time.isoformat().replace("+00:00", "Z")}
        or end not in {end_time.isoformat(), end_time.isoformat().replace("+00:00", "Z")}
        or start_time >= end_time
    ):
        raise RuntimeError("[REGISTRY_FIT_TRAIN_WINDOW_INVALID]")
    _require_immutable_file(
        observed["source_artifact"],
        expected_sha256=observed["source_sha256"],
        label="REGISTRY_FIT_SOURCE",
    )
    _require_immutable_file(
        observed["tape_manifest_artifact"],
        expected_sha256=observed["tape_manifest_sha256"],
        label="REGISTRY_FIT_TAPE_MANIFEST",
    )
    _require_immutable_file(
        observed["pair_manifest_artifact"],
        expected_sha256=observed["pair_manifest_sha256"],
        label="REGISTRY_FIT_PAIR_MANIFEST",
    )
    return observed


@dataclass(frozen=True)
class RegistryOutcomeStreamV1:
    """Threshold-independent candidate observations on the outer TRAIN tape.

    ``event_row`` is the first authoritative future outcome or policy-expiry
    row, or ``-1`` for an observation that remains right-censored at the outer
    TRAIN end.  A finite row with cause ``censored`` is therefore an observed
    TRAIN-fitted eligibility expiry, not a fabricated reaction.
    ``origin_row`` is the closed bar on which the geometric candidate becomes
    observable.  Every event must occur strictly after its origin.
    """

    origin_row: np.ndarray
    distance_atr: np.ndarray
    event_row: np.ndarray
    event_cause: tuple[str, ...]


def require_registry_outcome_stream_v1(
    value: Any,
    *,
    n_rows: int,
) -> RegistryOutcomeStreamV1:
    if not isinstance(value, RegistryOutcomeStreamV1):
        raise RuntimeError("[REGISTRY_OUTCOME_STREAM_TYPE_INVALID]")
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 3:
        raise RuntimeError("[REGISTRY_OUTCOME_STREAM_ROW_COUNT_INVALID]")
    origin = np.asarray(value.origin_row)
    distance = np.asarray(value.distance_atr)
    event_row = np.asarray(value.event_row)
    causes = tuple(value.event_cause)
    if (
        origin.ndim != 1
        or distance.ndim != 1
        or event_row.ndim != 1
        or not (len(origin) == len(distance) == len(event_row) == len(causes))
        or len(origin) < 4
        or origin.dtype.kind not in "iu"
        or event_row.dtype.kind not in "iu"
        or not np.isfinite(distance).all()
        or np.any(distance <= 0.0)
        or np.any(origin < 0)
        or np.any(origin >= n_rows - 1)
        or np.any((event_row != -1) & (event_row <= origin))
        or np.any(event_row >= n_rows)
        or any(cause not in REGISTRY_OUTCOMES_V1 for cause in causes)
        or any(
            event == -1 and cause != REGISTRY_OUTCOME_CENSORED
            for event, cause in zip(event_row.tolist(), causes)
        )
    ):
        raise RuntimeError("[REGISTRY_OUTCOME_STREAM_INVALID]")
    order = np.lexsort((distance.astype(np.float64), origin.astype(np.int64)))
    if not np.array_equal(order, np.arange(len(origin))):
        raise RuntimeError("[REGISTRY_OUTCOME_STREAM_ORDER_INVALID]")
    return RegistryOutcomeStreamV1(
        origin_row=np.ascontiguousarray(origin, dtype=np.int64),
        distance_atr=np.ascontiguousarray(distance, dtype=np.float64),
        event_row=np.ascontiguousarray(event_row, dtype=np.int64),
        event_cause=causes,
    )


def _frame_sha256(
    *,
    index_ns: np.ndarray,
    columns: Sequence[np.ndarray],
) -> str:
    index = np.ascontiguousarray(index_ns, dtype=np.int64)
    if index.ndim != 1 or len(index) < 3 or np.any(index[1:] <= index[:-1]):
        raise RuntimeError("[REGISTRY_FIT_FRAME_INDEX_INVALID]")
    digest = hashlib.sha256()
    digest.update(index.tobytes(order="C"))
    for column in columns:
        values = np.ascontiguousarray(column, dtype=np.float64)
        if values.shape != index.shape or not np.isfinite(values).all():
            raise RuntimeError("[REGISTRY_FIT_FRAME_VALUES_INVALID]")
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _stream_sha256(stream: RegistryOutcomeStreamV1) -> str:
    digest = hashlib.sha256()
    digest.update(stream.origin_row.tobytes(order="C"))
    digest.update(stream.distance_atr.tobytes(order="C"))
    digest.update(stream.event_row.tobytes(order="C"))
    digest.update("\n".join(stream.event_cause).encode("ascii"))
    return digest.hexdigest()


def _truncate_observations(
    stream: RegistryOutcomeStreamV1,
    indices: np.ndarray,
    *,
    last_observable_row: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    durations: list[int] = []
    distances: list[float] = []
    causes: list[str] = []
    for idx in indices.tolist():
        origin = int(stream.origin_row[idx])
        if origin >= last_observable_row:
            continue
        event = int(stream.event_row[idx])
        if event != -1 and event <= last_observable_row:
            duration = event - origin
            cause = stream.event_cause[idx]
        else:
            duration = last_observable_row - origin
            cause = REGISTRY_OUTCOME_CENSORED
        if duration <= 0:
            continue
        durations.append(duration)
        distances.append(float(stream.distance_atr[idx]))
        causes.append(cause)
    return (
        np.asarray(distances, dtype=np.float64),
        np.asarray(durations, dtype=np.int64),
        tuple(causes),
    )


def _hazard_counts(
    durations: np.ndarray,
    causes: tuple[str, ...],
    mask: np.ndarray,
    *,
    max_age: int,
) -> dict[str, list[int]]:
    selected_duration = durations[mask]
    selected_causes = tuple(cause for cause, keep in zip(causes, mask) if keep)
    if selected_duration.size == 0:
        raise RuntimeError("[REGISTRY_FIT_EMPTY_THRESHOLD_GROUP]")
    horizon = min(int(max_age), int(selected_duration.max()))
    duration_hist = np.bincount(
        selected_duration,
        minlength=horizon + 1,
    )[: horizon + 1]
    ended_before_age = np.cumsum(duration_hist, dtype=np.int64)[:horizon]
    at_risk_array = int(selected_duration.size) - ended_before_age
    if np.any(at_risk_array <= 0):
        positive = np.flatnonzero(at_risk_array > 0)
        if positive.size == 0:
            raise RuntimeError("[REGISTRY_FIT_EMPTY_THRESHOLD_GROUP]")
        horizon = int(positive[-1]) + 1
        at_risk_array = at_risk_array[:horizon]

    causes_array = np.asarray(selected_causes, dtype="U9")

    def event_hist(cause: str) -> np.ndarray:
        return np.bincount(
            selected_duration[causes_array == cause],
            minlength=horizon + 1,
        )[1 : horizon + 1]

    reaction_array = event_hist(REGISTRY_OUTCOME_REACTION)
    broken_array = event_hist(REGISTRY_OUTCOME_BREAK)
    censored_array = event_hist(REGISTRY_OUTCOME_CENSORED)
    return {
        "at_risk": [int(value) for value in at_risk_array.tolist()],
        "reaction_events": [int(value) for value in reaction_array.tolist()],
        "break_events": [int(value) for value in broken_array.tolist()],
        "censored": [int(value) for value in censored_array.tolist()],
    }


def _predictive_hazards(
    counts: Mapping[str, Sequence[int]],
    prior_alpha: Sequence[float] | None = None,
) -> np.ndarray:
    """Empirical-MLE or explicitly supplied learned-Dirichlet hazards."""

    risk = np.asarray(counts["at_risk"], dtype=np.float64)
    reaction = np.asarray(counts["reaction_events"], dtype=np.float64)
    broken = np.asarray(counts["break_events"], dtype=np.float64)
    if risk.size == 0 or reaction.shape != risk.shape or broken.shape != risk.shape:
        raise RuntimeError("[REGISTRY_FIT_HAZARD_COUNTS_INVALID]")
    survived = risk - reaction - broken
    if np.any(survived < 0.0):
        raise RuntimeError("[REGISTRY_FIT_HAZARD_COUNTS_INVALID]")
    if prior_alpha is None:
        alpha = np.zeros(3, dtype=np.float64)
    else:
        alpha = np.asarray(prior_alpha, dtype=np.float64)
        if (
            alpha.shape != (3,)
            or not np.isfinite(alpha).all()
            or np.any(alpha <= 0.0)
        ):
            raise RuntimeError("[REGISTRY_FIT_HAZARD_PRIOR_INVALID]")
    denominator = risk + float(alpha.sum())
    probs = np.column_stack(
        (
            (reaction + alpha[0]) / denominator,
            (broken + alpha[1]) / denominator,
            (survived + alpha[2]) / denominator,
        )
    )
    if (
        not np.isfinite(probs).all()
        or np.any(probs < 0.0)
        or not np.allclose(
            probs.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=8.0 * np.finfo(np.float64).eps,
        )
    ):
        raise RuntimeError("[REGISTRY_FIT_HAZARD_PROBABILITY_INVALID]")
    return probs


def _fit_dirichlet_hazard_prior(
    counts: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Learn base probabilities and concentration from exact inner-TRAIN counts."""

    risk = np.asarray(counts["at_risk"], dtype=np.int64)
    reaction = np.asarray(counts["reaction_events"], dtype=np.int64)
    broken = np.asarray(counts["break_events"], dtype=np.int64)
    survived = risk - reaction - broken
    rows = np.column_stack((reaction, broken, survived))
    totals = rows.sum(axis=0).astype(np.float64)
    if rows.size == 0 or np.any(totals <= 0.0):
        raise RuntimeError("[REGISTRY_FIT_HAZARD_PRIOR_UNIDENTIFIED]")
    base = totals / float(totals.sum())
    candidates = np.unique(risk[risk > 0]).astype(np.float64)
    if candidates.size == 0:
        raise RuntimeError("[REGISTRY_FIT_HAZARD_PRIOR_UNIDENTIFIED]")
    # Group the exact Dirichlet-multinomial terms by observed integer count.
    # This removes an O(age_rows*candidates) Python loop without dropping one
    # candidate or changing the likelihood.
    risk_values, risk_frequency = np.unique(risk, return_counts=True)
    component_histograms = [
        np.unique(rows[:, component], return_counts=True)
        for component in range(3)
    ]
    score_rows: list[tuple[float, float]] = []
    chunk_size = 256  # bounded numerical workspace; no market semantics
    for start in range(0, len(candidates), chunk_size):
        concentration = candidates[start : start + chunk_size]
        scores = len(rows) * gammaln(concentration)
        scores -= np.sum(
            gammaln(concentration[:, None] + risk_values[None, :])
            * risk_frequency[None, :],
            axis=1,
        )
        for component, (values, frequency) in enumerate(component_histograms):
            alpha = concentration * base[component]
            scores += np.sum(
                gammaln(alpha[:, None] + values[None, :])
                * frequency[None, :],
                axis=1,
            )
            scores -= len(rows) * gammaln(alpha)
        for concentration_value, score in zip(
            concentration.tolist(), scores.tolist()
        ):
            if math.isfinite(score):
                score_rows.append(
                    (float(concentration_value), float(score))
                )
    if not score_rows:
        raise RuntimeError("[REGISTRY_FIT_HAZARD_PRIOR_UNIDENTIFIED]")
    selected_concentration, selected_score = min(
        score_rows,
        key=lambda item: (-item[1], item[0]),
    )
    selected_alpha = (base * selected_concentration).tolist()
    candidate_payload = [
        {"concentration": concentration, "marginal_log_likelihood": score}
        for concentration, score in score_rows
    ]
    return {
        "contract": "inner_train_empirical_bayes_dirichlet_hazard_prior_v1",
        "base_probability_origin": (
            "pooled_inner_train_reaction_break_survive_conditional_trial_MLE"
        ),
        "base_probabilities": [float(value) for value in base.tolist()],
        "concentration_candidate_origin": (
            "all_distinct_positive_inner_train_at_risk_counts"
        ),
        "concentration_candidate_count": int(candidates.size),
        "concentration_candidate_population_sha256": canonical_json_sha256(
            [float(value) for value in candidates.tolist()]
        ),
        "concentration_score_stream_sha256": canonical_json_sha256(
            candidate_payload
        ),
        "selection_objective": "dirichlet_multinomial_marginal_log_likelihood",
        "tie_break": "minimum_concentration_on_exact_likelihood_tie",
        "selected_concentration": float(selected_concentration),
        "selected_marginal_log_likelihood": float(selected_score),
        "selected_alpha": [float(value) for value in selected_alpha],
    }


def _conditional_trial_tensor(
    durations: np.ndarray,
    causes: tuple[str, ...],
    *,
    horizon: int,
) -> np.ndarray:
    """Aggregate [reaction, break, survive] conditional trials by event age."""

    if (
        durations.ndim != 1
        or len(durations) != len(causes)
        or np.any(durations <= 0)
        or np.any(durations > horizon)
        or any(cause not in REGISTRY_OUTCOMES_V1 for cause in causes)
    ):
        raise RuntimeError("[REGISTRY_FIT_TRIAL_DURATION_INVALID]")
    duration_hist = np.bincount(durations, minlength=horizon + 1)
    at_risk = int(len(durations)) - np.cumsum(
        duration_hist, dtype=np.int64
    )[:horizon]
    cause_array = np.asarray(causes, dtype="U9")
    reaction = np.bincount(
        durations[cause_array == REGISTRY_OUTCOME_REACTION],
        minlength=horizon + 1,
    )[1 : horizon + 1]
    broken = np.bincount(
        durations[cause_array == REGISTRY_OUTCOME_BREAK],
        minlength=horizon + 1,
    )[1 : horizon + 1]
    survived = at_risk - reaction - broken
    if np.any(survived < 0):
        raise RuntimeError("[REGISTRY_FIT_TRIAL_DURATION_INVALID]")
    return np.column_stack((reaction, broken, survived)).astype(
        np.int64,
        copy=False,
    )


def _add_conditional_trials(
    tensor: np.ndarray,
    *,
    duration: int,
    cause: str,
) -> None:
    age = int(duration)
    if age > 1:
        tensor[: age - 1, 2] += 1
    if cause == REGISTRY_OUTCOME_REACTION:
        tensor[age - 1, 0] += 1
    elif cause == REGISTRY_OUTCOME_BREAK:
        tensor[age - 1, 1] += 1
    elif cause == REGISTRY_OUTCOME_CENSORED:
        tensor[age - 1, 2] += 1
    else:
        raise RuntimeError("[REGISTRY_FIT_TRIAL_CAUSE_INVALID]")


def _tensor_predictive_probabilities(
    fit_trials: np.ndarray,
    prior_alpha: np.ndarray,
) -> np.ndarray:
    risk = fit_trials.sum(axis=1, dtype=np.int64).astype(np.float64)
    denominator = risk + float(prior_alpha.sum())
    probabilities = (fit_trials.astype(np.float64) + prior_alpha) / denominator[:, None]
    if not np.isfinite(probabilities).all() or np.any(probabilities <= 0.0):
        raise RuntimeError("[REGISTRY_FIT_HAZARD_PROBABILITY_INVALID]")
    return probabilities


def _restricted_mean_survival_bars(
    counts: Mapping[str, Sequence[int]],
) -> tuple[float, int]:
    """Censor-aware Kaplan-Meier restricted mean and integer expiry."""

    risk = np.asarray(counts["at_risk"], dtype=np.float64)
    events = np.asarray(counts["reaction_events"], dtype=np.float64) + np.asarray(
        counts["break_events"], dtype=np.float64
    )
    if risk.size == 0 or events.shape != risk.shape or np.any(events > risk):
        raise RuntimeError("[REGISTRY_FIT_SURVIVAL_COUNTS_INVALID]")
    survival_at_start = 1.0
    restricted_mean = 0.0
    for at_risk, event_count in zip(risk.tolist(), events.tolist()):
        restricted_mean += survival_at_start
        survival_at_start *= 1.0 - event_count / at_risk
    if not math.isfinite(restricted_mean) or restricted_mean <= 0.0:
        raise RuntimeError("[REGISTRY_FIT_SURVIVAL_DEGENERATE]")
    return restricted_mean, int(math.ceil(restricted_mean))


def score_registry_runtime_episode_population_v1(
    *,
    origin_row: Sequence[int],
    end_row: Sequence[int],
    event_cause: Sequence[str],
    n_rows: int,
    inner_fit_end_exclusive: int,
) -> dict[str, Any]:
    """Proper chronological score for one exact runtime-replay population."""

    origins = np.asarray(origin_row, dtype=np.int64)
    ends = np.asarray(end_row, dtype=np.int64)
    causes = tuple(event_cause)
    if (
        origins.ndim != 1
        or ends.shape != origins.shape
        or len(origins) != len(causes)
        or len(origins) < 6
        or np.any(origins < 0)
        or np.any(origins >= n_rows - 1)
        or np.any(ends <= origins)
        or np.any(ends >= n_rows)
        or np.any(origins[1:] < origins[:-1])
        or any(cause not in REGISTRY_OUTCOMES_V1 for cause in causes)
        or isinstance(inner_fit_end_exclusive, bool)
        or not isinstance(inner_fit_end_exclusive, int)
        or not 1 < inner_fit_end_exclusive < n_rows - 1
    ):
        raise RuntimeError("[REGISTRY_RUNTIME_EPISODE_POPULATION_INVALID]")

    def _split(
        indices: np.ndarray,
        *,
        last_row: int,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        durations: list[int] = []
        observed_causes: list[str] = []
        for index in indices.tolist():
            origin = int(origins[index])
            if origin >= last_row:
                continue
            observed_end = int(ends[index])
            cause = causes[index]
            if observed_end > last_row:
                observed_end = last_row
                cause = REGISTRY_OUTCOME_CENSORED
            duration = observed_end - origin
            if duration > 0:
                durations.append(duration)
                observed_causes.append(cause)
        return np.asarray(durations, dtype=np.int64), tuple(observed_causes)

    fit_indices = np.flatnonzero(origins < inner_fit_end_exclusive)
    selection_indices = np.flatnonzero(origins >= inner_fit_end_exclusive)
    fit_duration, fit_cause = _split(
        fit_indices,
        last_row=inner_fit_end_exclusive - 1,
    )
    selection_duration, selection_cause = _split(
        selection_indices,
        last_row=n_rows - 1,
    )
    if fit_duration.size < 4 or selection_duration.size < 2:
        raise RuntimeError("[REGISTRY_RUNTIME_EPISODE_SUPPORT_INSUFFICIENT]")
    max_fit_age = int(fit_duration.max())
    fit_counts = _hazard_counts(
        fit_duration,
        fit_cause,
        np.ones(fit_duration.shape, dtype=bool),
        max_age=max_fit_age,
    )
    hazard_prior = _fit_dirichlet_hazard_prior(fit_counts)
    horizon = int(max(fit_duration.max(), selection_duration.max()))
    fit_trials = _conditional_trial_tensor(
        fit_duration,
        fit_cause,
        horizon=horizon,
    )
    selection_trials = _conditional_trial_tensor(
        selection_duration,
        selection_cause,
        horizon=horizon,
    )
    probabilities = _tensor_predictive_probabilities(
        fit_trials,
        np.asarray(hazard_prior["selected_alpha"], dtype=np.float64),
    )
    log_likelihood = float(np.sum(selection_trials * np.log(probabilities)))
    trial_count = int(selection_trials.sum())
    if not math.isfinite(log_likelihood) or trial_count <= 0:
        raise RuntimeError("[REGISTRY_RUNTIME_EPISODE_SCORE_INVALID]")
    return {
        "inner_fit_episode_count": int(fit_duration.size),
        "inner_selection_episode_count": int(selection_duration.size),
        "inner_selection_conditional_trial_count": trial_count,
        "selection_log_likelihood": log_likelihood,
        "selection_mean_log_likelihood": log_likelihood / float(trial_count),
        "inner_fit_hazard_counts": fit_counts,
        "hazard_prior_fit": hazard_prior,
    }


def fit_registry_competing_risk_threshold_v1(
    stream: RegistryOutcomeStreamV1,
    *,
    registry_kind: str,
    clock: str,
    n_rows: int,
    inner_fit_end_exclusive: int,
    index_ns: np.ndarray,
    frame_columns: Sequence[np.ndarray],
    source_provenance: Mapping[str, Any],
    population_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Select one empirical distance threshold on chronological inner-TRAIN.

    Candidate support is every distinct distance observed in the inner-fit
    prefix except the largest value (which would leave the ``far`` group
    empty).  No grid or quantile is accepted.  Equal likelihood resolves to
    the smaller threshold, the deterministic lower-complexity identity rule.
    """

    kind = _require_plain_string(registry_kind, label="REGISTRY_FIT_KIND")
    lane = _require_plain_string(clock, label="REGISTRY_FIT_CLOCK")
    if (
        isinstance(inner_fit_end_exclusive, bool)
        or not isinstance(inner_fit_end_exclusive, int)
        or not (1 < inner_fit_end_exclusive < n_rows - 1)
    ):
        raise RuntimeError("[REGISTRY_FIT_INNER_SPLIT_INVALID]")
    source = require_registry_fit_source_v1(source_provenance, clock=lane)
    if not isinstance(population_configuration, Mapping) or not population_configuration:
        raise RuntimeError("[REGISTRY_FIT_POPULATION_CONFIGURATION_INVALID]")
    population_config = dict(population_configuration)
    try:
        canonical_json_sha256(population_config)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "[REGISTRY_FIT_POPULATION_CONFIGURATION_INVALID]"
        ) from exc
    validated = require_registry_outcome_stream_v1(stream, n_rows=n_rows)
    frame_sha = _frame_sha256(index_ns=index_ns, columns=frame_columns)

    fit_indices = np.flatnonzero(validated.origin_row < inner_fit_end_exclusive)
    selection_indices = np.flatnonzero(
        validated.origin_row >= inner_fit_end_exclusive
    )
    fit_dist, fit_duration, fit_cause = _truncate_observations(
        validated,
        fit_indices,
        last_observable_row=inner_fit_end_exclusive - 1,
    )
    sel_dist, sel_duration, sel_cause = _truncate_observations(
        validated,
        selection_indices,
        last_observable_row=n_rows - 1,
    )
    if fit_dist.size < 4 or sel_dist.size < 2:
        raise RuntimeError("[REGISTRY_FIT_CHRONOLOGICAL_SUPPORT_INSUFFICIENT]")
    support = np.unique(fit_dist)
    if support.size < 3:
        raise RuntimeError("[REGISTRY_FIT_THRESHOLD_SUPPORT_DEGENERATE]")
    candidates = support[:-1]
    max_age = int(fit_duration.max())
    pooled_counts = _hazard_counts(
        fit_duration,
        fit_cause,
        np.ones(fit_duration.shape, dtype=bool),
        max_age=max_age,
    )
    hazard_prior = _fit_dirichlet_hazard_prior(pooled_counts)
    prior_alpha = hazard_prior["selected_alpha"]
    horizon = int(max(fit_duration.max(), sel_duration.max()))
    fit_total_trials = _conditional_trial_tensor(
        fit_duration, fit_cause, horizon=horizon
    )
    selection_total_trials = _conditional_trial_tensor(
        sel_duration, sel_cause, horizon=horizon
    )
    fit_near_trials = np.zeros_like(fit_total_trials)
    selection_near_trials = np.zeros_like(selection_total_trials)
    fit_order = np.argsort(fit_dist, kind="stable")
    selection_order = np.argsort(sel_dist, kind="stable")
    fit_pointer = 0
    selection_pointer = 0
    prior_alpha_array = np.asarray(prior_alpha, dtype=np.float64)
    prior_sum = float(prior_alpha_array.sum())

    def score_contribution(stop: int) -> np.ndarray:
        fit_near = fit_near_trials[:stop]
        fit_far = fit_total_trials[:stop] - fit_near
        selection_near = selection_near_trials[:stop]
        selection_far = selection_total_trials[:stop] - selection_near
        near_probability = (
            fit_near.astype(np.float64) + prior_alpha_array
        ) / (
            fit_near.sum(axis=1, dtype=np.int64).astype(np.float64)
            + prior_sum
        )[:, None]
        far_probability = (
            fit_far.astype(np.float64) + prior_alpha_array
        ) / (
            fit_far.sum(axis=1, dtype=np.int64).astype(np.float64)
            + prior_sum
        )[:, None]
        if (
            not np.isfinite(near_probability).all()
            or not np.isfinite(far_probability).all()
            or np.any(near_probability <= 0.0)
            or np.any(far_probability <= 0.0)
        ):
            raise RuntimeError("[REGISTRY_FIT_HAZARD_PROBABILITY_INVALID]")
        return np.sum(
            selection_near * np.log(near_probability)
            + selection_far * np.log(far_probability),
            axis=1,
            dtype=np.float64,
        )

    score_by_age = score_contribution(horizon)
    running_score = float(np.sum(score_by_age, dtype=np.float64))
    score_rows: list[tuple[float, float]] = []
    for threshold in candidates.tolist():
        affected_age = 0
        while (
            fit_pointer < len(fit_order)
            and float(fit_dist[fit_order[fit_pointer]]) <= threshold
        ):
            index = int(fit_order[fit_pointer])
            _add_conditional_trials(
                fit_near_trials,
                duration=int(fit_duration[index]),
                cause=fit_cause[index],
            )
            affected_age = max(affected_age, int(fit_duration[index]))
            fit_pointer += 1
        while (
            selection_pointer < len(selection_order)
            and float(sel_dist[selection_order[selection_pointer]]) <= threshold
        ):
            index = int(selection_order[selection_pointer])
            _add_conditional_trials(
                selection_near_trials,
                duration=int(sel_duration[index]),
                cause=sel_cause[index],
            )
            affected_age = max(affected_age, int(sel_duration[index]))
            selection_pointer += 1
        if affected_age <= 0:
            raise RuntimeError("[REGISTRY_FIT_THRESHOLD_SUPPORT_ORDER_INVALID]")
        refreshed = score_contribution(affected_age)
        running_score += float(
            np.sum(refreshed - score_by_age[:affected_age], dtype=np.float64)
        )
        score_by_age[:affected_age] = refreshed
        if (
            fit_pointer == 0
            or fit_pointer == len(fit_order)
            or selection_pointer == 0
            or selection_pointer == len(selection_order)
        ):
            continue
        score = float(running_score)
        if not math.isfinite(score):
            continue
        exact_threshold = float(threshold)
        score_rows.append((exact_threshold, float(score)))
    if not score_rows:
        raise RuntimeError("[REGISTRY_FIT_NO_SCOREABLE_THRESHOLD]")
    # max score, then narrower identity band as the declared complexity tie.
    selected_threshold, selected_score = min(
        score_rows,
        key=lambda item: (-item[1], item[0]),
    )
    selected_near = fit_dist <= selected_threshold
    near_counts = _hazard_counts(
        fit_duration,
        fit_cause,
        selected_near,
        max_age=max_age,
    )
    far_counts = _hazard_counts(
        fit_duration,
        fit_cause,
        ~selected_near,
        max_age=max_age,
    )
    restricted_mean, expiry_bars = _restricted_mean_survival_bars(near_counts)
    candidate_payload = [
        {"threshold_atr": threshold, "selection_log_likelihood": score}
        for threshold, score in score_rows
    ]
    payload: dict[str, Any] = {
        "schema_version": REGISTRY_HYPERPARAMETER_FIT_SCHEMA_V1,
        "owner": REGISTRY_HYPERPARAMETER_FIT_OWNER_V1,
        "registry_kind": kind,
        "clock": lane,
        "selection_objective": (
            "chronological_inner_train_empirical_bayes_dirichlet_discrete_"
            "competing_risk_log_likelihood"
        ),
        "tie_break": "minimum_threshold_atr_on_exact_log_likelihood_tie",
        "threshold_candidate_origin": (
            "all_distinct_inner_fit_distance_atr_values_except_population_max"
        ),
        "selected_threshold_atr": selected_threshold,
        "selected_selection_log_likelihood": selected_score,
        "candidate_count_total_empirical": int(candidates.size),
        "candidate_count_scoreable": int(len(score_rows)),
        "candidate_population_sha256": canonical_json_sha256(
            [float(value) for value in candidates.tolist()]
        ),
        "candidate_score_stream_sha256": canonical_json_sha256(candidate_payload),
        "outcome_stream_sha256": _stream_sha256(validated),
        "outcome_semantics": (
            "first_authoritative_reaction_or_break_after_candidate_origin; "
            "right-censored at its chronological inner-TRAIN boundary"
        ),
        "future_outcomes_usage": "TRAIN_hyperparameter_fit_only_not_apply_features",
        "outer_train_rows": int(n_rows),
        "inner_fit_end_exclusive": int(inner_fit_end_exclusive),
        "inner_fit_observation_count": int(fit_dist.size),
        "inner_selection_observation_count": int(sel_dist.size),
        "frame_sha256": frame_sha,
        "population_configuration": population_config,
        "source_provenance": source,
        "selected_near_hazard_counts": near_counts,
        "selected_far_hazard_counts": far_counts,
        "hazard_prior_fit": hazard_prior,
        "expiry_estimator": (
            "ceil_kaplan_meier_restricted_mean_survival_on_selected_near_"
            "inner_fit_population"
        ),
        "restricted_mean_survival_bars": float(restricted_mean),
        "learned_expiry_bars": int(expiry_bars),
        "lifetime_candidate_origin": (
            "censor_aware_restricted_mean_survival_point_estimate"
        ),
        "lifetime_candidate_count_total": 1,
        "lifetime_candidate_population_sha256": canonical_json_sha256(
            [int(expiry_bars)]
        ),
        "lifetime_score_stream_sha256": canonical_json_sha256(
            [
                {
                    "learned_expiry_bars": int(expiry_bars),
                    "selected_selection_log_likelihood": selected_score,
                }
            ]
        ),
        "state_population_contract": (
            "exact candidate origin/distance population and canonical outcome "
            "labels are artifact-bound; apply eligibility may not alter the "
            "fit population"
        ),
        "predictive_probability_convention": (
            "inner-TRAIN empirical-Bayes Dirichlet reaction/break/survive "
            "hazards with data-learned base probabilities and concentration"
        ),
    }
    payload["contract_sha256"] = canonical_json_sha256(payload)
    return require_registry_hyperparameter_payload_v1(
        payload,
        registry_kind=kind,
        clock=lane,
    )


def require_registry_hyperparameter_payload_v1(
    value: Any,
    *,
    registry_kind: str,
    clock: str,
) -> dict[str, Any]:
    """Exact v1 payload validator; legacy q/window payloads fail closed."""

    expected_keys = {
        "schema_version",
        "owner",
        "registry_kind",
        "clock",
        "selection_objective",
        "tie_break",
        "threshold_candidate_origin",
        "selected_threshold_atr",
        "selected_selection_log_likelihood",
        "candidate_count_total_empirical",
        "candidate_count_scoreable",
        "candidate_population_sha256",
        "candidate_score_stream_sha256",
        "outcome_stream_sha256",
        "outcome_semantics",
        "future_outcomes_usage",
        "outer_train_rows",
        "inner_fit_end_exclusive",
        "inner_fit_observation_count",
        "inner_selection_observation_count",
        "frame_sha256",
        "population_configuration",
        "source_provenance",
        "selected_near_hazard_counts",
        "selected_far_hazard_counts",
        "hazard_prior_fit",
        "expiry_estimator",
        "restricted_mean_survival_bars",
        "learned_expiry_bars",
        "lifetime_candidate_origin",
        "lifetime_candidate_count_total",
        "lifetime_candidate_population_sha256",
        "lifetime_score_stream_sha256",
        "state_population_contract",
        "predictive_probability_convention",
        "contract_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError("[REGISTRY_HYPERPARAMETER_PAYLOAD_SCHEMA_INVALID]")
    observed = dict(value)
    if (
        observed["schema_version"] != REGISTRY_HYPERPARAMETER_FIT_SCHEMA_V1
        or observed["owner"] != REGISTRY_HYPERPARAMETER_FIT_OWNER_V1
        or observed["registry_kind"] != registry_kind
        or observed["clock"] != clock
        or observed["selection_objective"]
        != (
            "chronological_inner_train_empirical_bayes_dirichlet_discrete_"
            "competing_risk_log_likelihood"
        )
        or observed["tie_break"]
        != "minimum_threshold_atr_on_exact_log_likelihood_tie"
        or observed["threshold_candidate_origin"]
        != "all_distinct_inner_fit_distance_atr_values_except_population_max"
        or observed["outcome_semantics"]
        != (
            "first_authoritative_reaction_or_break_after_candidate_origin; "
            "right-censored at its chronological inner-TRAIN boundary"
        )
        or observed["future_outcomes_usage"]
        != "TRAIN_hyperparameter_fit_only_not_apply_features"
        or observed["expiry_estimator"]
        != (
            "ceil_kaplan_meier_restricted_mean_survival_on_selected_near_"
            "inner_fit_population"
        )
        or observed["state_population_contract"]
        != (
            "exact candidate origin/distance population and canonical outcome "
            "labels are artifact-bound; apply eligibility may not alter the "
            "fit population"
        )
        or observed["predictive_probability_convention"]
        != (
            "inner-TRAIN empirical-Bayes Dirichlet reaction/break/survive "
            "hazards with data-learned base probabilities and concentration"
        )
    ):
        raise RuntimeError("[REGISTRY_HYPERPARAMETER_PAYLOAD_IDENTITY_INVALID]")
    _require_sha256(
        observed["candidate_population_sha256"],
        label="REGISTRY_CANDIDATE_POPULATION",
    )
    _require_sha256(
        observed["candidate_score_stream_sha256"],
        label="REGISTRY_CANDIDATE_SCORE_STREAM",
    )
    _require_sha256(
        observed["outcome_stream_sha256"],
        label="REGISTRY_OUTCOME_STREAM",
    )
    _require_sha256(observed["frame_sha256"], label="REGISTRY_FIT_FRAME")
    if not isinstance(observed["population_configuration"], Mapping) or not observed[
        "population_configuration"
    ]:
        raise RuntimeError("[REGISTRY_HYPERPARAMETER_POPULATION_CONFIG_INVALID]")
    _require_sha256(
        observed["lifetime_candidate_population_sha256"],
        label="REGISTRY_LIFETIME_CANDIDATE_POPULATION",
    )
    _require_sha256(
        observed["lifetime_score_stream_sha256"],
        label="REGISTRY_LIFETIME_SCORE_STREAM",
    )
    threshold = observed["selected_threshold_atr"]
    score = observed["selected_selection_log_likelihood"]
    rmst = observed["restricted_mean_survival_bars"]
    expiry = observed["learned_expiry_bars"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or float(threshold) <= 0.0
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or isinstance(rmst, bool)
        or not isinstance(rmst, (int, float))
        or not math.isfinite(float(rmst))
        or float(rmst) <= 0.0
        or isinstance(expiry, bool)
        or not isinstance(expiry, int)
        or expiry <= 0
        or expiry != math.ceil(float(rmst))
    ):
        raise RuntimeError("[REGISTRY_HYPERPARAMETER_PAYLOAD_VALUES_INVALID]")
    for key in (
        "candidate_count_total_empirical",
        "candidate_count_scoreable",
        "outer_train_rows",
        "inner_fit_end_exclusive",
        "inner_fit_observation_count",
        "inner_selection_observation_count",
        "lifetime_candidate_count_total",
    ):
        item = observed[key]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise RuntimeError("[REGISTRY_HYPERPARAMETER_PAYLOAD_COUNTS_INVALID]")
    if (
        observed["candidate_count_scoreable"]
        > observed["candidate_count_total_empirical"]
        or not 1 < observed["inner_fit_end_exclusive"]
        < observed["outer_train_rows"] - 1
    ):
        raise RuntimeError("[REGISTRY_HYPERPARAMETER_PAYLOAD_COUNTS_INVALID]")
    require_registry_fit_source_v1(observed["source_provenance"], clock=clock)
    prior = observed["hazard_prior_fit"]
    prior_keys = {
        "contract",
        "base_probability_origin",
        "base_probabilities",
        "concentration_candidate_origin",
        "concentration_candidate_count",
        "concentration_candidate_population_sha256",
        "concentration_score_stream_sha256",
        "selection_objective",
        "tie_break",
        "selected_concentration",
        "selected_marginal_log_likelihood",
        "selected_alpha",
    }
    if not isinstance(prior, Mapping) or set(prior) != prior_keys:
        raise RuntimeError("[REGISTRY_HYPERPARAMETER_HAZARD_PRIOR_INVALID]")
    base = np.asarray(prior["base_probabilities"], dtype=np.float64)
    alpha = np.asarray(prior["selected_alpha"], dtype=np.float64)
    concentration = prior["selected_concentration"]
    marginal_score = prior["selected_marginal_log_likelihood"]
    candidate_count = prior["concentration_candidate_count"]
    if (
        prior["contract"]
        != "inner_train_empirical_bayes_dirichlet_hazard_prior_v1"
        or prior["base_probability_origin"]
        != "pooled_inner_train_reaction_break_survive_conditional_trial_MLE"
        or prior["concentration_candidate_origin"]
        != "all_distinct_positive_inner_train_at_risk_counts"
        or prior["selection_objective"]
        != "dirichlet_multinomial_marginal_log_likelihood"
        or prior["tie_break"]
        != "minimum_concentration_on_exact_likelihood_tie"
        or base.shape != (3,)
        or alpha.shape != (3,)
        or not np.isfinite(base).all()
        or not np.isfinite(alpha).all()
        or np.any(base <= 0.0)
        or np.any(alpha <= 0.0)
        or not math.isclose(float(base.sum()), 1.0, rel_tol=0.0, abs_tol=1e-15)
        or isinstance(concentration, bool)
        or not isinstance(concentration, (int, float))
        or not math.isfinite(float(concentration))
        or float(concentration) <= 0.0
        or not np.array_equal(alpha, base * float(concentration))
        or isinstance(marginal_score, bool)
        or not isinstance(marginal_score, (int, float))
        or not math.isfinite(float(marginal_score))
        or isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count <= 0
    ):
        raise RuntimeError("[REGISTRY_HYPERPARAMETER_HAZARD_PRIOR_INVALID]")
    _require_sha256(
        prior["concentration_candidate_population_sha256"],
        label="REGISTRY_HAZARD_PRIOR_CANDIDATE_POPULATION",
    )
    _require_sha256(
        prior["concentration_score_stream_sha256"],
        label="REGISTRY_HAZARD_PRIOR_SCORE_STREAM",
    )
    for key in ("selected_near_hazard_counts", "selected_far_hazard_counts"):
        counts = observed[key]
        if not isinstance(counts, Mapping) or set(counts) != {
            "at_risk",
            "reaction_events",
            "break_events",
            "censored",
        }:
            raise RuntimeError("[REGISTRY_HYPERPARAMETER_HAZARD_SCHEMA_INVALID]")
        lengths = {len(counts[name]) for name in counts}
        if lengths == {0} or len(lengths) != 1:
            raise RuntimeError("[REGISTRY_HYPERPARAMETER_HAZARD_SCHEMA_INVALID]")
        for items in counts.values():
            if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in items):
                raise RuntimeError("[REGISTRY_HYPERPARAMETER_HAZARD_VALUES_INVALID]")
        _predictive_hazards(counts, alpha)
    _require_sha256(observed["contract_sha256"], label="REGISTRY_PAYLOAD")
    without_hash = dict(observed)
    without_hash.pop("contract_sha256")
    if observed["contract_sha256"] != canonical_json_sha256(without_hash):
        raise RuntimeError("[REGISTRY_HYPERPARAMETER_PAYLOAD_HASH_MISMATCH]")
    return observed


def write_registry_hyperparameter_artifact_v1(
    path: Path,
    payload: Mapping[str, Any],
    *,
    registry_kind: str,
    clock: str,
) -> dict[str, str]:
    """Write once and return the exact path/file/payload binding."""

    validated = require_registry_hyperparameter_payload_v1(
        payload,
        registry_kind=registry_kind,
        clock=clock,
    )
    target = Path(path).expanduser()
    if (
        not target.is_absolute()
        or target.exists()
        or target.is_symlink()
        or not target.parent.is_dir()
    ):
        raise RuntimeError("[REGISTRY_HYPERPARAMETER_ARTIFACT_TARGET_INVALID]")
    encoded = (json.dumps(validated, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    with target.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    resolved = target.resolve(strict=True)
    return {
        "artifact_path": str(resolved),
        "artifact_file_sha256": sha256_file(resolved),
        "artifact_payload_sha256": validated["contract_sha256"],
    }


def load_registry_hyperparameter_artifact_v1(
    binding: Mapping[str, Any],
    *,
    registry_kind: str,
    clock: str,
) -> dict[str, Any]:
    expected_keys = {
        "artifact_path",
        "artifact_file_sha256",
        "artifact_payload_sha256",
    }
    if not isinstance(binding, Mapping) or set(binding) != expected_keys:
        raise RuntimeError("[REGISTRY_HYPERPARAMETER_ARTIFACT_BINDING_INVALID]")
    artifact = _require_immutable_file(
        binding["artifact_path"],
        expected_sha256=binding["artifact_file_sha256"],
        label="REGISTRY_HYPERPARAMETER_ARTIFACT",
    )
    try:
        raw = json.loads(artifact.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("[REGISTRY_HYPERPARAMETER_ARTIFACT_JSON_INVALID]") from exc
    payload = require_registry_hyperparameter_payload_v1(
        raw,
        registry_kind=registry_kind,
        clock=clock,
    )
    if payload["contract_sha256"] != _require_sha256(
        binding["artifact_payload_sha256"],
        label="REGISTRY_HYPERPARAMETER_ARTIFACT_PAYLOAD",
    ):
        raise RuntimeError("[REGISTRY_HYPERPARAMETER_ARTIFACT_PAYLOAD_MISMATCH]")
    return payload


__all__ = [
    "REGISTRY_HYPERPARAMETER_FIT_SCHEMA_V1",
    "REGISTRY_OUTCOME_BREAK",
    "REGISTRY_OUTCOME_REACTION",
    "RETIRED_SOURCE_KEYS",
    "RegistryOutcomeStreamV1",
    "fit_registry_competing_risk_threshold_v1",
    "load_registry_hyperparameter_artifact_v1",
    "require_registry_fit_source_v1",
    "require_registry_hyperparameter_payload_v1",
    "require_registry_outcome_stream_v1",
    "write_registry_hyperparameter_artifact_v1",
]
