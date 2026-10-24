"""Source-declared GC Boolean domains; synthetic arrays and metadata only."""
from collections import Counter

import numpy as np
import pytest

from gx1.contracts import entry_gc_mtf_history_v1 as history
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.features import gc_native_context_v1 as gc


# Independent source-formula inventory, not values inferred from a fit sample.
_SOURCE_COLUMNS = {
    "structure_available", "bos_up", "bos_down", "sweep_up_event", "sweep_down_event",
    "ob_bull_active", "ob_bull_created", "ob_bull_first_retouch",
    "ob_bear_active", "ob_bear_created", "ob_bear_first_retouch",
}
_COUNTS = {"M5": 22, "M15": 21, "H1": 11, "H4": 11, "D1": 11}
_WIDTHS = {"M5": 639, "M15": 493, "H1": 453, "H4": 431, "D1": 399}
_SURFACES = [("signal", "ctx_cont."), ("ctx_cont", ""), ("gc_mtf_d1", "")]


def _expected_values():
    columns = _SOURCE_COLUMNS | {"observed_proxy_" + name for name in _SOURCE_COLUMNS}
    return {item["name"] for item in gc.REGISTRY["fields"] if item["column"] in columns}


def _sparse_pair(prefix, *, continuous=False):
    value = "gc_d1_observed_book_tw_imbalance" if continuous else "gc_d1_observed_proxy_sweep_up_event"
    names = [prefix + value, prefix + value + "_observed"]
    raw = np.zeros((117, 2), dtype=np.float32)
    raw[:, 1] = 1.
    raw[-1, 0] = 1.
    return names, raw


def test_exact_source_mapping_and_existing_masks_are_the_entire_domain():
    expected = _expected_values()
    assert len(expected) == 76
    assert isinstance(gc.SOURCE_BINARY_VALUE_FIELDS, frozenset)
    assert gc.SOURCE_BINARY_VALUE_FIELDS == expected
    assert Counter(history.SOURCE_TABLE_TIMEFRAMES[item["table"]]
        for item in gc.REGISTRY["fields"] if item["name"] in expected) == _COUNTS
    legacy = set(gc.OBSERVED_FIELDS) | gc.STRUCTURAL_VALUE_FIELDS
    assert gc.binary_fields(gc.CONTEXT_FIELDS) == legacy | expected
    assert gc.mtf_history_binary_fields(gc.CONTEXT_FIELDS) == legacy | expected
    aliases = ["ctx_cont." + name for name in gc.CONTEXT_FIELDS]
    assert gc.binary_fields(aliases) == {"ctx_cont." + name for name in legacy | expected}


@pytest.mark.parametrize("tf", list(_COUNTS))
def test_all_tf_names_masks_widths_categories_and_sources_are_preserved(tf):
    layout = history.gc_mtf_history_layout()
    original = [item for item in gc.REGISTRY["fields"] if history.SOURCE_TABLE_TIMEFRAMES[item["table"]] == tf]
    gaps = [item for item in gc.TRAILING_SUPPORT_FIELDS if history.SOURCE_TABLE_TIMEFRAMES[item["table"]] == tf]
    assert len(gaps) == 1
    names = [name for item in original for name in (item["name"], item["availability_field"])] + [gaps[0]["name"]]
    at = {name: i for i, name in enumerate(names)}
    pairs = [[at[item["name"]], at[item["availability_field"]]] for item in original + gaps]
    assert layout["field_names"][tf] == names
    assert layout["widths"][tf] == len(names) == _WIDTHS[tf]
    assert layout["value_mask_pairs"][tf] == pairs
    assert layout["source_fields"][tf] == original + gaps
    assert layout["categorical_domains"][tf] == {}
    assert layout["source_registry"] == {"path": gc.REGISTRY_PATH, "sha256": gc.REGISTRY_SHA256}
    assert gc.CONTEXT_FIELDS == tuple(name for item in gc.REGISTRY["fields"]
        for name in (item["name"], item["availability_field"]))
    legacy = set(gc.OBSERVED_FIELDS) | gc.STRUCTURAL_VALUE_FIELDS
    expected = (legacy | _expected_values()).intersection(names)
    assert set(layout["binary_fields"][tf]) == expected
    assert len(_expected_values().intersection(names)) == _COUNTS[tf]
    assert gaps[0]["name"] not in expected


@pytest.mark.parametrize("surface,prefix", _SURFACES)
def test_one_positive_of_117_is_binary_identity_in_base_and_gc(surface, prefix):
    names, raw = _sparse_pair(prefix)
    stats = norm.fit_surface_normalization(raw, surface=surface, field_names=names)
    norm.require_surface_normalization(stats, surface=surface, field_names=names)
    assert stats["binary_mask"] == [1, 1]
    assert stats["center"] == [0., 0.] and stats["scale"] == [1., 1.]
    assert stats["scale_source"] == ["semantic_binary_identity", "semantic_binary_identity"]
    assert stats["train_transformed_min"][0] == 0. and stats["train_transformed_max"][0] == 1.
    assert np.array_equal(norm.apply_surface_normalization(raw, stats), raw)
    missing = raw[:1].copy()
    missing[:, :] = 0.
    assert np.array_equal(norm.apply_surface_normalization(missing, stats), missing)


@pytest.mark.parametrize("surface,prefix", _SURFACES)
@pytest.mark.parametrize("stage", ["fit", "apply"])
def test_value_two_is_rejected_by_unchanged_binary_domain_guards(surface, prefix, stage):
    names, raw = _sparse_pair(prefix)
    if stage == "fit":
        raw[0, 0] = 2.
        with pytest.raises(RuntimeError, match="BINARY_DOMAIN_INVALID"):
            norm.fit_surface_normalization(raw, surface=surface, field_names=names)
    else:
        stats = norm.fit_surface_normalization(raw, surface=surface, field_names=names)
        raw[0, 0] = 2.
        with pytest.raises(RuntimeError, match="BINARY_VALUE_INVALID"):
            norm.apply_surface_normalization(raw, stats)


@pytest.mark.parametrize("surface,prefix", _SURFACES)
def test_zero_observations_remain_blocked_even_for_declared_binary(surface, prefix):
    names, raw = _sparse_pair(prefix)
    raw[:, :] = 0.
    with pytest.raises(RuntimeError, match="ENTRY_INPUT_NO_PHYSICAL_TRAIN_OBSERVATIONS"):
        norm.fit_surface_normalization(raw, surface=surface, field_names=names)


@pytest.mark.parametrize("surface,prefix", _SURFACES)
def test_same_sparse_genuine_continuous_input_remains_unscaleable(surface, prefix):
    names, raw = _sparse_pair(prefix, continuous=True)
    assert names[0] not in gc.binary_fields(names)
    with pytest.raises(RuntimeError, match="ENTRY_INPUT_NORMALIZATION_UNSCALEABLE"):
        norm.fit_surface_normalization(raw, surface=surface, field_names=names)


def test_unrelated_continuous_categorical_and_unregistered_names_do_not_change():
    names = ["gc_d1_observed_book_tw_imbalance", "smc_swing_state", "smc_sweep_up_event",
        "gc_d1_fake_sweep_up_event", "gc_m30_observed_proxy_sweep_up_event",
        "gc_d1_observed_proxy_ob_bear_age_bars", "gc_d1_observed_profile_rolling15_vwap_usd"]
    names += ["ctx_cont." + name for name in names]
    assert gc.binary_fields(names) == set()
    assert gc.mtf_history_binary_fields(names) == set()
    assert norm.SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS["smc_swing_state"] == (0, 1, 2, 3, 4)
