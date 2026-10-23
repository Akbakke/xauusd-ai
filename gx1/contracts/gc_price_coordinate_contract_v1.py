"""Source-only GC price-coordinate map; no array transformation or execution authority.

204 classifies the complete immutable049 value inventory. Existing independent
unit owners supply units; only the closed producer-derived sets below identify
absolute levels. Availability markers are retained separately, never inferred
from names. No native input, normalization, target or model function is changed.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path

SCHEMA_VERSION = "gx1_gc_price_coordinate_source_map_v1"
REGISTRY_PATH = "configs/research/NATIVE_GC_CONTEXT_FIELDS_20261018_049.json"
REGISTRY_SHA256 = "3e444777099a848cd8795688b57a19122d1ac5454f42902a3495e64aad73a757"
UNIT_OWNER_PATH = "gx1/scripts/audit_entry_gc_foundation_v1.py"
UNIT_OWNER_FUNCTIONS = (
    "event_book_reference_units", "trade_reference_units", "smc_reference_units",
    "profile_reference_units", "profile_dynamic_units",
)
SOURCE_SHA256 = {
    REGISTRY_PATH: REGISTRY_SHA256,
    UNIT_OWNER_PATH: "48c69ae0608b772db2c22211d39c24728ee0f4bb9abaf901b98303931f099c60",
    "gx1/features/gc_native_context_v1.py": "81d37563a5a473d7610aaff6d8eee71eeb9f6b8f3c95b454d1bf6ca6597b373c",
    "gx1/features/gc_order_flow_v1.py": "a1a40ed3516b22a7142ac9125f15d6bed1b4ce5c82cedd60f78a98ec5097d66d",
    "gx1/features/smc_v1.py": "56b9fe479e1cdb7dc524179b65e585d890cdd2fef128d47a6cada00692b94d9a",
    "gx1/contracts/gc_continuous_source_v1.py": "941ec83d463b99209ef77046e1f2cf36458a9d45fb9596cceb0494541ac4b964",
    "gx1/scripts/rebuild_native_gc_inputs_v1.py": "d44a0a1a5b14dd18d8e619f4782650009e475a15103f65e368cfcd8c9109ba67",
    "gx1/scripts/gc_order_flow_features_v1.py": "8ad8c50ad0427a7c45ac86a45a963da044e131019bf57021341ec04489e7b81f",
}
_REPO = Path(__file__).resolve().parents[2]

# Closed products of the actual producer's declared arguments/outputs. These are
# identities, not name patterns. In particular, USD units alone never imply level.
CANDLE_LEVEL_COLUMNS = frozenset(prefix + name
    for prefix in ("", "observed_")
    for name in ("open_usd", "high_usd", "low_usd", "close_usd"))
OB_LEVEL_COLUMNS = frozenset(prefix + "ob_" + side + "_" + edge
    for prefix in ("", "observed_proxy_")
    for side in ("bull", "bear") for edge in ("low", "high"))
HISTOGRAM_LEVEL_COLUMNS = frozenset(
    [prefix + name for prefix in ("", "observed_")
     for name in ("vwap_usd", "poc_usd", "q25_usd", "median_usd", "q75_usd")]
    + ["observed_val_usd", "observed_vah_usd", "buy_poc_usd", "sell_poc_usd",
       "observed_buy_known_poc_usd", "observed_sell_known_poc_usd"])
DYNAMIC_PROFILE_LEVEL_COLUMNS = frozenset(prefix + family + "_" + name
    for prefix in ("", "observed_profile_")
    for family in ("rolling15", "rolling60", "anchored_segment", "session_observed")
    for name in ("vwap_usd", "poc_usd", "val_usd", "vah_usd"))
USD_LEVEL_COLUMNS = (CANDLE_LEVEL_COLUMNS | OB_LEVEL_COLUMNS
    | HISTOGRAM_LEVEL_COLUMNS | DYNAMIC_PROFILE_LEVEL_COLUMNS)
TICK_LEVEL_COLUMNS = frozenset((
    "observed_book_opening_mid_tick", "observed_book_closing_mid_tick"))
SOURCE_DECISION_AGE_COLUMNS = frozenset((
    "observed_profile_snapshot_age_minutes", "observed_spot_basis_age_minutes"))
PROJECTED_M5_PROFILE_LEVEL_COLUMNS = frozenset("observed_profile_" + family + "_" + name
    for family in ("rolling15", "rolling60", "anchored_segment", "session_observed")
    for name in ("vwap_usd", "poc_usd", "val_usd", "vah_usd"))
PRICE_PRODUCER_EVIDENCE = {
    "candle": ["gc_order_flow_v1.py:82-101", "gc_order_flow_v1.py:120-184",
        "rebuild_native_gc_inputs_v1.py:643-667"],
    "order_block": ["gc_order_flow_v1.py:299-317", "gc_order_flow_v1.py:335-400",
        "smc_v1.py:1388-1468"],
    "histogram": ["gc_order_flow_v1.py:34-66", "gc_order_flow_v1.py:219-234"],
    "dynamic_profile": ["gc_order_flow_v1.py:946-950", "gc_order_flow_v1.py:978-1066"],
    "book_mid": ["gc_order_flow_v1.py:744-770", "gc_order_flow_v1.py:879-885",
        "gc_order_flow_v1.py:545-563"],
}


def _require(condition, reason):
    if not condition:
        raise RuntimeError("GC_PRICE_COORDINATE_" + reason)


def _read_bound_source(relative):
    path = _REPO / relative
    _require(path.is_file() and not path.is_symlink(), "SOURCE_FILE:" + relative)
    data = path.read_bytes()
    _require(hashlib.sha256(data).hexdigest() == SOURCE_SHA256[relative],
        "SOURCE_SHA256:" + relative)
    return data


def _merge_units(units, evidence, incoming, owner):
    for name, unit in incoming.items():
        _require(name not in units or units[name] == unit, "UNIT_CONFLICT:" + name)
        units[name] = unit
        evidence.setdefault(name, []).append(owner)


def _source_column_units(reference):
    units, evidence = {}, {}
    for name in UNIT_OWNER_FUNCTIONS:
        _merge_units(units, evidence, getattr(reference, name)(), name)
    # The retained strict M5 book surface uses these exact unprefixed names.
    # Its formulas are rebuild_native_gc_inputs_v1.legacy_strict_m5_surface.
    book_names = (*reference.GC_BOOK_REFERENCE_ADDITIVE,
        *reference.GC_BOOK_REFERENCE_MEANS, "mid_response_ticks")
    order_names = tuple(reference.GC_ORDER_REFERENCE_MEANS)
    _merge_units(units, evidence,
        {name: units["observed_book_" + name] for name in book_names},
        "event_book_reference_units / legacy_strict_m5_surface")
    _merge_units(units, evidence,
        {name: units["observed_order_" + name] for name in order_names},
        "event_book_reference_units / legacy_strict_m5_surface")
    _merge_units(units, evidence,
        {name: "binary" for name in ("ofi_valid", "bbo_valid", "order_count_valid")},
        "legacy_strict_m5_surface:649-652")
    # Existing basis reference declares these units inside its array-taking
    # reference function; do not invoke that function to obtain metadata.
    _merge_units(units, evidence, {"spot_basis_bps": "bps",
        "observed_spot_basis_bps": "bps", "observed_spot_basis_age_minutes": "minutes"},
        "reference_basis_aggregates:2201-2205")
    return units, evidence


def _coordinate_kind(column):
    if column in USD_LEVEL_COLUMNS:
        return "absolute_price_level_usd"
    if column in TICK_LEVEL_COLUMNS:
        return "absolute_price_level_ticks"
    return "unchanged"


def _price_evidence(column):
    for names, owner in ((CANDLE_LEVEL_COLUMNS, "candle"), (OB_LEVEL_COLUMNS, "order_block"),
            (HISTOGRAM_LEVEL_COLUMNS, "histogram"),
            (DYNAMIC_PROFILE_LEVEL_COLUMNS, "dynamic_profile"), (TICK_LEVEL_COLUMNS, "book_mid")):
        if column in names:
            return owner
    return None


def _compile_fields(fields, units, evidence):
    _require(isinstance(fields, list) and bool(fields), "REGISTRY_FIELDS")
    records, names, masks = [], set(), set()
    for index, item in enumerate(fields):
        name, column, mask = item["name"], item["column"], item["availability_field"]
        _require(name not in names and mask not in masks and name != mask,
            "DUPLICATE_FIELD_OR_MASK:" + name)
        _require(column in units and column in evidence, "UNIT_UNDECLARED:" + column)
        names.add(name); masks.add(mask)
        kind = _coordinate_kind(column)
        _require(kind != "absolute_price_level_usd" or units[column] == "USD", "USD_UNIT:" + column)
        _require(kind != "absolute_price_level_ticks" or units[column] == "ticks", "TICK_UNIT:" + column)
        record = {"registry_index": index, "name": name, "availability_field": mask,
            "table": item["table"], "column": column, "owner": item["owner"],
            "source_role": item.get("role"), "source_unit": units[column],
            "coordinate_kind": kind, "unit_evidence": list(evidence[column]),
            "price_producer_evidence": _price_evidence(column),
            "source_clock_column": item.get("source_clock_column") or "available_ns",
            "validity_column": item.get("validity_column"),
            "source_transform": item.get("transform") or (
                "decision_age_minutes" if column in SOURCE_DECISION_AGE_COLUMNS else None)}
        if kind != "unchanged":
            if column in PROJECTED_M5_PROFILE_LEVEL_COLUMNS:
                record["contract_provenance"] = {
                    "mode": "exact_m5_profile_snapshot_join",
                    "source_table": "TRADE_FEATURES_M5.parquet",
                    "source_clock_column": "available_ns", "contract_column": "contract_index",
                    "join_clock_column": "observed_profile_snapshot_available_ns",
                    "outer_tf_contract_is_authority": False}
            else:
                record["contract_provenance"] = {"mode": "selected_source_table_row",
                    "source_table": item["table"], "contract_column": "contract_index"}
        records.append(record)
    _require(not names.intersection(masks), "VALUE_MASK_COLLISION")
    return records


def build_gc_price_coordinate_source_contract():
    """Return the complete source-bound JSON catalog; never read market payloads.

    The existing independent unit owners are imported lazily and only their five
    metadata functions are called. No fit, array replay or native forward occurs.
    This map does not implement log(P/A), normalization or anchor selection.
    """
    bound = {name: _read_bound_source(name) for name in SOURCE_SHA256}
    registry = json.loads(bound[REGISTRY_PATH])
    reference = importlib.import_module("gx1.scripts.audit_entry_gc_foundation_v1")
    _require(Path(reference.__file__).resolve() == (_REPO / UNIT_OWNER_PATH).resolve(), "UNIT_OWNER_PATH")
    units, evidence = _source_column_units(reference)
    fields = _compile_fields(registry["fields"], units, evidence)
    counts = dict(Counter(item["coordinate_kind"] for item in fields))
    _require(len(fields) == 1205 and len({item["column"] for item in fields}) == 318,
        "COMPLETE_REGISTRY_COUNT")
    _require(counts == {"absolute_price_level_usd": 232,
        "absolute_price_level_ticks": 10, "unchanged": 963}, "COMPLETE_KIND_COUNT")
    _require({item["column"] for item in fields if item["coordinate_kind"] != "unchanged"}
        == USD_LEVEL_COLUMNS | TICK_LEVEL_COLUMNS, "COMPLETE_LEVEL_SET")
    levels = [item["name"] for item in fields if item["coordinate_kind"] != "unchanged"]
    _require("gc_book_close_usd" in levels, "ANCHOR_FIRST_SOURCE")
    priority = ["gc_book_close_usd"] + [name for name in levels if name != "gc_book_close_usd"]
    return {"schema_version": SCHEMA_VERSION, "scope": "static_source_map_only",
        "numerical_transformation_implemented": False, "execution_authorized": False,
        "source_sha256": dict(SOURCE_SHA256), "unit_owner_functions": list(UNIT_OWNER_FUNCTIONS),
        "fields": fields, "field_count": len(fields), "coordinate_kind_counts": counts,
        "source_column_count": 318, "source_unit_counts": dict(sorted(Counter(
            item["source_unit"] for item in fields).items())),
        "level_source_column_counts": {"USD": len(USD_LEVEL_COLUMNS), "ticks": len(TICK_LEVEL_COLUMNS)},
        "price_producer_evidence": dict(PRICE_PRODUCER_EVIDENCE),
        "anchor_source_priority": priority, "anchor_unobserved_source_id": -1,
        "anchor_source_id_semantics": "zero-based index into fixed priority; provenance, not model input",
        "anchor_policy": "book-close first, then registry order over complete observed positive levels; invalid observed levels fail",
        "tick_to_usd": {"formula": "level_ticks * tick_size / PRICE_SCALE",
            "price_scale": 1000000000, "tick_size_source": "bound ROLL_SCHEDULE.metadata_identity[*][2]",
            "uniform_tick_size_guard": "rebuild_native_gc_inputs_v1.py:898-902",
            "price_scale_owner": "gc_order_flow_v1.py:21",
            "positive_price_owner": "gc_continuous_source_v1.py:90-92,301-305",
            "midpoint_grid": "BBO midpoints may be half ticks; no integer-midpoint requirement"},
        "future_adapter_requirements": {
            "all_values_and_original_masks": "canonical F32 value/mask parity before any transform; aliases once",
            "required_numeric_provenance": ["chosen canonical field ID and fixed source identity",
                "bound decision clock", "own original availability mask", "bound tick scale"],
            "per_row_contract_and_absolute_source_clock": "not stored in raw snap/ctx or198; unknown unless separately projected",
            "row_contract_metadata_required_for_numeric_transform": False,
            "optional_metadata_projection": ["selected source table row", "source available_ns",
                "actual source contract_index"],
            "projected_profile_contract": "if reconstructed, join exact M5 snapshot clock to M5 contract; outer TF can differ at roll",
            "metadata_availability": "retained GC tables preserve clocks/contracts; no fresh metadata projection is claimed",
            "same_contract_or_tradable_support_resistance_claimed": False,
            "clock_meaning": "state availability is not necessarily price-origin event time (OB/profile carry)",
            "tick_size": "fresh bound roll metadata, positive and identical across source contracts",
            "no_observed_levels": "anchor unavailable; no fabricated price or changed original masks",
            "observed_invalid_level": "hard failure, never silently remove observation",
            "normalization_and_roundtrip": "separate reviewed numerical design; not implemented here"}}
