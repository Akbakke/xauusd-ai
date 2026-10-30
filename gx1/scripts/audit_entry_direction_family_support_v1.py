"""Streaming current-profile input audit; zero model/target/fit operations.

This module is an adapter for the existing qualified physical provider and
capped readguard/receipt harness. It deliberately supplies no standalone launch
or file loader. The harness authenticates source/payload manifests, prepares
immutable inputs, binds this plan once, and publishes only aggregate output.
"""
from __future__ import annotations
import hashlib
import time
import numpy as np
from gx1.contracts import entry_direction_family_intervention_v1 as family
from gx1.contracts import entry_gc_mtf_history_v1 as history
from gx1.contracts import entry_model_native_input_normalization_v1 as normalization
from gx1.contracts import entry_native_direction_feature_view_v1 as view_owner

SCHEMA_VERSION = 'gx1_direction_family_support_plan_v1'
PLAN_KEYS = frozenset({'schema_version', 'mode', 'train_parent_rows', 'later_parent_rows',
    'expected_train_rows', 'expected_later_rows', 'train_cutoff_ns', 'later_cutoff_ns',
    'feature_view_sha256', 'intervention_sha256', 'max_batch_rows', 'max_wall_seconds', 'contract_sha256'})


def require(ok, reason):
    if not ok:
        raise RuntimeError('FAMILY_SUPPORT_' + reason)


def axis_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype='<i8').tobytes()).hexdigest()


class SurfaceSupport:
    """Fixed-width accumulators, including exact unique physical-row support."""
    def __init__(self, names, mask_pairs):
        self.names = list(names); self.width = len(names); self.pairs = list(mask_pairs)
        self.parents = 0; self.cells = 0; self.unique_rows = set(); self.digest = hashlib.sha256()
        self.count = np.zeros(self.width, np.int64); self.nonzero = self.count.copy()
        self.normalized_nonzero = self.count.copy(); self.unique_count = self.count.copy()
        self.unique_nonzero = self.count.copy(); self.outside_train = self.count.copy()
        self.minimum = np.full(self.width, np.inf); self.maximum = np.full(self.width, -np.inf)
        self.history_min = np.full(self.width, np.iinfo(np.int64).max, np.int64)
        self.history_max = np.zeros(self.width, np.int64)
        self.train_supported = None

    def update(self, raw, normalized, physical_rows, *, train_range=None):
        require(raw.shape == normalized.shape and raw.shape[-1] == self.width
            and raw.ndim in (2, 3) and np.isfinite(raw).all() and np.isfinite(normalized).all(), 'SURFACE_SHAPE_FINITE')
        require(physical_rows.shape == raw.shape[:-1] and physical_rows.dtype == np.int64
            and np.all(physical_rows >= 0), 'PHYSICAL_ROW_IDENTITY')
        observed = np.ones(raw.shape, dtype=bool)
        for value, mask in self.pairs:
            require(np.isin(raw[..., mask], (0, 1)).all() and np.all(raw[..., value][raw[..., mask] == 0] == 0), 'MASKED_STORAGE')
            observed[..., value] = raw[..., mask] == 1
        flat, available, z = (x.reshape(-1, self.width) for x in (raw, observed, normalized))
        self.parents += raw.shape[0]; self.cells += len(flat); self.digest.update(np.ascontiguousarray(raw).tobytes())
        self.count += available.sum(axis=0); self.nonzero += ((flat != 0) & available).sum(axis=0)
        self.normalized_nonzero += ((z != 0) & available).sum(axis=0)
        self.minimum = np.minimum(self.minimum, np.where(available, flat, np.inf).min(axis=0))
        self.maximum = np.maximum(self.maximum, np.where(available, flat, -np.inf).max(axis=0))
        per_parent = available.reshape(raw.shape[0], -1, self.width).sum(axis=1)
        self.history_min = np.minimum(self.history_min, per_parent.min(axis=0))
        self.history_max = np.maximum(self.history_max, per_parent.max(axis=0))
        ids, first = np.unique(physical_rows.reshape(-1), return_index=True)
        fresh = np.fromiter((int(i) not in self.unique_rows for i in ids), dtype=bool, count=len(ids))
        ix = first[fresh]; self.unique_rows.update(int(i) for i in ids[fresh])
        self.unique_count += available[ix].sum(axis=0)
        self.unique_nonzero += ((flat[ix] != 0) & available[ix]).sum(axis=0)
        if train_range is not None:
            low, high, supported = train_range
            self.train_supported = supported
            self.outside_train += (available & supported & ((flat < low) | (flat > high))).sum(axis=0)

    def finish(self):
        seen = self.count > 0
        return {'parent_rows': self.parents, 'consumed_physical_row_occurrences': self.cells,
            'unique_physical_rows': len(self.unique_rows), 'unique_physical_rows_sha256': axis_sha(sorted(self.unique_rows)),
            'raw_consumed_bytes_sha256': self.digest.hexdigest(),
            'observed_consumed': self.count.tolist(), 'missing_consumed': (self.cells - self.count).tolist(),
            'nonzero_observed_consumed': self.nonzero.tolist(), 'nonzero_normalized_observed_consumed': self.normalized_nonzero.tolist(),
            'observed_unique_physical': self.unique_count.tolist(), 'nonzero_observed_unique_physical': self.unique_nonzero.tolist(),
            'observed_min': [float(x) if ok else None for x, ok in zip(self.minimum, seen)],
            'observed_max': [float(x) if ok else None for x, ok in zip(self.maximum, seen)],
            'constant_when_observed': [bool(a == b) if ok else None for a, b, ok in zip(self.minimum, self.maximum, seen)],
            'observed_history_rows_per_parent_min': self.history_min.tolist(),
            'observed_history_rows_per_parent_mean': (self.count / self.parents).tolist(),
            'observed_history_rows_per_parent_max': self.history_max.tolist(),
            'outside_observed_train_range_consumed': None if self.train_supported is None else [int(n) if ok else None for n, ok in zip(self.outside_train, self.train_supported)]}


def _physical_rows(key, raw, prepared, selected):
    witness = raw['witness']
    if key == 'seq_x':
        positions = np.asarray(witness['source_positions'], np.int64)
        length = raw['features'][key].shape[1]
        return positions[:, None] - np.arange(length - 1, -1, -1, dtype=np.int64)[None, :]
    if key == 'snap_x':
        return np.asarray(witness['source_positions'], np.int64)
    if key in ('ctx_cont', 'ctx_cat'):
        return selected
    tf = key.rsplit('_', 1)[-1].upper()
    return np.asarray([x['native_row_indices'] for x in witness['multi_tf'][tf]], np.int64)


def _stats_for(key, base_normalization, gc_normalization):
    if key == 'ctx_cat': return None
    if key.startswith('gc_seq_'):
        return gc_normalization['surfaces']['gc_mtf_' + key.removeprefix('gc_seq_')]
    surface = 'signal' if key in ('seq_x', 'snap_x') else 'ctx_cont' if key == 'ctx_cont' else 'mtf_' + key.removeprefix('seq_')
    return base_normalization['surfaces'][surface]


def audit_prepared_family_support(prepared, view, intervention_contract, *, plan, expected_plan_sha256,
                                  base_normalization, gc_normalization, release_pages=None):
    """One bounded streaming pass over declared inputs, no model/targets needed.

    Pilot selection must be predeclared and totals at most 256 parents. Full
    mode consumes every exact TRAIN-prefix and Q1 parent. The wrapper must use
    existing capped execution, authenticated immutable payloads and bookends.
    """
    start = time.monotonic()
    require(set(plan) == PLAN_KEYS and plan['schema_version'] == SCHEMA_VERSION
        and plan['contract_sha256'] == expected_plan_sha256
        and family._sha({k: v for k, v in plan.items() if k != 'contract_sha256'}) == expected_plan_sha256, 'PLAN_IDENTITY')
    view_owner._require_view(view)
    family.require_family_intervention(intervention_contract, expected_sha256=plan['intervention_sha256'])
    require(view['contract_sha256'] == plan['feature_view_sha256']
        and view['entry_execution_profile'] == intervention_contract['profile_reference']
        and view['prefix_normalization']['base_sha256'] == base_normalization['contract_sha256'] == intervention_contract['base_normalization_sha256']
        and view['prefix_normalization']['gc_normalization_sha256'] == gc_normalization['contract_sha256'] == intervention_contract['gc_normalization_sha256'], 'INPUT_IDENTITIES')
    require(plan['mode'] in ('pilot', 'full') and type(plan['max_batch_rows']) is int
        and 0 < plan['max_batch_rows'] <= view['max_batch_rows'] and type(plan['max_wall_seconds']) is int
        and plan['max_wall_seconds'] > 0 and plan['expected_train_rows'] == view['prefix']['rows']
        and plan['train_cutoff_ns'] == view['prefix']['cutoff_ns']
        and plan['train_cutoff_ns'] < plan['later_cutoff_ns'] <= history.TEST_BOUNDARY_NS, 'PLAN_SCOPE')
    axes = prepared['arrays']; count = plan['expected_train_rows']; later_count = plan['expected_later_rows']
    require(type(later_count) is int and later_count > 0 and count + later_count <= len(axes['parent_rows']), 'LATER_ROWS')
    for key in ('parent_rows', 'bar_start_ns', 'decision_ns'):
        require(axes[key].dtype == np.int64 and axes[key].shape == (view['full_axes']['rows'],)
            and axis_sha(axes[key]) == view['full_axes'][key + '_sha256'], 'FULL_AXIS:' + key)
    require(np.all(np.diff(axes['parent_rows']) > 0) and np.all(np.diff(axes['decision_ns']) > 0), 'AXIS_ORDER')
    require(all(isinstance(plan[name + '_parent_rows'], list) and all(type(i) is int for i in plan[name + '_parent_rows']) for name in ('train', 'later')), 'SELECTION_INTEGER_IDS')
    selections = {name: np.asarray(plan[name + '_parent_rows'], dtype=np.int64) for name in ('train', 'later')}
    require(all(x.ndim == 1 and len(x) > 0 and np.all(np.diff(x) > 0) for x in selections.values()), 'SELECTIONS')
    complete = {'train': axes['parent_rows'][:count], 'later': axes['parent_rows'][count:count + later_count]}
    for name, selected in selections.items():
        require(np.isin(selected, complete[name]).all(), 'SELECTION_BOUNDARY:' + name)
        if plan['mode'] == 'full': require(np.array_equal(selected, complete[name]), 'FULL_COHORT:' + name)
    if plan['mode'] == 'pilot': require(sum(map(len, selections.values())) <= 256, 'PILOT_BOUND')
    require(np.all(axes['decision_ns'][:count] < plan['train_cutoff_ns'])
        and np.all(axes['decision_ns'][count:count + later_count] >= plan['train_cutoff_ns'])
        and np.all(axes['decision_ns'][count:count + later_count] < plan['later_cutoff_ns']), 'SPLIT_CLOCK')
    expected_fields = {name: intervention_contract['surfaces']['seq_x' if name == 'signal' else 'ctx_cont' if name == 'ctx_cont' else 'seq_' + name.removeprefix('mtf_')]['field_names'] for name in normalization.EXPECTED_SURFACES}
    checked_base = normalization.require_input_normalization_contract(base_normalization, expected_field_names=expected_fields, expected_ctx_cat_names=intervention_contract['surfaces']['ctx_cat']['field_names'])
    checked_gc = history.require_gc_mtf_history_normalization(gc_normalization, layout=prepared['gc']['layout'])
    require(checked_base['contract_sha256'] == intervention_contract['base_normalization_sha256'] and checked_gc['contract_sha256'] == intervention_contract['gc_normalization_sha256'], 'NORMALIZATION_IDENTITIES')
    inventory = intervention_contract['surfaces']
    reports, reference_ranges = {}, {}
    logical_bytes = 0; batches = 0; gather_seconds = 0.0; summarize_seconds = 0.0
    for partition in ('train', 'later'):
        collectors = {key: SurfaceSupport(item['field_names'], item['mask_pairs']) for key, item in inventory.items()}
        selected = selections[partition]
        parent_digest = hashlib.sha256(); decision_digest = hashlib.sha256()
        for left in range(0, len(selected), plan['max_batch_rows']):
            require(time.monotonic() - start < plan['max_wall_seconds'], 'WALL_LIMIT')
            ids = selected[left:left + plan['max_batch_rows']]
            t = time.monotonic()
            raw = view_owner.gather_native_direction_feature_batch(view, parent_rows=axes['parent_rows'],
                bar_start_ns=axes['bar_start_ns'], decision_ns=axes['decision_ns'], entry_sequence_source=prepared['entry_sequence_source'],
                snap_x=prepared['current']['snap'], ctx_cont=prepared['current']['ctx_cont'], ctx_cat=prepared['current']['ctx_cat'],
                multi_tf_sources=prepared['cache'], gc_surfaces=prepared['gc']['surfaces'], gc_layout=prepared['gc']['layout'], batch_parent_rows=ids)
            gather_seconds += time.monotonic() - t; t = time.monotonic()
            family._validate_inputs(raw['features'], intervention_contract)
            require(np.array_equal(np.asarray(raw['witness']['parent_rows'], np.int64), ids), 'GATHER_PARENT_IDENTITY')
            parent_digest.update(ids.astype('<i8', copy=False).tobytes())
            at = np.searchsorted(axes['parent_rows'], ids)
            require(np.array_equal(np.asarray(raw['witness']['decision_ns'], np.int64), axes['decision_ns'][at]) and np.array_equal(np.asarray(raw['witness']['bar_start_ns'], np.int64), axes['bar_start_ns'][at]), 'GATHER_CLOCK_IDENTITY')
            decision_digest.update(np.asarray(raw['witness']['decision_ns'], dtype='<i8').tobytes())
            for key, values in raw['features'].items():
                stats = _stats_for(key, base_normalization, gc_normalization)
                z = values if stats is None else normalization.apply_surface_normalization(values, surface_contract=stats)
                collectors[key].update(values, z, _physical_rows(key, raw, prepared, ids), train_range=reference_ranges.get(key))
                logical_bytes += values.nbytes
            batches += 1; summarize_seconds += time.monotonic() - t
            if release_pages is not None: release_pages()
            del raw
        require(parent_digest.hexdigest() == axis_sha(selected), 'COMPLETE_STREAM')
        reports[partition] = {'selected_parent_rows': len(selected), 'parent_rows_sha256': parent_digest.hexdigest(),
            'decision_ns_sha256': decision_digest.hexdigest(), 'surfaces': {key: c.finish() for key, c in collectors.items()}}
        if partition == 'train':
            reference_ranges = {key: (c.minimum.copy(), c.maximum.copy(), c.count > 0) for key, c in collectors.items()}
            for key, c in collectors.items():
                reports[partition]['surfaces'][key]['outside_observed_train_range_consumed'] = None
    compact_inventory = {key: {k: item[k] for k in ('field_names', 'family_indices', 'preserved_mask_indices',
        'preserved_constant_indices', 'mask_pairs', 'source_constants', 'normalization_sha256')} for key, item in inventory.items()}
    for key, item in compact_inventory.items():
        active = sorted({i for indices in item['family_indices'].values() for i in indices})
        item['active_indices'] = active
        item['parked_indices'] = sorted(set(range(len(item['field_names']))) - set(active))
    return {'schema_version': 'gx1_direction_family_support_result_v1', 'status': 'COMPLETE',
        'plan_sha256': expected_plan_sha256, 'feature_view_sha256': view['contract_sha256'],
        'intervention_sha256': intervention_contract['contract_sha256'], 'mode': plan['mode'],
        'full_population_measured': plan['mode'] == 'full', 'inventory': compact_inventory, 'partitions': reports,
        'raw_gather_logical_bytes': logical_bytes, 'disk_read_bytes': None, 'batches': batches,
        'gather_seconds': gather_seconds, 'summarize_seconds': summarize_seconds, 'elapsed_seconds': time.monotonic() - start,
        'structural_history_padding': {'rows': 0, 'evidence': 'canonical gather fails on insufficient history/warmup; no padding or repeat'},
        'missingness_reason': 'UNKNOWN; masks identify unavailable values, not why unavailable',
        'constant_scope': 'declared source constants distinct from empirically constant observed coordinates; counts distinguish unique rows and repeated history exposure',
        'limits': 'Input/support audit only; no learned value, generalization, source-formula correctness or profitability claim. Pilot TRAIN range is sampled support, not full TRAIN support.',
        'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False}
