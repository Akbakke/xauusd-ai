"""Build the exact current 4003-field tabular view over immutable 253-field caches.

All old cells are copied in their original name order. The twelve new cells
come from qualified causal span sidecars with exact source-row/time joins.
No labels, statistics fit, model, or cache mutation occurs here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.audit import qualify_market_state_ta_anchor_span_inputs_v1 as old_source
from gx1.audit import qualify_market_state_ta_fivefold_span_extension_v1 as spans
from gx1.contracts import market_state_tabular_feature_view_v1 as tabular
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.contracts.entry_exit_feature_surface_v1 import load_m1_feature_surface_times
from gx1.features import htf_features as htf
from gx1.features.entry_specialist_feature_groups_v1 import compile_compact_entry_execution_profile

TF = tabular.TIMEFRAMES
SPAN_SIGNAL = ('chart.geomline_above_anchor_span_bars',
               'chart.geomline_below_anchor_span_bars')
SPAN_MTF = ('geomline_above_anchor_span_bars',
            'geomline_below_anchor_span_bars')


def need(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError('CURRENT_TABULAR_SPAN_' + code)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_insert(old: np.ndarray, addition: np.ndarray,
                   names: list[str], extra: tuple[str, str]) -> np.ndarray:
    """Copy every old float32 bit and insert the two named span columns."""
    need(isinstance(old, np.ndarray) and old.dtype == np.float32
         and old.ndim == 2 and isinstance(addition, np.ndarray)
         and addition.dtype == np.float32 and addition.shape == (len(old), 2)
         and len(names) == old.shape[1] + 2
         and all(names.count(name) == 1 for name in extra), 'INSERT_SHAPE')
    old_positions = [i for i, name in enumerate(names) if name not in extra]
    extra_positions = [names.index(name) for name in extra]
    out = np.empty((len(old), len(names)), dtype=np.float32)
    out[:, old_positions] = old
    out[:, extra_positions] = addition
    need(np.array_equal(np.ascontiguousarray(out[:, old_positions]).view('<u4'),
                        np.ascontiguousarray(old).view('<u4')),
         'OLD_BITS')
    return out


def build(provider, provider_plan: dict, info: dict, *, old_plan: dict,
          span_result: dict, old_schema: dict) -> tuple[tabular.MarketStateTabularFeatureView, dict]:
    """Read only the model-used source intervals and construct the current view."""
    profile = compile_compact_entry_execution_profile()['physical_field_names']
    signal_names = list(profile['signal'])
    mtf_names = list(profile['mtf'])
    old_signal_names = [name for name in signal_names if name not in SPAN_SIGNAL]
    old_mtf_names = [name for name in mtf_names if name not in SPAN_MTF]
    current_names = (['signal.' + name for name in signal_names]
                     + ['ctx_cat.' + name for name in profile['ctx_cat']]
                     + ['mtf_' + tf.lower() + '.' + name
                        for tf in TF for name in mtf_names])
    old_names = [name for name in current_names if not name.endswith('_anchor_span_bars')]
    need(len(current_names) == 4003 and len(old_names) == 3991
         and old_schema['feature_names'] == old_names
         and old_schema['width'] == 3991
         and old_schema['no_future_targets_in_features'] is True,
         'FEATURE_NAMES')
    cache_dir = Path(provider_plan['native_cache_dir'])
    manifest_path = cache_dir / 'manifest.json'
    need(manifest_path.is_file() and sha(manifest_path) ==
         provider_plan['base_lineage']['mtf_cache_manifest_sha256'],
         'LEGACY_MANIFEST_HASH')
    manifest = json.loads(manifest_path.read_text())
    need(manifest['feature_count'] == len(old_mtf_names) == 253
         and manifest['feature_names'] == old_mtf_names
         and list(htf.MULTI_TF_PER_BAR_FEATURES_V4) == mtf_names,
         'LEGACY_TO_CURRENT_FIELDS')
    axes, expected_axes = provider.read_axes(provider_plan, info, np)
    groups = provider.map_reuse(info['reuse'], np)
    source, current = groups['source050'], groups['current051']
    need(source['signal'].shape == tuple(old_plan['local_signal']['shape'])
         and old_source.stat(old_plan['local_signal']['path']) ==
             old_plan['local_signal']['stat']
         and current['snap'].shape == (len(axes['parent_rows']), len(old_signal_names))
         and current['snap'].dtype == np.float32,
         'LEGACY_SIGNAL_SHAPE')
    clock_path = Path(old_plan['local_clock_path'])
    need(str(clock_path) == info['source']['output_parquet']
         and old_source.stat(clock_path) == old_plan['local_clock_stat'],
         'SOURCE_CLOCK_IDENTITY')
    source_times = load_m1_feature_surface_times(
        clock_path, context='CURRENT_TABULAR_SPAN_SOURCE_CLOCK',
        expected_bar_seconds=300)
    source_ns = source_times.asi8
    positions = np.searchsorted(source_ns, axes['bar_start_ns'])
    need(len(source_ns) == provider.SOURCE_ROWS
         and np.all(positions >= 0) and np.all(positions < len(source_ns))
         and np.array_equal(source_ns[positions], axes['bar_start_ns'])
         and np.all(np.diff(positions) > 0), 'SOURCE_MODEL_JOIN')
    sidecars = {}
    for lane in ('local', *TF):
        item = span_result['lanes'][lane]['full_artifact']
        need(spans.ref(Path(item['path'])) == item, 'SIDECAR_HASH:' + lane)
        with np.load(item['path'], allow_pickle=False) as data:
            need(set(data.files) == spans.KEYS, 'SIDECAR_KEYS:' + lane)
            sidecars[lane] = {key: data[key].copy() for key in spans.KEYS}
    local = sidecars['local']
    start, end = int(local['row_indices'][0]), int(local['row_indices'][-1]) + 1
    need(np.array_equal(local['row_indices'], np.arange(start, end, dtype=np.int64))
         and np.all((positions >= start) & (positions < end))
         and np.array_equal(local['times_ns'][positions - start], axes['bar_start_ns']),
         'LOCAL_SPAN_JOIN')
    signal = ordered_insert(np.asarray(current['snap'], dtype=np.float32),
                            local['anchor_span_bars'][positions - start],
                            signal_names, SPAN_SIGNAL)
    cache = {}
    bounds = {'local': [start, end]}
    for tf in TF:
        lane = sidecars[tf]
        first, stop = int(lane['row_indices'][0]), int(lane['row_indices'][-1]) + 1
        need(np.array_equal(lane['row_indices'], np.arange(first, stop, dtype=np.int64))
             and manifest['tfs'][tf]['causal_warmup_rows'] <= first,
             'SPAN_SOURCE_ROWS:' + tf)
        refs = old_plan['caches'][tf]
        old_ts_path = Path(refs['ts']['path'])
        old_feats_path = Path(refs['feats']['path'])
        need(sha(old_ts_path) == refs['ts']['sha256']
             and sha(old_feats_path) == refs['feats']['sha256'],
             'OLD_CACHE_HASH:' + tf)
        old_ts = np.load(old_ts_path, mmap_mode='r', allow_pickle=False)
        old_feats = np.load(old_feats_path, mmap_mode='r', allow_pickle=False)
        need(old_feats.dtype == np.float32 and old_feats.shape[1] == 253
             and old_ts.dtype == np.int64 and old_ts.shape[0] == old_feats.shape[0]
             and stop <= len(old_ts)
             and np.array_equal(old_ts[first:stop], lane['times_ns']),
             'TF_SOURCE_JOIN:' + tf)
        values = ordered_insert(np.asarray(old_feats[first:stop]),
                                lane['anchor_span_bars'], mtf_names, SPAN_MTF)
        frame = pd.DataFrame(values, columns=mtf_names)
        frame.attrs['htf_feature_contract'] = htf.HTF_V4_MATRIX_CONTRACT
        frame.attrs['ts_int64'] = lane['times_ns']
        frame.attrs['feats_np'] = values
        frame.attrs['causal_warmup_rows'] = 0
        cache[tf] = frame
        bounds[tf] = [first, stop]
    view = tabular.MarketStateTabularFeatureView(
        parent_rows=axes['parent_rows'], bar_start_ns=axes['bar_start_ns'],
        decision_ns=axes['decision_ns'], expected_axes=expected_axes,
        signal=signal, ctx_cont=current['ctx_cont'], ctx_cat=current['ctx_cat'],
        multi_tf_sources=cache)
    need(view.width == 4003 and list(view.feature_names) == current_names
         and view.parent_rows.shape[0] == old_schema['rows']
         and tabular._axis_hash(view.parent_rows) ==
             old_schema['model_parent_rows_sha256']
         and tabular._axis_hash(view.decision_ns) ==
             old_schema['model_decision_ns_sha256'],
         'CURRENT_VIEW_IDENTITY')
    return view, {'source_bounds': bounds, 'old_feature_count': 3991,
                  'new_feature_count': 12, 'current_feature_count': 4003,
                  'source_rows': len(view.parent_rows),
                  'old_cache_manifest_sha256': sha(manifest_path),
                  'no_future_targets_in_features': True}
