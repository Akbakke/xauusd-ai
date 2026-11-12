"""Bounded composition of immutable parent features and qualified anchor spans.

The caller verifies artifact/file hashes before this in-memory owner is used.
Local features are assembled only for requested rows. Only the qualified MTF
row unions are assembled once; the unchanged native window owner slices them.
No feature calculation, normalization, target, padding or resampling occurs.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import hashlib
import json
from pathlib import PurePosixPath

import numpy as np
import pandas as pd

from gx1.features import entry_specialist_feature_groups_v1 as owners
from gx1.features import htf_features as htf
from gx1.contracts.entry_anchor_span_normalization_v1 import SPAN_FIELDS

TIMEFRAMES = ('M5','M15','H1','H4','D1')
LATER_COVERAGE_SCHEMA = 'entry_anchor_span_later_coverage_v1'
LATER_BATCH_INDICES = tuple(i * 1108 // 255 for i in range(256))
LATER_SCORE_START_NS = 1727740800000000000  # 2024-10-01 UTC
LATER_SCORE_END_NS = 1735689600000000000  # 2025-01-01 UTC


def _axis_sha(value):
    return hashlib.sha256(value.astype('<i8', copy=False).tobytes()).hexdigest()


def _require_later_coverage(view, coverage, axes, populations):
    """Bind only the fixed fold1 development panel; this grants no execution.

    The caller must verify the referenced COMPLETE qualification and sidecar
    file hashes. This owner verifies their declared axes and actual arrays.
    Normalization provenance remains the original TRAIN prefix.
    """
    keys = {'schema_version', 'status', 'qualification', 'view_sha256', 'full_axes',
            'normalization_prefix', 'score_parent_start', 'score_rows',
            'score_start_ns', 'score_end_exclusive_ns', 'batch_size', 'batch_indices',
            'selected_parent_rows_sha256', 'selected_decision_ns_sha256',
            'physical_populations'}
    need(type(coverage) is dict and set(coverage) == keys, 'LATER_COVERAGE_KEYS')
    need(coverage['schema_version'] == LATER_COVERAGE_SCHEMA
         and coverage['status'] == 'COMPLETE'
         and coverage['view_sha256'] == view['contract_sha256']
         and coverage['full_axes'] == view['full_axes']
         and coverage['normalization_prefix'] == view['prefix']
         and coverage['physical_populations'] == populations, 'LATER_COVERAGE_BINDING')
    need(view['full_axes']['rows'] == 118249 and view['prefix']['rows'] == 29574
         and view['prefix']['cutoff_ns'] == 1727654400000000000, 'LATER_FOLD1_PREFIX')
    fixed = dict(score_parent_start=29849, score_rows=17753, batch_size=16,
                 score_start_ns=LATER_SCORE_START_NS, score_end_exclusive_ns=LATER_SCORE_END_NS)
    need(all(type(coverage[k]) is int and coverage[k] == v for k, v in fixed.items())
         and type(coverage['batch_indices']) is list
         and all(type(i) is int for i in coverage['batch_indices'])
         and tuple(coverage['batch_indices']) == LATER_BATCH_INDICES, 'LATER_FIXED_PANEL')
    ref = coverage['qualification']
    need(type(ref) is dict and set(ref) == {'path', 'sha256'}
         and type(ref['path']) is str and type(ref['sha256']) is str, 'LATER_QUALIFICATION_REF')
    path = PurePosixPath(ref['path'])
    need(str(path) == ref['path'] and path.is_absolute() and '..' not in path.parts
         and ref['path'].startswith('/home/andre2/GX1_RUNS/') and path.name == 'RESULT.json'
         and len(ref['sha256']) == 64 and all(c in '0123456789abcdef' for c in ref['sha256']),
         'LATER_QUALIFICATION_REF')
    need(type(axes) is dict and set(axes) == {'parent_rows', 'bar_start_ns', 'decision_ns'},
         'LATER_FULL_AXIS_KEYS')
    for name, value in axes.items():
        need(isinstance(value, np.ndarray) and value.dtype == np.int64
             and value.shape == (view['full_axes']['rows'],) and value.flags.c_contiguous
             and not value.flags.writeable and np.all(np.diff(value) > 0)
             and _axis_sha(value) == view['full_axes'][name + '_sha256'], 'LATER_FULL_AXIS:' + name)
    need(np.array_equal(axes['bar_start_ns'] + 300 * 10**9, axes['decision_ns']),
         'LATER_DECISION_CLOCK')
    score_times = axes['decision_ns'][29849:47602]
    need(np.all(score_times >= LATER_SCORE_START_NS) and np.all(score_times < LATER_SCORE_END_NS),
         'LATER_SCORE_INTERVAL')
    positions = (29849 + np.asarray(LATER_BATCH_INDICES, dtype=np.int64)[:, None] * 16
                 + np.arange(16, dtype=np.int64)).reshape(-1)
    parents, decisions = axes['parent_rows'][positions], axes['decision_ns'][positions]
    need(_axis_sha(parents) == coverage['selected_parent_rows_sha256']
         and _axis_sha(decisions) == coverage['selected_decision_ns_sha256'], 'LATER_SELECTED_AXES')
    digest = hashlib.sha256(json.dumps(coverage, sort_keys=True, separators=(',', ':'),
                                       allow_nan=False).encode()).hexdigest()
    return dict(later_parent_positions=owned_readonly(positions, np.int64),
                later_parent_rows=owned_readonly(parents, np.int64),
                later_decision_ns=owned_readonly(decisions, np.int64),
                later_coverage_sha256=digest, later_qualification=(ref['path'], ref['sha256']))


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('ANCHOR_SPAN_FEATURE_INPUT_' + reason)


def owned_readonly(value, dtype):
    result=np.array(value,dtype=dtype,copy=True,order='C')
    result.setflags(write=False)
    return result


def column_plan(parent_names, new_names, output_names):
    """Exactly partition current output names into old and new input columns."""
    parent,new,output=map(list,(parent_names,new_names,output_names))
    need(len(parent)==len(set(parent)) and len(new)==len(set(new))
         and len(output)==len(set(output)) and not set(parent).intersection(new)
         and set(output)<=set(parent)|set(new) and set(new)<=set(output),'COLUMN_NAMES')
    old_at={name:i for i,name in enumerate(parent)}
    new_at={name:i for i,name in enumerate(new)}
    old_dest=[i for i,n in enumerate(output) if n in old_at]
    new_dest=[i for i,n in enumerate(output) if n in new_at]
    return dict(parent_take=[old_at[output[i]] for i in old_dest],parent_dest=old_dest,
                span_take=[new_at[output[i]] for i in new_dest],span_dest=new_dest,width=len(output))


def exact_span_positions(span_rows, span_times, requested_rows, requested_times):
    rows,times=np.asarray(requested_rows),np.asarray(requested_times)
    need(rows.dtype==times.dtype==np.int64 and rows.ndim==times.ndim==1
         and rows.shape==times.shape and len(rows)>0,'REQUESTED_AXES')
    at=np.searchsorted(span_rows,rows)
    need(np.all(at<len(span_rows)) and np.array_equal(span_rows[at],rows)
         and np.array_equal(span_times[at],times),'UNCOVERED_OR_DIFFERENT_CLOCK')
    return at


def combine_selected(parent, parent_positions, spans, *, columns):
    """Copy only selected parent columns; preserve order and float32 bits."""
    ids=np.asarray(parent_positions)
    need(isinstance(parent,np.ndarray) and parent.ndim==2 and parent.dtype==np.float32
         and ids.dtype==np.int64 and ids.ndim==1 and len(ids)>0
         and np.all(ids>=0) and np.all(ids<len(parent))
         and isinstance(spans,np.ndarray) and spans.dtype==np.float32
         and spans.shape==(len(ids),len(columns['span_take'])),'BATCH_HEADERS')
    result=np.empty((len(ids),columns['width']),dtype=np.float32)
    result[:,columns['parent_dest']]=parent[np.ix_(ids,columns['parent_take'])]
    result[:,columns['span_dest']]=spans[:,columns['span_take']]
    need(np.isfinite(result).all(),'NONFINITE_SELECTED_FEATURES')
    return result


@dataclass(frozen=True)
class QualifiedAnchorSpanInputs:
    view_sha256: str
    parent_source: np.ndarray
    parent_times: np.ndarray
    parent_sample_positions: np.ndarray
    parent_snap: np.ndarray
    parent_context: np.ndarray
    parent_categories: np.ndarray
    local_rows: np.ndarray
    local_times: np.ndarray
    local_spans: np.ndarray
    local_columns: dict
    mtf_sources: dict
    mtf_parent_rows: dict
    mtf_backings: dict
    fit_parent_rows: int
    parent_signal_width: int
    later_parent_positions: np.ndarray | None = None
    later_parent_rows: np.ndarray | None = None
    later_decision_ns: np.ndarray | None = None
    later_coverage_sha256: str | None = None
    later_qualification: tuple | None = None

    def require_binding(self, view, *, entry_sequence_source, snap_x, ctx_cont, ctx_cat, multi_tf_sources):
        need(view.get('contract_sha256')==self.view_sha256
             and 'anchor_span_input_extension' in view
             and entry_sequence_source['signal'] is self.parent_source
             and entry_sequence_source['times_ns'] is self.parent_times
             and entry_sequence_source['sample_positions'] is self.parent_sample_positions
             and snap_x is self.parent_snap and ctx_cont is self.parent_context
             and ctx_cat is self.parent_categories and multi_tf_sources is self.mtf_sources,
             'EXACT_VIEW_AND_PARENT_BACKINGS')
        need(all(not x.flags.writeable for x in
                 (self.parent_source,self.parent_times,self.parent_sample_positions,self.parent_snap,
                  self.parent_context,self.parent_categories,self.local_rows,self.local_times,self.local_spans)),
             'READONLY_BACKINGS')
        for tf, (clock, matrix, rows) in self.mtf_backings.items():
            frame = self.mtf_sources[tf]
            need(frame.attrs['ts_int64'] is clock and frame.attrs['feats_np'] is matrix
                 and frame.attrs['anchor_span_parent_row_indices'] is rows
                 and self.mtf_parent_rows[tf] is rows
                 and all(not a.flags.writeable for a in (clock, matrix, rows)),
                 'IMMUTABLE_MTF_BACKINGS:' + tf)

    def require_later_batch(self, positions, parents, decisions):
        """A later adapter accepts only its exact fixed panel, never TRAIN."""
        arrays = (self.later_parent_positions, self.later_parent_rows, self.later_decision_ns)
        need(all(isinstance(a, np.ndarray) and a.dtype == np.int64 and a.ndim == 1
                 and a.shape == arrays[0].shape and not a.flags.writeable for a in arrays)
             and self.later_coverage_sha256 is not None and self.later_qualification is not None,
             'READONLY_LATER_BINDING')
        at = np.searchsorted(self.later_parent_positions, positions)
        need(np.all(at < len(self.later_parent_positions))
             and np.array_equal(self.later_parent_positions[at], positions)
             and np.array_equal(self.later_parent_rows[at], parents)
             and np.array_equal(self.later_decision_ns[at], decisions), 'UNQUALIFIED_LATER_PARENT')

    def local_batch(self, parent, parent_positions, physical_rows, times):
        need(parent is self.parent_source or parent is self.parent_snap,'PARENT_VALUE_OWNER')
        at=exact_span_positions(self.local_rows,self.local_times,physical_rows,times)
        return combine_selected(parent,parent_positions,self.local_spans[at],columns=self.local_columns)

    def original_mtf_rows(self, timeframe, cropped_rows):
        ids=np.asarray(cropped_rows)
        parent=self.mtf_parent_rows[timeframe]
        need(ids.dtype==np.int64 and ids.ndim==1 and np.all(ids>=0) and np.all(ids<len(parent)),
             'MTF_ROW_MAPPING')
        return np.ascontiguousarray(parent[ids],dtype=np.int64)


def build_anchor_span_inputs(view, *, entry_sequence_source, snap_x, ctx_cont, ctx_cat,
                             parent_signal_names, parent_mtf_names, parent_mtf_values,
                             parent_mtf_times, span_payloads, physical_populations,
                             unobserved_units_profile, later_coverage=None, later_axes=None):
    """Compose TRAIN sidecars, or a separately qualified fixed later panel.

    Later envelopes remain contiguous so the existing native window owner sees
    all intervening rows. They never alter the TRAIN normalization cutoff.
    """
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.models.entry_v10.entry_v10_input_normalization import _hash_int64_indices

    views._require_view(view,unobserved_units_profile=unobserved_units_profile)
    need('anchor_span_input_extension' in view and view.get('full_ta_feature_profile') is True,
         'EXPLICIT_TA_EXTENSION')
    need(isinstance(entry_sequence_source,Mapping)
         and set(entry_sequence_source)=={'signal','times_ns','sample_positions','audit'},'LOCAL_SOURCE_KEYS')
    profile=owners.compile_full_ta_entry_execution_profile()
    parent_signal=list(parent_signal_names);parent_mtf=list(parent_mtf_names)
    new_local=['chart.'+name for name in SPAN_FIELDS]
    need(parent_signal==[n for n in profile['source_field_names']['signal'] if n not in new_local]
         and parent_mtf==[n for n in htf.MULTI_TF_PER_BAR_FEATURES_V4 if n not in SPAN_FIELDS]
         and len(parent_signal)==2725 and len(parent_mtf)==253,'EXACT_PARENT_NAMES')
    source,times,positions=(entry_sequence_source[k] for k in ('signal','times_ns','sample_positions'))
    total=view['full_axes']['rows']
    need(isinstance(source,np.ndarray) and source.dtype==np.float32 and source.ndim==2
         and source.shape[1]==2725 and isinstance(times,np.ndarray) and times.dtype==np.int64
         and times.shape==(len(source),) and isinstance(positions,np.ndarray)
         and positions.dtype==np.int64 and positions.shape==(total,),'LOCAL_PARENT_HEADERS')
    for a,width,dtype in ((snap_x,2725,np.float32),(ctx_cont,2542,np.float32),(ctx_cat,1,np.int64)):
        need(isinstance(a,np.ndarray) and a.dtype==dtype and a.shape==(total,width),'CURRENT_PARENT_HEADERS')
    need(all(not a.flags.writeable for a in (source,times,positions,snap_x,ctx_cont,ctx_cat)),
         'PARENT_READONLY')
    need(tuple(parent_mtf_values)==tuple(parent_mtf_times)==TIMEFRAMES
         and set(span_payloads)==set(physical_populations)=={'local',*TIMEFRAMES},'SIX_SURFACES')
    if later_coverage is None:
        need(later_axes is None, 'UNBOUND_LATER_AXES')
        later_binding = {}
        coverage_cutoff = view['prefix']['cutoff_ns']
    else:
        later_binding = _require_later_coverage(view, later_coverage, later_axes, physical_populations)
        coverage_cutoff = LATER_SCORE_END_NS
        need(np.all(positions >= 0) and np.all(positions < len(times))
             and np.array_equal(times[positions], later_axes['bar_start_ns']), 'LATER_PARENT_SOURCE_CLOCK')
    selected={}
    for lane in ('local',*TIMEFRAMES):
        payload=span_payloads[lane];pop=physical_populations[lane]
        need(set(payload)=={'row_indices','times_ns','native_bar_positions','anchor_span_bars'},'SIDECAR_KEYS')
        rows,clocks,native,values=(payload[k] for k in
            ('row_indices','times_ns','native_bar_positions','anchor_span_bars'))
        n=pop['selected_unique_row_count']
        need(rows.dtype==clocks.dtype==native.dtype==np.int64
             and rows.shape==clocks.shape==native.shape==(n,)
             and values.dtype==np.float32 and values.shape==(n,2)
             and np.all(np.diff(rows)==1) and np.all(np.diff(clocks)>0) and np.all(np.diff(native)>0)
             and np.isfinite(values).all() and np.all(values>=0) and np.all(values==np.floor(values)),
             'SIDECAR_ARRAYS:' + lane)
        namespace='entry_local_m5_rows' if lane=='local' else 'mtf:'+lane
        duration=300*10**9 if lane=='local' else int(htf.MULTI_TF_SHIFT[lane].value)
        available=np.ascontiguousarray(clocks+duration,dtype='<i8')
        need(_hash_int64_indices(rows,namespace=namespace)==pop['selected_row_indices_sha256']
             and hashlib.sha256(available.tobytes()).hexdigest()==pop['available_ns_sha256']
             and np.all(available<coverage_cutoff),'QUALIFIED_POPULATION:' + lane)
        selected[lane]=(owned_readonly(rows,np.int64),owned_readonly(clocks,np.int64),
                        owned_readonly(values,np.float32))
    local_rows,local_times,local_spans=selected['local']
    need(np.all(local_rows>=0) and np.all(local_rows<len(source))
         and np.array_equal(times[local_rows],local_times),'LOCAL_PHYSICAL_CLOCK')
    local_columns=column_plan(parent_signal,new_local,profile['physical_field_names']['signal'])
    mtf_columns=column_plan(parent_mtf,SPAN_FIELDS,htf.MULTI_TF_PER_BAR_FEATURES_V4)
    frames,mapped_rows,backings={},{},{}
    for tf in TIMEFRAMES:
        rows,clock,spans=selected[tf]
        parent=parent_mtf_values[tf];parent_clock=parent_mtf_times[tf]
        need(isinstance(parent,np.ndarray) and parent.dtype==np.float32 and parent.ndim==2
             and parent.shape[1]==253 and not parent.flags.writeable
             and isinstance(parent_clock,np.ndarray) and parent_clock.dtype==np.int64
             and parent_clock.shape==(len(parent),) and not parent_clock.flags.writeable
             and np.all(rows>=0) and np.all(rows<len(parent))
             and np.array_equal(parent_clock[rows],clock),'MTF_PARENT_CLOCK:' + tf)
        matrix=combine_selected(parent,rows,spans,columns=mtf_columns)
        for j,side in enumerate(('above','below')):
            active=matrix[:,list(htf.MULTI_TF_PER_BAR_FEATURES_V4).index('geomline_'+side+'_active_count')]>0
            need(np.all(spans[~active,j]==0) and np.all(spans[active,j]>0),'SPAN_OCCUPANCY:' + tf)
        matrix.setflags(write=False)
        frame=pd.DataFrame(matrix,index=pd.DatetimeIndex(clock.astype('datetime64[ns]'),tz='UTC'),
                           columns=htf.MULTI_TF_PER_BAR_FEATURES_V4,copy=False)
        frame.attrs.update(ts_int64=clock,feats_np=matrix,htf_feature_contract=htf.HTF_V4_MATRIX_CONTRACT,
                           causal_warmup_rows=0,anchor_span_parent_row_indices=rows)
        frames[tf]=frame;mapped_rows[tf]=rows;backings[tf]=(clock,matrix,rows)
    htf.require_multi_tf_v4_frames(frames)
    return QualifiedAnchorSpanInputs(view['contract_sha256'],source,times,positions,snap_x,ctx_cont,ctx_cat,
        local_rows,local_times,local_spans,local_columns,frames,mapped_rows,backings,
        view['prefix']['rows'],len(parent_signal),**later_binding)
