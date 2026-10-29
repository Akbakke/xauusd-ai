"""One guarded, resumable pure-direction prefix trajectory; no launch authority.

The canonical CUDA guard precedes strict context and numerical imports.  Immutable
267/264 providers supply raw physical inputs only.  Every prefix starts its own
combined ONLINE and measured0U baseline; private273 weights are never loaded.
The admitted order consumes the entire prefix, including missing/tied/immature rows.
Only the separate260 loss decides eligible cells.  Context tags are report-only.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

SOURCE_DESIGN = {'path':'/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_PREFIX_FIT_SOURCE_20261025_275/DESIGN.json',
    'sha256':'f73b5a516494ec82cab64e3d43526f337d10438dc11d46dcbb59f724c8c80e19'}
INPUT_QUALIFICATION = {'path':'/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_GC_BOOLEAN_PHYSICAL_PREFIX_NORMALIZATION_20261024_267/ACTUAL_QUALIFICATION_REVIEW.json',
    'sha256':'8b3e0b953e6e760e787f51e44bb8c701dd22e8e06ee2d7f934f50f86f5cf84c4'}
PRIVATE_QUALIFICATION = {'path':'/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_CANONICAL_PRIVATE_GPU_RESUME_20261024_273/ACTUAL_QUALIFICATION_REVIEW.json',
    'sha256':'59a2e9918ec7df392b483d840b07b48beeb01a35fb0220483b41bce301242d0a'}
MANIFEST_KEYS = frozenset(('schema_version','source_design','physical_provider_plan','physical_provider_worker',
    'prefix_normalization','constructor_metadata','view_source_bindings','targets','archive_members',
    'state_qualification','native_private_qualification','input_qualification'))
PRICE_MEMBERS = ('parent_rows','bar_start_ns','dynamic_entry_m1_row','decision_ns','terminal_time_ns',
    'elapsed_wall_seconds','terminal_source_row','observed','mid_move','mid_return_bps','sign')
TARGET_MEMBERS = ('terminal_time_ns','observed','sign')
SEED = 20260911
BATCH_SIZE = 16
MINUTE_NS = 60_000_000_000


def _require(ok, reason):
    if not ok:raise RuntimeError('NATIVE_DIRECTION_PREFIX_WINDOW_'+reason)


def _weighted_historical_source(ref, manifest, design):
    transition = design['historical_component_transition']
    _require(isinstance(transition, dict) and set(transition) == {'original', 'current'}
        and transition['original'] == manifest['view_source_bindings']['direction_component_source']
        and transition['current']['path'] == transition['original']['path'], 'WEIGHT_HISTORICAL_COMPONENT')
    return transition['current'] if ref == transition['original'] else ref


def _metadata(launch, common, *, gc_layout_metadata_parity=None, independent_completion=None, weight_qualification_design=None, order_qualification=None, dense_source_transition=None):
    metadata={};manifest=common._read(launch['input_manifest'],metadata)
    _require(set(manifest)==MANIFEST_KEYS and manifest['schema_version']=='gx1_native_direction_prefix_fit_input_manifest_v1'
        and manifest['source_design']==SOURCE_DESIGN and manifest['state_qualification']==common.STATE_QUALIFICATION
        and manifest['native_private_qualification']==PRIVATE_QUALIFICATION and manifest['input_qualification']==INPUT_QUALIFICATION
        and manifest['archive_members']==list(PRICE_MEMBERS),'INPUT_MANIFEST')
    design=common._read(SOURCE_DESIGN,metadata);fold=launch['fold']
    _require(fold in design['folds'],'SOURCE_DESIGN_FOLD')
    qualified=common._read(INPUT_QUALIFICATION,metadata);common._read(PRIVATE_QUALIFICATION,metadata)
    common._read(manifest['state_qualification'],metadata)
    item=manifest['prefix_normalization']
    _require(item in qualified['normalizations'] and item['fold_id']==fold['fold_id']
        and item['prefix_parent_rows']==fold['prefix_rows'] and item['cutoff_ns']==fold['train_cutoff_ns'],'QUALIFIED_PREFIX_RECORD')
    physical=common._module(manifest['physical_provider_worker']);ancestor=physical.provider_module()
    old_plan=common._read(manifest['physical_provider_plan'],metadata)
    _old_design,ancestral_plan,info=physical.validate_inputs(old_plan,ancestor)
    _require(manifest['targets']==ancestral_plan['axes']['source'] and old_plan['worker']==manifest['physical_provider_worker'],
        'SAME_PHYSICAL_AXES_AND_PROVIDER')
    fitted=common._read(item['output'],metadata);constructor=common._read(manifest['constructor_metadata'],metadata)
    _require(fitted['contract_sha256']==item['combined_contract_sha256']
        and fitted['base_normalization']['contract_sha256']==item['base_normalization_sha256']
        and fitted['gc_mtf_history']['layout']['contract_sha256']==item['gc_layout_sha256']
        and fitted['gc_mtf_history']['normalization']['contract_sha256']==item['gc_normalization_sha256'],'PREFIX_IDENTITIES')
    independent_design = None
    if independent_completion is not None:
        from gx1.contracts import entry_native_direction_frozen_extraction_v1 as frozen
        independent_design = frozen.read_independent_design(independent_completion,
            lambda ref: common._read(ref, metadata))
        original = independent_design['original_bindings']
        _require(launch['input_manifest'] == original['input_manifest'] and fold == original['fold']
            and launch['numerical_recipe'] == original['original_numerical_recipe']
            and all(manifest[key] == original[key] for key in ('prefix_normalization', 'constructor_metadata',
                'targets', 'archive_members', 'view_source_bindings', 'physical_provider_plan', 'physical_provider_worker')),
            'INDEPENDENT_ORIGINAL_BINDINGS')
    weight_design = None
    if weight_qualification_design is not None:
        from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
        _require(launch.get('run_scope') == authority.WEIGHT_SCOPE
            and weight_qualification_design == authority.WEIGHT_DESIGN, 'WEIGHT_SOURCE_SCOPE')
        weight_design = common._read(weight_qualification_design, metadata)
        _require(weight_design['input_manifest'] == launch['input_manifest'] and weight_design['fold'] == fold
            and weight_design['original_initial_predictions'] == authority.WEIGHT_INITIAL_PREDICTIONS, 'WEIGHT_ORIGINAL_BINDINGS')
    order_transition = None
    if order_qualification is not None:
        from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as authority
        _require(launch.get('run_scope') == authority.ORDER_SCOPE
            and independent_design is None and weight_design is None, 'ORDER_METADATA_SCOPE')
        order_transition = order_qualification['historical_component_transition']
        _require(order_transition['original'] == manifest['view_source_bindings']['direction_component_source']
            and order_transition['current'] == launch['source_owners'][authority.ORDER_COMPONENT_SOURCE]
            and order_transition['original']['path'] == order_transition['current']['path'], 'ORDER_METADATA_TRANSITION')
    dense_transitions = {}
    if dense_source_transition is not None:
        from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as authority
        names = {'owner_profile_source':'gx1/features/entry_specialist_feature_groups_v1.py',
            'native_constructor_source':'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py',
            'direction_component_source':'gx1/contracts/entry_native_direction_research_v1.py'}
        _require(launch.get('run_scope') in (authority.DENSE_SCOPE, authority.DENSE_DECIMAL_PRIVATE_SCOPE) and independent_design is None
            and weight_design is None and order_transition is None
            and isinstance(dense_source_transition,dict) and set(dense_source_transition)==set(names), 'DENSE_TRANSITION_SCOPE')
        for key, source in names.items():
            item=dense_source_transition[key]
            _require(isinstance(item,dict) and set(item)=={'original','current'}
                and item['original']==manifest['view_source_bindings'][key]
                and item['current']==launch['source_owners'][source]
                and item['original']['path']==item['current']['path'], 'DENSE_SOURCE_TRANSITION:'+key)
            dense_transitions[item['original']['path']]=item
    for ref in manifest['view_source_bindings'].values():
        if ref['path'].endswith('.json'):common._read(ref,metadata)
        else:
            expected = ref
            if gc_layout_metadata_parity is not None:
                from gx1.contracts import entry_native_direction_frozen_extraction_v1 as frozen
                expected = frozen.historical_gc_view_source(ref, gc_layout_metadata_parity)
            if independent_design is not None:
                transition = independent_design['historical_source_verification']['only_new_manifest_source_transition']
                if ref['path'] == transition['original']['path']:
                    _require(ref == transition['original'], 'INDEPENDENT_HISTORICAL_COMPONENT')
                    expected = transition['current']
            if weight_design is not None:
                weighted_expected = _weighted_historical_source(ref, manifest, weight_design)
                if weighted_expected != ref:
                    expected = weighted_expected
            if order_transition is not None and ref == order_transition['original']:
                expected = order_transition['current']
            if ref['path'] in dense_transitions:
                item=dense_transitions[ref['path']]
                _require(ref==item['original'],'DENSE_HISTORICAL_SOURCE')
                expected=item['current']
            _require(common._hash_file(ref['path'])==expected,'VIEW_SOURCE_SHA')
    info['metadata'].update(metadata)
    return {'manifest':manifest,'design':design,'physical':physical,'ancestor':ancestor,'ancestor_plan':ancestral_plan,
        'info':info,'fitted':fitted,'constructor':constructor}


def _launch_metadata(launch, common):
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as authority
    if launch.get('run_scope') in (authority.DENSE_SCOPE, authority.DENSE_DECIMAL_PRIVATE_SCOPE):
        decimal=launch['run_scope']==authority.DENSE_DECIMAL_PRIVATE_SCOPE
        metadata = {}
        proof=common._read(launch['dense_source_transition_proof'],metadata)
        target_report=common._read(launch['dense_target_result'],metadata)
        _require(proof.get('schema_version')=='gx1_dense_source_transition_proof_v1'
            and proof.get('status')=='SOURCE_ONLY_VERIFIED_NO_DATA_OR_TRAINING'
            and proof.get('test_data_used') is False and proof.get('genuine_training_authorized') is False, 'DENSE_SOURCE_PROOF')
        target=target_report.get('target',{})
        if decimal:
            from gx1.contracts import entry_native_direction_research_campaign_v1 as private
            refs=launch['decimal_target_refs']; qualification=common._read(launch['decimal_input_audit'],metadata)
            _require(launch['decimal_input_audit']==private.DECIMAL_INPUT_AUDIT
                and qualification.get('schema_version')=='gx1_decimal_bound_input_actual_audit_v1'
                and qualification.get('status')=='COMPLETE' and qualification.get('test_data_used') is False
                and qualification.get('inputs')==refs and refs['original']==private.DENSE_TARGET
                and refs['original_result']==private.DENSE_TARGET_RESULT
                and refs['corrected']==launch['dense_target'] and refs['corrected_result']==launch['dense_target_result']
                and target_report.get('schema_version')=='gx1_direction_decimal_targets_result_v1'
                and target_report.get('status')=='COMPLETE' and target_report.get('test_data_used') is False
                and target_report.get('target_identity')=='exact_original_decimal_midpoint_v1'
                and target_report.get('parent_rows')==118249 and target_report.get('horizons_minutes')==list(range(1,61))
                and target_report.get('target')==launch['dense_target'], 'DECIMAL_TARGET_RESULT')
            receipts={key:common._read(refs[key],metadata) for key in
                ('original_result','consumer_plan','consumer_result','materialization_plan','audit_result')}
        else:
            _require(target_report.get('status')=='COMPLETE_REAL_TARGETS_ONLY'
                and target_report.get('test_data_used') is False and target_report.get('parents')==118249
                and target_report.get('horizons_minutes')==list(range(1,61))
                and target_report.get('legacy_anchor_byte_parity')=={'1':True,'5':True,'15':True}
                and launch['dense_target']=={key:target.get(key) for key in ('path','sha256')}, 'DENSE_TARGET_RESULT')
        context=_metadata(launch,common,gc_layout_metadata_parity=launch['gc_layout_metadata_parity'],
            dense_source_transition=proof['source_transitions'])
        context['info']['metadata'].update(metadata)
        context['dense_target']=dict(launch['dense_target'])
        context['dense_target_stat']=common._stat(context['dense_target']['path'])
        _require(common._hash_file(context['dense_target']['path'])==context['dense_target'], 'DENSE_TARGET_SHA')
        if decimal:
            _require(context['manifest']['targets']==refs['legacy'], 'DECIMAL_LEGACY_SOURCE')
            context['decimal_target_refs']=dict(refs)
            context['decimal_original']=dict(refs['original'])
            context['decimal_original_stat']=common._stat(context['decimal_original']['path'])
            _require(common._hash_file(context['decimal_original']['path'])==context['decimal_original'], 'DECIMAL_ORIGINAL_SHA')
            context['decimal_sign_audit']=receipts['audit_result']
    elif launch.get('run_scope') == authority.ORDER_SCOPE:
        metadata = {}
        proof = common._read(launch['native_order_qualification'], metadata)
        initial = common._read(launch['order_origin']['initial_measurement'], metadata)
        context = _metadata(launch, common, gc_layout_metadata_parity=launch['gc_layout_metadata_parity'],
            order_qualification=proof)
        context['info']['metadata'].update(metadata)
        context['order_original_initial'] = initial
    elif 'native_continuation_qualification' in launch:
        from gx1.contracts.entry_native_direction_frozen_extraction_v1 import CONTINUATION_PRIOR_RECIPE
        metadata = {}
        prior = common._read(CONTINUATION_PRIOR_RECIPE,metadata)
        original = common._read(launch['continuation_origin']['launch_recipe'],metadata)
        historical_launch = {**launch,'numerical_recipe':original['numerical_recipe']}
        context = _metadata(historical_launch,common,gc_layout_metadata_parity=launch.get('gc_layout_metadata_parity'),
            independent_completion=prior['independent_completion'])
        context['info']['metadata'].update(metadata)
    else:
        context = _metadata(launch, common, gc_layout_metadata_parity=launch.get('gc_layout_metadata_parity'))
    if 'native_continuation_qualification' in launch:
        metadata = context['info']['metadata']
        for ref in (launch['gc_layout_metadata_parity'],launch['native_continuation_qualification'],
                launch['continuation_origin']['terminal_train_measurement']):
            common._read(ref,metadata)
    if 'native_bps_qualification' in launch:
        metadata = context['info']['metadata']
        common._read(launch['gc_layout_metadata_parity'], metadata)
        proof = common._read(launch['native_bps_qualification'], metadata)
        context['bps_qualified_scale_ref'] = proof['own_prefix_bps_scale']
        context['bps_qualified_scale'] = common._read(proof['own_prefix_bps_scale'], metadata)
    return context


def _qualified_bps_scale(context, view):
    if 'bps_qualified_scale' in context:
        _require(view.get('bps_scale') == context['bps_qualified_scale'], 'QUALIFIED_OWN_PREFIX_BPS_SCALE')
        return context['bps_qualified_scale_ref']
    _require('bps_scale' not in view, 'UNQUALIFIED_BPS_SCALE')
    return None


def _qualify_axes_targets(axes, targets, fold, np, *, signed_bps=False, magnitude_weighted=False, dense_minutes=False):
    parents,starts,decisions=(axes[k] for k in ('parent_rows','bar_start_ns','decision_ns'))
    size=len(parents)
    _require(size==118249 and parents.dtype==starts.dtype==decisions.dtype==np.int64
        and parents.shape==starts.shape==decisions.shape==(size,) and np.all(parents[1:]>parents[:-1])
        and np.all(decisions[1:]>decisions[:-1]) and np.all(starts<=np.iinfo(np.int64).max-300_000_000_000)
        and np.array_equal(decisions,starts+300_000_000_000),'FULL_AXES')
    _require(int(np.searchsorted(decisions,fold['train_cutoff_ns'],side='left'))==fold['prefix_rows']
        and fold['oof_start_row']==fold['prefix_rows']
        and int(np.searchsorted(decisions,fold['oof_cutoff_ns'],side='left'))==fold['oof_end_row'],'FOLD_CLOCK_AND_ROW_COHORT')
    _require(type(signed_bps) is bool and type(magnitude_weighted) is bool and type(dense_minutes) is bool
        and sum((signed_bps,magnitude_weighted,dense_minutes))<=1,'OBJECTIVE_MODE')
    has_bps=signed_bps or magnitude_weighted
    horizons=np.arange(1,61,dtype=np.int64) if dense_minutes else np.asarray([1,5,15],np.int64)
    width=len(horizons)
    _require(set(targets)==set(TARGET_MEMBERS + (('mid_return_bps',) if has_bps else ())) and targets['terminal_time_ns'].dtype==np.int64
        and targets['terminal_time_ns'].shape==(size,width) and targets['observed'].dtype==np.bool_
        and targets['observed'].shape==(size,width) and targets['sign'].dtype==np.int8 and targets['sign'].shape==(size,width),
        'TARGET_HEADERS')
    _require(np.all(decisions>0) and np.all(decisions<=np.iinfo(np.int64).max-int(horizons[-1])*MINUTE_NS)
        and np.array_equal(targets['terminal_time_ns'],decisions[:,None]+horizons[None,:]*MINUTE_NS)
        and np.isin(targets['sign'],(-1,0,1)).all() and np.all(targets['sign'][~targets['observed']]==0),'TARGET_CLOCK_AND_DOMAIN')
    if has_bps:
        value=targets['mid_return_bps']
        _require(value.dtype==np.float64 and value.shape==(size,3) and np.isfinite(value).all()
            and np.all(value[~targets['observed']]==0)
            and np.array_equal(np.sign(value).astype(np.int8),targets['sign']),'BPS_TARGET_DOMAIN')


def _fixed_order(parents, prefix_rows, seed, np):
    _require(type(prefix_rows) is int and 256<=prefix_rows<=len(parents) and seed==SEED,'ORDER_ARGUMENTS')
    return np.ascontiguousarray(parents[:prefix_rows][np.random.RandomState(seed).permutation(prefix_rows)],dtype=np.int64)


def _presentation_order(parents, prefix_rows, seed, np, *, launch):
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as authority
    if launch.get('run_scope') == authority.ORDER_SCOPE:
        _require(launch.get('parent_order_algorithm') == authority.ORDER_ALGORITHM
            and type(prefix_rows) is int and 256 <= prefix_rows <= len(parents)
            and seed == SEED and parents.dtype == np.int64 and parents.ndim == 1
            and np.all(parents[1:prefix_rows] > parents[:prefix_rows-1]), 'CHRONOLOGICAL_ORDER')
        return np.ascontiguousarray(parents[:prefix_rows], dtype=np.int64)
    return _fixed_order(parents, prefix_rows, seed, np)


def _select_price_targets(archive, *, signed_bps=False, magnitude_weighted=False):
    _require(type(signed_bps) is bool and type(magnitude_weighted) is bool and not (signed_bps and magnitude_weighted),'OBJECTIVE_MODE')
    _require(archive.files==list(PRICE_MEMBERS),'EXACT_ELEVEN_ARCHIVE_NAMES')
    members=TARGET_MEMBERS + (('mid_return_bps',) if signed_bps or magnitude_weighted else ())
    return {key:archive[key] for key in members}


def _prepare(context, launch, np, *, signed_bps=False, magnitude_weighted=False, dense_minutes=False):
    from gx1.contracts import entry_native_direction_feature_view_v1 as view_owner
    from gx1.contracts import entry_native_direction_context_report_v1 as report_owner
    from gx1.features.htf_features import discard_multi_tf_v4_cache_pages
    physical,ancestor,plan,info=(context[k] for k in ('physical','ancestor','ancestor_plan','info'))
    prepared=physical.prepare_inputs(ancestor,plan,info,np)
    physical.release_pages(ancestor,prepared,discard_multi_tf_v4_cache_pages)
    manifest=context['manifest']
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    _require(prepared['gc']['layout']==common._read(manifest['view_source_bindings']['raw_gc_layout'],info['metadata'])
        and prepared['gc']['transition']==common._read(manifest['view_source_bindings']['raw_gc_transition'],info['metadata']),
        'CURRENT_QUALIFIED_RAW_GC_DOMAIN')
    _require(type(dense_minutes) is bool and not (dense_minutes and (signed_bps or magnitude_weighted)), 'PREPARE_MODE')
    with np.load(manifest['targets']['path'],allow_pickle=False) as archive:
        if dense_minutes:
            from gx1.contracts.entry_native_direction_dense_data_v1 import AXES, TARGETS
            _require(archive.files==list(PRICE_MEMBERS),'EXACT_ELEVEN_ARCHIVE_NAMES')
            legacy={key:archive[key] for key in AXES+TARGETS}
        else:
            targets=_select_price_targets(archive,signed_bps=signed_bps,magnitude_weighted=magnitude_weighted)
    if dense_minutes:
        from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as authority
        _require(launch.get('run_scope') in (authority.DENSE_SCOPE,authority.DENSE_DECIMAL_PRIVATE_SCOPE)
            and context.get('dense_target')==launch.get('dense_target'), 'DENSE_EXACT_SCOPE')
        physical_axes={**prepared['arrays'],'dynamic_entry_m1_row':legacy['dynamic_entry_m1_row']}
        if launch['run_scope']==authority.DENSE_DECIMAL_PRIVATE_SCOPE:
            from gx1.contracts.entry_native_direction_decimal_input_v1 import load_bound_decimal_direction_targets
            refs=launch['decimal_target_refs']
            _require(context.get('decimal_target_refs')==refs and refs['legacy']==manifest['targets']
                and refs['original']==context.get('decimal_original') and refs['corrected']==context['dense_target'],
                'DECIMAL_EXACT_SCOPE')
            bound=load_bound_decimal_direction_targets(refs=refs,axes=physical_axes,legacy=legacy)
            targets={key:bound[key] for key in TARGET_MEMBERS}
            del bound
        else:
            from gx1.contracts.entry_native_direction_dense_data_v1 import DENSE_MEMBERS, bind_dense_targets
            ref=context['dense_target']; before=common._stat(ref['path'])
            _require(common._hash_file(ref['path'])==ref, 'DENSE_ARCHIVE_SHA')
            with np.load(ref['path'],allow_pickle=False) as archive:
                _require(set(archive.files)==set(DENSE_MEMBERS) and len(archive.files)==len(DENSE_MEMBERS), 'DENSE_ARCHIVE_MEMBERS')
                dense={key:archive[key] for key in DENSE_MEMBERS}
            _require(common._stat(ref['path'])==before and common._hash_file(ref['path'])==ref, 'DENSE_ARCHIVE_BOOKEND')
            targets=bind_dense_targets(axes=physical_axes,legacy=legacy,dense=dense)
            del dense
        del legacy
    _qualify_axes_targets(prepared['arrays'],targets,launch['fold'],np,signed_bps=signed_bps,
        magnitude_weighted=magnitude_weighted,dense_minutes=dense_minutes)
    bps_scale=None; magnitude_weight_contract=None
    if signed_bps:
        from gx1.contracts.entry_native_direction_bps_v1 import fit_direction_bps_scale
        n=launch['fold']['prefix_rows']
        bps_scale=fit_direction_bps_scale(parent_rows=prepared['arrays']['parent_rows'][:n],
            decision_ns=prepared['arrays']['decision_ns'][:n], **{key:value[:n] for key,value in targets.items()},
            cutoff_ns=launch['fold']['train_cutoff_ns'],target_source=manifest['targets'],fit_partition='train')
    if magnitude_weighted:
        from gx1.contracts.entry_native_direction_bps_v1 import fit_direction_magnitude_weight_contract
        n=launch['fold']['prefix_rows']
        magnitude_weight_contract=fit_direction_magnitude_weight_contract(parent_rows=prepared['arrays']['parent_rows'][:n],
            decision_ns=prepared['arrays']['decision_ns'][:n], **{key:value[:n] for key,value in targets.items()},
            cutoff_ns=launch['fold']['train_cutoff_ns'],target_source=manifest['targets'],fit_partition='train')
    for value in prepared['arrays'].values():value.setflags(write=False)
    for value in targets.values():value.setflags(write=False)
    for key in ('times_ns','sample_positions'):prepared['entry_sequence_source'][key].setflags(write=False)
    sources=dict(manifest['view_source_bindings'])
    if dense_minutes:
        owners=launch['source_owners']
        sources.update({'dense_direction_component_source':owners['gx1/contracts/entry_native_direction_dense_v1.py'],
            'dense_profile_source':owners['gx1/features/entry_specialist_feature_groups_v1.py'],
            'dense_native_constructor_source':owners['gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py']})
    view=view_owner.bind_native_direction_feature_view(prefix_normalization=context['fitted'],
        prefix_normalization_ref=manifest['prefix_normalization']['output'],constructor_metadata=context['constructor'],
        constructor_metadata_ref=manifest['constructor_metadata'],source_bindings=sources,max_batch_rows=BATCH_SIZE,
        bps_scale=bps_scale,magnitude_weight_contract=magnitude_weight_contract,dense_minutes=dense_minutes)
    context_map=report_owner.native_direction_context_map(gc_layout=prepared['gc']['layout'],
        mtf_field_names=context['fitted']['base_normalization']['surfaces']['mtf_m5']['field_names'])
    order=_presentation_order(prepared['arrays']['parent_rows'],launch['fold']['prefix_rows'],SEED,np,launch=launch)
    return prepared,targets,view,context_map,order


def _batch(prepared, targets, view, selected, np):
    from gx1.contracts.entry_native_direction_feature_view_v1 import gather_native_direction_feature_batch
    raw=gather_native_direction_feature_batch(view,parent_rows=prepared['arrays']['parent_rows'],
        bar_start_ns=prepared['arrays']['bar_start_ns'],decision_ns=prepared['arrays']['decision_ns'],
        entry_sequence_source=prepared['entry_sequence_source'],snap_x=prepared['current']['snap'],
        ctx_cont=prepared['current']['ctx_cont'],ctx_cat=prepared['current']['ctx_cat'],multi_tf_sources=prepared['cache'],
        gc_surfaces=prepared['gc']['surfaces'],gc_layout=prepared['gc']['layout'],batch_parent_rows=selected,
        **({'m1_ofi_source':prepared['m1_ofi_source'], 'm1_ofi_clock':prepared['m1_ofi_clock']}
            if 'm1_ofi_normalization' in view else {}))
    at=np.searchsorted(prepared['arrays']['parent_rows'],selected)
    _require(np.array_equal(prepared['arrays']['parent_rows'][at],selected),'TARGET_EXACT_PARENT_JOIN')
    target={key:np.ascontiguousarray(value[at]) for key,value in targets.items()}
    target['decision_ns']=np.ascontiguousarray(prepared['arrays']['decision_ns'][at])
    return raw,target


def _device_batch(raw, target, torch, device):
    return ({key:torch.from_numpy(value).to(device) for key,value in raw['features'].items()},
        {key:torch.from_numpy(value).to(device) for key,value in target.items()})


def _prefix_prior(axes, targets, fold, np):
    n=fold['prefix_rows'];eligible=targets['observed'][:n]&(targets['sign'][:n]!=0)&(targets['terminal_time_ns'][:n]<fold['train_cutoff_ns'])
    up=np.count_nonzero(eligible&(targets['sign'][:n]==1),axis=0);down=np.count_nonzero(eligible&(targets['sign'][:n]==-1),axis=0)
    _require(np.all(up>0) and np.all(down>0),'OWN_PREFIX_BOTH_CLASSES')
    return {'eligible_up':up.astype(np.int64).tolist(),'eligible_down':down.astype(np.int64).tolist(),
        'probability_up':(up/(up+down)).tolist(),'logit_up':np.log(up.astype(np.float64)/down).tolist(),
        'scope':'own mature observed nonzero prefix labels; no weights fit or OOF use'}


def _forward(model, features, *, kind, counters, ceiling, torch, grad=False, return_outputs=False):
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as hybrid
    counters['forward_calls']+=1;counters['forward_kind_calls'][kind]=counters['forward_kind_calls'].get(kind,0)+1
    _require(counters['forward_calls']<=ceiling,'DERIVED_FORWARD_CEILING')
    with trainer._training_mha_fastpath_context(model.backbone), (torch.enable_grad() if grad else torch.no_grad()):
        _require(not torch.backends.mha.get_fastpath_enabled() and not hybrid._BATCH_EQUAL_LENGTH_MTF_EVAL.get(),'CANONICAL_FP32_MHA')
        output=trainer._model_forward_fp32(model,**features)
    counters['forward_returns']+=1
    return output if return_outputs else output['logit_up']


def _online_identity(model, device, tree_sha, trainer):
    return {'model':tree_sha(model.state_dict()),'all_buffers':tree_sha(dict(model.named_buffers())),
        'rng':tree_sha(trainer._attended_session_rng_state(device=device)),
        'modes':{name:bool(module.training) for name,module in model.named_modules()}}


def _evaluate(model, prepared, targets, view, context_map, selected, *, cutoff, kind, counters, ceiling,
        device, torch, np, tree_sha, deadline=None, parity=False):
    from gx1.contracts import entry_native_direction_context_report_v1 as reporter
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    before=_online_identity(model,device,tree_sha,trainer);modes=[(module,module.training) for module in model.modules()]
    signed_bps=getattr(model,'signed_bps',False)
    predictions=[];bps_predictions=[];bps_targets=[];tags=[];decisions=[];sign=[];observed=[];terminal=[];starts=[];completed=0;parity_passed=not parity;context_header=None
    _require(all(p.grad is None for p in model.parameters()),'EVAL_PARTIAL_GRADIENT')
    try:
        model.eval()
        for left in range(0,len(selected),BATCH_SIZE):
            if deadline is not None and time.monotonic()>=deadline:break
            ids=selected[left:left+BATCH_SIZE];raw,target=_batch(prepared,targets,view,ids,np)
            context=reporter.tag_native_direction_contexts(raw['features'],context_map=context_map,witness=raw['witness'])
            header={key:value for key,value in context.items() if key not in ('parent_rows','decision_ns','tags')}
            if context_header is None:context_header=header
            else:_require(header==context_header,'CONTEXT_HEADER_PRESERVATION')
            _require(np.array_equal(context['parent_rows'],ids) and np.array_equal(context['decision_ns'],target['decision_ns']),
                'REPORT_CONTEXT_EXACT_INPUT_JOIN')
            features,_unused=_device_batch(raw,target,torch,device)
            outputs=(_forward(model,features,kind=kind,counters=counters,ceiling=ceiling,torch=torch,return_outputs=True)
                if signed_bps else {'logit_up':_forward(model,features,kind=kind,counters=counters,ceiling=ceiling,torch=torch)})
            logits=outputs['logit_up']
            if signed_bps:
                bps_predictions.append(outputs['mean_return_bps'].detach().cpu().numpy().copy())
                bps_targets.append(target['mid_return_bps'])
            if parity and left==0:
                with_grad=(_forward(model,features,kind='INITIAL_PARITY_GRAD',counters=counters,ceiling=ceiling,torch=torch,grad=True,return_outputs=True)
                    if signed_bps else {'logit_up':_forward(model,features,kind='INITIAL_PARITY_GRAD',counters=counters,ceiling=ceiling,torch=torch,grad=True)})
                _require(set(outputs)==set(with_grad)
                    and all(torch.equal(outputs[k].detach(),with_grad[k].detach()) for k in outputs if k!='horizons_minutes') and all(p.grad is None for p in model.parameters()),
                    'OWN_INITIAL_GRAD_NOGRAD_LOGIT_PARITY')
                del with_grad;parity_passed=True
            predictions.append(logits.detach().cpu().numpy().copy());tags.append(context['tags'])
            decisions.append(target['decision_ns']);sign.append(target['sign']);observed.append(target['observed']);terminal.append(target['terminal_time_ns'])
            starts.append(np.asarray(raw['witness']['bar_start_ns'],np.int64));completed+=len(ids)
            del logits,outputs,features,_unused,raw,target,context
    finally:
        for module,training in modes:module.training=training
        _require(_online_identity(model,device,tree_sha,trainer)==before and all(p.grad is None for p in model.parameters()),
            'EVAL_ONLINE_BUFFERS_RNG_MODES_UNCHANGED')
    if completed!=len(selected):return None,completed
    arrays={'parent_rows':np.ascontiguousarray(selected),'bar_start_ns':np.concatenate(starts),
        'decision_ns':np.concatenate(decisions),'terminal_time_ns':np.concatenate(terminal),
        'observed':np.concatenate(observed),'sign':np.concatenate(sign),'logit_up':np.concatenate(predictions)}
    tag_arrays={key:np.concatenate([item[key] for item in tags]) for key in tags[0]}
    context={**context_header,'parent_rows':arrays['parent_rows'],
        'decision_ns':arrays['decision_ns'],'tags':tag_arrays}
    if view.get('dense_minute_supervision', False):
        from gx1.contracts.entry_native_direction_dense_report_v1 import score_dense_direction_logits
        report=score_dense_direction_logits(arrays['logit_up'],sign=arrays['sign'],observed=arrays['observed'],
            decision_ns=arrays['decision_ns'],terminal_time_ns=arrays['terminal_time_ns'],cutoff_ns=cutoff)
    else:
        report=reporter.score_native_direction_logits(arrays['logit_up'],sign=arrays['sign'],observed=arrays['observed'],
            decision_ns=arrays['decision_ns'],terminal_time_ns=arrays['terminal_time_ns'],cutoff_ns=cutoff,context_tags=context)
    if signed_bps:
        from gx1.contracts.entry_native_direction_bps_v1 import score_direction_bps
        arrays['mean_return_bps']=np.concatenate(bps_predictions)
        arrays['mid_return_bps']=np.concatenate(bps_targets)
        report['signed_bps']=score_direction_bps(arrays['mean_return_bps'],mid_return_bps=arrays['mid_return_bps'],
            observed=arrays['observed'],terminal_time_ns=arrays['terminal_time_ns'],cutoff_ns=cutoff,scale=model.bps_scale)
    arrays.update({'context__'+key:value for key,value in tag_arrays.items()})
    return {'arrays':arrays,'report':report,'context_tags':context,'state_preservation':before,'initial_parity_passed':parity_passed},completed


def _save_predictions(path, arrays, *, np, common, tree_sha):
    path=Path(path);_require(not path.exists() and not path.is_symlink(),'PREDICTION_EXISTS')
    _require(arrays and all(isinstance(a,np.ndarray) and a.flags.c_contiguous and a.dtype.kind in 'biuf'
        and np.isfinite(a).all() for a in arrays.values()),'PREDICTION_ARRAY_DOMAIN')
    fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:np.savez(handle,**arrays);handle.flush();os.fsync(handle.fileno())
        before=common._stat(temporary);staged=common._hash_file(temporary)
        with np.load(temporary,allow_pickle=False) as archive:
            _require(archive.files==list(arrays) and tree_sha({k:archive[k] for k in archive.files})==tree_sha(arrays),
                'PREDICTION_STAGED_ROUNDTRIP')
        _require(common._stat(temporary)==before and common._hash_file(temporary)==staged and before['st_size']<=sum(a.nbytes for a in arrays.values())+65536,
            'PREDICTION_STAGED_BOUND')
        os.link(temporary,path);path.chmod(0o444)
        from gx1.models.entry_v10.entry_v10_ctx_train_v3 import _fsync_directory
        _fsync_directory(path.parent)
    finally:
        if Path(temporary).exists():Path(temporary).unlink()
    return common._hash_file(path)


def _require_original_initial_match(measured, original_ref, *, common, np):
    before = common._stat(original_ref['path'])
    _require(common._hash_file(original_ref['path']) == original_ref, 'ORDER_INITIAL_PREDICTION_SHA')
    arrays = measured['arrays']
    with np.load(original_ref['path'], allow_pickle=False) as archive:
        _require(set(archive.files) == set(arrays), 'ORDER_INITIAL_PREDICTION_FIELDS')
        for key, value in arrays.items():
            original = archive[key]
            _require(original.dtype == value.dtype and original.shape == value.shape
                and np.ascontiguousarray(original).tobytes() == np.ascontiguousarray(value).tobytes(),
                'ORDER_INITIAL_PREDICTION:' + key)
    _require(common._stat(original_ref['path']) == before, 'ORDER_INITIAL_PREDICTION_STABLE')


def _publish_measurement(name, measured, *, output, view, prior, own_initial_state, fresh_identity,
        context_map, counters, launch, np, common, tree_sha):
    from gx1.contracts.entry_native_direction_context_report_v1 import score_native_direction_logits
    arrays=measured['arrays'];prediction=_save_predictions(output/(name+'.npz'),arrays,np=np,common=common,tree_sha=tree_sha)
    horizons=arrays['logit_up'].shape[1]
    _require(horizons in (3,60) and len(prior['logit_up'])==horizons,'PRIOR_HORIZONS')
    prior_logits=np.broadcast_to(np.asarray(prior['logit_up'],np.float64),(len(arrays['parent_rows']),horizons)).copy()
    cutoff=launch['fold']['train_cutoff_ns'] if name!='OOF' else launch['fold']['oof_cutoff_ns']
    if horizons==60:
        from gx1.contracts.entry_native_direction_dense_report_v1 import score_dense_direction_logits
        prior_report=score_dense_direction_logits(prior_logits,sign=arrays['sign'],observed=arrays['observed'],
            decision_ns=arrays['decision_ns'],terminal_time_ns=arrays['terminal_time_ns'],cutoff_ns=cutoff)
    else:
        prior_report=score_native_direction_logits(prior_logits,sign=arrays['sign'],observed=arrays['observed'],
            decision_ns=arrays['decision_ns'],terminal_time_ns=arrays['terminal_time_ns'],cutoff_ns=cutoff,context_tags=measured['context_tags'])
    return common._json(output/(name+'.json'),{'schema_version':'gx1_native_direction_bps_prefix_measurement_v1' if 'mean_return_bps' in arrays else 'gx1_native_direction_prefix_measurement_v1',
        'role':name,'fold':launch['fold'],'feature_view_identity':view,'context_map':context_map,
        'predictions':prediction,'model_metrics':measured['report'],'own_prefix_prior':prior,'prior_metrics':prior_report,
        'own_initial_state':own_initial_state,'fresh_identity_sha256':fresh_identity,'preservation':measured['state_preservation'],
        'initial_grad_nograd_parity':measured['initial_parity_passed'],'test_data_used':False,
        'scope':'reused TRAIN development periods; binary direction conditional observed nonzero; no trading or generalization authority'},
        counters,launch['max_published_json_bytes'])


def _gradient_summary(model, torch):
    _require(all(p.grad is None for p in model.parameters() if not p.requires_grad),'INACTIVE_GRADIENT')
    _require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'FINITE_GRADIENT')
    head=model.direction_head
    out={'head_weight_rows':[int(torch.count_nonzero(row)) for row in head.weight.grad] if head.weight.grad is not None else [0]*3,
        'head_bias_rows':[int(value!=0) for value in head.bias.grad] if head.bias.grad is not None else [0]*3}
    if hasattr(model,'dense_aux_head'):
        aux=model.dense_aux_head
        out['dense_aux_head_weight_rows']=[int(torch.count_nonzero(row)) for row in aux.weight.grad] if aux.weight.grad is not None else [0]*57
        out['dense_aux_head_bias_rows']=[int(value!=0) for value in aux.bias.grad] if aux.bias.grad is not None else [0]*57
    if getattr(model,'signed_bps',False):
        mean=model.bps_mean_head
        out['bps_head_weight_rows']=[int(torch.count_nonzero(row)) for row in mean.weight.grad] if mean.weight.grad is not None else [0]*3
        out['bps_head_bias_rows']=[int(value!=0) for value in mean.bias.grad] if mean.bias.grad is not None else [0]*3
    for tf in ('m5','m15','h1','h4','d1'):
        for label,modules in (('gc_projection',model.backbone.gc_mtf_family_proj),('gc_gate',model.backbone.gc_mtf_feature_context_gate)):
            out[label+'_'+tf]=sum(int(torch.count_nonzero(p.grad)) for key in modules if key.startswith(tf+'__')
                for p in modules[key].parameters() if p.grad is not None)
    return out


def _train_batch(model, optimizer, features, target, *, cutoff, counters, ceiling, torch, diagnostics):
    from gx1.contracts.entry_native_direction_research_v1 import native_direction_binary_loss
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    _require(model.training and all(module.training for module in model.modules()),'TRAIN_MODE')
    signed_bps=getattr(model,'signed_bps',False); magnitude_weighted=getattr(model,'magnitude_weighted',False)
    _require(not (signed_bps and magnitude_weighted),'OBJECTIVE_MODE')
    if signed_bps:
        outputs=_forward(model,features,kind='TRAIN',counters=counters,ceiling=ceiling,torch=torch,grad=True,return_outputs=True)
        from gx1.contracts.entry_native_direction_bps_v1 import native_direction_bps_loss
        loss=native_direction_bps_loss(outputs['logit_up'],outputs['mean_return_bps'],
            rms_bps=model.bps_rms,**target,cutoff_ns=cutoff)
    elif magnitude_weighted:
        logits=_forward(model,features,kind='TRAIN',counters=counters,ceiling=ceiling,torch=torch,grad=True)
        from gx1.contracts.entry_native_direction_bps_v1 import native_direction_magnitude_weighted_loss
        loss=native_direction_magnitude_weighted_loss(logits,rms_bps=model.magnitude_rms_bps,
            raw_weight_mean=model.magnitude_raw_weight_mean,**target,cutoff_ns=cutoff)
        _require(loss['research_objective_sha256']==model.research_objective['contract_sha256'],'WEIGHTED_LOSS_IDENTITY')
    else:
        logits=_forward(model,features,kind='TRAIN',counters=counters,ceiling=ceiling,torch=torch,grad=True)
        from gx1.contracts.entry_native_direction_dense_v1 import DenseNativeEntryDirectionResearch, dense_direction_binary_loss
        if type(model) is DenseNativeEntryDirectionResearch:
            loss=dense_direction_binary_loss(logits,**target,cutoff_ns=cutoff)
            _require(loss['research_objective_sha256']==model.research_objective['contract_sha256'],'DENSE_LOSS_IDENTITY')
        else:
            loss=native_direction_binary_loss(logits,**target,cutoff_ns=cutoff)
    record={'eligible_cells':loss['counts']['eligible_cells'],'counts':loss['counts'],'per_horizon_counts':loss['per_horizon_counts']}
    if magnitude_weighted:
        for key in ('unweighted_direction_loss','magnitude_weighted_loss'):
            record[key]=None if loss[key] is None else float(loss[key].detach())
    if signed_bps:
        record['bps_eligible_cells']=loss['bps_eligible_cells']
        record['bps_per_horizon_counts']=loss['bps_per_horizon_counts']
        for key in ('direction_loss','bps_loss','bps_raw_sse','bps_normalized_sse'):
            record[key]=None if loss[key] is None else float(loss[key].detach())
    if not loss['update_allowed']:
        _require(loss['loss'] is None and all(p.grad is None for p in model.parameters()),'EMPTY_BATCH_NO_UPDATE')
        counters['empty_batches']+=1;return record,False
    counters['backward_calls']+=1;loss['loss'].backward();counters['backward_returns']+=1
    if diagnostics:record['gradient_nonzero_counts']=_gradient_summary(model,torch)
    counters['optimizer_step_calls']+=1
    norm=trainer._optimizer_step_with_finite_gradients(model=model,optimizer=optimizer)
    counters['optimizer_step_returns']+=1
    record.update(loss=float(loss['loss'].detach()),gradient_norm=float(norm));return record,True


def _next_training_parent_range(cursor, n, train_passes):
    _require(type(n) is int and n > 0 and type(train_passes) is int and train_passes in (1,3)
        and type(cursor) is int and 0 <= cursor < train_passes*n, 'TRAIN_PARENT_RANGE')
    within=cursor%n
    _require(within%BATCH_SIZE == 0, 'TRAIN_PARENT_BOUNDARY')
    right=min(within+BATCH_SIZE,n)
    return within,right,cursor+right-within


def _save_snapshot(model, optimizer, *, context, recipe, order, parent_cursor, updates, state_directory,
        name, launch, counters, torch, device, common, state_owner, train_passes=1):
    used=sum(common._stat(p)['st_size'] for p in state_directory.glob('*.pt'))
    remaining=launch['max_checkpoint_total_bytes']-used
    _require(remaining>0,'PREFIX_CHECKPOINT_TOTAL_BUDGET')
    state=state_owner.capture_native_direction_research_state(model,optimizer,recipe=recipe,
        feature_view_identity=model.feature_view_identity,parent_order=order,cursor=parent_cursor,optimizer_updates=updates,device=device,train_passes=train_passes)
    ref=common._save_state(state_directory/(name+'.pt'),state,torch=torch,
        maximum=min(launch['max_checkpoint_bytes'],remaining),counters=counters)
    seal=state['contract_sha256'];del state
    return ref,seal


def _latest_cursor(path, value, *, common, counters, limit):
    from gx1.models.entry_v10.entry_v10_ctx_train_v3 import _attended_session_atomic_write_json, _fsync_directory
    data=(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode()
    _require(counters['json_bytes']+len(data)<=limit,'PREFIX_JSON_BUDGET')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    _attended_session_atomic_write_json(path,value);_fsync_directory(path.parent);counters['json_bytes']+=len(data)
    return common._hash_file(path)


def _fit_input_launch(launch, authority):
    """Reuse the qualified private target adapter for the separate full-fit scope."""
    if launch['run_scope'] == authority.DENSE_DECIMAL_FIT_SCOPE:
        return {**launch, 'run_scope': authority.DENSE_DECIMAL_PRIVATE_SCOPE}
    return launch


def run_native_direction_prefix_fit_window(*, policy_path: Path, policy_file_sha256: str, progress_path: Path) -> dict:
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    limits=require_guarded_cuda_trainer_execution()
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as authority
    window,plan,invocation,marker,launch=authority.require_native_direction_prefix_fit_context(policy_path,policy_file_sha256,progress_path)
    sys.dont_write_bytecode=True
    import numpy as np
    import torch
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.contracts.entry_model_native_train_recipe_v1 import require_native_direction_research_recipe
    from gx1.contracts import entry_native_direction_research_state_v1 as state_owner
    from gx1.contracts.entry_native_direction_feature_view_v1 import build_native_direction_research_model
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.features.htf_features import discard_multi_tf_v4_cache_pages
    from gx1.scripts.run_unified_exit_random_access_full_train_v1 import _dynamic_m5_technical_tree_sha256 as tree_sha
    recipe=require_native_direction_research_recipe(launch['numerical_recipe']);fold=launch['fold'];number=window['invocation_number']
    continuation=launch['run_scope']==authority.CONTINUATION_SCOPE;order_variant=launch['run_scope']==authority.ORDER_SCOPE;train_passes=launch['train_passes']
    decimal_fit=launch['run_scope']==authority.DENSE_DECIMAL_FIT_SCOPE
    input_launch=_fit_input_launch(launch,authority)
    binding_ref=common._hash_file(launch['execution_binding_path']);extra={};binding=common._read(binding_ref,extra)
    context=_launch_metadata(input_launch,common);context['info']['metadata'].update(extra)
    if decimal_fit:
        metadata=context['info']['metadata']
        review=common._read(launch['native_dense_qualification'],metadata)
        private_plan=common._read(review['plan'],metadata)
        for ref in (private_plan['recipe'],review['result'],review['receipt'],review['idle_restore'],
                launch['dense_evaluation_protocol']):
            common._read(ref,metadata)
        initial=launch['decimal_initial_predictions']
        context['decimal_initial_predictions']=dict(initial)
        context['decimal_initial_predictions_stat']=common._stat(initial['path'])
        _require(common._hash_file(initial['path'])==initial,'DECIMAL_INITIAL_PREDICTIONS_SHA')
    origin=launch.get('continuation_origin') if continuation else None
    original_recipe=None;original_initial_measurement=None
    if continuation:
        original_launch=common._read(origin['launch_recipe'],context['info']['metadata'])
        original_recipe=require_native_direction_research_recipe(original_launch['numerical_recipe'])
        original_initial_measurement=common._read(origin['initial_measurement'],context['info']['metadata'])
    for ref in (window['recipe'],binding['plan'],{'path':str(policy_path),'sha256':policy_file_sha256}):common._read(ref,context['info']['metadata'])
    common._freeze(launch,binding,context)
    output=Path(window['output_directory']);output.mkdir(parents=True,exist_ok=True)
    work=output/f'window-{number:04d}';work.mkdir();state_directory=output/'states';state_directory.mkdir(exist_ok=True)
    denied=common._install_read_guard(context,output,Path(launch['runtime_root']),
        extra_read_paths=(origin['terminal_state']['path'],origin['initial_state']['path']) if continuation else (),
        original_initial_prediction=launch['order_origin']['initial_predictions']['path'] if order_variant else None,
        dense_target=context.get('dense_target'),
        decimal_original=context.get('decimal_original') if decimal_fit else None,
        decimal_initial_predictions=context.get('decimal_initial_predictions') if decimal_fit else None)
    counters={k:0 for k in ('model_constructions','model_construction_returns','optimizer_constructions','forward_calls','forward_returns',
        'backward_calls','backward_returns','optimizer_step_calls','optimizer_step_returns','empty_batches',
        'staged_state_load_calls','staged_state_load_returns','resume_state_load_calls','resume_state_load_returns','forbidden_legacy_calls')}
    counters.update(json_bytes=sum(common._stat(p)['st_size'] for p in output.rglob('*.json'))
        +sum(common._stat(p)['st_size'] for folder in ('prefix-cursor','progress') for p in (Path(launch['runtime_root'])/folder).glob('*.json')),forward_kind_calls={})
    started=time.monotonic();deadline=started+window['max_invocation_seconds'];before=common._io();device=torch.device('cuda',0)
    initial_ref=initial_measurement=final_measurement=full_state=None;view_identity=None;parent_cursor=updates=oof_cursor=0
    phase='TRAINING';outcome='FAILED';error=None;source_unchanged=False;checkpoint_refs=[];first_diagnostic=last_diagnostic=None
    prepared=model=optimizer=None;order_sha='0'*64;prior_cursor=None;fresh_identity=None;prior=None;initial_inactive=None
    ceiling=((2 if continuation else 1)*((fold['prefix_rows']+15)//16)
        +33+(fold['oof_end_row']-fold['oof_start_row']+15)//16)
    durable=None;observed_progress=None;bps_scale_ref=None
    common._json(work/'CLAIM.json',{'recipe':window['recipe'],'fold':fold,'invocation_number':number,
        'source_head':binding['source_head'],'plan_sha256':plan['plan_sha256'],'limits':limits},counters,launch['max_published_json_bytes'])
    previous_profile=sys.getprofile()
    try:
        sys.setprofile(common._deny_legacy_work(previous_profile,counters))
        prepared,targets,view,context_map,ordered=_prepare(context,input_launch,np,signed_bps=recipe.get('signed_bps_supervision',False),
            magnitude_weighted=recipe.get('magnitude_weighted_direction',False),dense_minutes=recipe.get('dense_minute_supervision',False))
        bps_scale_ref=_qualified_bps_scale(context,view)
        order=torch.from_numpy(ordered.copy());order_sha=tree_sha(order)
        prior=_prefix_prior(prepared['arrays'],targets,fold,np)
        if decimal_fit:
            from gx1.scripts.run_entry_native_direction_dense_qualification_v1 import require_decimal_prefix_prior
            require_decimal_prefix_prior(prior,context['decimal_sign_audit'])
        if number>1:
            cursor_ref={'path':window['campaign_cursor_path'],'sha256':marker['pointer_before_sha256']}
            saved_cursor=json.loads(Path(cursor_ref['path']).read_text())
            _require(common._hash_file(cursor_ref['path'])==cursor_ref,'PREVIOUS_CURSOR_SHA')
            prior_cursor=authority.require_prefix_fit_cursor(saved_cursor,expected_recipe=window['recipe'],expected_fold=fold,verify_files=True)
            _require(prior_cursor['parent_order_sha256']==order_sha,'SAME_FIXED_PERMUTATION')
        trainer._set_deterministic(recipe['seed'],device,recipe['precision_policy'])
        counters['model_constructions']+=1
        model=build_native_direction_research_model(view,prefix_normalization=context['fitted'],constructor_metadata=context['constructor'],device=device)
        counters['model_construction_returns']+=1;model.train()
        optimizer,_optimizer_binding=state_owner.build_native_direction_research_optimizer(model,recipe=recipe);counters['optimizer_constructions']+=1
        view_identity=model.feature_view_identity
        fresh=state_owner.capture_native_direction_research_state(model,optimizer,recipe=original_recipe if continuation else recipe,feature_view_identity=view_identity,
            parent_order=order,cursor=0,optimizer_updates=0,device=device)
        fresh_identity=tree_sha({key:fresh[key] for key in ('binding','model_state','buffer_state','optimizer_state','rng_state','module_training')})
        initial_inactive=tree_sha({name:p for name,p in model.named_parameters() if not p.requires_grad});del fresh
        if order_variant:
            old_initial=context['order_original_initial']
            _require(old_initial['fresh_identity_sha256']==fresh_identity and old_initial['own_prefix_prior']==prior, 'ORDER_FRESH_BASELINE_IDENTITY')
        diagnostic_ids=(_fixed_order(prepared['arrays']['parent_rows'],fold['prefix_rows'],SEED,np)[:256]
            if order_variant else ordered[:256])
        if prior_cursor is None and continuation:
            initial_ref=origin['initial_state'];initial_measurement=origin['initial_measurement']
            _require(original_initial_measurement['fresh_identity_sha256']==fresh_identity
                and original_initial_measurement['own_prefix_prior']==prior, 'CONTINUATION_SAME_OWN_FRESH_BASELINE')
            loaded=common._load_state(origin['terminal_state'],torch=torch,maximum=launch['max_checkpoint_bytes'],counters=counters)
            migrated=state_owner.migrate_one_pass_terminal_to_three_pass(loaded,model,optimizer,
                original_recipe=original_recipe,continuation_recipe=recipe,feature_view_identity=view_identity,parent_order=order,device=device)
            parent_cursor=migrated['cursor'];updates=migrated['optimizer_updates']
            full_state,seal=_save_snapshot(model,optimizer,context=context,recipe=recipe,order=order,parent_cursor=parent_cursor,updates=updates,
                state_directory=state_directory,name='CONTINUATION_INITIAL_1PASS',launch=launch,counters=counters,torch=torch,device=device,
                common=common,state_owner=state_owner,train_passes=train_passes)
            _require(seal==migrated['contract_sha256'],'CONTINUATION_MIGRATED_STATE_PARITY')
            del loaded,migrated
        elif prior_cursor is None:
            initial_ref,_seal=_save_snapshot(model,optimizer,context=context,recipe=recipe,order=order,parent_cursor=0,updates=0,
                state_directory=state_directory,name='INITIAL_0U',launch=launch,counters=counters,torch=torch,device=device,common=common,state_owner=state_owner)
            measured,_=_evaluate(model,prepared,targets,view,context_map,diagnostic_ids,cutoff=fold['train_cutoff_ns'],kind='INITIAL_TRAIN',
                counters=counters,ceiling=ceiling,device=device,torch=torch,np=np,tree_sha=tree_sha,parity=True)
            if order_variant:
                _require_original_initial_match(measured,launch['order_origin']['initial_predictions'],common=common,np=np)
            if decimal_fit:
                from gx1.scripts.run_entry_native_direction_dense_qualification_v1 import require_decimal_initial_prediction_parity
                require_decimal_initial_prediction_parity(measured['arrays'],context['decimal_initial_predictions'],np=np)
            initial_measurement=_publish_measurement('INITIAL_TRAIN',measured,output=work,view=view_identity,prior=prior,
                own_initial_state=initial_ref,fresh_identity=fresh_identity,context_map=context_map,counters=counters,launch=launch,np=np,common=common,tree_sha=tree_sha)
            del measured
        else:
            _require(prior_cursor['feature_view_identity']==view_identity,'RESUME_VIEW_IDENTITY')
            initial_ref=prior_cursor['own_initial_state'];initial_measurement=prior_cursor['initial_measurement']
            original=common._read(initial_measurement,context['info']['metadata'])
            _require(original['fresh_identity_sha256']==fresh_identity and original['own_prefix_prior']==prior,'SAME_OWN_FRESH_BASELINE')
            loaded=common._load_state(prior_cursor['full_state'],torch=torch,maximum=launch['max_checkpoint_bytes'],counters=counters)
            restored=state_owner.restore_native_direction_research_state(loaded,model,optimizer,recipe=recipe,
                feature_view_identity=view_identity,parent_order=order,device=device,train_passes=train_passes)
            _require(restored['cursor']==prior_cursor['parent_cursor'] and restored['optimizer_updates']==prior_cursor['optimizer_updates'],
                'RESTORED_PROGRESS')
            restored_state=state_owner.capture_native_direction_research_state(model,optimizer,recipe=recipe,feature_view_identity=view_identity,
                parent_order=order,cursor=restored['cursor'],optimizer_updates=restored['optimizer_updates'],device=device,train_passes=train_passes)
            _require(restored_state['contract_sha256']==loaded['contract_sha256'],'EXACT_SERIALIZED_FULL_RESUME');del restored_state
            del loaded;parent_cursor=restored['cursor'];updates=restored['optimizer_updates'];phase=prior_cursor['phase_state']
            final_measurement=prior_cursor['final_train_measurement'];full_state=prior_cursor['full_state']
        durable={'state':full_state or initial_ref,'parent_cursor':parent_cursor,'optimizer_updates':updates}
        while parent_cursor<train_passes*len(ordered) and time.monotonic()<deadline:
            within,right,next_cursor=_next_training_parent_range(parent_cursor,len(ordered),train_passes)
            raw,target=_batch(prepared,targets,view,ordered[within:right],np)
            features,device_target=_device_batch(raw,target,torch,device)
            diagnostic=(counters['optimizer_step_returns']==0 or right==len(ordered))
            record,did_update=_train_batch(model,optimizer,features,device_target,cutoff=fold['train_cutoff_ns'],
                counters=counters,ceiling=ceiling,torch=torch,diagnostics=diagnostic)
            parent_cursor=next_cursor;updates+=int(did_update)
            if did_update and first_diagnostic is None:first_diagnostic={'next_parent_cursor':parent_cursor,**record}
            if right==len(ordered):last_diagnostic={'next_parent_cursor':parent_cursor,**record}
            _require(updates<=recipe['max_updates'],'PREFIX_UPDATE_CEILING')
            del raw,target,features,device_target
            if did_update and updates%launch['checkpoint_every_updates']==0:
                ref,seal=_save_snapshot(model,optimizer,context=context,recipe=recipe,order=order,parent_cursor=parent_cursor,updates=updates,
                    state_directory=state_directory,name=f'w{number:04d}_u{updates:06d}_p{parent_cursor:06d}',launch=launch,
                    counters=counters,torch=torch,device=device,common=common,state_owner=state_owner,train_passes=train_passes)
                durable={'state':ref,'contract_sha256':seal,'optimizer_updates':updates,'parent_cursor':parent_cursor}
                checkpoint_refs.append(durable)
                common._json(work/f'CHECKPOINT_u{updates:06d}_p{parent_cursor:06d}.json',{'recipe':window['recipe'],'fold':fold,
                    'feature_view_identity':view_identity,'parent_order_sha256':order_sha,**durable},counters,launch['max_published_json_bytes'])
        if parent_cursor==train_passes*len(ordered):phase='SCORING'
        if phase=='SCORING' and final_measurement is None and time.monotonic()<deadline:
            measured,_=_evaluate(model,prepared,targets,view,context_map,diagnostic_ids,cutoff=fold['train_cutoff_ns'],kind='FINAL_TRAIN',
                counters=counters,ceiling=ceiling,device=device,torch=torch,np=np,tree_sha=tree_sha)
            final_measurement=_publish_measurement('FINAL_TRAIN',measured,output=work,view=view_identity,prior=prior,
                own_initial_state=initial_ref,fresh_identity=fresh_identity,context_map=context_map,counters=counters,launch=launch,np=np,common=common,tree_sha=tree_sha)
            del measured
        oof_ref=None
        if phase=='SCORING' and final_measurement is not None:
            selected=prepared['arrays']['parent_rows'][fold['oof_start_row']:fold['oof_end_row']]
            measured,oof_cursor=_evaluate(model,prepared,targets,view,context_map,selected,cutoff=fold['oof_cutoff_ns'],kind='OOF',
                counters=counters,ceiling=ceiling,device=device,torch=torch,np=np,tree_sha=tree_sha,deadline=deadline)
            if measured is not None:
                oof_ref=_publish_measurement('OOF',measured,output=work,view=view_identity,prior=prior,own_initial_state=initial_ref,
                    fresh_identity=fresh_identity,context_map=context_map,counters=counters,launch=launch,np=np,common=common,tree_sha=tree_sha)
                del measured;phase='COMPLETE'
        _require(tree_sha({name:p for name,p in model.named_parameters() if not p.requires_grad})==initial_inactive,'INACTIVE_WEIGHTS_UNCHANGED')
        full_state,seal=_save_snapshot(model,optimizer,context=context,recipe=recipe,order=order,parent_cursor=parent_cursor,updates=updates,
            state_directory=state_directory,name=f'w{number:04d}_TERMINAL_u{updates:06d}_p{parent_cursor:06d}',launch=launch,
            counters=counters,torch=torch,device=device,common=common,state_owner=state_owner,train_passes=train_passes)
        durable={'state':full_state,'contract_sha256':seal,'optimizer_updates':updates,'parent_cursor':parent_cursor};checkpoint_refs.append(durable)
        _require(counters['model_constructions']==counters['model_construction_returns']==counters['optimizer_constructions']==1
            and counters['forward_calls']==counters['forward_returns'] and counters['backward_calls']==counters['backward_returns']
            ==counters['optimizer_step_calls']==counters['optimizer_step_returns'] and counters['forbidden_legacy_calls']==0,'ACTUAL_CALL_RETURNS')
        common._freeze(launch,binding,context);source_unchanged=True
        outcome='COMPLETE' if phase=='COMPLETE' else 'RESUMABLE'
    except BaseException as exc:
        error=type(exc).__name__+':'+str(exc)
        observed_progress={'parent_cursor':parent_cursor,'optimizer_updates':updates,'oof_cursor':oof_cursor}
        if durable is not None:
            full_state=durable['state'];parent_cursor=durable['parent_cursor'];updates=durable['optimizer_updates']
            phase='SCORING' if parent_cursor==train_passes*fold['prefix_rows'] else 'TRAINING'
            if phase=='TRAINING':final_measurement=None;oof_cursor=0
        else:parent_cursor=updates=oof_cursor=0;phase='TRAINING';full_state=initial_ref
        try:common._freeze(launch,binding,context);source_unchanged=True
        except Exception:source_unchanged=False
    finally:
        sys.setprofile(previous_profile)
        if prepared is not None:context['physical'].release_pages(context['ancestor'],prepared,discard_multi_tf_v4_cache_pages)
    after=common._io()
    if after['rchar']>context['ancestor_plan']['budget']['max_rchar_bytes']:
        outcome='FAILED';error=(error+';' if error else '')+'INHERITED_PHYSICAL_RCHAR_CEILING'
        if phase=='COMPLETE':phase='SCORING'
    result=common._json(work/'RESULT.json',{'schema_version':'gx1_native_direction_prefix_window_result_v1','outcome':outcome,'phase_state':phase,
        'recipe':window['recipe'],'fold':fold,'feature_view_identity':view_identity,'parent_order_sha256':order_sha,
        'parent_cursor':parent_cursor,'optimizer_updates':updates,'oof_cursor':oof_cursor,'own_initial_state':initial_ref,
        'initial_measurement':initial_measurement,'final_train_measurement':final_measurement,'oof_measurement':locals().get('oof_ref'),
        'full_state':full_state,'durable_checkpoint':durable,'observed_progress_before_failure':observed_progress,'checkpoints':checkpoint_refs,'first_update_diagnostic':first_diagnostic,'last_batch_diagnostic':last_diagnostic,
        'counts':{k:v for k,v in counters.items() if k!='json_bytes'},'counter_scope':'actual admitted caller boundaries; no independent delegated-loader callback claim',
        'source_unchanged':source_unchanged,'metadata_bookends':context['info']['metadata'],'payload_stat_bookends':context['ancestor_plan']['payloads'],
        'forbidden_open_paths':denied,'limits':limits,'elapsed_seconds':time.monotonic()-started,'io_before':before,'io_after':after,
        'rchar_delta':after['rchar']-before['rchar'],'rchar_scope':'process reads; mmap fault bytes excluded','error':error,
        'test_data_used':False,'qualification_weights_reused':False,'predictive_value_qualified':False,
        **({'own_prefix_bps_scale':bps_scale_ref} if bps_scale_ref is not None else {})},counters,launch['max_published_json_bytes'])
    cursor=authority.build_prefix_fit_cursor(recipe_ref=window['recipe'],fold=fold,feature_view_identity=view_identity,phase_state=phase,outcome=outcome,
        optimizer_updates=updates,parent_cursor=parent_cursor,parent_order_sha256=order_sha,full_state=full_state,own_initial_state=initial_ref,
        initial_measurement=initial_measurement,final_train_measurement=final_measurement,oof_cursor=oof_cursor,result=result,train_passes=train_passes)
    common._json(work/'CURSOR.json',cursor,counters,launch['max_published_json_bytes'])
    pointer=_latest_cursor(window['campaign_cursor_path'],cursor,common=common,counters=counters,limit=launch['max_published_json_bytes'])
    progress=authority.build_prefix_fit_progress(window=window,plan=plan,invocation=invocation,checkpoint_pointer=pointer,cursor=cursor)
    common._json(progress_path,progress,counters,launch['max_published_json_bytes'])
    authority.require_prefix_fit_progress(progress,plan_sha256=plan['plan_sha256'],invocation=invocation,
        expected_selection_receipt_sha256=None,verify_file=True)
    common._freeze(launch,binding,context)
    if error is not None:raise RuntimeError('NATIVE_DIRECTION_PREFIX_WINDOW_FAILED:'+error)
    return {'decision':outcome,'result':result,'checkpoint_pointer':pointer,'counts':counters,'progress_path':str(progress_path)}
