"""Pure clocks/support for an authenticated row-membership artifact; no file IO or RNG.

Callers authenticate the manifest/order raw-file refs before supplying their
decoded values. This module verifies the bound array and clocks; it never
regenerates the order or grants training, model, checkpoint or launch authority.
"""
from __future__ import annotations

import copy
import hashlib
import json
import numpy as np

ROWS=354550
FULL_ROWS=354528
BATCH_ROWS=96
FULL_BATCHES=3693
BATCHES=3694
TAIL_ROWS=22
ALGORITHM='PCG64_permute_full_prefix_sort_within_B96_keep_original_tail'
DESCRIPTOR_SCHEMA='gx1_native_provider_hlc_row_membership_descriptor_v1'
SCHEDULE_SCHEMA='gx1_native_provider_hlc_row_membership_schedule_v1'
STEP_SCHEMA='gx1_native_provider_hlc_row_membership_step_binding_v1'
CURSOR_SCHEMA='gx1_native_provider_hlc_row_membership_cursor_binding_v1'
_HASH_FIELDS=('batch_row_positions_sha256','batch_source_rows_sha256','batch_issue_ns_sha256','batch_activity_mask_sha256')
_DESCRIPTOR_KEYS={'schema_version','canonical_schedule_sha256','source_rows_sha256','issue_ns_sha256',
    'recipe_sha256','manifest','manifest_content_sha256','order','order_array','seed','algorithm',
    'rows','full_rows','batch_rows','full_batches','last_batch_rows','passes','contract_sha256'}
_SCHEDULE_KEYS={'schema_version','descriptor','canonical_schedule','rows','batch_rows','passes',
    'batches_per_pass','logical_batches','optimizer_updates','skipped_batches','update_flags',
    'source_rows_sha256','issue_ns_sha256','activity_mask_sha256','activity_denominator_total',
    'batch_activity_known','batch_D_activity_all_TF','shuffle','drop_last','support_rule','contract_sha256',*_HASH_FIELDS}


def _need(ok,reason):
    if not bool(ok):
        raise ValueError('ROW_MEMBERSHIP_SCHEDULE_'+reason)


def _json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _seal(value):
    return {**value,'contract_sha256':_json_sha(value)}


def _sha(value):
    return type(value)is str and len(value)==64 and all(c in'0123456789abcdef'for c in value)


def _reference(value):
    return (type(value)is dict and set(value)=={'path','sha256'} and type(value['path'])is str
            and value['path'].startswith('/') and '..'not in value['path'].split('/') and _sha(value['sha256']))


def _sealed(value,reason):
    _need(type(value)is dict and _sha(value.get('contract_sha256')),reason)
    _need(_json_sha({k:v for k,v in value.items()if k!='contract_sha256'})==value['contract_sha256'],reason)


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _canonical(canonical):
    _sealed(canonical,'CANONICAL_SEAL')
    expected={'schema_version':'gx1_provider_hlc_whole_train_schedule_v1','rows':ROWS,
              'batch_rows':BATCH_ROWS,'passes':1,'batches_per_pass':BATCHES,'logical_batches':BATCHES,
              'shuffle':False,'drop_last':False}
    _need(all(type(canonical.get(k))is type(v)and canonical[k]==v for k,v in expected.items()),'CANONICAL_GEOMETRY')
    _need(_sha(canonical.get('source_rows_sha256'))and _sha(canonical.get('issue_ns_sha256')),'CANONICAL_CLOCK_REFS')
    for field in('batch_source_rows_sha256','batch_issue_ns_sha256'):
        _need(type(canonical.get(field))is list and len(canonical[field])==BATCHES
              and all(_sha(v)for v in canonical[field]),'CANONICAL_BATCH_INVENTORY')
    flags=canonical.get('update_flags')
    _need(type(flags)is list and len(flags)==BATCHES and all(type(v)is bool for v in flags)
          and type(canonical.get('optimizer_updates'))is int and canonical['optimizer_updates']==sum(flags)
          and type(canonical.get('skipped_batches'))is int and canonical['skipped_batches']==BATCHES-sum(flags),'CANONICAL_POSITION')


def schedule_descriptor(canonical_schedule,recipe,manifest,manifest_ref):
    """Metadata only; callers authenticate the manifest's exact raw-file ref."""
    _canonical(canonical_schedule)
    _need(type(recipe)is dict and type(recipe.get('seed'))is int
          and type(recipe.get('train_batch_rows'))is int and recipe['train_batch_rows']==BATCH_ROWS
          and type(recipe.get('passes'))is int and recipe['passes']==1,'RECIPE')
    _need(_reference(manifest_ref)and type(manifest)is dict,'MANIFEST_REF')
    expected={'schema_version':'gx1_source_only_row_membership_candidate_v1',
              'status':'SEALED_SOURCE_ONLY_BEFORE_THIS_CANDIDATE_LABEL_EVALUATION',
              'rows':ROWS,'full_rows':FULL_ROWS,'batch_rows':BATCH_ROWS,'full_batches':FULL_BATCHES,
              'tail_rows':TAIL_ROWS,'seed':recipe['seed'],'algorithm':ALGORITHM,
              'whole_population_once':True,'within_batch_sorted':True,'original_tail_unchanged':True,
              'selection_uses_labels_predictions_support':False,'global_NumPy_RNG_unchanged':True,
              'candidate_orders_generated':1,'label_prediction_or_target_payload_reads':0}
    _need(all(type(manifest.get(k))is type(v)and manifest[k]==v for k,v in expected.items()),'MANIFEST_CONTRACT')
    header=manifest.get('order_array')
    _need(_reference(manifest.get('order'))and _reference(manifest.get('original_epoch_plan'))
          and _reference(manifest.get('source_generator'))and _sha(manifest.get('raw_draw_sha256'))
          and manifest.get('numpy_version')==np.__version__,'MANIFEST_BINDINGS')
    _need(type(header)is dict and set(header)=={'dtype','shape','sha256','nbytes'}
          and header['dtype']=='<i8'and header['shape']==[ROWS]and type(header['nbytes'])is int
          and header['nbytes']==ROWS*8 and _sha(header['sha256']),'ORDER_HEADER')
    return _seal({'schema_version':DESCRIPTOR_SCHEMA,
        'canonical_schedule_sha256':canonical_schedule['contract_sha256'],
        'source_rows_sha256':canonical_schedule['source_rows_sha256'],'issue_ns_sha256':canonical_schedule['issue_ns_sha256'],
        'recipe_sha256':_json_sha(recipe),'manifest':copy.deepcopy(manifest_ref),'manifest_content_sha256':_json_sha(manifest),
        'order':copy.deepcopy(manifest['order']),'order_array':copy.deepcopy(header),'seed':recipe['seed'],
        'algorithm':ALGORITHM,'rows':ROWS,'full_rows':FULL_ROWS,'batch_rows':BATCH_ROWS,
        'full_batches':FULL_BATCHES,'last_batch_rows':TAIL_ROWS,'passes':1})


def _descriptor(descriptor):
    _need(type(descriptor)is dict and set(descriptor)==_DESCRIPTOR_KEYS,'DESCRIPTOR_KEYS')
    _sealed(descriptor,'DESCRIPTOR_SEAL')
    expected={'schema_version':DESCRIPTOR_SCHEMA,'algorithm':ALGORITHM,'rows':ROWS,'full_rows':FULL_ROWS,
              'batch_rows':BATCH_ROWS,'full_batches':FULL_BATCHES,'last_batch_rows':TAIL_ROWS,'passes':1}
    _need(all(type(descriptor.get(k))is type(v)and descriptor[k]==v for k,v in expected.items())
          and type(descriptor['seed'])is int,'DESCRIPTOR_GEOMETRY')
    _need(all(_sha(descriptor[k])for k in('canonical_schedule_sha256','source_rows_sha256','issue_ns_sha256',
          'recipe_sha256','manifest_content_sha256'))and _reference(descriptor['manifest'])
          and _reference(descriptor['order']),'DESCRIPTOR_BINDINGS')
    h=descriptor['order_array']
    _need(type(h)is dict and set(h)=={'dtype','shape','sha256','nbytes'}and h['dtype']=='<i8'
          and h['shape']==[ROWS]and type(h['nbytes'])is int and h['nbytes']==ROWS*8
          and _sha(h['sha256']),'DESCRIPTOR_ORDER_HEADER')


def _axes(source_rows,issue_ns):
    _need(all(isinstance(a,np.ndarray)and a.dtype.str=='<i8'and a.shape==(ROWS,)
              and np.all(np.diff(a)>0)for a in(source_rows,issue_ns)),'SOURCE_ISSUE_AXES')


def _canonical_clock(canonical,source_rows,issue_ns):
    _axes(source_rows,issue_ns)
    _need(_array_sha(source_rows)==canonical['source_rows_sha256'],'CANONICAL_SOURCE_CLOCK')
    _need(_array_sha(issue_ns)==canonical['issue_ns_sha256'],'CANONICAL_ISSUE_CLOCK')
    for j,start in enumerate(range(0,ROWS,BATCH_ROWS)):
        stop=min(start+BATCH_ROWS,ROWS)
        _need(_array_sha(source_rows[start:stop])==canonical['batch_source_rows_sha256'][j],'CANONICAL_BATCH_SOURCE_CLOCK')
        _need(_array_sha(issue_ns[start:stop])==canonical['batch_issue_ns_sha256'][j],'CANONICAL_BATCH_ISSUE_CLOCK')


def _order(order,descriptor):
    _need(isinstance(order,np.ndarray)and order.dtype.str=='<i8'and order.shape==(ROWS,),'ORDER_ARRAY')
    _need(np.array_equal(np.sort(order),np.arange(ROWS,dtype=np.int64)),'ORDER_BIJECTION')
    _need(np.array_equal(order[FULL_ROWS:],np.arange(FULL_ROWS,ROWS,dtype=np.int64)),'ORDER_TAIL')
    _need(np.all(np.diff(order[:FULL_ROWS].reshape(FULL_BATCHES,BATCH_ROWS),axis=1)>0),'ORDER_PACKET_SORT')
    _need(_array_sha(order)==descriptor['order_array']['sha256']and order.nbytes==descriptor['order_array']['nbytes'],'ORDER_ARRAY_SHA')


def _activity(activity_mask):
    _need(isinstance(activity_mask,np.ndarray)and activity_mask.dtype==np.bool_
          and activity_mask.shape==(ROWS,5),'ACTIVITY_MASK')


def _runtime_batches(source_rows,issue_ns,order,activity_mask):
    values={k:[]for k in _HASH_FIELDS};values.update(batch_activity_known=[],batch_D_activity_all_TF=[])
    for start in range(0,ROWS,BATCH_ROWS):
        take=order[start:min(start+BATCH_ROWS,ROWS)];mask=activity_mask[take];counts=mask.sum(0,dtype=np.int64)
        values['batch_row_positions_sha256'].append(_array_sha(take))
        values['batch_source_rows_sha256'].append(_array_sha(source_rows[take]))
        values['batch_issue_ns_sha256'].append(_array_sha(issue_ns[take]))
        values['batch_activity_mask_sha256'].append(_array_sha(mask))
        values['batch_activity_known'].append(counts.tolist());values['batch_D_activity_all_TF'].append(int(counts.sum()))
    return values


def build_schedule(canonical_schedule,recipe,source_rows,issue_ns,order,manifest,manifest_ref,activity_mask):
    """After actual context/order load: bind source clocks and own native support."""
    descriptor=schedule_descriptor(canonical_schedule,recipe,manifest,manifest_ref)
    _canonical_clock(canonical_schedule,source_rows,issue_ns);_order(order,descriptor);_activity(activity_mask)
    batches=_runtime_batches(source_rows,issue_ns,order,activity_mask)
    flags=[D>0 for D in batches['batch_D_activity_all_TF']]
    return _seal({'schema_version':SCHEDULE_SCHEMA,'descriptor':descriptor,'canonical_schedule':copy.deepcopy(canonical_schedule),
        'rows':ROWS,'batch_rows':BATCH_ROWS,'passes':1,'batches_per_pass':BATCHES,'logical_batches':BATCHES,
        'optimizer_updates':sum(flags),'skipped_batches':BATCHES-sum(flags),'update_flags':flags,
        'source_rows_sha256':descriptor['source_rows_sha256'],'issue_ns_sha256':descriptor['issue_ns_sha256'],
        'activity_mask_sha256':_array_sha(activity_mask),'activity_denominator_total':sum(batches['batch_D_activity_all_TF']),
        'shuffle':True,'drop_last':False,'support_rule':'Own validated native activity_mask per candidate batch; D=sum all5TF activity_known; update iff D>0.',**batches})


def validate_schedule(schedule):
    _need(type(schedule)is dict and set(schedule)==_SCHEDULE_KEYS,'SCHEDULE_KEYS')
    _sealed(schedule,'SCHEDULE_SEAL');_descriptor(schedule['descriptor']);_canonical(schedule['canonical_schedule'])
    d=schedule['descriptor'];c=schedule['canonical_schedule']
    _need(d['canonical_schedule_sha256']==c['contract_sha256']
          and schedule['source_rows_sha256']==d['source_rows_sha256']==c['source_rows_sha256']
          and schedule['issue_ns_sha256']==d['issue_ns_sha256']==c['issue_ns_sha256'],'SCHEDULE_CANONICAL_BINDING')
    expected={'schema_version':SCHEDULE_SCHEMA,'rows':ROWS,'batch_rows':BATCH_ROWS,'passes':1,
              'batches_per_pass':BATCHES,'logical_batches':BATCHES,'shuffle':True,'drop_last':False}
    _need(all(type(schedule.get(k))is type(v)and schedule[k]==v for k,v in expected.items()),'SCHEDULE_GEOMETRY')
    for field in _HASH_FIELDS:
        _need(type(schedule[field])is list and len(schedule[field])==BATCHES and all(_sha(h)for h in schedule[field]),'BATCH_HASH_INVENTORY')
    _need(_sha(schedule['activity_mask_sha256']),'ACTIVITY_MASK_SHA_REF')
    counts=schedule['batch_activity_known'];denominators=schedule['batch_D_activity_all_TF'];flags=schedule['update_flags']
    _need(type(counts)is type(denominators)is type(flags)is list and len(counts)==len(denominators)==len(flags)==BATCHES,'SUPPORT_INVENTORY')
    for j,(cells,D,flag)in enumerate(zip(counts,denominators,flags)):
        rows=TAIL_ROWS if j==FULL_BATCHES else BATCH_ROWS
        _need(type(cells)is list and len(cells)==5 and all(type(v)is int and 0<=v<=rows for v in cells)
              and type(D)is int and D==sum(cells)and type(flag)is bool and flag==(D>0),'SUPPORT_FLAGS')
    _need(type(schedule['optimizer_updates'])is int and schedule['optimizer_updates']==sum(flags)
          and type(schedule['skipped_batches'])is int and schedule['skipped_batches']==BATCHES-sum(flags)
          and type(schedule['activity_denominator_total'])is int and schedule['activity_denominator_total']==sum(denominators),'SUPPORT_TOTALS')
    return schedule


def resolve_batches(context,schedule,order,activity_mask):
    """Verify all loaded axes/order/support, then return readonly candidate takes."""
    validate_schedule(schedule);source=context['source_rows'];issues=context['truth']['issue_ns']
    _canonical_clock(schedule['canonical_schedule'],source,issues);_order(order,schedule['descriptor']);_activity(activity_mask)
    _need(_array_sha(activity_mask)==schedule['activity_mask_sha256'],'ACTIVITY_MASK_SHA')
    actual=_runtime_batches(source,issues,order,activity_mask)
    _need(all(actual[k]==schedule[k]for k in actual),'RUNTIME_BATCH_CLOCK_SUPPORT')
    batches=[]
    for start in range(0,ROWS,BATCH_ROWS):
        take=order[start:min(start+BATCH_ROWS,ROWS)].copy();take.flags.writeable=False;batches.append(take)
    return batches


def _expected_step_binding(schedule,index):
    return {'schema_version':STEP_SCHEMA,'schedule_sha256':schedule['contract_sha256'],
        'candidate_batch_index':index,'rows':TAIL_ROWS if index==FULL_BATCHES else BATCH_ROWS,
        'row_positions_sha256':schedule['batch_row_positions_sha256'][index],
        'source_rows_sha256':schedule['batch_source_rows_sha256'][index],
        'issue_ns_sha256':schedule['batch_issue_ns_sha256'][index],
        'activity_mask_sha256':schedule['batch_activity_mask_sha256'][index],
        'activity_known':list(schedule['batch_activity_known'][index]),
        'D_activity_all_TF':schedule['batch_D_activity_all_TF'][index],'skip_update':not schedule['update_flags'][index]}


def expected_step_binding(schedule,index):
    validate_schedule(schedule);_need(type(index)is int and 0<=index<BATCHES,'STEP_INDEX')
    return _expected_step_binding(schedule,index)


def step_binding(context,schedule,index,take,packet,activity_mask):
    """Verify an actual gathered packet against its exact candidate clock/support."""
    expected=expected_step_binding(schedule,index);_activity(activity_mask)
    _need(isinstance(take,np.ndarray)and take.dtype.str=='<i8'and take.shape==(expected['rows'],)
          and np.all(take>=0)and np.all(take<ROWS)and np.all(np.diff(take)>0)
          and _array_sha(take)==expected['row_positions_sha256'],'ACTUAL_PACKET_POSITIONS')
    source=context['source_rows'];truth=context['truth'];issues=truth['issue_ns']
    _need(_array_sha(source[take])==expected['source_rows_sha256']
          and _array_sha(issues[take])==expected['issue_ns_sha256'],'ACTUAL_PACKET_CLOCK')
    _need(type(packet)is dict and type(packet.get('truth'))is dict and set(packet['truth'])==set(truth),'PACKET_TRUTH_KEYS')
    for key,value in truth.items():
        received=packet['truth'][key]
        if isinstance(value,np.ndarray):
            _need(value.ndim>0 and len(value)==ROWS and isinstance(received,np.ndarray)
                  and received.dtype==value.dtype and np.array_equal(received,value[take]),'PACKET_TRUTH_JOIN')
        else:
            _need(received==value,'PACKET_TRUTH_METADATA')
    mask=activity_mask[take]
    _need(np.array_equal(packet['truth']['activity_known'],mask)
          and _array_sha(mask)==expected['activity_mask_sha256']
          and mask.sum(0,dtype=np.int64).tolist()==expected['activity_known']
          and int(mask.sum())==expected['D_activity_all_TF'],'ACTUAL_PACKET_SUPPORT')
    return expected


def _cursor_binding(schedule,cursor):
    updates=sum(schedule['update_flags'][:cursor]);eof=cursor==BATCHES
    position={'logical_cursor':cursor,'completed_passes':cursor//BATCHES,'next_batch_in_pass':cursor%BATCHES,
              'optimizer_updates':updates,'skipped_batches':cursor-updates}
    next_step=None if eof else _expected_step_binding(schedule,cursor)
    return {'schema_version':CURSOR_SCHEMA,'schedule_sha256':schedule['contract_sha256'],
        'descriptor_sha256':schedule['descriptor']['contract_sha256'],'position':position,
        'processed_rows':min(cursor*BATCH_ROWS,ROWS),'phase':'FINAL_SCORE'if eof else'TRAIN',
        'next_candidate_batch_index':None if eof else cursor,
        'next_row_positions_sha256':None if eof else next_step['row_positions_sha256'],
        'next_source_rows_sha256':None if eof else next_step['source_rows_sha256'],
        'next_issue_ns_sha256':None if eof else next_step['issue_ns_sha256'],
        'next_activity_mask_sha256':None if eof else next_step['activity_mask_sha256'],
        'expected_prefix_sha256':_json_sha([_expected_step_binding(schedule,j)for j in range(cursor)])}


def cursor_binding(schedule,cursor):
    validate_schedule(schedule);_need(type(cursor)is int and 0<=cursor<=BATCHES,'LOGICAL_CURSOR')
    return _cursor_binding(schedule,cursor)


def validate_prefix(schedule,steps,cursor):
    validate_schedule(schedule);_need(type(cursor)is int and 0<=cursor<=BATCHES,'LOGICAL_CURSOR')
    _need(type(steps)is list and len(steps)==cursor,'PREFIX_LENGTH')
    for j,step in enumerate(steps):
        _need(type(step)is dict and type(step.get('logical_step'))is int and step['logical_step']==j+1
              and type(step.get('skip_update'))is bool and step['skip_update']==(not schedule['update_flags'][j])
              and step.get('order_binding')==_expected_step_binding(schedule,j),'PREFIX_EXECUTION_BINDING')
    return _cursor_binding(schedule,cursor)
