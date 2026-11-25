"""Pure synthetic clock/support tests; no files, model, checkpoint or real data."""
import copy
import hashlib
import json

import numpy as np
import pytest

from gx1.audit import native_provider_hlc_row_membership_schedule_v1 as m


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def seal(value):
    body={k:v for k,v in value.items()if k!='contract_sha256'}
    return {**body,'contract_sha256':hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()}


def reference(name):
    return {'path':'/synthetic/'+name,'sha256':hashlib.sha256(name.encode()).hexdigest()}


@pytest.fixture(scope='module')
def f():
    source=np.arange(m.ROWS,dtype=np.int64)+1000
    issues=np.arange(m.ROWS,dtype=np.int64)*300_000_000_000+1_609_459_200_000_000_000
    activity=np.ones((m.ROWS,5),np.bool_)
    # Synthetic discontiguous memberships exercise the pure artifact contract;
    # generation provenance/raw-file authentication belongs to the caller.
    prefix=np.arange(m.FULL_ROWS,dtype=np.int64).reshape(m.BATCH_ROWS,m.FULL_BATCHES).T.copy().reshape(-1)
    order=np.r_[prefix,np.arange(m.FULL_ROWS,m.ROWS,dtype=np.int64)]
    canonical=seal({'schema_version':'gx1_provider_hlc_whole_train_schedule_v1','rows':m.ROWS,
        'batch_rows':m.BATCH_ROWS,'passes':1,'batches_per_pass':m.BATCHES,'logical_batches':m.BATCHES,
        'source_rows_sha256':array_sha(source),'issue_ns_sha256':array_sha(issues),
        'batch_source_rows_sha256':[array_sha(source[a:a+96])for a in range(0,m.ROWS,96)],
        'batch_issue_ns_sha256':[array_sha(issues[a:a+96])for a in range(0,m.ROWS,96)],
        'update_flags':[True]*m.BATCHES,'optimizer_updates':m.BATCHES,'skipped_batches':0,'shuffle':False,'drop_last':False})
    recipe={'seed':20260911,'train_batch_rows':96,'passes':1,'optimizer':{'lr':.0001}}
    manifest={'schema_version':'gx1_source_only_row_membership_candidate_v1',
        'status':'SEALED_SOURCE_ONLY_BEFORE_THIS_CANDIDATE_LABEL_EVALUATION','rows':m.ROWS,'full_rows':m.FULL_ROWS,
        'batch_rows':96,'full_batches':m.FULL_BATCHES,'tail_rows':22,'seed':recipe['seed'],'algorithm':m.ALGORITHM,
        'whole_population_once':True,'within_batch_sorted':True,'original_tail_unchanged':True,
        'selection_uses_labels_predictions_support':False,'global_NumPy_RNG_unchanged':True,
        'candidate_orders_generated':1,'label_prediction_or_target_payload_reads':0,
        'numpy_version':np.__version__,'order':reference('order.npy'),'original_epoch_plan':reference('original.json'),
        'source_generator':reference('generator.py'),'raw_draw_sha256':'a'*64,
        'order_array':{'dtype':'<i8','shape':[m.ROWS],'sha256':array_sha(order),'nbytes':m.ROWS*8}}
    manifest_ref=reference('manifest.json')
    schedule=m.build_schedule(canonical,recipe,source,issues,order,manifest,manifest_ref,activity)
    truth={'issue_ns':issues,'activity_known':activity,'side_up':np.zeros_like(activity),
           'certificate':reference('certificate.json')}
    context={'source_rows':source,'truth':truth}
    takes=m.resolve_batches(context,schedule,order,activity)
    return dict(source=source,issues=issues,activity=activity,order=order,canonical=canonical,recipe=recipe,
                manifest=manifest,manifest_ref=manifest_ref,schedule=schedule,context=context,takes=takes)


def build(f,**overrides):
    args={k:f[k]for k in('canonical','recipe','source','issues','order','manifest','manifest_ref','activity')}
    args.update(overrides)
    return m.build_schedule(args['canonical'],args['recipe'],args['source'],args['issues'],args['order'],
                            args['manifest'],args['manifest_ref'],args['activity'])


def packet(context,take):
    return {'features':{},'truth':{k:v[take].copy()if isinstance(v,np.ndarray)else copy.deepcopy(v)
                                 for k,v in context['truth'].items()}}


def test_actual_discontiguous_geometry_clocks_and_own_support(f):
    s=f['schedule'];takes=f['takes'];take=takes[0]
    assert len(takes)==3694 and [len(t)for t in takes]==[96]*3693+[22]
    assert np.all(np.diff(take)>1) and np.array_equal(takes[-1],np.arange(m.FULL_ROWS,m.ROWS))
    assert not take.flags.writeable
    assert s['batch_source_rows_sha256'][0]==array_sha(f['source'][take])
    assert s['batch_issue_ns_sha256'][0]==array_sha(f['issues'][take])
    assert s['batch_row_positions_sha256'][0]==array_sha(take)
    assert s['batch_activity_mask_sha256'][0]==array_sha(f['activity'][take])
    assert s['batch_D_activity_all_TF']==[480]*3693+[110]
    assert s['batch_activity_known']==[[96]*5]*3693+[[22]*5]
    assert s['optimizer_updates']==3694 and s['skipped_batches']==0
    assert m.step_binding(f['context'],s,0,take,packet(f['context'],take),f['activity'])==m.expected_step_binding(s,0)
    assert m.validate_schedule(s)==s


def test_descriptor_has_no_payload_io_or_rng_draws(f,monkeypatch):
    before=copy.deepcopy(np.random.get_state())
    def denied(*args,**kwargs):
        raise AssertionError('Payload IO or RNG draw before actual context')
    with monkeypatch.context()as patch:
        patch.setattr('builtins.open',denied);patch.setattr(np,'load',denied)
        patch.setattr(np.random,'Generator',denied);patch.setattr(np.random,'PCG64',denied)
        descriptor=m.schedule_descriptor(f['canonical'],f['recipe'],f['manifest'],f['manifest_ref'])
    after=np.random.get_state()
    assert descriptor==f['schedule']['descriptor']
    assert before[0]==after[0]and np.array_equal(before[1],after[1])and before[2:]==after[2:]


@pytest.mark.parametrize('kind,gate',[
    ('duplicate','ORDER_BIJECTION'),('unsorted','ORDER_PACKET_SORT'),('tail','ORDER_TAIL'),
    ('valid_changed_hash','ORDER_ARRAY_SHA'),('wrong_dtype','ORDER_ARRAY')])
def test_order_tampering_reaches_exact_gate(f,kind,gate):
    order=f['order'].copy()
    if kind=='duplicate':order[1]=order[0]
    elif kind=='unsorted':order[:2]=order[1::-1]
    elif kind=='tail':order[0],order[-1]=order[-1],order[0]
    elif kind=='valid_changed_hash':
        order[0],order[96]=order[96],order[0];order[:96].sort();order[96:192].sort()
    else:order=order.astype(np.int32)
    with pytest.raises(ValueError,match='ROW_MEMBERSHIP_SCHEDULE_'+gate+'$'):
        build(f,order=order)


@pytest.mark.parametrize('key,gate',[('source','CANONICAL_SOURCE_CLOCK'),('issues','CANONICAL_ISSUE_CLOCK')])
def test_full_original_clock_tamper(f,key,gate):
    bad=f[key].copy();bad[-1]+=1
    with pytest.raises(ValueError,match=gate+'$'):
        build(f,**{key:bad})


def test_individual_canonical_batch_clock_is_verified(f):
    canonical=copy.deepcopy(f['canonical']);canonical['batch_source_rows_sha256'][0]='0'*64;canonical=seal(canonical)
    with pytest.raises(ValueError,match='CANONICAL_BATCH_SOURCE_CLOCK$'):
        build(f,canonical=canonical)


@pytest.mark.parametrize('key,value,gate',[
    ('seed',True,'RECIPE'),('train_batch_rows',64,'RECIPE'),('passes',2,'RECIPE')])
def test_recipe_does_not_silently_change(f,key,value,gate):
    recipe={**f['recipe'],key:value}
    with pytest.raises(ValueError,match=gate+'$'):
        build(f,recipe=recipe)


def test_support_is_computed_not_inherited_and_D0_is_not_an_update(f):
    activity=f['activity'].copy();activity[f['order'][:96]]=False
    s=build(f,activity=activity)
    assert s['batch_D_activity_all_TF'][0]==0 and s['batch_activity_known'][0]==[0]*5
    assert s['update_flags'][0]is False and s['optimizer_updates']==3693 and s['skipped_batches']==1
    assert f['canonical']['update_flags'][0]is True and f['schedule']['update_flags'][0]is True
    context={'source_rows':f['source'],'truth':{**f['context']['truth'],'activity_known':activity}}
    takes=m.resolve_batches(context,s,f['order'],activity)
    binding=m.step_binding(context,s,0,takes[0],packet(context,takes[0]),activity)
    cursor=m.validate_prefix(s,[{'logical_step':1,'skip_update':True,'order_binding':binding}],1)
    assert cursor['position']=={'logical_cursor':1,'completed_passes':0,'next_batch_in_pass':1,'optimizer_updates':0,'skipped_batches':1}


def test_runtime_activity_mismatch_is_rejected(f):
    activity=f['activity'].copy();activity[0,0]=False
    with pytest.raises(ValueError,match='ACTIVITY_MASK_SHA$'):
        m.resolve_batches(f['context'],f['schedule'],f['order'],activity)
    with pytest.raises(ValueError,match='ACTIVITY_MASK$'):
        build(f,activity=f['activity'].astype(np.int8))


def test_resealed_inconsistent_support_flags_are_rejected(f):
    bad=copy.deepcopy(f['schedule']);bad['update_flags'][0]=False
    bad['optimizer_updates']-=1;bad['skipped_batches']+=1;bad=seal(bad)
    with pytest.raises(ValueError,match='SUPPORT_FLAGS$'):
        m.validate_schedule(bad)


def test_runtime_rechecks_even_resealed_candidate_batch_clock(f):
    bad=copy.deepcopy(f['schedule']);bad['batch_issue_ns_sha256'][0]='0'*64;bad=seal(bad)
    assert m.validate_schedule(bad)==bad
    with pytest.raises(ValueError,match='RUNTIME_BATCH_CLOCK_SUPPORT$'):
        m.resolve_batches(f['context'],bad,f['order'],f['activity'])


def test_wrong_packet_truth_or_candidate_batch_is_rejected(f):
    take=f['takes'][0];p=packet(f['context'],take);p['truth']['side_up'][0,0]=True
    with pytest.raises(ValueError,match='PACKET_TRUTH_JOIN$'):
        m.step_binding(f['context'],f['schedule'],0,take,p,f['activity'])
    with pytest.raises(ValueError,match='ACTUAL_PACKET_POSITIONS$'):
        m.step_binding(f['context'],f['schedule'],0,f['takes'][1],packet(f['context'],f['takes'][1]),f['activity'])


def test_prefix_records_have_candidate_identity_and_exact_next_cursor(f):
    steps=[{'logical_step':j+1,'skip_update':False,'order_binding':m.expected_step_binding(f['schedule'],j)}for j in range(3)]
    c=m.validate_prefix(f['schedule'],steps,3)
    assert c==m.cursor_binding(f['schedule'],3)
    assert c['next_candidate_batch_index']==3 and c['processed_rows']==288
    assert c['next_row_positions_sha256']==f['schedule']['batch_row_positions_sha256'][3]
    assert c['next_source_rows_sha256']==f['schedule']['batch_source_rows_sha256'][3]
    assert 'original_batch_id'not in steps[0]['order_binding']
    bad=copy.deepcopy(steps);bad[0]['order_binding'],bad[1]['order_binding']=bad[1]['order_binding'],bad[0]['order_binding']
    with pytest.raises(ValueError,match='PREFIX_EXECUTION_BINDING$'):
        m.validate_prefix(f['schedule'],bad,3)
    bad=copy.deepcopy(steps);bad[0]['skip_update']=True
    with pytest.raises(ValueError,match='PREFIX_EXECUTION_BINDING$'):
        m.validate_prefix(f['schedule'],bad,3)
    with pytest.raises(ValueError,match='PREFIX_LENGTH$'):
        m.validate_prefix(f['schedule'],steps,2)


def test_cross_schedule_and_legacy_journal_cannot_restore(f):
    steps=[{'logical_step':1,'skip_update':False,'order_binding':m.expected_step_binding(f['schedule'],0)}]
    steps[0]['order_binding']['schema_version']='gx1_old_block_binding'
    steps[0]['order_binding']['original_batch_id']=0
    with pytest.raises(ValueError,match='PREFIX_EXECUTION_BINDING$'):
        m.validate_prefix(f['schedule'],steps,1)
    bad=copy.deepcopy(f['schedule']);bad['activity_mask_sha256']='0'*64;bad=seal(bad)
    steps[0]['order_binding']=m.expected_step_binding(f['schedule'],0)
    with pytest.raises(ValueError,match='PREFIX_EXECUTION_BINDING$'):
        m.validate_prefix(bad,steps,1)


def test_EOF_and_empty_prefix_are_explicit(f):
    start=m.validate_prefix(f['schedule'],[],0);end=m.cursor_binding(f['schedule'],3694)
    assert start['processed_rows']==0 and start['next_candidate_batch_index']==0
    assert end['processed_rows']==m.ROWS and end['phase']=='FINAL_SCORE'
    assert end['position']=={'logical_cursor':3694,'completed_passes':1,'next_batch_in_pass':0,'optimizer_updates':3694,'skipped_batches':0}
    assert all(end[k]is None for k in('next_candidate_batch_index','next_row_positions_sha256',
        'next_source_rows_sha256','next_issue_ns_sha256','next_activity_mask_sha256'))
    assert m.expected_step_binding(f['schedule'],3693)['rows']==22
    for cursor in(-1,3695,True):
        with pytest.raises(ValueError,match='LOGICAL_CURSOR$'):
            m.cursor_binding(f['schedule'],cursor)
    with pytest.raises(ValueError,match='STEP_INDEX$'):
        m.expected_step_binding(f['schedule'],3694)


def test_manifest_and_schedule_seals_are_fail_closed(f):
    manifest={**f['manifest'],'selection_uses_labels_predictions_support':True}
    with pytest.raises(ValueError,match='MANIFEST_CONTRACT$'):
        build(f,manifest=manifest)
    bad=copy.deepcopy(f['schedule']);bad['update_flags'][0]=False
    with pytest.raises(ValueError,match='SCHEDULE_SEAL$'):
        m.validate_schedule(bad)
