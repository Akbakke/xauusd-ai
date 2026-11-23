"""Small synthetic checks for declared input masks; never load a native payload."""
from __future__ import annotations
import copy,sys
import numpy as np
import pytest
from gx1.audit import audit_native_input_observation_v1 as audit


def relation(rule_id,*,values=('value',),mask='known'):
    return {'id':rule_id,'availability_field':mask,'value_fields':list(values),'kind':'synthetic_declared_relation'}


def counted(rule_id,rows,*,values=('value',)):
    return audit.count_relation(np.asarray(rows,dtype=np.float32),[*values,'known'],relation(rule_id,values=values))


def assert_counts(result,*,true,false,invalid=0,violations=0):
    assert result['support_true']==true
    assert result['support_false']==false
    assert result['invalid_predicate']==invalid
    assert result['raw_storage_violations']==violations


def test_range_zero_can_be_known_or_undefined_without_changing_values():
    result=counted('micro_range',[[0,0],[0,1],[.25,1]])
    assert_counts(result,true=2,false=1)


def test_range_absence_requires_parked_raw_zero():
    assert_counts(counted('micro_range',[[.2,0],[0,1]]),true=1,false=1,violations=1)


def test_invalid_binary_certificate_is_not_known_or_absent_support():
    assert_counts(counted('micro_range',[[0,.5],[0,1]]),true=1,false=0,invalid=1)


def test_swing_present_genuine_zero_remains_known():
    assert_counts(counted('swing_impulse',[[0,1],[-2,1],[0,0]]),true=2,false=1)


def test_envelope_positive_width_does_not_clip_positions_to_unit_interval():
    assert_counts(counted('smc_envelope',[[-2,1],[2.5,2],[0,0]]),true=2,false=1)


def test_envelope_zero_width_requires_zero_position():
    assert_counts(counted('smc_envelope',[[.5,0],[0,0]]),true=0,false=2,violations=1)


def test_negative_envelope_width_is_not_a_valid_absence_certificate():
    assert_counts(counted('smc_envelope',[[0,-1],[.5,1]]),true=1,false=0,invalid=1)


def test_mtf_envelope_and_swing_aliases_use_the_same_declared_semantics():
    assert_counts(counted('smc_envelope_mtf_alias',[[0,0],[2,1]]),true=1,false=1)
    assert_counts(counted('swing_impulse_mtf_alias',[[0,0],[0,1]]),true=1,false=1)


@pytest.mark.parametrize('rule_id',['smc_sweep_up','smc_sweep_down'])
def test_sweep_state_absence_is_distinct_from_missing_market_observation(rule_id):
    assert_counts(counted(rule_id,[[0,0],[.5,1]]),true=1,false=1)
    assert_counts(counted(rule_id,[[.5,0],[0,1]]),true=1,false=1,violations=2)


@pytest.mark.parametrize('rule_id',['level_below','level_above2','level_below2'])
def test_level_presence_preserves_genuine_signed_zero(rule_id):
    assert_counts(counted(rule_id,[[-0.,1],[0.,0],[2.,1]]),true=2,false=1)


def test_level_touch_count_certificate_requires_nonnegative_integers():
    assert_counts(counted('level_above',[[0,0],[0,1],[2,3],[0,.5],[0,-1]]),true=2,false=1,invalid=2)


@pytest.mark.parametrize('rule_id',['geomline_above','geomline_below'])
def test_geomline_count_zero_requires_all_paired_slot_attributes_zero(rule_id):
    result=counted(rule_id,[[0,0,0],[0,2,1],[1,0,0]],values=('distance','slope'))
    assert_counts(result,true=1,false=2,violations=1)


def test_geomchannel_flag_controls_its_own_five_values_only():
    rows=[[0,0,0,0,0,0],[1,-2,0,3,.5,1]]
    assert_counts(counted('geomchan',rows,values=('width','position','slope','converging','apex')),true=1,false=1)


@pytest.mark.parametrize('rule_id',['level_above_pending_retest','level_below_pending_retest'])
def test_pending_pair_uses_eligible_age_and_strictly_positive_distance(rule_id):
    assert_counts(counted(rule_id,[[0,0],[.5,1],[2,4]]),true=2,false=1)
    assert_counts(counted(rule_id,[[.5,0],[0,1],[-.5,2]]),true=2,false=1,violations=3)


def test_pending_fractional_or_negative_age_is_not_an_occupancy_certificate():
    assert_counts(counted('level_above_pending_retest',[[0,-1],[1,.5],[1,1]]),true=1,false=0,invalid=2)


def test_unrecognized_rule_or_missing_paired_field_fails_closed():
    with pytest.raises(ValueError,match='RELATION_ID'):
        counted('arbitrary_age_is_known',[[0,1]])
    with pytest.raises(KeyError,match='known'):
        audit.count_relation(np.zeros((1,1),np.float32),['value'],relation('micro_range'))


def test_relation_counting_does_not_mutate_readonly_values():
    raw=np.array([[0,0],[0,1]],np.float32);before=raw.copy();raw.flags.writeable=False
    audit.count_relation(raw,['value','known'],relation('micro_range'))
    assert np.array_equal(raw,before)


def test_no_positive_condition_support_is_reported_without_vacuous_pass():
    total=counted('micro_range',[[0,0],[0,0]])
    result=audit.summarize_relation(total)
    assert result['qualification_status']=='NO_SUPPORT'
    assert result['absent_storage_status']!='NOT_EXERCISED'


def test_unexercised_absence_rule_is_reported_even_when_all_values_known():
    result=audit.summarize_relation(counted('micro_range',[[0,1],[.3,1]]))
    assert result['qualification_status']=='COUNTS_COMPLETE'
    assert result['absent_storage_status']=='NOT_EXERCISED'


def test_storage_violation_is_failure_even_when_positive_condition_has_no_support():
    result=audit.summarize_relation(counted('micro_range',[[.5,0]]))
    assert result['qualification_status']=='FAIL'


def contract(names,center,binary):
    return {'surface':'ctx_cont','field_names':names,'center':center,'scale':[1.]*len(names),
        'binary_mask':binary,'categorical_mask':[False]*len(names),'categorical_domains':{},
        'scale_source':['synthetic_test_scale']*len(names),'continuous_transform':audit.norm.CONTINUOUS_TRANSFORM}


def test_canonical_asinh_does_not_require_general_structural_raw_zero_to_stay_zero():
    raw=np.array([[0.]],np.float32)
    transformed=audit.norm.apply_surface_normalization(raw,contract(['ordinary_structural_value'],[2.],[False]))
    assert transformed[0,0]!=0 and np.isfinite(transformed).all()


def test_canonical_micro_range_mask_requires_normalized_zero_only_when_undefined():
    names=['close_distance_below_high_range_fraction','close_range_observed']
    raw=np.array([[0,0],[0,1]],np.float32)
    transformed=audit.norm.apply_surface_normalization(raw,contract(names,[2.,0.],[False,True]))
    assert transformed[0,0]==0 and transformed[1,0]!=0


def test_census_surface_keeps_transform_blocks_bounded_and_uses_only_selected_rows(monkeypatch):
    names=['close_distance_below_high_range_fraction','close_range_observed']
    raw=np.array([[.5,1],[0,0],[0,1],[.2,1],[0,0],[.4,1]],np.float32)
    observed=[];original=audit.norm.apply_surface_normalization
    def transform(values,spec):
        observed.append(values.copy());return original(values,spec)
    monkeypatch.setattr(audit.norm,'apply_surface_normalization',transform)
    audit.census_surface(raw,np.array([1,2,4],np.int64),names,[relation('micro_range',values=(names[0],),mask=names[1])],contract(names,[2.,0.],[False,True]),block_rows=2)
    assert len(observed)==2 and max(len(x)for x in observed)<=2
    assert np.array_equal(np.concatenate(observed),raw[[1,2,4]])


def test_nonfinite_input_is_counted_and_canonical_transform_failure_is_retained():
    names=['ordinary_structural_value'];raw=np.array([[np.nan],[0.]],np.float32)
    result=audit.census_surface(raw,np.array([0,1],np.int64),names,[],contract(names,[2.],[False]))
    assert result['field_counts'][0]['nonfinite']==1
    assert result['field_counts'][0]['source_known_status']=='NOT_ESTABLISHED_FROM_FINITE_OR_ZERO'
    assert result['normalization_apply_calls']==0 and len(result['normalization_block_errors'])==1
    assert result['qualification_status']=='FAIL'


def test_bound_rule_inventory_is_exactly_86_and_pending_errata_does_not_mutate_source():
    ids=sorted(audit.IDS);chosen=ids[:14]
    source={'counts':{'field_records':1731},'cache_measurable_rules':{
        surface:[relation(rule_id)for rule_id in (chosen if surface!='ctx_cont'else['micro_range','swing_impulse'])]
        for surface in audit.UNION_ROWS}}
    source['cache_measurable_rules']['ctx_cat']=[]
    errata={'original_census':audit.CENSUS,'affected_rule_ids':['level_above_pending_retest','level_below_pending_retest'],
        'corrected_rule':{'preconditions':['canonical eligible pending-slot selection only']}}
    before=copy.deepcopy(source);result=audit.observation_rules(source,errata)
    assert sum(map(len,result.values()))==86 and source==before
    pending=next(x for x in result['signal']if x['id']=='level_above_pending_retest')
    assert 'distance>0'in pending['absent_storage'] and pending['preconditions']==errata['corrected_rule']['preconditions']
    source['cache_measurable_rules']['signal'].pop()
    with pytest.raises(ValueError,match='EXACT_86_RELATIONS'):audit.observation_rules(source,errata)


def test_cli_preflight_performs_no_array_load_claim_or_calculation(monkeypatch,capsys,tmp_path):
    binding={'path':'plan','sha256':'a'*64};pref={'path':'policy','sha256':'b'*64};seen=[]
    monkeypatch.setattr(audit,'ROOT',tmp_path/'never_created')
    monkeypatch.setattr(audit,'inspect',lambda sha,*,admitted:(seen.append(('inspect',admitted))or({},binding,pref)))
    monkeypatch.setattr(audit,'checked_inputs',lambda plan:seen.append(('checked',)))
    def forbidden(*args,**kwargs):raise AssertionError('payload/claim/calculation forbidden in preflight')
    monkeypatch.setattr(audit.np,'load',forbidden);monkeypatch.setattr(audit.cache,'load_snapshot',forbidden)
    monkeypatch.setattr(audit,'calculate',forbidden);monkeypatch.setattr(audit,'publish',forbidden)
    monkeypatch.setattr(sys,'argv',['audit','--plan-sha256','a'*64,'--preflight']);audit.main()
    import json
    result=json.loads(capsys.readouterr().out)
    assert result['claim_created']is False and result['arrays_decoded']==result['models']==result['fits']==result['forwards']==0
    assert seen==[('inspect',False),('checked',)] and not audit.ROOT.exists()


def run_harness(monkeypatch,tmp_path,calculate):
    binding={'path':'plan','sha256':'a'*64};pref={'path':'policy','sha256':'b'*64};published=[];checked=[]
    monkeypatch.setattr(audit,'ROOT',tmp_path/'output')
    original_read_text=audit.Path.read_text
    def read_text(path,*args,**kwargs):
        return '123'if str(path)=='/sys/fs/cgroup/synthetic/memory.peak'else original_read_text(path,*args,**kwargs)
    monkeypatch.setattr(audit.Path,'read_text',read_text)
    monkeypatch.setattr(audit,'require_capped_cpu_audit_execution',lambda:{'memory_max':4*1024**3,'memory_high':4*1024**3,
        'swap':512*1024**2,'numerical_threads':1,'cgroup_path':'/sys/fs/cgroup/synthetic'})
    monkeypatch.setattr(audit,'require_capped_lock_ancestry',lambda:{'scope':'/synthetic'})
    monkeypatch.setattr(audit,'inspect',lambda sha,*,admitted:({},binding,pref))
    monkeypatch.setattr(audit,'checked_inputs',lambda plan:checked.append(True))
    monkeypatch.setattr(audit,'source_clean',lambda plan:True)
    monkeypatch.setattr(audit,'ref',lambda path:binding if path==audit.PLAN else pref)
    monkeypatch.setattr(audit,'calculate',calculate)
    def publish(name,value):
        published.append((name,copy.deepcopy(value)));return {'path':name,'sha256':'c'*64}
    monkeypatch.setattr(audit,'publish',publish)
    return published,checked


def test_claimed_success_publishes_both_result_and_terminal_after_input_bookends(monkeypatch,tmp_path):
    published,checked=run_harness(monkeypatch,tmp_path,lambda plan,**kwargs:{'qualification_status':'synthetic'})
    result=audit.run('a'*64)
    assert [name for name,_ in published]==['CLAIM.json','RESULT.json','TERMINAL.json']
    terminal=published[-1][1]
    assert terminal['status']=='COMPLETE' and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False
    assert result['result']['path']=='RESULT.json' and len(checked)==2


def test_claimed_failure_is_consumed_and_retains_progress_without_result(monkeypatch,tmp_path):
    def calculate(plan,*,progress):
        progress['phase']='synthetic_failure';raise RuntimeError('synthetic stop')
    published,_=run_harness(monkeypatch,tmp_path,calculate)
    with pytest.raises(RuntimeError,match='synthetic stop'):audit.run('a'*64)
    assert [name for name,_ in published]==['CLAIM.json','TERMINAL.json']
    terminal=published[-1][1]
    assert terminal['status']=='FAILED_CONSUMED' and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False
    assert terminal['result']is None and terminal['progress']['phase']=='synthetic_failure'


def test_source_bookend_failure_prevents_publication_of_success_result(monkeypatch,tmp_path):
    published,_=run_harness(monkeypatch,tmp_path,lambda plan,**kwargs:{})
    monkeypatch.setattr(audit,'source_clean',lambda plan:False)
    with pytest.raises(ValueError,match='SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND'):audit.run('a'*64)
    assert [name for name,_ in published]==['CLAIM.json','TERMINAL.json']
    assert published[-1][1]['status']=='FAILED_CONSUMED'


def test_inspection_keeps_each_of_five_global_training_flags_closed(monkeypatch,tmp_path):
    binding={'path':str(audit.PLAN),'sha256':'a'*64};pref={'path':'policy','sha256':'b'*64}
    plan={'inputs':{'design':{}},'source_base_commit':'synthetic'}
    policy={flag:False for flag in audit.raw.saved.FLAGS};policy[audit.STAGE]=audit.gate_expected(binding,False)
    monkeypatch.setattr(audit,'REPO',audit.Path.cwd());monkeypatch.setattr(audit,'ROOT',tmp_path/'fresh')
    monkeypatch.setattr(audit,'build_plan',lambda **kwargs:plan);monkeypatch.setattr(audit,'source_clean',lambda value:True)
    monkeypatch.setattr(audit,'ref',lambda path:pref)
    monkeypatch.setattr(audit,'read',lambda value:plan if value==binding else policy)
    assert audit.inspect('a'*64,admitted=False)==(plan,binding,pref)
    for flag in audit.raw.saved.FLAGS:
        policy[flag]=True
        with pytest.raises(ValueError,match='EXACT_ONE_USE_ALL_TRAINING_CLOSED'):audit.inspect('a'*64,admitted=False)
        policy[flag]=False


def test_raw_zero_counters_separate_present_absent_and_exclude_invalid_certificate():
    result=counted('micro_range',[[0,1],[0,0],[0,.5],[2,1]])
    assert_counts(result,true=2,false=1,invalid=1)
    assert result['present_raw_zero_by_value']=={'value':1}
    assert result['absent_raw_zero_by_value']=={'value':1}
