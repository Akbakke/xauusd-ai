"""Bind the M5-only full fit to its own completed private qualification.

Metadata validation only: no model, market data or execution authority.
"""
from __future__ import annotations
import ast
import hashlib
import subprocess
from pathlib import Path
from typing import Mapping
from gx1.contracts import local_random_access_campaign_v2 as campaign
COMPARISON_SOURCES=('gx1/scripts/score_entry_direction_dense_onepass_v1.py',
 'gx1/scripts/score_entry_direction_dense_minutes_v1.py','gx1/scripts/compare_entry_direction_matched_exposure_v1.py',
 'gx1/scripts/compare_entry_direction_saved_bps_v1.py','gx1/contracts/entry_direction_block_rank_v1.py',
 'gx1/contracts/entry_direction_probability_reference_v1.py','gx1/contracts/entry_native_direction_context_report_v1.py')

_ROOT='/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63'
QUALIFICATION={'path':_ROOT+'/DIRECTION_M5_PRICE_ONLY_PRIVATE_QUALIFICATION_20261029_001/ACTUAL_ROOT_REVIEW_001.json','sha256':'98aea1482cebefcadb8276a299cee4e8e1e1a2bb73924787a5d965a3d25726f8'}
INITIAL={'path':_ROOT+'/DIRECTION_M5_PRICE_ONLY_PRIVATE_QUALIFICATION_20261029_001/RUNTIME/private-research/INITIAL_TRAIN.npz','sha256':'90c298464bf2282ee17f5b71a02fc05c1fd942d9254a11e7edad63d631bf7792'}
BASELINE={
 'predictions':{'path':_ROOT+'/DENSE_DECIMAL_ONEPASS_FIT_20261028_001/RUNTIME/prefix-fit/window-0001/OOF.npz','sha256':'083d76cb29e778e119cc6e9979af1284d111e7e288f1674f8b8918ba07031482'},
 'report':{'path':_ROOT+'/DENSE_DECIMAL_ONEPASS_FIT_20261028_001/RUNTIME/prefix-fit/window-0001/OOF.json','sha256':'a773849f2895a77c3af0567a3d9421f9f8724fb5592f8e1c43204af497218757'},
 'result':{'path':_ROOT+'/DENSE_DECIMAL_ONEPASS_FIT_20261028_001/RUNTIME/prefix-fit/window-0001/RESULT.json','sha256':'a9f4fe99fbccd89178fddabaf2cf30c22fd66843e545bdcba3b2a24b41c6fbd8'},
}
TRANSITIONS={
 'gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py':{'changed':('require_prefix_fit_launch_recipe',),'new_assignments':('M5_PRICE_FIT_SCOPE',)},
 'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py':{'changed':('_fit_input_launch','run_native_direction_prefix_fit_window'),'new_assignments':()},
 'gx1/scripts/run_entry_native_direction_research_window_v1.py':{'changed':('_install_read_guard',),'new_assignments':()},
}
COMPARISON_FUNCTION='gx1.scripts.score_entry_direction_dense_onepass_v1.compare_saved_predictions; validate all 60 target and clock arrays first; paired 5-minute comparison; candidate simultaneous 60-horizon bands and descriptive baseline panels.'

def _require(ok,why):
    if not ok:raise campaign.RandomAccessCampaignError('M5_PRICE_FULLFIT_'+why)

def _read(ref,label):
    checked=campaign.require_binding(ref,label=label,verify_file=True)
    return campaign.read_bound_json(Path(checked['path']),checked['sha256'])

def require_qualified_source_transition(source_repo,qualified_commit,qualified_owners,owners):
    """Allow only specified launch orchestration changes, preserving computation."""
    _require(type(qualified_commit) is str and campaign._COMMIT.fullmatch(qualified_commit) is not None,'QUALIFIED_COMMIT')
    for path,allowed in TRANSITIONS.items():
        before_ref=qualified_owners.get(path);after_ref=owners.get(path)
        _require(isinstance(before_ref,Mapping) and isinstance(after_ref,Mapping)
            and before_ref.get('path')==after_ref.get('path')==str(source_repo/path),'TRANSITION_PATH:'+path)
        before=subprocess.check_output(['git','-C',str(source_repo),'show',qualified_commit+':'+path]);after=(source_repo/path).read_bytes()
        _require(hashlib.sha256(before).hexdigest()==before_ref.get('sha256') and hashlib.sha256(after).hexdigest()==after_ref.get('sha256'),'TRANSITION_SHA:'+path)
        changed=set(allowed['changed']);new=set(allowed['new_assignments'])
        def kept(raw,original):
            nodes=[];found_changed=[];found_new=[]
            for node in ast.parse(raw,filename=path).body:
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in changed:
                    found_changed.append(node.name);continue
                if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in new:
                    found_new.append(node.targets[0].id);continue
                nodes.append(ast.dump(node,include_attributes=False))
            _require(len(found_changed)==len(changed) and set(found_changed)==changed
                and len(found_new)==(0 if original else len(new)) and set(found_new)==(set() if original else new),'TRANSITION_NODE_SET:'+path)
            return nodes
        _require(kept(before,True)==kept(after,False),'TRANSITION_AST:'+path)

def require_protocol(protocol,recipe):
    _require(protocol.get('schema_version')=='gx1_m5_price_only_direction_protocol_v1'
        and protocol.get('status')=='PREDECLARED_NO_TRAINING_AUTHORITY'
        and protocol.get('candidate_input_mode')=='price_only' and protocol.get('m1_values_or_masks_as_entry_inputs') is False
        and protocol.get('fold')==recipe['fold'] and protocol.get('target')==recipe['dense_target']
        and protocol.get('future_outcome_horizons_minutes')==list(range(1,61))
        and protocol.get('fit')=={'batch_size':16,'fresh_initial_state':True,'maximum_updates':2976,'train_parent_rows':47602,'train_passes':1,'use_existing_corrected_60_head_objective':True,'seed':20260911}
        and protocol.get('comparison')=={'later_data':'reused Q1 2025 development; strict mature observed nonzero labels',
            'limitation':'Price fields and associated model projection/gating change together; this is not a single-feature or equal-capacity ablation.',
            'metrics':['auc','accuracy_at_0.5','logloss','brier'],'primary_minutes':5,
            'references':['fixed_old_model','always_up','own_train_prior'],'secondary_minutes':'1..60',
            'simultaneous_auc_bands_all_60':True,'paired_comparison_scope':'5-minute primary only; remaining horizons descriptive against baseline','threshold_or_checkpoint_selection':False,
            'utc_day_and_week_paired_bootstrap_replicates':10000}
        and protocol.get('hardware')=={'existing_thermal_guards_preserved':True,'gpu_power_limit_w_during_and_after':250,'physical_restart_before_each_training_window':True}
        and protocol.get('test_data_used') is False and protocol.get('broker_orders_paper_live_promotion_authorized') is False
        and protocol.get('economic_edge_claimed') is False,'PROTOCOL')
    baseline=protocol.get('baseline');_require(isinstance(baseline,Mapping) and all(baseline.get(k)==v for k,v in BASELINE.items()),'BASELINE')
    for key,ref in BASELINE.items():campaign.require_binding(ref,label='M5 baseline '+key,verify_file=True)
    sources=protocol.get('comparison_sources');repo=Path(recipe['source_repo'])
    _require(isinstance(sources,Mapping) and set(sources)==set(COMPARISON_SOURCES)
        and protocol.get('comparison_function')==COMPARISON_FUNCTION,'COMPARISON_SOURCES')
    for name,ref in sources.items():
        checked=campaign.require_binding(ref,label='M5 comparison source',verify_file=True)
        _require(checked['path']==str(repo/name),'COMPARISON_SOURCE_PATH:'+name)

def require_qualification(recipe,owners,*,source_repo,private_scope):
    _require(recipe.get('candidate_mode')=='price_only' and 'candidate_m1_refs' not in recipe
        and 'm1_pair_initial_parity' not in recipe and recipe.get('native_dense_qualification')==QUALIFICATION
        and recipe.get('decimal_initial_predictions')==INITIAL,'QUALIFICATION_BINDING')
    review=_read(QUALIFICATION,'M5 private review')
    _require(review.get('schema_version')=='gx1_m5_price_private_root_review_v1'
        and review.get('status')=='PASS_CONSUMED_PRICE_ONLY_PRIVATE_NATIVE_MECHANICS'
        and review.get('candidate_mode')=='price_only' and review.get('m1_values_and_masks_as_entry_inputs') is False
        and review.get('initial_predictions')==INITIAL and review.get('initial_parent_count')==256
        and review.get('all_sixty_train_priors_checked') is True
        and review.get('all_sixty_heads_have_gradient_and_eligible_labels_on_fixed_batches') is True
        and review.get('four_price_features_shared_projection_gradients_nonzero_in_both_trajectories') is True
        and review.get('bit_exact_resume_reported_and_bound') is True and review.get('private_optimizer_updates')==4
        and review.get('qualification_weights_for_main_fit') is False and review.get('test_data_used') is False,'PRIVATE_REVIEW')
    plan=_read(review['plan'],'M5 private plan');prior_ref=plan['recipe'];prior=_read(prior_ref,'M5 private recipe')
    result=_read(review['result'],'M5 private result');receipt=_read(review['receipt'],'M5 private receipt')
    power=_read(review['power_after'],'M5 post-campaign power')
    campaign.require_binding(INITIAL,label='M5 private initial predictions',verify_file=True)
    _require(power.get('campaign_task_state')=='Disabled' and power.get('configured_power_limit_w')==250
        and power.get('actual_power_limit_w')==250 and power.get('keeper_state')=='Running' and power.get('bridge_state')=='Running','PRIVATE_POWER')
    _require(plan.get('source_commit')==review.get('source_commit') and receipt.get('plan_sha256')==plan.get('plan_sha256')
        and receipt.get('outcome')=='COMPLETE' and receipt.get('guard_decision')=='PASS'
        and receipt.get('trainer_guard_exit_code')==receipt.get('progress_observer_exit_code')==0
        and receipt.get('test_data_used') is False,'PRIVATE_RECEIPT')
    _require(result.get('schema_version')=='gx1_native_m1_price_private_qualification_result_v1'
        and result.get('decision')=='PASS_M1_PRICE_PRIVATE_MECHANICS' and result.get('recipe')==prior_ref
        and result.get('candidate_mode')=='price_only' and result.get('candidate_m1_refs') is None
        and result.get('initial_predictions')==INITIAL and result.get('decimal_prior_against_sign_audit') is True
        and result.get('bit_exact_resume') is True and result.get('physical_optimizer_updates')==4
        and result.get('source_unchanged') is True and result.get('test_data_used') is False
        and result.get('private_weights_for_main_fit') is False,'PRIVATE_RESULT')
    keys=('input_manifest','fold','dense_target','dense_target_result','dense_source_transition_proof','decimal_input_audit','decimal_target_refs','candidate_mode','parent_order_algorithm')
    _require(prior.get('run_scope')==private_scope and 'candidate_m1_refs' not in prior
        and all(prior.get(k)==recipe.get(k) for k in keys) and prior.get('qualification_weights_for_main_fit') is False
        and {k:v for k,v in prior['numerical_recipe'].items() if k not in ('max_updates','contract_sha256')}
            =={k:v for k,v in recipe['numerical_recipe'].items() if k not in ('max_updates','contract_sha256')},'PRIVATE_RECIPE')
    old=prior['source_owners'];_require(set(old)-set(TRANSITIONS) <= set(owners)
        and all(old[k]==owners[k] for k in set(old)-set(TRANSITIONS)),'QUALIFIED_UNCHANGED_SOURCES')
    require_qualified_source_transition(source_repo,review['source_commit'],old,owners)

# The completed private qualification is a mechanics witness, not fit weights.
FACTOR_QUALIFICATION = {'path': _ROOT+'/DIRECTION_M5_DENSE_BPS_PRIVATE_20261030_001/ACTUAL_ROOT_REVIEW_001.json',
    'sha256':'11a0245c0a753c359f8ef9f29474f7a4144609b677c8babcabca67715cce2d8e'}
FACTOR_PROTOCOL_NAME = 'DIRECTION_M5_SIGN_BPS_EXPOSURE_PROTOCOL_20261030.json'
FACTOR_RUNTIME_ROOTS = {
    'sign_one_pass': _ROOT+'/DIRECTION_M5_FACTOR_SIGN_ONEPASS_20261030_001/RUNTIME',
    'bps_one_pass': _ROOT+'/DIRECTION_M5_FACTOR_BPS_ONEPASS_20261030_001/RUNTIME',
    'sign_two_pass': _ROOT+'/DIRECTION_M5_FACTOR_SIGN_TWOPASS_20261030_001/RUNTIME',
}
FACTOR_TRANSITIONS = {
    'gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py': {
        'functions':('require_prefix_fit_launch_recipe',),
        'new_functions':('guarded_prefix_power_limit',),
        'new_assignments':('M5_FACTOR_SIGN_SCOPE','M5_FACTOR_BPS_SCOPE','M5_FACTOR_CONTINUATION_SCOPE')},
    'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py': {
        'functions':('_fit_input_launch','run_native_direction_prefix_fit_window')},
    'gx1/scripts/run_entry_native_direction_research_window_v1.py': {'functions':('_install_read_guard',)},
}


def require_factor_protocol(protocol, recipe):
    """Bind both hypotheses before either full fit; grant no execution authority."""
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    from gx1.contracts import entry_native_direction_research_campaign_v1 as private
    scopes={prefix.M5_FACTOR_SIGN_SCOPE:'sign_one_pass',prefix.M5_FACTOR_BPS_SCOPE:'bps_one_pass',
        prefix.M5_FACTOR_CONTINUATION_SCOPE:'sign_two_pass'}
    arm=scopes.get(recipe.get('run_scope'))
    _require(arm is not None and isinstance(protocol,Mapping)
        and protocol.get('schema_version')=='gx1_m5_direction_bps_exposure_comparison_v1'
        and protocol.get('status')=='PREDECLARED_NO_EXECUTION_AUTHORITY'
        and protocol.get('fold')==recipe['fold'] and protocol.get('target')==recipe['dense_target']
        and protocol.get('candidate_input_mode')=='price_only' and protocol.get('m1_values_or_masks_as_entry_inputs') is False
        and protocol.get('future_outcome_horizons_minutes')==list(range(1,61))
        and protocol.get('private_qualification')==FACTOR_QUALIFICATION
        and protocol.get('initial_predictions')==private.M5_DENSE_BPS_BASELINE_INITIAL
        and protocol.get('initial_report')==private.M5_DENSE_BPS_BASELINE_REPORT
        and protocol.get('bps_scale')==private.M5_DENSE_BPS_SCALE
        and protocol.get('runtime_roots')==FACTOR_RUNTIME_ROOTS and recipe['runtime_root']==FACTOR_RUNTIME_ROOTS[arm],
        'FACTOR_PROTOCOL_IDENTITY')
    _require(protocol.get('fit')=={'batch_size':16,'seed':20260911,'train_parent_rows':47602,
        'updates_per_pass':2976,'same_source_and_parent_order_required':True,
        'same_initial_shared_parameters_buffers_rng_and_256_predictions_required':True,'private_weights_reused':False}
        and protocol.get('arms')=={
            'sign_one_pass':{'passes':1,'signed_bps_auxiliary':False,'origin':'fresh'},
            'bps_one_pass':{'passes':1,'signed_bps_auxiliary':True,'origin':'fresh'},
            'sign_two_pass':{'passes':2,'signed_bps_auxiliary':False,'origin':'exact_sign_one_pass_checkpoint'}}
        and protocol.get('comparison')=={'later_data':'reused Q1 2025 development','fixed_parent_range':[47602,64890],
            'primary_minutes':5,'all_horizons':True,
            'separate_contrasts':['bps_one_pass_minus_sign_one_pass','sign_two_pass_minus_sign_one_pass'],
            'metrics':['auc','accuracy_at_0.5','logloss','brier'],'paired_utc_day_and_week_bootstrap_replicates':10000,
            'simultaneous_auc_bands_all_60':True,'threshold_or_checkpoint_selection':False}
        and protocol.get('hardware')=={'gpu_power_limit_w_during_and_after':250,
            'physical_restart_before_each_training_window':True,'existing_thermal_guards_preserved':True}
        and protocol.get('test_data_used') is False and protocol.get('broker_orders_paper_live_exit_promotion_authorized') is False
        and protocol.get('economic_edge_claimed') is False, 'FACTOR_PROTOCOL_DESIGN')
    sources=protocol.get('comparison_sources');repo=Path(recipe['source_repo'])
    _require(isinstance(sources,Mapping) and set(sources)==set(COMPARISON_SOURCES), 'FACTOR_COMPARISON_SOURCES')
    for name, ref in sources.items():
        checked=campaign.require_binding(ref,label='factor comparison source',verify_file=True)
        _require(checked['path']==str(repo/name),'FACTOR_COMPARISON_SOURCE_PATH:'+name)


def require_factor_source_transition(source_repo, source_commit, qualified_commit, old, owners):
    """Keep the qualified numerical owners; permit only declared launch plumbing."""
    from gx1.contracts import entry_native_direction_five_priority_private_v1 as ast_owner
    _require(set(owners)==set(old)|{'gx1/contracts/entry_native_direction_m5_fullfit_v1.py'},'FACTOR_SOURCE_SET')
    _require(subprocess.run(['git','merge-base','--is-ancestor',qualified_commit,source_commit],
        cwd=source_repo,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0,'FACTOR_SOURCE_ANCESTRY')
    guard='scripts/gx1_capped_run.sh'
    for name,original in old.items():
        current=owners[name]
        if name not in FACTOR_TRANSITIONS and name != guard:
            _require(current==original,'FACTOR_UNCHANGED_SOURCE:'+name);continue
        _require(original['path']==current['path']==str(source_repo/name),'FACTOR_SOURCE_PATH:'+name)
        before=subprocess.check_output(['git','show',qualified_commit+':'+name],cwd=source_repo)
        after=(source_repo/name).read_bytes()
        _require(hashlib.sha256(before).hexdigest()==original['sha256'] and hashlib.sha256(after).hexdigest()==current['sha256'],
            'FACTOR_SOURCE_SHA:'+name)
        if name != guard:
            spec=FACTOR_TRANSITIONS[name]
            _require(ast_owner._unchanged_ast(before,spec,original=True)==ast_owner._unchanged_ast(after,spec,original=False),
                'FACTOR_SOURCE_AST:'+name)
        else:
            old_block=b'if window.get("schema_version") != "gx1_native_direction_research_window_policy_v1":\n    print(300)\nelse:\n'
            new_block=b'''if window.get("schema_version") == "gx1_native_direction_prefix_fit_window_policy_v1":
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    window = campaign.read_bound_json(Path(sys.argv[1]), sys.argv[2])
    window = prefix.require_prefix_fit_window_policy(window)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    recipe = prefix.require_prefix_fit_launch_recipe(window["recipe"], source_repo=repo, source_commit=head)
    print(prefix.guarded_prefix_power_limit(recipe))
elif window.get("schema_version") != "gx1_native_direction_research_window_policy_v1":
    print(300)
else:
'''
            old_comment=b"# This single research scope retains the user's configured 250 W ceiling."
            new_comment=b"# Exact matched research scopes retain the user's configured 250 W ceiling."
            _require(before.count(old_block)==1 and before.count(old_comment)==1
                and before.replace(old_block,new_block,1).replace(old_comment,new_comment,1)==after,
                'FACTOR_GUARD_TRANSITION')


def require_factor_qualification(recipe, owners, *, source_repo, source_commit):
    """Require the actual bounded bps witness and matching sign/bps fit numerics."""
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    from gx1.contracts import entry_native_direction_research_campaign_v1 as private
    from gx1.contracts.entry_model_native_train_recipe_v1 import native_direction_research_recipe
    bps=recipe['run_scope']==prefix.M5_FACTOR_BPS_SCOPE
    continuation=recipe['run_scope']==prefix.M5_FACTOR_CONTINUATION_SCOPE
    _require(recipe.get('candidate_mode')=='price_only' and 'candidate_m1_refs' not in recipe
        and recipe.get('native_dense_qualification')==FACTOR_QUALIFICATION
        and recipe.get('decimal_initial_predictions')==private.M5_DENSE_BPS_BASELINE_INITIAL
        and recipe.get('factor_initial_report')==private.M5_DENSE_BPS_BASELINE_REPORT
        and (recipe.get('dense_bps_scale')==private.M5_DENSE_BPS_SCALE if bps else 'dense_bps_scale' not in recipe),
        'FACTOR_QUALIFICATION_BINDING')
    expected=native_direction_research_recipe(batch_size=16,max_updates=5952 if continuation else 2976,
        dense_minutes=True,dense_bps=bps)
    _require(recipe['numerical_recipe']==expected,'FACTOR_CANONICAL_NUMERICS')
    review=_read(FACTOR_QUALIFICATION,'matched factor private review')
    _require(review.get('status')=='PASS_VERIFIED_PRIVATE_MECHANICS_NOT_EDGE' and review.get('physical_boot_id')==583
        and review.get('unique_train_parents')==32 and review.get('physical_optimizer_updates')==4
        and review.get('initial_predictions')==256 and review.get('initial_shared_state_and_rng_parity') is True
        and review.get('signed_bps_gradient_rows')==60 and review.get('bps_only_joint_projection_nonzero',0)>0
        and review.get('bit_exact_resume') is True and review.get('m1_entry_input') is False
        and review.get('private_weights_for_main_fit') is False and review.get('test_data_used') is False,'FACTOR_PRIVATE_REVIEW')
    plan=_read(review['plan'],'factor private plan');prior_ref=plan['recipe'];prior=_read(prior_ref,'factor private recipe')
    result=_read(review['result'],'factor private result');receipt=_read(review['receipt'],'factor private receipt')
    power=_read(review['power_after'],'factor post-campaign power')
    _require(plan.get('source_commit')==review.get('source_commit') and receipt.get('plan_sha256')==plan.get('plan_sha256')
        and receipt.get('outcome')=='COMPLETE' and receipt.get('guard_decision')=='PASS'
        and receipt.get('trainer_guard_exit_code')==receipt.get('progress_observer_exit_code')==0
        and receipt.get('test_data_used') is False and result.get('recipe')==prior_ref and result.get('decision')=='PASS_M5_DENSE_BPS_PRIVATE_MECHANICS'
        and result.get('dense_bps_initial_prediction_and_shared_state_parity') is True
        and result.get('bit_exact_resume') is True and result.get('source_unchanged') is True
        and result.get('private_weights_for_main_fit') is False and result.get('test_data_used') is False
        and power.get('task_state')=='Disabled' and power.get('configured_power_limit_w')==250, 'FACTOR_PRIVATE_TERMINAL')
    keys=('input_manifest','fold','dense_target','dense_target_result','dense_source_transition_proof',
        'decimal_input_audit','decimal_target_refs','candidate_mode','parent_order_algorithm')
    omitted={'max_updates','contract_sha256','dense_signed_bps_supervision','research_objective_sha256'}
    _require(prior.get('run_scope')==prefix.M5_DENSE_BPS_PRIVATE_SCOPE
        and all(prior.get(k)==recipe.get(k) for k in keys)
        and {k:v for k,v in prior['numerical_recipe'].items() if k not in omitted}
            =={k:v for k,v in recipe['numerical_recipe'].items() if k not in omitted}, 'FACTOR_PRIVATE_INPUTS')
    for key in ('factor_initial_report','decimal_initial_predictions'):
        campaign.require_binding(recipe[key],label='factor '+key,verify_file=True)
    if bps: campaign.require_binding(recipe['dense_bps_scale'],label='factor bps scale',verify_file=True)
    require_factor_source_transition(source_repo,source_commit,review['source_commit'],prior['source_owners'],owners)
    if continuation: require_factor_continuation(recipe,owners,source_commit=source_commit)


def require_factor_continuation(recipe, owners, *, source_commit):
    """Accept only the terminal checkpoint of the exact matched sign fit."""
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    origin=recipe.get('continuation_origin');keys={'review','launch_recipe','terminal_state','initial_state',
        'initial_measurement','terminal_train_measurement','terminal_train_predictions'}
    _require(isinstance(origin,Mapping) and set(origin)==keys, 'FACTOR_ORIGIN_FIELDS')
    expected_review=FACTOR_RUNTIME_ROOTS['sign_one_pass'].removesuffix('/RUNTIME')+'/ACTUAL_FIT_REVIEW_001.json'
    _require(origin['review'].get('path')==expected_review,'FACTOR_ORIGIN_REVIEW_PATH')
    review=_read(origin['review'],'matched sign fit review');old=_read(origin['launch_recipe'],'matched sign fit recipe')
    receipt=_read(review['receipt'],'matched sign terminal receipt');plan=_read(review['plan'],'matched sign fit plan')
    actual_review=review.get('status')=='PASS_SIGN_ONEPASS_TERMINAL_CHAIN_NOT_EDGE'
    result_ref=review.get('fit_result') if actual_review else review.get('result')
    _require(isinstance(result_ref,Mapping),'FACTOR_ORIGIN_RESULT_REF')
    result=_read(result_ref,'matched sign fit result');terminal=_read(origin['terminal_train_measurement'],'matched sign terminal train')
    initial=_read(origin['initial_measurement'],'matched sign initial train')
    original_commit=plan.get('source_commit') if actual_review else review.get('source_commit')
    review_valid=(
        review.get('source_unchanged') is True and review.get('guard_decision')=='PASS'
        and review.get('guard_exit_code')==0 and review.get('no_prior_private_weights_reused') is True
        and review.get('forbidden_opens')==0 and review.get('trainer_processes_remaining')==0
        and review.get('terminal_checkpoint')==origin['terminal_state']
        and review.get('initial_predictions')==initial.get('predictions')
        and review.get('boot')==receipt.get('boot') and review.get('oof_parents')==17288
    ) if actual_review else (
        review.get('status')=='PASS_CONSUMED_M5_FACTOR_SIGN_ONEPASS'
        and review.get('recipe')==origin['launch_recipe']
        and review.get('training_source_unchanged') is True and review.get('signed_guard_pass') is True
        and review.get('private_weights_reused') is False
    )
    _require(type(original_commit) is str and campaign._COMMIT.fullmatch(original_commit) is not None
        and subprocess.run(['git','merge-base','--is-ancestor',original_commit,source_commit],
            cwd=Path(recipe['source_repo']),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0,
        'FACTOR_ORIGIN_SOURCE_ANCESTRY')
    _require(review_valid and review.get('optimizer_updates')==2976
        and review.get('train_parents')==47602 and review.get('test_data_used') is False
        and plan.get('source_commit')==original_commit and plan.get('recipe')==origin['launch_recipe']
        and receipt.get('plan_sha256')==plan.get('plan_sha256')
        and receipt.get('outcome')=='COMPLETE' and receipt.get('guard_decision')=='PASS'
        and receipt.get('trainer_guard_exit_code')==receipt.get('progress_observer_exit_code')==0
        and receipt.get('test_data_used') is False
        and old.get('run_scope')==prefix.M5_FACTOR_SIGN_SCOPE
        and old.get('input_manifest')==recipe['input_manifest'] and old.get('fold')==recipe['fold']
        and old.get('dense_target')==recipe['dense_target'] and old.get('dense_evaluation_protocol')==recipe['dense_evaluation_protocol']
        and old.get('factor_initial_report')==recipe['factor_initial_report']
        and result.get('recipe')==origin['launch_recipe'] and result.get('source_unchanged') is True
        and result.get('test_data_used') is False and result.get('qualification_weights_reused') is False
        and result.get('outcome')=='COMPLETE' and result.get('parent_cursor')==47602
        and result.get('optimizer_updates')==2976 and result.get('full_state')==origin['terminal_state']
        and result.get('own_initial_state')==origin['initial_state']
        and result.get('initial_measurement')==origin['initial_measurement']
        and result.get('final_train_measurement')==origin['terminal_train_measurement']
        and initial.get('role')=='INITIAL_TRAIN' and initial.get('own_initial_state')==origin['initial_state']
        and initial.get('fold')==recipe['fold'] and initial.get('test_data_used') is False
        and terminal.get('role')=='FINAL_TRAIN' and terminal.get('fold')==recipe['fold']
        and terminal.get('own_initial_state')==origin['initial_state'] and terminal.get('test_data_used') is False
        and terminal.get('predictions')==origin['terminal_train_predictions']
        and {k:v for k,v in old.get('numerical_recipe',{}).items() if k not in ('max_updates','contract_sha256')}
            =={k:v for k,v in recipe['numerical_recipe'].items() if k not in ('max_updates','contract_sha256')},
        'FACTOR_ORIGIN_TERMINAL')
    old_owners=old.get('source_owners');control='gx1/contracts/entry_native_direction_m5_fullfit_v1.py'
    _require(isinstance(old_owners,Mapping) and isinstance(owners,Mapping) and set(old_owners)==set(owners),'FACTOR_SOURCE_SET')
    if old_owners!=owners:
        _require({key for key in owners if owners[key]!=old_owners[key]}=={control},'FACTOR_SOURCE_DIFFERENCE')
        source_repo=Path(recipe['source_repo']);before_ref=old_owners[control];after_ref=owners[control]
        _require(before_ref.get('path')==after_ref.get('path')==str(source_repo/control),'FACTOR_REPAIR_PATH')
        before=subprocess.check_output(['git','-C',str(source_repo),'show',original_commit+':'+control]);after=(source_repo/control).read_bytes()
        _require(hashlib.sha256(before).hexdigest()==before_ref.get('sha256')
            and hashlib.sha256(after).hexdigest()==after_ref.get('sha256'),'FACTOR_REPAIR_SHA')
        def unchanged_outside_continuation(raw):
            tree=ast.parse(raw,filename=control)
            nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))
                and node.name=='require_factor_continuation']
            _require(len(nodes)==1,'FACTOR_REPAIR_FUNCTION')
            nodes[0].body=[ast.Pass()]
            return ast.dump(tree,include_attributes=False)
        _require(unchanged_outside_continuation(before)==unchanged_outside_continuation(after),'FACTOR_REPAIR_AST')
    for key in ('terminal_state','initial_state','initial_measurement','terminal_train_predictions'):
        campaign.require_binding(origin[key],label='matched sign '+key,verify_file=True)
