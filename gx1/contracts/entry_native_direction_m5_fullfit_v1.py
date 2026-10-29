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
