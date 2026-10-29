"""Bind matched M1/M5 full fits to their actual private qualifications.

Metadata only; this module neither reads market arrays nor grants execution.
"""
from __future__ import annotations
import ast
import hashlib
import subprocess
from pathlib import Path
from typing import Mapping
from gx1.contracts import local_random_access_campaign_v2 as campaign

_ROOT='/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63'
QUALIFICATIONS={
 'price_and_m1_coarsened':{'path':_ROOT+'/DIRECTION_MATCHED_M1_COARSENED_PRIVATE_QUALIFICATION_20261029_001/ACTUAL_ROOT_REVIEW_001.json','sha256':'50a940e16257abc6e9a1ad440a45082301af4ea227e9448cf15512e2a6c45c22'},
 'price_and_m1':{'path':_ROOT+'/DIRECTION_MATCHED_M1_ORDERED_PRIVATE_QUALIFICATION_20261029_001/ACTUAL_ROOT_REVIEW_001.json','sha256':'e179fbde91a492b78aa704246b90672c847e9df8fe1f6f30da6b07f2bb9476ae'},
}
INITIALS={
 mode:{'path':_ROOT+'/DIRECTION_MATCHED_M1_'+('COARSENED' if mode=='price_and_m1_coarsened' else 'ORDERED')+'_PRIVATE_QUALIFICATION_20261029_001/RUNTIME/private-research/INITIAL_TRAIN.npz',
       'sha256':'90c298464bf2282ee17f5b71a02fc05c1fd942d9254a11e7edad63d631bf7792'}
 for mode in QUALIFICATIONS
}
PAIR_PARITY={'path':_ROOT+'/DIRECTION_MATCHED_M1_INITIAL_PARITY_20261029_001/RESULT.json','sha256':'0361289d022d168d158473f2d97d3bb45ef061883a3bdec1c11d2c19ba179b67'}
TRANSITIONS={
 'gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py':{'changed':('require_prefix_fit_launch_recipe',),'new_assignments':('M1_PRICE_FIT_SCOPE',)},
 'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py':{'changed':('_fit_input_launch','run_native_direction_prefix_fit_window'),'new_assignments':()},
 'gx1/scripts/run_entry_native_direction_research_window_v1.py':{'changed':('_install_read_guard',),'new_assignments':()},
}


def _require(ok, why):
    if not ok:raise campaign.RandomAccessCampaignError('M1_MATCHED_FULLFIT_'+why)


def _read(ref, label):
    checked=campaign.require_binding(ref,label=label,verify_file=True)
    return campaign.read_bound_json(Path(checked['path']),checked['sha256'])


def require_qualified_source_transition(source_repo,qualified_commit,qualified_owners,owners):
    """Allow only declared orchestration changes; every other top-level AST is equal."""
    _require(type(qualified_commit) is str and campaign._COMMIT.fullmatch(qualified_commit) is not None,'QUALIFIED_COMMIT')
    for path,allowed in TRANSITIONS.items():
        before_ref=qualified_owners.get(path);after_ref=owners.get(path)
        _require(isinstance(before_ref,Mapping) and isinstance(after_ref,Mapping)
            and before_ref.get('path')==after_ref.get('path')==str(source_repo/path),'TRANSITION_PATH:'+path)
        before=subprocess.check_output(['git','-C',str(source_repo),'show',qualified_commit+':'+path])
        after=(source_repo/path).read_bytes()
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


COMPARISON_SOURCES=('gx1/scripts/score_entry_direction_dense_onepass_v1.py',
 'gx1/scripts/score_entry_direction_dense_minutes_v1.py','gx1/scripts/compare_entry_direction_matched_exposure_v1.py',
 'gx1/scripts/compare_entry_direction_saved_bps_v1.py','gx1/contracts/entry_direction_block_rank_v1.py',
 'gx1/contracts/entry_direction_probability_reference_v1.py','gx1/contracts/entry_native_direction_context_report_v1.py')


def require_protocol(protocol,recipe):
    _require(protocol.get('schema_version')=='gx1_matched_m1_structure_fullfit_protocol_v1'
        and protocol.get('status')=='PREDECLARED_BEFORE_MATCHED_FULL_FIT_NO_EXECUTION_AUTHORITY'
        and protocol.get('candidate_modes')==['price_and_m1_coarsened','price_and_m1']
        and protocol.get('fold')==recipe['fold'] and protocol.get('corrected_target')==recipe['dense_target']
        and protocol.get('training')=={'train_passes':1,'batch_size':16,'max_optimizer_updates':2976,'seed':20260911,'fresh_initial_per_arm':True,'initial_parity_rows':256,'qualification_weights_reused':False}
        and protocol.get('evaluation')=={'primary_horizon_minutes':5,'horizons_minutes':'1..60','data':'reused 2025 Q1 development, strict mature observed nonzero targets','primary_metrics':['paired_auc','accuracy_at_0.5','logloss','brier'],'references':['always_up','own_train_class_prior'],'paired_utc_day_week_bootstrap_replicates':10000,'all_horizon_auc_bands':'simultaneous across all 60, UTC-day and UTC-week','threshold_or_checkpoint_selection':False}
        and protocol.get('execution')=={'sequential_arms':True,'maximum_windows_per_arm':4,'physical_restart_before_each_window':True,'physical_gpu_power_limit_w':250,'restore_idle_limit_w':160}
        and protocol.get('test_data_used') is False and protocol.get('native_launch_authorized') is False and protocol.get('economic_edge_claimed') is False,'PROTOCOL')
    sources=protocol.get('comparison_sources');repo=Path(recipe['source_repo'])
    _require(isinstance(sources,Mapping) and set(sources)==set(COMPARISON_SOURCES)
        and protocol.get('comparison_function')=='gx1.scripts.score_entry_direction_dense_onepass_v1.compare_saved_predictions; call once per 60-head arm with the other arm reduced to 1/5/15-minute anchors; require exact matching corrected target/clock arrays before scoring.', 'COMPARISON_SOURCES')
    for name,ref in sources.items():
        checked=campaign.require_binding(ref,label='M1 matched comparison source',verify_file=True)
        _require(checked['path']==str(repo/name),'COMPARISON_SOURCE_PATH:'+name)


def require_qualification(recipe,owners,*,source_repo,private_scope):
    mode=recipe.get('candidate_mode');_require(mode in QUALIFICATIONS and recipe.get('native_dense_qualification')==QUALIFICATIONS[mode]
        and recipe.get('decimal_initial_predictions')==INITIALS[mode] and recipe.get('m1_pair_initial_parity')==PAIR_PARITY,'QUALIFICATION_BINDING')
    review=_read(QUALIFICATIONS[mode],'M1 private review');pair=_read(PAIR_PARITY,'M1 pair initial parity')
    suffix='COARSENED' if mode=='price_and_m1_coarsened' else 'ORDERED'
    _require(review.get('schema_version')=='gx1_m1_price_private_root_review_v1'
        and review.get('status')=='PASS_CONSUMED_'+suffix+'_PRIVATE_NATIVE_MECHANICS'
        and review.get('all_sixty_train_priors_checked') is True
        and review.get('all_sixty_heads_have_gradient_and_eligible_labels_on_fixed_batches') is True
        and review.get('m1_value_weight_gradient_nonzero_both_second_steps') is True
        and review.get('four_price_features_shared_projection_gradients_nonzero_in_both_trajectories') is True
        and review.get('bit_exact_resume_reported_and_bound') is True and review.get('private_optimizer_updates')==4
        and review.get('qualification_weights_for_main_fit') is False and review.get('test_data_used') is False,'PRIVATE_REVIEW')
    _require(pair.get('status')=='PASS_MATCHED_INITIAL_PREDICTIONS' and pair.get('test_data_used') is False
        and pair.get('fixed_32_train_parents_identical') is True and pair.get('all_60_train_priors_identical') is True
        and all(pair.get('comparison',{}).get(k,{}).get('byte_identical') is True for k in ('parent_rows','bar_start_ns','decision_ns','logit_up')),'PAIR_INITIAL_PARITY')
    plan=_read(review['plan'],'M1 private plan');prior_ref=plan['recipe'];prior=_read(prior_ref,'M1 private recipe')
    result=_read(review['result'],'M1 private result');receipt=_read(review['receipt'],'M1 private receipt')
    campaign.require_binding(review['idle_restore'],label='M1 idle restoration',verify_file=True)
    campaign.require_binding(INITIALS[mode],label='M1 private initial predictions',verify_file=True)
    _require(plan.get('source_commit')==review.get('source_commit') and receipt.get('plan_sha256')==plan.get('plan_sha256')
        and receipt.get('outcome')=='COMPLETE' and receipt.get('guard_decision')=='PASS' and receipt.get('test_data_used') is False,'PRIVATE_RECEIPT')
    _require(result.get('schema_version')=='gx1_native_m1_price_private_qualification_result_v1'
        and result.get('decision')=='PASS_M1_PRICE_PRIVATE_MECHANICS' and result.get('recipe')==prior_ref
        and result.get('candidate_mode')==mode and result.get('candidate_m1_refs')==recipe['candidate_m1_refs']
        and result.get('initial_predictions')==INITIALS[mode] and result.get('decimal_prior_against_sign_audit') is True
        and result.get('bit_exact_resume') is True and result.get('physical_optimizer_updates')==4
        and result.get('source_unchanged') is True and result.get('test_data_used') is False
        and result.get('private_weights_for_main_fit') is False,'PRIVATE_RESULT')
    keys=('input_manifest','fold','dense_target','dense_target_result','dense_source_transition_proof','decimal_input_audit','decimal_target_refs','candidate_mode','candidate_m1_refs','parent_order_algorithm')
    _require(prior.get('run_scope')==private_scope
        and all(prior.get(k)==recipe.get(k) for k in keys) and prior.get('qualification_weights_for_main_fit') is False
        and {k:v for k,v in prior['numerical_recipe'].items() if k not in ('max_updates','contract_sha256')}
            =={k:v for k,v in recipe['numerical_recipe'].items() if k not in ('max_updates','contract_sha256')},'PRIVATE_RECIPE')
    old=prior['source_owners'];_require(set(old)-set(TRANSITIONS) <= set(owners)
        and all(old[k]==owners[k] for k in set(old)-set(TRANSITIONS)),'QUALIFIED_UNCHANGED_SOURCES')
    require_qualified_source_transition(source_repo,review['source_commit'],old,owners)
