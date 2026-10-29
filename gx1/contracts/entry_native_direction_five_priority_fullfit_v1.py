"""Bind a one-pass M5 five-minute-priority fit to its private qualification.

Metadata checks only: execution still requires a separately bound campaign,
existing resource guards, and a fresh physical boot. No trading authority.
"""
from __future__ import annotations
import hashlib
import subprocess
from pathlib import Path
from typing import Mapping
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import entry_native_direction_five_priority_private_v1 as private
from gx1.contracts import entry_native_direction_m5_fullfit_v1 as m5

_ROOT='/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63'
QUALIFICATION={'path':_ROOT+'/DIRECTION_M5_FIVE_PRIORITY_PRIVATE_20261029_001/ACTUAL_ROOT_REVIEW_001.json',
               'sha256':'6281b90b9cf0fc994ac24d6fac4bd1a78b4197d754bb20f870ae6d27e18a6126'}
PRIVATE_INITIAL={'path':_ROOT+'/DIRECTION_M5_FIVE_PRIORITY_PRIVATE_20261029_001/RUNTIME/private-research/INITIAL_TRAIN.npz',
                 'sha256':m5.INITIAL['sha256']}
OWN_SOURCE='gx1/contracts/entry_native_direction_five_priority_fullfit_v1.py'
TRANSITIONS={
    'gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py':
        {'functions':('require_prefix_fit_launch_recipe',), 'new_assignments':('M5_FIVE_PRIORITY_FIT_SCOPE',)},
    'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py':
        {'functions':('_fit_input_launch','run_native_direction_prefix_fit_window')},
}


def _require(ok, why):
    if not ok: raise campaign.RandomAccessCampaignError('M5_FIVE_PRIORITY_FULLFIT_'+why)


def _read(ref, label):
    checked=campaign.require_binding(ref,label=label,verify_file=True)
    return campaign.read_bound_json(Path(checked['path']),checked['sha256'])


def require_protocol(design, recipe):
    candidate=design.get('candidate',{});fit=design.get('fit',{});comparison=design.get('comparison',{})
    _require(design.get('schema_version')=='gx1_m5_five_priority_candidate_design_v1'
        and design.get('status')=='PREDECLARED_NO_EXECUTION_AUTHORITY'
        and candidate.get('five_minute_weight')==10 and candidate.get('other_horizon_weight')==1
        and candidate.get('horizon_minutes')==list(range(1,61))
        and candidate.get('preserve_model_inputs_targets_and_seed') is True
        and fit.get('train_parent_rows')==47602 and fit.get('batch_size')==16
        and fit.get('max_optimizer_updates')==2976 and fit.get('passes')==1
        and all(fit.get(k) is True for k in ('fresh_own_initial_state','baseline_same_initial_logits_required',
            'resume_private_qualification_required','separate_one_shot_plan_required',
            'physical_restart_before_each_training_window','existing_power_draw_and_temperature_guards_preserved'))
        and fit.get('qualification_weights_reused') is False and fit.get('gpu_limit_during_and_after_w')==250
        and comparison.get('primary_minutes')==5 and comparison.get('report_all_horizons') is True
        and comparison.get('validate_exact_parent_clock_targets_before_scores') is True
        and comparison.get('paired_day_and_week_intervals_for_primary') is True
        and comparison.get('simultaneous_candidate_auc_intervals_all_60') is True
        and comparison.get('no_horizon_or_parameter_search') is True
        and comparison.get('training_predictive_or_economic_edge_claimed') is False
        and design.get('test_data_used') is False
        and design.get('broker_orders_live_paper_exit_or_promotion_authorized') is False,'DESIGN')
    _require(recipe.get('dense_evaluation_protocol')==private.DESIGN
        and recipe.get('priority_baseline_initial_report')==private.BASELINE_INITIAL_REPORT
        and design.get('baseline',{}).get('recipe')==private.BASELINE_RECIPE,'DESIGN_BINDING')
    for key in ('recipe','fit_result','later_report'):
        campaign.require_binding(design['baseline'][key],label='priority baseline '+key,verify_file=True)


def require_source_transition(source_repo, source_commit, qualified_commit, old, owners):
    _require(set(owners)==set(old)|{OWN_SOURCE},'SOURCE_SET')
    _require(subprocess.run(['git','merge-base','--is-ancestor',qualified_commit,source_commit],
        cwd=source_repo,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0,'SOURCE_ANCESTRY')
    for name,original_ref in old.items():
        current_ref=owners[name]
        if name not in TRANSITIONS:
            _require(current_ref==original_ref,'UNCHANGED_SOURCE:'+name)
            continue
        _require(original_ref['path']==current_ref['path']==str(source_repo/name),'SOURCE_PATH:'+name)
        before=subprocess.check_output(['git','show',qualified_commit+':'+name],cwd=source_repo)
        after=(source_repo/name).read_bytes()
        _require(hashlib.sha256(before).hexdigest()==original_ref['sha256']
            and hashlib.sha256(after).hexdigest()==current_ref['sha256'],'SOURCE_SHA:'+name)
        spec=TRANSITIONS[name]
        _require(private._unchanged_ast(before,spec,original=True)==private._unchanged_ast(after,spec,original=False),
            'SOURCE_AST:'+name)


def require_qualification(recipe, owners, *, source_repo, source_commit, private_scope):
    _require(recipe.get('candidate_mode')=='price_only' and 'candidate_m1_refs' not in recipe
        and 'm1_pair_initial_parity' not in recipe and recipe.get('native_dense_qualification')==QUALIFICATION
        and recipe.get('decimal_initial_predictions')==m5.INITIAL
        and recipe.get('numerical_recipe',{}).get('five_minute_priority') is True,'QUALIFICATION_BINDING')
    review=_read(QUALIFICATION,'priority private review')
    _require(review.get('schema_version')=='gx1_m5_five_priority_private_root_review_v1'
        and review.get('status')=='PASS_CONSUMED_PRICE_ONLY_FIVE_PRIORITY_PRIVATE_MECHANICS'
        and review.get('candidate_mode')=='price_only' and review.get('m1_values_and_masks_as_entry_inputs') is False
        and all(review.get(k) is True for k in ('initial_model_and_buffers_exactly_matched_previous_m5_fit',
            'initial_predictions_256_exactly_matched_previous_m5_fit','all_sixty_heads_have_eligible_labels_and_gradient_evidence',
            'four_price_projection_columns_nonzero_gradient_on_each_step','bit_exact_resume','campaign_task_disabled','source_unchanged'))
        and review.get('private_optimizer_updates')==4 and review.get('unique_train_parents')==32
        and review.get('configured_gpu_limit_after_w')==250 and review.get('qualification_weights_for_main_fit') is False
        and review.get('test_data_used') is False,'PRIVATE_REVIEW')
    plan=_read(review['plan'],'priority private plan');prior_ref=plan['recipe']
    prior=_read(prior_ref,'priority private recipe');result=_read(review['result'],'priority private result')
    receipt=_read(review['receipt'],'priority private receipt');power=_read(review['power_after'],'priority post-campaign power')
    baseline=_read(private.BASELINE_RECIPE,'priority baseline recipe')
    initial=_read(private.BASELINE_INITIAL_REPORT,'priority baseline initial')
    campaign.require_binding(PRIVATE_INITIAL,label='priority initial predictions',verify_file=True)
    campaign.require_binding(m5.INITIAL,label='baseline initial predictions',verify_file=True)
    gpu=power.get('gpu')
    gpu_fields=gpu.split(',') if isinstance(gpu,str) else []
    _require(power.get('campaign_task_state')=='Disabled' and power.get('configured_power_limit_w')==250
        and len(gpu_fields)==6 and gpu_fields[1].strip()=='250.00' and power.get('keeper_state')=='Running'
        and power.get('bridge_state')=='Running' and power.get('test_data_used') is False,'PRIVATE_POWER')
    _require(plan.get('source_commit')==review.get('source_commit') and receipt.get('plan_sha256')==plan.get('plan_sha256')
        and receipt.get('outcome')=='COMPLETE' and receipt.get('guard_decision')=='PASS'
        and receipt.get('trainer_guard_exit_code')==receipt.get('progress_observer_exit_code')==0
        and receipt.get('test_data_used') is False,'PRIVATE_RECEIPT')
    _require(result.get('schema_version')=='gx1_native_m1_price_private_qualification_result_v1'
        and result.get('decision')=='PASS_M1_PRICE_PRIVATE_MECHANICS' and result.get('recipe')==prior_ref
        and result.get('candidate_mode')=='price_only' and result.get('candidate_m1_refs') is None
        and result.get('initial_predictions')==PRIVATE_INITIAL and result.get('priority_design')==private.DESIGN
        and result.get('priority_baseline_initial_report')==private.BASELINE_INITIAL_REPORT
        and result.get('priority_initial_prediction_and_model_buffer_byte_parity') is True
        and result.get('decimal_prior_against_sign_audit') is True and result.get('bit_exact_resume') is True
        and result.get('physical_optimizer_updates')==4 and result.get('source_unchanged') is True
        and result.get('test_data_used') is False and result.get('private_weights_for_main_fit') is False,'PRIVATE_RESULT')
    keys=('input_manifest','fold','dense_target','dense_target_result','dense_source_transition_proof',
        'decimal_input_audit','decimal_target_refs','candidate_mode','parent_order_algorithm','gc_layout_metadata_parity')
    _require(prior.get('run_scope')==private_scope and 'candidate_m1_refs' not in prior
        and all(prior.get(k)==recipe.get(k)==baseline.get(k) for k in keys)
        and prior.get('priority_design')==private.DESIGN
        and prior.get('priority_baseline_initial_report')==private.BASELINE_INITIAL_REPORT
        and prior.get('qualification_weights_for_main_fit') is False
        and {k:v for k,v in prior['numerical_recipe'].items() if k not in ('max_updates','contract_sha256')}
            =={k:v for k,v in recipe['numerical_recipe'].items() if k not in ('max_updates','contract_sha256')}
        and {k:v for k,v in baseline['numerical_recipe'].items() if k not in ('contract_sha256','research_objective_sha256')}
            =={k:v for k,v in recipe['numerical_recipe'].items() if k not in ('contract_sha256','research_objective_sha256','five_minute_priority')},
        'MATCHED_RECIPE')
    _require(initial.get('role')=='INITIAL_TRAIN' and initial.get('test_data_used') is False
        and initial.get('fold')==recipe.get('fold') and initial.get('predictions',{}).get('sha256')==m5.INITIAL['sha256']
        and initial.get('preservation',{}).get('model') and initial.get('preservation',{}).get('all_buffers'),'BASELINE_INITIAL')
    require_source_transition(source_repo,source_commit,review['source_commit'],prior['source_owners'],owners)
