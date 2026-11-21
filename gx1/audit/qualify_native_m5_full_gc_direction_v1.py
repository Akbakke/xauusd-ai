"""Genuine joint native TA/GC input, backward and coordinate-control qualification."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import time

if __name__=='__main__':
    from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
    install_native_m5_direction_io_firewall()

import numpy as np
from gx1.audit import qualify_market_state_native_m5_pipeline_v1 as ta_audit
from gx1.audit import build_native_m5_gc_raw_components_v1 as common

REPO=common.REPO;POLICY=REPO/'NEXT_RUN_POLICY.json'
STAGE='native_m5_full_gc_joint_qualification_20261119_001'
PLAN=REPO/'configs/research/NATIVE_M5_FULL_GC_JOINT_QUALIFICATION_20261119_001.json'
ROOT=Path('/home/andre2/GX1_RUNS/NATIVE_M5_FULL_GC_JOINT_QUALIFICATION_20261119_001')
INPUT_KEYS={'ta_result','ta_terminal','gc_result','gc_terminal'}
EXECUTION={'class':'producer','memory':'12G','swap':'512M','outer_seconds':1800,
    'internal_seconds':1770,'kill_after_seconds':10,'numerical_threads':1,
    'model_constructions':1,'model_forwards':3,'autograd_calls':1,'optimizer_steps':0}
SOURCE_PATHS=tuple(dict.fromkeys((*ta_audit.SOURCE_PATHS,
    'gx1/audit/qualify_native_m5_full_gc_direction_v1.py',
    'tests/test_qualify_native_m5_full_gc_direction_v1.py',
    'gx1/audit/build_native_m5_gc_raw_components_v1.py',
    'gx1/audit/assemble_native_m5_gc_inputs_v1.py',
    'gx1/contracts/market_state_native_m5_gc_feature_inputs_v1.py',
    'gx1/features/gc_native_m5_features_v1.py','gx1/features/gc_order_flow_v1.py',
    'gx1/contracts/gc_native_m5_source_inputs_v1.py','gx1/contracts/gc_continuous_source_v1.py',
    'gx1/contracts/gc_source_disposition_v1.py','gx1/contracts/gc_order_flow_source_v1.py')))
sha,ref,read,publish=common.sha,common.ref,common.read,common.publish


def need(ok,reason):
    if not bool(ok):raise ValueError('NATIVE_M5_JOINT_'+reason)


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def batch_hashes(batch):
    return {key:digest(value)for key,value in batch.items()}


def complete_parent(result_ref,terminal_ref,status,stage):
    result,terminal=read(result_ref),read(terminal_ref)
    plan_ref,claim_ref=result['plan'],result['claim']
    plan,claim=read(plan_ref),read(claim_ref)
    need(result['status']==status and result['stage']==terminal['stage']==plan['stage']==claim['stage']==stage
        and terminal['status']=='COMPLETE'and terminal['exit_code']==0
        and terminal['result']==result_ref and terminal['plan']==claim['plan']==plan_ref
        and terminal['claim']==claim_ref and result['source_unchanged']is True
        and result['test_data_used']is terminal['test_data_used']is False
        and result['optimizer_steps']==0,'COMPLETE_PARENT:'+stage)
    return result,plan,plan_ref,claim_ref


def input_metadata(inputs):
    from gx1.contracts import market_state_native_m5_gc_feature_inputs_v1 as gc
    need(type(inputs)is dict and set(inputs)==INPUT_KEYS,'INPUT_ROLES')
    need(inputs['ta_result']=={'path':'/home/andre2/GX1_RUNS/NATIVE_M5_PIPELINE_20261119_006/RESULT.json',
        'sha256':'5c015b3af8a17c8a16459be0eeba5ca017b7625db91e193aa84b101e664e7bca'}
        and inputs['ta_terminal']=={'path':'/home/andre2/GX1_RUNS/NATIVE_M5_PIPELINE_20261119_006/TERMINAL.json',
        'sha256':'849bcda1483e088ca5fa2ec8671e8ed6b003654331a6556c70af53dcfc11c874'},'EXACT_QUALIFIED_TA006')
    ta,ta_plan,ta_plan_ref,ta_claim_ref=complete_parent(inputs['ta_result'],inputs['ta_terminal'],
        'COMPLETE_NATIVE_M5_PIPELINE_TECHNICAL_QUALIFICATION','native_m5_pipeline_20261119_006')
    assembled,gc_plan,gc_plan_ref,gc_claim_ref=complete_parent(inputs['gc_result'],inputs['gc_terminal'],
        'COMPLETE_NATIVE_GC_INPUTS_ONLY','gc_native_m5_assembly_20261119_001')
    need(ta['model_forwards']==1 and assembled['model_forwards']==0 and assembled['raw_replay']is False,
         'PARENT_EXECUTION_SCOPE')
    ta_support_ref=ta['report']['feature_support_artifact'];support=read(ta_support_ref)
    need(ta['report']['fit_population']==support['fit_population'],'QUALIFIED_AXIS_RECEIPT')
    source_ref=assembled['report']['source_binding']
    source=gc.require_native_m5_gc_source_binding(read(source_ref))
    need(source['native_xau_m5_manifest']==ta_plan['feature_source_spec']['manifest']
        and source['packet_artifacts']==assembled['report']['packet_artifacts']
        and source['assembly_proof']==assembled['report']['assembly_proof'],'NATIVE_SOURCE_JOIN')
    proof=read(source['assembly_proof'])
    gc._raw_component_lineage(proof,gc_plan['metadata']['raw_source_binding'])
    need(proof['status']=='COMPLETE'and proof['source_unchanged']is True
        and proof['m1_data_or_metadata_used']is False and proof['test_data_used']is False
        and proof['packet_artifacts']==source['packet_artifacts'],'ASSEMBLY_PROOF')
    dependencies={'ta_plan':ta_plan_ref,'ta_claim':ta_claim_ref,'ta_support':ta_support_ref,
        'ta_admission':ta['report']['feature_inputs_requalification'],
        'ta_candidate_snapshot':ta_plan['inputs']['candidate_snapshot'],
        'ta_candidate_archive':ta_plan['inputs']['candidate_archive'],
        'target_tape':ta_plan['inputs']['target_tape'],'gc_plan':gc_plan_ref,'gc_claim':gc_claim_ref,
        'gc_source_binding':source_ref,'gc_assembly_proof':source['assembly_proof'],
        'gc_packet_json':source['packet_artifacts']['json'],'gc_packet_npz':source['packet_artifacts']['npz']}
    dependencies.update({key:proof[key]for key in ('raw_components_plan','raw_components_claim','raw_components_result','raw_components_terminal')})
    dependencies.update({'raw_component_'+str(i):value for i,value in enumerate(proof['raw_components'])})
    return {'dependencies':dependencies,'ta_plan':ta_plan,'ta_support':support,
        'native_gc_source_binding':source,'architecture':ta_plan['architecture'],'seed':ta_plan['seed'],
        'fold_upper_ns':ta_plan['fit_end_ns'],'target_contract':ta_plan['target_contract'],
        'qualified_fit_population':support['fit_population'],
        'qualified_normalization_population':ta['report']['normalization_population']}


def build_plan(*,inputs,source_base_commit=None):
    from gx1.features.gc_native_m5_features_v1 import native_m5_field_catalog,native_m5_source_unit_policy
    from gx1.contracts.native_m5_direction_io_firewall_v1 import contract
    metadata=input_metadata(inputs)
    head=source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    return {'schema_version':'gx1_native_m5_full_gc_joint_qualification_plan_v1','stage':STAGE,
        'status':'PREPARED_NOT_ADMITTED','output_root':str(ROOT),'source_base_commit':head,
        'source_owners':{path:sha(REPO/path)for path in SOURCE_PATHS},'inputs':inputs,'metadata':metadata,
        'execution':EXECUTION,'io_firewall':contract(),
        'catalog_contract_sha256':native_m5_field_catalog()['contract_sha256'],
        'unit_policy_sha256':native_m5_source_unit_policy()['contract_sha256'],
        'fit_axis':'exact_qualified_TA006_no_GC_observedness_row_filter',
        'panel_rule':'first_16_fixed_fit_rows_with_observed_nonBoolean_market_in_each_declared_supported_GC_TF_family_window',
        'control_rule':'first_catalog_order_observed_nonBoolean_transformNone_primitive_latest_local_and_own_TF_covisible_source_bar_one_positive_declared_unit_with_typed_bounds',
        'model_constructions':1,'model_forwards':3,'autograd_calls':1,'optimizer_steps':0,
        'training_authorized':False,'test_data_used':False,'economic_edge_proven':False}


def source_clean(plan):
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    return(subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current'
        and not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)
        and subprocess.run(['git','merge-base','--is-ancestor',plan['source_base_commit'],head],
            cwd=REPO,capture_output=True).returncode==0
        and subprocess.run(['git','diff','--quiet',plan['source_base_commit'],head,'--','gx1','scripts'],cwd=REPO).returncode==0
        and set(plan['source_owners'])==set(SOURCE_PATHS)
        and all(sha(REPO/path)==value for path,value in plan['source_owners'].items()))


def checked_inputs(plan):
    for binding in [*plan['inputs'].values(),*plan['metadata']['dependencies'].values()]:
        need(ref(binding['path'])==binding,'INPUT_HASH_BOOKEND')


def inspect(plan_sha,*,admitted):
    need(Path.cwd().resolve()==REPO and not ROOT.exists(),'CWD_OUTPUT')
    plan_ref={'path':str(PLAN),'sha256':plan_sha};plan=read(plan_ref)
    need(plan==build_plan(inputs=plan['inputs'],source_base_commit=plan['source_base_commit'])
         and source_clean(plan),'PLAN_SOURCE')
    policy_ref=ref(POLICY);policy=read(policy_ref);gate=policy.get(STAGE,{})
    need(all(policy.get(key)is False for key in ta_audit.data.CLOSED_FLAGS)
        and gate.get('plan')==plan_ref and gate.get('output_root')==str(ROOT)
        and gate.get('relaunch_allowed')is False and gate.get('max_invocations')==1
        and gate.get('claimed_invocations')==gate.get('completed_invocations')==0
        and gate.get('training_authorized')is False and gate.get('test_authorized')is False
        and gate.get('model_forwards')==3 and gate.get('autograd_calls')==1 and gate.get('optimizer_steps')==0,
        'CLOSED_POLICY')
    expected=('ADMITTED_ONE_USE',True,1)if admitted else('PREPARED_NOT_ADMITTED',False,0)
    need((gate.get('status'),gate.get('execution_authorized'),gate.get('remaining_invocations'))==expected,'ONE_USE_GATE')
    checked_inputs(plan);return plan,plan_ref,policy_ref


def market_routes(provider):
    result={}
    for tf in provider.layout['timeframes']:
        names=provider.layout['field_names'][tf];positions={name:i for i,name in enumerate(names)}
        for family,indices in provider.layout['family_indices'][tf].items():
            if not indices:continue
            allowed=set(indices)
            fields=[field for field in provider.fields if field['timeframe']==tf
                and not field['structural_mask_value'] and not field['binary_value']
                and field['transform']is None and positions[field['name']]in allowed]
            result[tf.lower()+'__'+family]={'tf':tf,'family':family,'indices':list(indices),'fields':fields}
    return result


def choose_witness_panel(provider,fit_rows,*,batch_rows=16):
    eligible=np.ones(len(fit_rows),bool);report={}
    routes=market_routes(provider)
    need(routes,'NO_GC_PROJECTION_ROUTES')
    for key,route in routes.items():
        tf=route['tf'];names=provider.layout['field_names'][tf]
        fields=route['fields'];matrix=provider.gc_surfaces[tf]['features']
        if not fields:
            report[key]={'nonBoolean_market_fields':0,'criterion':'not_applicable_no_declared_nonBoolean_market_value'}
            continue
        values=np.asarray([names.index(field['name'])for field in fields],np.int64)
        gates=np.asarray([names.index(field['availability_field'])for field in fields],np.int64)
        observed=np.any((matrix[:,gates]==1)&np.isfinite(matrix[:,values]),axis=1)
        total=np.r_[0,np.cumsum(observed,dtype=np.int64)]
        tails=provider.ta.tf_tails[tf][fit_rows]
        # Geometry is source-owned; the TA provider already validates each tail.
        from gx1.contracts.market_state_native_m5_feature_inputs_v1 import WINDOWS
        window=WINDOWS[tf];ready=(tails>=window-1)
        support=np.zeros(len(fit_rows),bool)
        support[ready]=(total[tails[ready]+1]-total[tails[ready]-window+1])>0
        eligible&=support
        report[key]={'nonBoolean_market_fields':len(fields),'eligible_fit_rows':int(support.sum())}
    candidates=fit_rows[eligible]
    need(len(candidates)>=batch_rows,'INSUFFICIENT_INPUT_ONLY_GC_PANEL')
    rows=candidates[:batch_rows]
    return rows,{'source_only':True,'future_outcomes_inspected':False,'fit_axis_rows_unchanged':len(fit_rows),
        'input_eligible_rows':len(candidates),'selected_rows':len(rows),
        'selected_source_rows_sha256':digest(rows),'per_route':report}


def choose_control(provider,packet,panel):
    from gx1.contracts import market_state_native_m5_gc_feature_inputs_v1 as gc
    for field in provider.fields:
        if field['binary_value']or field['structural_mask_value']or field['transform']is not None:continue
        table=gc._table(packet,field);clock=np.asarray(table['available_ns'])
        tf=field['timeframe'];tails=provider.ta.tf_tails[tf][panel]
        from gx1.contracts.market_state_native_m5_feature_inputs_v1 import WINDOWS,MODEL_NATIVE_SEQ_LEN
        take=tails[:,None]+np.arange(-WINDOWS[tf]+1,1,dtype=np.int64)
        side_available=np.unique(provider.gc_surfaces[tf]['available_ns'][take])
        local_positions=provider.ta._feature_positions(panel)
        local_ids=local_positions[:,None]+np.arange(-MODEL_NATIVE_SEQ_LEN+1,1,dtype=np.int64)
        local_available=provider.local_gc_bucket_available_ns[tf][local_ids]
        available=np.intersect1d(side_available,local_available[local_available>=0])[::-1]
        for when in available:
            at=np.searchsorted(clock,when)
            if at>=len(clock)or clock[at]!=when:continue
            value,observed=gc._field_values(field,table,np.array([when],np.int64),exact=True)
            if not observed[0]:continue
            old=float(np.asarray(table[field['column']])[at]);delta=float(field['unit_scale']);new=old+delta
            bounds=gc._diagnostic_scalar_bounds(field)
            if not(np.isfinite(new)and np.float32(old)!=np.float32(new)):continue
            if bounds['lower']is not None and not(new>=bounds['lower']if bounds['lower_inclusive']else new>bounds['lower']):continue
            if bounds['upper']is not None and new>bounds['upper']:continue
            return {'field_name':field['name'],'source_available_ns':int(when),'delta':delta,
                'selection':'catalog_order_then_latest_local_and_own_TF_covisible_source_bar_input_only','old':old,'new':new}
    raise ValueError('NATIVE_M5_JOINT_NO_TYPED_OBSERVED_PRIMITIVE')


def normalized_surface(model,key,value):
    if key=='ctx_cat':return value
    if key.startswith('gc_seq_'):
        return model.backbone._normalize_gc_history_surface(value,timeframe=key.removeprefix('gc_seq_').upper())
    surface='signal'if key in('seq_x','snap_x')else'ctx_cont'if key=='ctx_cont'else'mtf_'+key.removeprefix('seq_')
    return model.backbone._normalize_input_surface(value,surface=surface)


def normalization_report(model,tensors,*,reference=None):
    report={}
    for key,value in tensors.items():
        normalized=normalized_surface(model,key,value)
        raw=value.detach().cpu().numpy();out=normalized.detach().cpu().numpy()
        need(np.isfinite(raw).all()and np.isfinite(out).all(),'NORMALIZED_FINITE:'+key)
        report[key]={'shape':list(raw.shape),'raw_sha256':digest(raw),'normalized_sha256':digest(out)}
        if reference is not None:
            previous=reference[key]
            old=normalized_surface(model,key,previous).detach().cpu().numpy()
            report[key]['raw_delta_max_abs']=float(np.max(np.abs(raw.astype(np.float64)-previous.detach().cpu().numpy())))
            report[key]['normalized_delta_max_abs']=float(np.max(np.abs(out.astype(np.float64)-old)))
    need(len(report)==14,'FOURTEEN_KEYS');return report


def gradients(model,batch,truth,target_contract,fold_upper_ns,provider):
    import torch
    from gx1.contracts import entry_model_native_input_normalization_v1 as norm
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_direction_loss
    features={key:torch.from_numpy(value).requires_grad_(value.dtype.kind=='f')for key,value in batch.items()}
    captured={};handles=[]
    for key,layer in model.backbone.gc_mtf_family_proj.items():
        def hook(module,args,output,key=key):captured[key]=(args[0],output)
        handles.append(layer.register_forward_hook(hook))
    try:output=model(features)
    finally:
        for handle in handles:handle.remove()
    loss=native_bar_direction_loss(model_output=output,truth=truth,fold_upper_ns=fold_upper_ns,
                                  target_contract_sha256=target_contract['contract_sha256'])
    need(not loss['skip_update']and bool(torch.isfinite(loss['loss'])),'GENUINE_LOSS')
    float_keys=[key for key,value in features.items()if value.dtype.is_floating_point]
    projectors=list(model.backbone.gc_mtf_family_proj)
    need(set(captured)==set(projectors),'ACTUAL_GC_PROJECTION_CALLS')
    heads=[(kind,tf,head)for kind,collection in(('open',model.open_heads),('close',model.close_heads))
           for tf,head in collection.items()]
    targets=[features[key]for key in float_keys]+[captured[key][1]for key in projectors]
    targets +=[model.backbone.gc_mtf_family_proj[key].weight for key in projectors]
    targets +=[parameter for kind,tf,head in heads for parameter in(head.weight,head.bias)]
    returned=torch.autograd.grad(loss['loss'],targets,allow_unused=True)
    need(all(value is None or bool(torch.isfinite(value).all()) for value in returned),'FINITE_RETURNED_GRADIENTS')
    at=0;input_grads={}
    for key in float_keys:input_grads[key]=returned[at];at+=1
    output_grads={key:returned[at+i]for i,key in enumerate(projectors)};at+=len(projectors)
    weight_grads={key:returned[at+i]for i,key in enumerate(projectors)};at+=len(projectors)
    input_report={};zero_unknown=True
    market_names={field['name'] for field in provider.fields if not field['structural_mask_value'] and not field['binary_value'] and field['transform']is None}
    for key,gradient in input_grads.items():
        if key.startswith('gc_seq_'):names=provider.layout['field_names'][key.removeprefix('gc_seq_').upper()]
        elif key in('seq_x','snap_x'):names=model.backbone._input_normalization_field_names['signal']
        elif key=='ctx_cont':names=model.backbone._input_normalization_field_names['ctx_cont']
        else:continue
        raw=batch[key];g=np.zeros_like(raw)if gradient is None else gradient.detach().cpu().numpy()
        pairs=norm.native_gc_masking_pairs(names)
        unknown=0;unknown_nonzero=0;observed_norm=0.;gate_norm=0.
        for value,gate in pairs:
            missing=raw[...,gate]==0;known=~missing
            unknown+=int(missing.sum());unknown_nonzero+=int(np.count_nonzero(g[...,value][missing]))
            if str(names[value]).removeprefix('ctx_cont.')in market_names:
                observed_norm+=float(np.square(g[...,value][known].astype(np.float64)).sum())
            gate_norm+=float(np.square(g[...,gate].astype(np.float64)).sum())
        zero_unknown&=unknown_nonzero==0
        input_report[key]={'unknown_value_cells':unknown,'unknown_value_nonzero_gradients':unknown_nonzero,
            'observed_market_value_gradient_l2':observed_norm**.5,'observation_gate_gradient_l2':gate_norm**.5}
    routes=market_routes(provider);projection_report={}
    for key in projectors:
        activation,projected=captured[key];weight_gradient=weight_grads[key];out_gradient=output_grads[key]
        norm_weight=0. if weight_gradient is None else float(torch.linalg.vector_norm(weight_gradient))
        norm_output=0. if out_gradient is None else float(torch.linalg.vector_norm(out_gradient))
        route=routes[key];tf=route['tf'];names=provider.layout['field_names'][tf]
        globals_=list(model.backbone._gc_mtf_specialist_index_tuples[key])
        market={names.index(field['name'])for field in route['fields']}
        columns=[j for j,index in enumerate(globals_)if index in market]
        market_weight=0. if weight_gradient is None or not columns else float(torch.linalg.vector_norm(weight_gradient[:,columns]))
        raw_gc=batch['gc_seq_'+tf.lower()];input_gradient=input_grads['gc_seq_'+tf.lower()]
        observed_market_cells=0;market_input_square=0.
        for field in route['fields']:
            value_col=names.index(field['name']);gate_col=names.index(field['availability_field'])
            observed=raw_gc[...,gate_col]==1;observed_market_cells+=int(observed.sum())
            if input_gradient is not None:
                selected_grad=input_gradient[...,value_col].detach().cpu().numpy()[observed]
                market_input_square+=float(np.square(selected_grad.astype(np.float64)).sum())
        projection_report[key]={'shape':list(activation.shape),'actual_input_sha256':digest(activation.detach().cpu().numpy()),
            'actual_projection_sha256':digest(projected.detach().cpu().numpy()),
            'projection_output_gradient_l2':norm_output,'projection_weight_gradient_l2':norm_weight,
            'nonBoolean_market_weight_gradient_l2':market_weight,
            'observed_nonBoolean_market_cells':observed_market_cells,
            'observed_nonBoolean_market_input_gradient_l2':market_input_square**.5,'declared_indices':globals_}
        need(norm_output>0 and norm_weight>0,'GC_PROJECTION_GRADIENT:'+key)
        if route['fields']:
            need(observed_market_cells>0 and market_input_square>0,'OBSERVED_MARKET_INPUT_GRADIENT:'+key)
            need(market_weight>0,'MARKET_VALUE_PROJECTION_WEIGHT_GRADIENT:'+key)
    head_report={}
    for kind,tf,head in heads:
        w,b=returned[at:at+2];at+=2;j=list(model.close_heads).index(tf)
        valid_count=int(loss['counts'][kind][j])
        head_report[kind+'_'+tf]={'loss_valid_cells':valid_count,
            'weight_gradient_l2':0. if w is None else float(torch.linalg.vector_norm(w)),
            'bias_gradient_l2':0. if b is None else float(torch.linalg.vector_norm(b))}
        if valid_count:
            need(w is not None and b is not None and (bool(torch.count_nonzero(w))
                or bool(torch.count_nonzero(b))),'VALID_HEAD_GRADIENT:'+kind+'_'+tf)
    need(zero_unknown,'UNKNOWN_MARKET_VALUE_GRADIENT')
    need(all(parameter.grad is None for parameter in model.parameters()),'PARAMETER_GRAD_ALLOCATED')
    return {key:value.detach().cpu().numpy().copy()for key,value in output.items()},{
        'actual_bar_loss':float(loss['loss']),'input_gradients':input_report,'gc_projections':projection_report,
        'heads':head_report,'unknown_value_gradients_exact_zero':zero_unknown,'autograd_calls':1,
        'parameter_grad_fields_allocated':0}


def qualify(plan):
    import torch
    from gx1.contracts import market_state_native_m5_gc_feature_inputs_v1 as gc
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_bar_direction_model
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_direction_score_sums
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    meta=plan['metadata'];ta,admission=ta_audit.load_candidate_provider(meta['ta_plan'])
    rows,truth,population=ta_audit.target_axis(meta['ta_plan'],ta)
    need(population==meta['qualified_fit_population'],'EXACT_QUALIFIED_FIT_AXIS')
    packet=gc.load_native_m5_gc_feature_packet(meta['native_gc_source_binding'])
    provider=gc.NativeM5GCFeatureProvider(ta,packet,meta['native_gc_source_binding'])
    need(np.array_equal(provider.input_ready,ta.input_ready),'GC_ROW_FILTER_FORBIDDEN')
    panel,panel_report=choose_witness_panel(provider,rows)
    normalization=gc.fit_native_m5_gc_feature_normalization(provider,fit_source_rows=rows,cutoff_ns=meta['fold_upper_ns'])
    need(normalization['fit_population']==meta['qualified_normalization_population'],'EXACT_12_SURFACE_POPULATION')
    normalization_ref=publish(ROOT/'NORMALIZATION.json',normalization)
    input_binding=provider.input_binding(normalization)
    positions=np.searchsorted(rows,panel)
    batch_truth={key:value[positions]for key,value in truth.items()}
    raw=provider.gather(panel,max_rows=16)
    choice=choose_control(provider,packet,panel)
    original_norm_sha=gc._sha(normalization);original_source_sha=gc._sha(provider.source_binding)
    diagnostic=gc.diagnostic_regather_native_m5_gc_primitive(provider,source_packet=packet,
        input_normalization=normalization,source_rows=panel,max_rows=16,
        field_name=choice['field_name'],source_available_ns=choice['source_available_ns'],delta=choice['delta'])
    need(batch_hashes(diagnostic['before'])==batch_hashes(raw),'DIAGNOSTIC_GENUINE_BATCH_PARITY')
    torch.set_num_threads(1);torch.manual_seed(meta['seed'])
    model=build_native_bar_direction_model(input_normalization=normalization,input_binding=input_binding,
        architecture=meta['architecture'],device=torch.device('cpu'),gc_mtf_history=normalization['gc_mtf_history']).eval()
    forward_count=[0]
    def counted(module,args):
        forward_count[0]+=1
        need(torch.is_grad_enabled()and not module.training and forward_count[0]<=3,'FORWARD_LANE_OR_LIMIT')
    count_handle=model.register_forward_pre_hook(counted)
    before=canonical_model_state_sha256(model.state_dict())
    head_states={kind+'_'+tf:{'weight_shape':list(head.weight.shape),'bias_shape':list(head.bias.shape),'weight_sha256':digest(head.weight.detach().cpu().numpy()),
        'bias_sha256':digest(head.bias.detach().cpu().numpy())}
        for kind,collection in(('open',model.open_heads),('close',model.close_heads))for tf,head in collection.items()}
    features={key:torch.from_numpy(value)for key,value in raw.items()}
    with torch.enable_grad():
        normalized=normalization_report(model,features)
        first=model(features)
    initial={key:value.detach().cpu().numpy().copy()for key,value in first.items()};del first,features
    with torch.enable_grad():
        repeated,witness=gradients(model,raw,batch_truth,meta['target_contract'],meta['fold_upper_ns'],provider)
    need(all(np.array_equal(initial[key],repeated[key])for key in initial),'SAME_GENUINE_FORWARD_LANE')
    with torch.enable_grad():
        control_features={key:torch.from_numpy(value)for key,value in diagnostic['after'].items()}
        control_surfaces=normalization_report(model,control_features,
            reference={key:torch.from_numpy(value)for key,value in raw.items()})
        controlled=model(control_features)
    del control_features
    control={key:value.detach().cpu().numpy().copy()for key,value in controlled.items()};del controlled
    need(all(np.isfinite(value).all()for value in control.values()),'CONTROL_FINITE')
    need(any(np.any(control[key]!=initial[key])for key in initial),'CONTROL_LOGIT_CONTACT')
    count_handle.remove();need(forward_count[0]==3,'EXACT_THREE_FORWARDS')
    need(canonical_model_state_sha256(model.state_dict())==before
        and gc._sha(normalization)==original_norm_sha
        and gc._sha(provider.source_binding)==original_source_sha
        and provider.input_binding(normalization)==input_binding
        and all(parameter.grad is None for parameter in model.parameters()),'MODEL_PARAMETER_BOOKEND')
    checkpoint_binding={'input':input_binding,'normalization':normalization_ref,'model':model.research_objective,
        'target':meta['target_contract'],'architecture':meta['architecture'],'seed':meta['seed'],'model_state_sha256':before}
    stream=io.BytesIO();torch.save({'model_state':model.state_dict(),'bindings':checkpoint_binding},stream)
    payload=stream.getvalue();stream.close()
    loaded=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
    need(loaded['bindings']==checkpoint_binding and canonical_model_state_sha256(loaded['model_state'])==before,
         'CHECKPOINT_SERIALIZATION_PARITY')
    model.load_state_dict(loaded['model_state'],strict=True)
    checkpoint=publish(ROOT/'INITIAL_MODEL.pt',payload)
    capture={'source_row':panel,**{'raw_'+key:value for key,value in raw.items()},**{'truth_'+key:value for key,value in batch_truth.items()},
        **{'initial_'+key:value for key,value in initial.items()},**{'diagnostic_'+key:value for key,value in control.items()}}
    stream=io.BytesIO();np.savez_compressed(stream,**capture);capture_ref=publish(ROOT/'CAPTURE.npz',stream.getvalue());stream.close()
    diagnostic_ref=publish(ROOT/'DIAGNOSTIC_COUNTERFACTUAL.json',diagnostic['diagnostic'])
    score=native_bar_direction_score_sums(model_output={key:torch.from_numpy(value)for key,value in initial.items()},
        truth=batch_truth,fold_upper_ns=meta['fold_upper_ns'],target_contract_sha256=meta['target_contract']['contract_sha256'])
    return {'status':'GENUINE_JOINT_TECHNICAL_ONLY','fit_population':population,
        'normalization_population':normalization['fit_population'],'normalization':normalization_ref,
        'input_binding':input_binding,'input_support':provider.support_report(rows),
        'base_issue_coverage':provider.support_report(np.flatnonzero((provider.issue_ns>=meta['ta_plan']['fit_start_ns'])
            &(provider.issue_ns<meta['fold_upper_ns']))),
        'source_issue_rows':population['window_issue_rows'],'fit_issue_rows':len(rows),'panel':panel_report,
        'fourteen_forward_surfaces':normalized,'fourteen_counterfactual_surfaces':control_surfaces,'gradient_witness':witness,'ten_initial_heads':head_states,
        'checkpoint':checkpoint,'capture':capture_ref,'diagnostic':diagnostic_ref,'control_primitive':choice,
        'control_max_abs_logit_change':{key:float(np.max(np.abs(control[key]-initial[key])))for key in initial},
        'control_logit_change_l2':{key:float(np.linalg.norm(control[key].astype(np.float64)-initial[key]))for key in initial},
        'initial_score_sums':score,'initial_state_sha256':before,'strict_load_state_sha256':canonical_model_state_sha256(model.state_dict()),
        'model_constructions':1,'model_forwards':forward_count[0],'autograd_calls':1,'optimizer_steps':0,
        'old_statistics_or_weights_loaded':False,'predictive_quality_measured':False,'economic_edge_proven':False}


def run(plan_sha):
    began=time.monotonic();common.firewall_state()
    cap=common.require_capped_cpu_research_producer_execution();lock=common.require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==12*1024**3 and cap['swap']==512*1024**2
        and cap['numerical_threads']==1 and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):]
        and signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'CAP_LIMITS')
    plan,plan_ref,policy_ref=inspect(plan_sha,admitted=True);ROOT.mkdir(parents=True,exist_ok=False)
    claim=publish(ROOT/'CLAIM.json',{'stage':STAGE,'status':'CLAIMED_ONE_USE','plan':plan_ref,
        'policy_at_claim':policy_ref,'max_invocations':1,'remaining_invocations':0,'test_data_used':False})
    previous=signal.getsignal(signal.SIGALRM);result=None
    def expired(*args):raise TimeoutError('NATIVE_M5_JOINT_INTERNAL_LIMIT')
    try:
        remaining=plan['execution']['internal_seconds']-(time.monotonic()-began)
        need(remaining>0,'TIME_BEFORE_PHYSICAL_INPUTS')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
        report=qualify(plan);checked_inputs(plan)
        need(source_clean(plan)and ref(POLICY)==policy_ref and ref(PLAN)==plan_ref,'SOURCE_POLICY_BOOKEND')
        result=publish(ROOT/'RESULT.json',{'stage':STAGE,'status':'COMPLETE_GENUINE_JOINT_TECHNICAL_QUALIFICATION',
            'plan':plan_ref,'claim':claim,'report':report,'source_unchanged':True,'model_forwards':3,'autograd_calls':1,
            'optimizer_steps':0,'training_authorized':False,'test_data_used':False,'economic_edge_proven':False,
            'clean_process_io_firewall':common.firewall_state(),'elapsed_seconds':time.monotonic()-began})
        terminal=publish(ROOT/'TERMINAL.json',{'stage':STAGE,'status':'COMPLETE','exit_code':0,'plan':plan_ref,
            'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        return {'result':result,'terminal':terminal}
    except BaseException as error:
        publish(ROOT/'TERMINAL.json',{'stage':STAGE,'status':'FAILED_CONSUMED','exit_code':1,'plan':plan_ref,
            'claim':claim,'result':result,'error':type(error).__name__+':'+str(error)[:2000],
            'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--out-dir',required=True);parser.add_argument('--preflight',action='store_true');parser.add_argument('--device',choices=('cpu','cuda'))
    args=parser.parse_args();common.firewall_state()
    if Path(args.plan).name.startswith(('NATIVE_M5_TA_ATR03_FIVE_YEAR_WINDOW_','NATIVE_M5_TA_ATR03_FIVE_YEAR_CONTINUE_')):
        return checkpoint_window_cli(args)
    if args.plan in(str(LARGER_PREPARE_PLAN),str(FIVE_YEAR_PREPARE_PLAN)):return larger_prepare_cli(args)
    if args.plan==str(THROUGHPUT_PLAN):return throughput_cli(args)
    if args.plan==str(NOISE_PREPARE_PLAN):return noise_prepare_cli(args)
    if args.plan==str(SHARED_GRADIENT_PLAN):return shared_gradient_cli(args)
    if args.plan!=str(PLAN):return fit_cli(args)
    need(args.device in(None,'cpu'),'QUALIFICATION_CPU_ONLY')
    need(args.plan==str(PLAN)and args.out_dir==str(ROOT),'CLI_SCOPE')
    if args.preflight:
        _,plan_ref,policy_ref=inspect(args.plan_sha256,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_COMPLETED_PARENT_METADATA_ONLY','plan':plan_ref,'policy':policy_ref,
            'claim_created':False,'raw_values_decoded':0,'feature_values_decoded':0,'model_forwards':0,'autograd_calls':0}))
    else:print(json.dumps(run(args.plan_sha256),sort_keys=True))


# The consumed joint qualification remains a separate, immutable one-use scope.
FIT_START_NS=1714521600000000000
FIT_END_NS=1725148800000000000
CONTROL_END_NS=1727740800000000000
FIT_SCOPES={
    'mechanics':('native_m5_full_gc_fit_mechanics_20261119_002',
        REPO/'configs/research/NATIVE_M5_FULL_GC_FIT_MECHANICS_20261119_002.json',
        Path('/home/andre2/GX1_RUNS/NATIVE_M5_FULL_GC_FIT_MECHANICS_20261119_002')),
    'main':('native_m5_full_gc_short_fit_20261119_002',
        REPO/'configs/research/NATIVE_M5_FULL_GC_SHORT_FIT_20261119_002.json',
        Path('/home/andre2/GX1_RUNS/NATIVE_M5_FULL_GC_SHORT_FIT_20261119_002')),
    'ta_mechanics':('native_m5_ta_prior_mechanics_20261119_001',
        REPO/'configs/research/NATIVE_M5_TA_PRIOR_MECHANICS_20261119_001.json',
        Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_PRIOR_MECHANICS_20261119_001')),
    'ta_main':('native_m5_ta_prior_learning_curve_20261119_001',
        REPO/'configs/research/NATIVE_M5_TA_PRIOR_LEARNING_CURVE_20261119_001.json',
        Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_PRIOR_LEARNING_CURVE_20261119_001')),
    'ta_shuffle':('native_m5_ta_prior_block_order_20261119_001',
        REPO/'configs/research/NATIVE_M5_TA_PRIOR_BLOCK_ORDER_20261119_001.json',
        Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_PRIOR_BLOCK_ORDER_20261119_001')),
    'ta_primary_mechanics':('native_m5_ta_primary_backbone_mechanics_20261119_001',
        REPO/'configs/research/NATIVE_M5_TA_PRIMARY_BACKBONE_MECHANICS_20261119_001.json',
        Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_PRIMARY_BACKBONE_MECHANICS_20261119_001')),
    'ta_primary_main':('native_m5_ta_primary_backbone_fit_20261119_001',
        REPO/'configs/research/NATIVE_M5_TA_PRIMARY_BACKBONE_FIT_20261119_001.json',
        Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_PRIMARY_BACKBONE_FIT_20261119_001'))}
ADAMW_RECIPE={'lr':1e-4,'betas':[.9,.999],'eps':1e-8,'weight_decay':1e-4,
    'amsgrad':False,'foreach':None,'maximize':False,'capturable':False,
    'differentiable':False,'fused':None}
FIT_SOURCE_PATHS=tuple(dict.fromkeys((*SOURCE_PATHS,
    'gx1/contracts/market_state_ta_fold2_m5_fit_v1.py',
    'tests/test_gx1_capped_run_contract.py','tests/test_market_state_ta_integrated_objective_v1.py','tests/test_entry_model_native_input_normalization.py')))


TA_MATCHED_PRIOR_REFERENCE={
    'path':'/home/andre2/GX1_RUNS/NATIVE_M5_FULL_GC_SHORT_FIT_20261119_002/MATCHED_FIT_PRIOR.json',
    'sha256':'122f285e0eeddf66158079c3c018679d6b1d580976cd09311c3c4bf5fa05c691'}
TA_BLOCK_ORDER_REFERENCE={
    'chronological_result':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_TA_PRIOR_LEARNING_CURVE_20261119_001/RESULT.json',
        'sha256':'3bf639efeb58b5526600b2c3478f1139eb7aa52704ca8d33a97ea6272c241d1d'},
    'chronological_terminal':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_TA_PRIOR_LEARNING_CURVE_20261119_001/TERMINAL.json',
        'sha256':'3b4a0c0b3c5883662f99d9c3c49728852951e3f6cf7fa2bb7fb5d2cc21f9b728'},
    'chronological_review':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_TA_PRIOR_LEARNING_CURVE_20261119_001/MATCHED_RESULT_REVIEW.json',
        'sha256':'98b9d778887b1dff1a687a75cdd4a009075bb6a774105a82add34c16231ab3b4'}}
TA_FIT_SOURCE_PATHS=tuple(dict.fromkeys((*ta_audit.SOURCE_PATHS,
    'gx1/audit/qualify_native_m5_full_gc_direction_v1.py',
    'tests/test_qualify_native_m5_full_gc_direction_v1.py',
    'gx1/contracts/market_state_ta_fold2_m5_fit_v1.py',
    'tests/test_gx1_capped_run_contract.py')))


def _ta_fit(plan):
    return plan.get('mode') in ('ta_mechanics','ta_main','ta_shuffle','ta_primary_mechanics','ta_primary_main','ta_noise_mechanics','ta_noise_main','ta_larger_mechanics','ta_larger_main','ta_five_year_mechanics','ta_five_year_main')


def _ta_main(mode):
    return mode in ('ta_main','ta_shuffle','ta_primary_main','ta_noise_main','ta_larger_main','ta_five_year_main')


def _mechanics(mode):
    return mode in ('mechanics','ta_mechanics','ta_primary_mechanics','ta_noise_mechanics','ta_larger_mechanics','ta_five_year_mechanics')


def _ta_pool_axis(ta_plan,*,return_source_rows=False):
    # Selection reads only two admitted clock/index arrays, never features or labels.
    snapshot=read(ta_plan['inputs']['candidate_snapshot'])
    archive_ref=ta_plan['inputs']['candidate_archive']
    need(ref(archive_ref['path'])==archive_ref,'TA_CLOCK_ARCHIVE')
    with np.load(archive_ref['path'],allow_pickle=False)as archive:
        arrays={name:archive[name]for name in ('bar_start_ns','retained_issue_rows')}
    for name,value in arrays.items():
        header=snapshot['arrays'][name]
        need(value.dtype==np.int64 and value.ndim==1 and value.dtype.str==header['dtype']
            and list(value.shape)==header['shape']and digest(value)==header['sha256'],'TA_CLOCK_BYTES:'+name)
    starts=arrays['bar_start_ns'];rows=arrays['retained_issue_rows']
    need(len(rows)>0 and rows[0]>=0 and rows[-1]<len(starts)and np.all(np.diff(rows)>0)
        and np.all(np.diff(starts)>0),'TA_CLOCK_AXIS')
    issues=starts[rows]+300_000_000_000;axis={};sources={}
    for name,start,end in [('TRAIN',FIT_START_NS,FIT_END_NS),('CONTROL',FIT_END_NS,CONTROL_END_NS)]:
        chosen=(issues>=start)&(issues<end);source=rows[chosen];times=issues[chosen]
        need(len(source)>0,'TA_CLOCK_POPULATION:'+name)
        sources[name]=source
        axis[name]={'input_ready_rows':len(source),'input_not_ready_rows':0,
            'source_rows_sha256':digest(source),'issue_ns_sha256':digest(times)}
    need(ref(archive_ref['path'])==archive_ref,'TA_CLOCK_ARCHIVE_BOOKEND')
    return (axis,sources)if return_source_rows else axis


def ta_fit_metadata(inputs,mode):
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_train_prior_contract
    fields={'ta_result','ta_terminal','matched_prior_reference'}|(
        {'mechanics_result','mechanics_terminal'}if _ta_main(mode)else set())
    if mode=='ta_shuffle':fields.update(TA_BLOCK_ORDER_REFERENCE)
    need(type(inputs)is dict and set(inputs)==fields,'TA_FIT_INPUT_ROLES')
    need(inputs['ta_result']=={'path':'/home/andre2/GX1_RUNS/NATIVE_M5_PIPELINE_20261119_006/RESULT.json',
        'sha256':'5c015b3af8a17c8a16459be0eeba5ca017b7625db91e193aa84b101e664e7bca'}
        and inputs['ta_terminal']=={'path':'/home/andre2/GX1_RUNS/NATIVE_M5_PIPELINE_20261119_006/TERMINAL.json',
        'sha256':'849bcda1483e088ca5fa2ec8671e8ed6b003654331a6556c70af53dcfc11c874'},'TA_FIT_EXACT_PARENT')
    need(inputs['matched_prior_reference']==TA_MATCHED_PRIOR_REFERENCE,'TA_FIT_EXACT_MATCHED_PRIOR')
    if mode=='ta_shuffle':
        need(all(inputs[key]==value for key,value in TA_BLOCK_ORDER_REFERENCE.items()),'TA_BLOCK_EXACT_CHRONOLOGICAL_REFERENCE')
    result,ta_plan,plan_ref,claim_ref=complete_parent(inputs['ta_result'],inputs['ta_terminal'],
        'COMPLETE_NATIVE_M5_PIPELINE_TECHNICAL_QUALIFICATION','native_m5_pipeline_20261119_006')
    need(result['model_forwards']==1 and result['clean_process_io_firewall']['denied_attempts']==0,
        'TA_FIT_CLEAN_PARENT')
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    reference=objective.require_native_bar_train_prior(read(inputs['matched_prior_reference']))
    need(reference['fit_population']['issue_rows']==1536
        and reference['fit_population']['cutoff_ns']==FIT_END_NS,'TA_FIT_MATCHED_PRIOR_SCOPE')
    dependencies={'ta_plan':plan_ref,'ta_claim':claim_ref,'ta_support':result['report']['feature_support_artifact'],
        'ta_admission':result['report']['feature_inputs_requalification'],
        'ta_candidate_snapshot':ta_plan['inputs']['candidate_snapshot'],
        'ta_candidate_archive':ta_plan['inputs']['candidate_archive'],'target_tape':ta_plan['inputs']['target_tape']}
    readiness=result['report']['input_readiness']
    need(readiness['rows']==readiness['input_ready']==29964
        and readiness['input_not_ready']==0,'TA_FIT_QUALIFIED_NO_ROW_FILTER')
    axis=_ta_pool_axis(ta_plan)
    need(axis['TRAIN']['input_ready_rows']==24175 and axis['CONTROL']['input_ready_rows']==5789,'TA_FIT_FIXED_POOLS')
    metadata={'dependencies':dependencies,'ta_plan':ta_plan,'architecture':ta_plan['architecture'],
        'seed':ta_plan['seed'],'target_contract':ta_plan['target_contract'],'fit_start_ns':FIT_START_NS,
        'fit_end_ns':FIT_END_NS,'control_end_ns':CONTROL_END_NS,'prior_contract':native_bar_train_prior_contract(),
        'source_pool':{'result':inputs['ta_result'],'axis':axis},
        'input_view':'native_M5_full_TA_9_surfaces_GC_deferred',
        'normalization_policy':'fresh_whole_earlier_TRAIN_consumption_union_before_source_rank_sampling',
        'initialization':'fresh_backbone_zero_final_head_weights_TRAIN_Laplace1_log_biases'}
    if _ta_main(mode):
        stage,_,root=FIT_SCOPES['ta_mechanics']
        need(inputs['mechanics_result']['path']==str(root/'RESULT.json')
            and inputs['mechanics_terminal']['path']==str(root/'TERMINAL.json'),'TA_FIT_MECHANICS_PATH')
        done,terminal=read(inputs['mechanics_result']),read(inputs['mechanics_terminal'])
        parent,claim=read(done['plan']),read(done['claim'])
        need(done['status']=='COMPLETE_NATIVE_M5_TA_PRIOR_MECHANICS'
            and done['stage']==terminal['stage']==parent['stage']==claim['stage']==stage
            and terminal['status']=='COMPLETE'and terminal['exit_code']==0
            and terminal['result']==inputs['mechanics_result']and terminal['plan']==done['plan']==claim['plan']
            and terminal['claim']==done['claim']and done['source_unchanged']is True
            and done['test_data_used']is terminal['test_data_used']is False
            and done['report']['cold_resume']['bitexact']is True
            and done['report']['prior_initialization']['all_selected_initial_logits_equal_prior']is True
            and done['report']['gradient_transition']['first_backbone_gradient_l2']==0
            and done['report']['gradient_transition']['second_backbone_gradient_l2']>0
            and done['optimizer_steps']==3 and done['report']['normalization_fits']==done['report']['prior_fits']==1
            and done['clean_process_io_firewall']['denied_attempts']==0,
            'TA_FIT_COMPLETED_MECHANICS')
        need(parent['mode']=='ta_mechanics'and parent['recipe']==fit_recipe(parent['recipe'],'ta_mechanics')
            and parent['cohort_selection']==small_cohort_manifest(metadata)
            and parent['metadata']==metadata and all(parent['inputs'][key]==inputs[key]
            for key in ('ta_result','ta_terminal','matched_prior_reference')),'TA_FIT_MECHANICS_DOMAINS')
        # No numerically relevant source change is allowed between these two plans.
        compatibility=ta_block_order_source_bridge(parent)if mode=='ta_shuffle'else None
        if compatibility is None:
            need(set(parent['source_owners'])==set(TA_FIT_SOURCE_PATHS)
                and all(sha(REPO/name)==value for name,value in parent['source_owners'].items()),'TA_FIT_MECHANICS_SOURCE')
        report=done['report']
        metadata['mechanics']={'result':inputs['mechanics_result'],'terminal':inputs['mechanics_terminal'],
            'plan':done['plan'],'claim':done['claim'],'normalization':report['normalization'],'prior':report['prior'],
            'axis':report['axis'],'normalization_population':report['normalization_population'],
            'initial_state_sha256':report['initial_state_sha256'],
            'physical_windows_boot_id':done['physical_windows_boot_id'],
            **({'source_compatibility':compatibility}if compatibility is not None else{})}
        dependencies.update({'mechanics_plan':done['plan'],'mechanics_claim':done['claim'],
            'train_normalization':report['normalization'],'train_prior':report['prior']})
    if mode=='ta_shuffle':metadata['chronological_reference']=ta_chronological_reference(inputs,metadata)
    return metadata


def ta_chronological_reference(inputs,metadata):
    result,terminal,review=(read(inputs[key])for key in TA_BLOCK_ORDER_REFERENCE)
    parent,claim=read(result['plan']),read(result['claim']);report=result['report']
    initial='d30d41feb2d8c37ba624785b6504cd58ecf4530d06d785191219ecdc9a36712d'
    need(result['status']=='COMPLETE_NATIVE_M5_TA_PRIOR_LEARNING_CURVE'
        and result['stage']==terminal['stage']==parent['stage']==claim['stage']==FIT_SCOPES['ta_main'][0]
        and terminal['status']=='COMPLETE'and terminal['exit_code']==0
        and terminal['result']==inputs['chronological_result']and terminal['plan']==result['plan']==claim['plan']
        and terminal['claim']==result['claim']and result['source_unchanged']is True
        and result['test_data_used']is terminal['test_data_used']is False
        and result['optimizer_steps']==report['canonical_optimizer_updates']==384
        and report['declared_passes']==4 and report['actual_counts']=={
            'model_constructions':1,'model_forwards':544,'backward_calls':384,'optimizer_steps':384}
        and report['normalization_fits']==report['prior_fits']==0
        and report['qualification_weights_loaded']is False
        and result['clean_process_io_firewall']['denied_attempts']==0,'TA_BLOCK_COMPLETE_CHRONOLOGICAL')
    mechanics=metadata['mechanics']
    need(parent['mode']=='ta_main'and parent['recipe']==fit_recipe(parent['recipe'],'ta_main')
        and parent['recipe']['passes']==4 and parent['cohort_selection']==small_cohort_manifest(metadata)
        and report['normalization']==mechanics['normalization']and report['prior']==mechanics['prior']
        and report['axis']==mechanics['axis']and report['normalization_population']==mechanics['normalization_population']
        and report['initial_state_sha256']==mechanics['initial_state_sha256']==initial
        and all(parent['metadata'][key]==metadata[key]for key in (
            'architecture','seed','target_contract','fit_start_ns','fit_end_ns','control_end_ns','prior_contract','source_pool'))
        and all(parent['inputs'][key]==inputs[key]for key in (
            'ta_result','ta_terminal','matched_prior_reference','mechanics_result','mechanics_terminal')),
        'TA_BLOCK_MATCHED_CHRONOLOGICAL_DOMAINS')
    for name in ('FIT','CONTROL'):
        item=report['selected_cohorts'][name];expected=small_cohort_manifest(metadata)['cohorts'][name]
        need(item['rows']==expected['rows']and item['frozen_selection_identity_sha256']==expected['selection_identity_sha256']
            and item['first_and_last_pool_rows_included']is True,'TA_BLOCK_MATCHED_COHORT:'+name)
    need(review['schema_version']=='gx1_native_m5_ta_prior_fourpass_result_review_v1'
        and review['result']==inputs['chronological_result']and review['terminal']==inputs['chronological_terminal']
        and review['plan']==result['plan']and review['initial_state_sha256']==initial
        and review['fixed_passes']==4 and review['actual_counts']==report['actual_counts']
        and review['normalization']==report['normalization']and review['prior']==report['prior']
        and review['selected_cohorts']==report['selected_cohorts']and review['no_best_checkpoint_selection']is True
        and review['GC_used']is review['test_data_used']is False and review['M1_attempts']==0,'TA_BLOCK_CHRONOLOGICAL_REVIEW')
    metadata['dependencies'].update({'chronological_plan':result['plan'],'chronological_claim':result['claim']})
    return {**{key:inputs[key]for key in TA_BLOCK_ORDER_REFERENCE},'plan':result['plan'],'claim':result['claim'],
        'normalization':report['normalization'],'prior':report['prior'],'selected_cohorts':report['selected_cohorts'],
        'initial_state_sha256':initial,'reference_weights_loaded':False}


def fit_recipe(value,mode):
    if mode in LARGER_FIT_MODES:
        need(False,'LARGER_RECIPE_REQUIRES_BOUND_METADATA')
    fields={'optimizer','gradient_clip_norm','scheduler','seed','batch_rows','score_batch_rows','fit_batches',
        'score_batches_per_split','checkpoint_every_updates','outer_seconds','internal_seconds',
        'required_windows_boot_id_greater_than'}
    if mode in ('ta_mechanics','ta_main','ta_shuffle','ta_primary_mechanics','ta_primary_main'):fields.add('passes')
    need(type(value)is dict and set(value)==fields,'FIT_RECIPE_FIELDS')
    need(value['optimizer']==ADAMW_RECIPE and value['gradient_clip_norm']==1.
        and value['scheduler']is None and value['seed']==20260911 and value['batch_rows']==16
        and type(value['score_batch_rows'])is int and value['score_batch_rows']==64
        and type(value['required_windows_boot_id_greater_than'])is int
        and value['required_windows_boot_id_greater_than']==666,'FIT_NUMERIC_RECIPE')
    for key in ('fit_batches','checkpoint_every_updates','outer_seconds','internal_seconds'):
        need(type(value[key])is int and value[key]>0,'FIT_INTEGER:'+key)
    need(value['checkpoint_every_updates']<=value['fit_batches']
        and value['internal_seconds']==value['outer_seconds']-30 and value['outer_seconds']<=10800,'FIT_BUDGET')
    need(mode in FIT_SCOPES and(not _mechanics(mode)or value['fit_batches']==2),'FIT_MODE')
    if mode in ('ta_mechanics','ta_main','ta_shuffle','ta_primary_mechanics','ta_primary_main'):
        need(type(value['passes'])is int and value['passes']>0
            and value['fit_batches']==value['passes']*(1 if mode in('ta_mechanics','ta_primary_mechanics','ta_noise_mechanics')else 96)
            and(mode not in('ta_mechanics','ta_primary_mechanics','ta_noise_mechanics')or value['passes']==2)
            and value['checkpoint_every_updates']==(1 if mode in('ta_mechanics','ta_primary_mechanics','ta_noise_mechanics')else 96)
            and value['score_batches_per_split']is None
            and value['outer_seconds']==1800 and value['internal_seconds']==1770,'TA_FIT_FIXED_RECIPE')
        if mode in('ta_shuffle','ta_primary_main','ta_noise_main'):need(value['passes']==4 and value['fit_batches']==384,'TA_BLOCK_FIXED_FOUR_PASSES')
        return dict(value)
    if mode=='main':
        need(value['fit_batches']==96 and value['checkpoint_every_updates']==32
            and value['outer_seconds']==1800 and value['internal_seconds']==1770,'FIT_SMALL_FIXED_BUDGET')
    need((mode=='main'and value['score_batches_per_split']is None)or
        (mode=='mechanics'and type(value['score_batches_per_split'])is int and value['score_batches_per_split']>0),
        'FIT_SCORE_SCOPE')
    return dict(value)


def fit_metadata(inputs,mode):
    if mode in('ta_primary_mechanics','ta_primary_main'):return primary_backbone_metadata(inputs,mode)
    if mode in ('ta_mechanics','ta_main','ta_shuffle','ta_primary_mechanics','ta_primary_main'):return ta_fit_metadata(inputs,mode)
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    fields={'joint_result','joint_terminal'}|({'mechanics_result','mechanics_terminal'}if mode=='main'else {'stats_result','stats_terminal'})
    need(type(inputs)is dict and set(inputs)==fields,'FIT_INPUT_ROLES')
    need(inputs['joint_result']=={'path':str(ROOT/'RESULT.json'),
        'sha256':'55fd0cdb2ff308984b72764abc090c5751f54aada7831583b898d9909f698aff'}
        and inputs['joint_terminal']=={'path':str(ROOT/'TERMINAL.json'),
        'sha256':'9ee1354e7ed2c8e4a97509de9560541a6ec47f16cd5b5d6284704af654016696'},'FIT_GENUINE_JOINT_PARENT')
    result,parent,parent_ref,claim_ref=complete_parent(inputs['joint_result'],inputs['joint_terminal'],
        'COMPLETE_GENUINE_JOINT_TECHNICAL_QUALIFICATION',STAGE)
    metadata=input_metadata(parent['inputs'])
    need(metadata==parent['metadata'] and result['report']['model_forwards']==3
        and result['clean_process_io_firewall']['denied_attempts']==0,'FIT_QUALIFIED_INPUTS')
    dependencies=dict(metadata['dependencies'])
    dependencies.update({'joint_plan':parent_ref,'joint_claim':claim_ref,**parent['inputs']})
    metadata={**metadata,'dependencies':dependencies,'fit_start_ns':FIT_START_NS,
        'fit_end_ns':FIT_END_NS,'control_end_ns':CONTROL_END_NS,
        'prior_contract':objective.native_bar_train_prior_contract()}
    parent_role='mechanics'if mode=='main'else'statistics_parent'
    parent=_fit_statistics_parent(inputs,metadata,mode)
    metadata[parent_role]=parent
    dependencies.update({parent_role+'_plan':parent['plan'],parent_role+'_claim':parent['claim'],
        'train_normalization':parent['normalization'],'train_prior':parent['prior']})
    return metadata


def _fit_statistics_parent(inputs,metadata,mode):
    role='mechanics'if mode=='main'else'stats'
    result_ref,terminal_ref=inputs[role+'_result'],inputs[role+'_terminal']
    if mode=='mechanics':
        root=Path('/home/andre2/GX1_RUNS/NATIVE_M5_FULL_GC_FIT_MECHANICS_20261119_001')
        need(result_ref=={'path':str(root/'RESULT.json'),
            'sha256':'1a5a3a0ec76400d552df59d2346154bf7442adfa7dba61154a54af6d2d6ebafe'}
            and terminal_ref=={'path':str(root/'TERMINAL.json'),
            'sha256':'24454ce94896ecc4ccbcf9331a5cba499bf637ef59b43b4ca99479fd5dd533bc'},'FIT_EXACT_STATISTICS_PARENT')
        expected_stage='native_m5_full_gc_fit_mechanics_20261119_001'
    else:
        expected_stage,_,root=FIT_SCOPES['mechanics']
        need(result_ref['path']==str(root/'RESULT.json')and terminal_ref['path']==str(root/'TERMINAL.json'),
            'FIT_EXACT_MECHANICS_PARENT')
    done,terminal=read(result_ref),read(terminal_ref)
    parent_plan,parent_claim=read(done['plan']),read(done['claim'])
    need(done['status']=='COMPLETE_NATIVE_M5_FULL_GC_FIT_MECHANICS'
        and done['stage']==terminal['stage']==parent_plan['stage']==parent_claim['stage']==expected_stage
        and terminal['status']=='COMPLETE'and terminal['exit_code']==0
        and terminal['result']==result_ref and terminal['plan']==done['plan']==parent_claim['plan']
        and terminal['claim']==done['claim']and done['source_unchanged']is True
        and done['test_data_used']is terminal['test_data_used']is False
        and done['report']['cold_resume']['bitexact']is True,'FIT_COMPLETED_MECHANICS')
    need(parent_plan['metadata']['fit_start_ns']==FIT_START_NS
        and parent_plan['metadata']['fit_end_ns']==FIT_END_NS
        and parent_plan['metadata']['control_end_ns']==CONTROL_END_NS
        and parent_plan['metadata']['native_gc_source_binding']==metadata['native_gc_source_binding']
        and parent_plan['metadata']['target_contract']==metadata['target_contract']
        and parent_plan['metadata']['prior_contract']==metadata['prior_contract']
        and parent_plan['metadata']['architecture']==metadata['architecture']
        and all(parent_plan['inputs'][key]==inputs[key]for key in ('joint_result','joint_terminal')),'FIT_MECHANICS_DOMAINS')
    if mode=='main':
        compatibility=fit_orchestration_source_bridge(parent_plan)
    report=done['report']
    return {'role':'frozen_TRAIN_statistics_only_no_weights','result':result_ref,'terminal':terminal_ref,
        'plan':done['plan'],'claim':done['claim'],
        'normalization':report['normalization'],'prior':report['prior'],
        'axis':report['axis'],'normalization_population':report['normalization_population'],
        'initial_state_sha256':report['initial_state_sha256'],
        'physical_windows_boot_id':done['physical_windows_boot_id'],
        'source_base_commit':parent_plan['source_base_commit'],
        **({'source_compatibility':compatibility}if mode=='main'else{})}


def ta_block_order_numeric_identity(source):
    import ast
    tree=ast.parse(source)
    leaves=('fresh_fit_model','fit_step','fit_score','prepare_ta_native_fit','fit_axes',
        'seed_native_fit','training_rng','restore_training_rng','state_sha','slice_truth',
        'fit_checkpoint','publish_checkpoint','restore_fit_checkpoint','fit_state_proof',
        '_ta_initial_prior_proof','cold_resume_fit','require_cold_resume_proof')
    functions={node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    need(set(leaves)<=set(functions),'TA_BLOCK_NUMERICAL_FUNCTIONS')
    sha_ast=lambda node:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
    unchanged={name:sha_ast(functions[name])for name in leaves}
    constants={target.id:node for node in tree.body if isinstance(node,ast.Assign)
        for target in node.targets if isinstance(target,ast.Name)}
    unchanged['ADAMW_RECIPE']=sha_ast(constants['ADAMW_RECIPE'])
    # Only named scope/selection/schedule/launch orchestration may differ.
    orchestration={'_ta_fit','_ta_main','_ta_pool_axis','ta_fit_metadata','ta_chronological_reference',
        'ta_block_order_numeric_identity','ta_block_order_source_bridge','ta_block_schedule_from_rows','ta_block_schedule',
        'fit_recipe','fit_metadata','fit_limits','build_fit_plan','fit_boot_state',
        '_ta_training_batches','_ta_schedule_position','prepare_native_fit','fit_physical','run_native_fit'}
    tree.body=[node for node in tree.body if not(
        isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))and node.name in orchestration)and not(
        isinstance(node,ast.Assign)and any(isinstance(target,ast.Name)and target.id in (
            'FIT_SCOPES','TA_BLOCK_ORDER_REFERENCE')for target in node.targets))]
    return {'remaining_module_ast_sha256':sha_ast(tree),'unchanged_numeric_leaf_ast_sha256':unchanged}


def ta_block_order_source_bridge(parent_plan):
    owner='gx1/audit/qualify_native_m5_full_gc_direction_v1.py'
    allowed={owner,'tests/test_qualify_native_m5_full_gc_direction_v1.py',
        'scripts/gx1_capped_run.sh','tests/test_gx1_capped_run_contract.py'}
    need(parent_plan['mode']=='ta_mechanics'and set(parent_plan['source_owners'])==set(TA_FIT_SOURCE_PATHS),
        'TA_BLOCK_MECHANICS_SOURCE_SET')
    current={name:sha(REPO/name)for name in TA_FIT_SOURCE_PATHS}
    need(all(current[name]==value for name,value in parent_plan['source_owners'].items()if name not in allowed),
        'TA_BLOCK_UNCHANGED_MODEL_LOSS_STATS_OWNERS')
    original=subprocess.check_output(['git','show',parent_plan['source_base_commit']+':'+owner],cwd=REPO)
    need(hashlib.sha256(original).hexdigest()==parent_plan['source_owners'][owner],'TA_BLOCK_ORIGINAL_DRIVER_BLOB')
    numerical=ta_block_order_numeric_identity(original.decode())
    need(numerical==ta_block_order_numeric_identity((REPO/owner).read_text()),'TA_BLOCK_UNCHANGED_NUMERICAL_AST')
    return {'schema_version':'gx1_native_m5_ta_block_order_source_bridge_v1',
        'original_mechanics_source_base_commit':parent_plan['source_base_commit'],
        'allowed_changed_source_paths':sorted(allowed),'all_other_source_owners_exact':True,**numerical,
        'source_changes':{name:{'original_sha256':parent_plan['source_owners'][name],'current_sha256':current[name]}
            for name in sorted(allowed)if current[name]!=parent_plan['source_owners'][name]},
        'scope':'only_ta_shuffle_block_order001_same_numeric_model_loss_optimizer_statistics_score_reload',
        'guard_change_scope':'one_additional_exact_plan_output_path_no_resource_sensor_cgroup_change'}


def ta_block_schedule_from_rows(rows,recipe,selection_sha):
    rows=np.asarray(rows)
    need(rows.dtype==np.int64 and rows.shape==(1536,)and np.all(np.diff(rows)>0)
        and recipe['passes']==4 and recipe['batch_rows']==16 and recipe['seed']==20260911,'TA_BLOCK_SCHEDULE_POPULATION')
    generator=np.random.Generator(np.random.PCG64(recipe['seed']))
    orders=np.stack([generator.permutation(96)for _ in range(4)]).astype(np.int64)
    blocks=rows.reshape(96,16);ordered=blocks[orders]
    body={'schema_version':'gx1_native_m5_ta_whole_block_order_schedule_v1',
        'rng':'dedicated_local_numpy_Generator_PCG64_no_global_RNG_state',
        'seed':recipe['seed'],'passes':4,'batches_per_pass':96,'rows_per_batch':16,
        'selection_identity_sha256':selection_sha,'ordered_FIT_source_rows_sha256':digest(rows),
        'base_block_source_rows_sha256':[digest(block)for block in blocks],
        'block_orders':orders.tolist(),'block_orders_sha256':digest(orders),
        'per_pass_source_rows_sha256':[digest(value)for value in ordered],
        'source_row_schedule_sha256':digest(ordered),
        'window_interior_permuted':False,'labels_or_CONTROL_used_for_order':False}
    body['contract_sha256']=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
    return body


def ta_block_schedule(metadata,recipe):
    axis,pools=_ta_pool_axis(metadata['ta_plan'],return_source_rows=True)
    need(axis==metadata['source_pool']['axis'],'TA_BLOCK_SCHEDULE_SOURCE_AXIS')
    ranks=spread_rank_blocks(len(pools['TRAIN']),96,16);rows=pools['TRAIN'][ranks]
    expected=metadata['chronological_reference']['selected_cohorts']['FIT']
    control=pools['CONTROL'][spread_rank_blocks(len(pools['CONTROL']),8,64)]
    need(digest(rows)==expected['source_rows_sha256']
        and digest(control)==metadata['chronological_reference']['selected_cohorts']['CONTROL']['source_rows_sha256'],
        'TA_BLOCK_CHRONOLOGICAL_SOURCE_ROWS')
    return ta_block_schedule_from_rows(rows,recipe,small_cohort_manifest(metadata)['cohorts']['FIT']['selection_identity_sha256'])


def fit_orchestration_source_bridge(parent_plan):
    import ast
    owner='gx1/audit/qualify_native_m5_full_gc_direction_v1.py'
    allowed={owner,'tests/test_qualify_native_m5_full_gc_direction_v1.py',
        'scripts/gx1_capped_run.sh','tests/test_gx1_capped_run_contract.py'}
    need(set(parent_plan['source_owners'])==set(FIT_SOURCE_PATHS),'FIT_MECHANICS_SOURCE_SET')
    current={name:sha(REPO/name)for name in FIT_SOURCE_PATHS}
    need(all(current[name]==value for name,value in parent_plan['source_owners'].items()
        if name not in allowed),'FIT_MECHANICS_NUMERICAL_SOURCE')
    original=subprocess.check_output(['git','show',parent_plan['source_base_commit']+':'+owner],cwd=REPO)
    need(hashlib.sha256(original).hexdigest()==parent_plan['source_owners'][owner],'FIT_MECHANICS_ORIGINAL_BLOB')
    numerical=fit_orchestration_ast_sha(original.decode())
    need(numerical==fit_orchestration_ast_sha((REPO/owner).read_text()),'FIT_MECHANICS_NUMERICAL_AST')
    return {'schema_version':'gx1_native_m5_small_cohort_source_bridge_v1',
        'original_source_base_commit':parent_plan['source_base_commit'],
        'unchanged_numerical_module_ast_sha256':numerical,'all_other_source_owners_exact':True,
        'allowed_changed_source_paths':sorted(allowed),
        'source_changes':{name:{'original_sha256':parent_plan['source_owners'][name],'current_sha256':current[name]}
            for name in sorted(allowed)if current[name]!=parent_plan['source_owners'][name]},
        'guard_change_scope':'new_exact_002_path_and_1800_deadline_only; execution_sensor_cgroup_owners_unchanged'}


def fit_orchestration_ast_sha(source):
    import ast
    # All imports, math constants, model/optimizer/score, RNG and checkpoint
    # leaves remain byte-equivalent at the AST level. These are only recipe,
    # metadata and the explicitly reviewed cohort/reference orchestration.
    allowed={'fit_recipe','fit_metadata','_fit_statistics_parent','fit_limits','build_fit_plan','fit_physical',
        'fit_orchestration_source_bridge','fit_orchestration_ast_sha','spread_rank_blocks',
        'small_cohort_manifest','selected_small_cohorts','prepare_matched_prior'}
    tree=ast.parse(source)
    tree.body=[node for node in tree.body
        if not(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))and node.name in allowed)
        and not(isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='FIT_SCOPES'for t in node.targets))]
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def spread_rank_blocks(population,blocks,batch):
    need(all(type(value)is int and value>0 for value in(population,blocks,batch))
        and population>=blocks*batch,'FIT_SOURCE_RANK_POPULATION')
    starts=np.linspace(0,population-batch,blocks,dtype=np.int64)
    need(len(np.unique(starts))==blocks and (blocks==1 or np.all(np.diff(starts)>=batch))
        and starts[0]==0 and (blocks==1 or starts[-1]+batch==population),'FIT_SOURCE_RANK_BLOCKS')
    ranks=(starts[:,None]+np.arange(batch,dtype=np.int64)).reshape(-1)
    need(len(ranks)==blocks*batch and np.all(np.diff(ranks)>0),'FIT_SOURCE_RANK_ORDER')
    return ranks


def small_cohort_manifest(metadata):
    authority=metadata.get('source_pool',metadata.get('mechanics'))
    result={'schema_version':'gx1_native_m5_source_rank_cohorts_v1',
        'selection':'linspace_source_ranks_before_outcomes_no_observedness_or_class_filter',
        'pool_axis_authority':authority['result'],'cohorts':{}}
    for name,pool,blocks,batch in [('FIT','TRAIN',96,16),('CONTROL','CONTROL',8,64)]:
        axis=authority['axis'][pool]
        need(axis['input_not_ready_rows']==0,'FIT_UNFILTERED_SOURCE_POOL')
        ranks=spread_rank_blocks(axis['input_ready_rows'],blocks,batch)
        identity={'ordered_source_pool_sha256':axis['source_rows_sha256'],'selected_rank_sha256':digest(ranks)}
        result['cohorts'][name]={'pool':pool,'pool_rows':axis['input_ready_rows'],
            'pool_source_rows_sha256':axis['source_rows_sha256'],'pool_issue_ns_sha256':axis['issue_ns_sha256'],
            'blocks':blocks,'block_rows':batch,'rows':len(ranks),'first_rank':int(ranks[0]),
            'last_rank':int(ranks[-1]),'selected_rank_sha256':digest(ranks),
            'selection_identity_sha256':hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()}
    return result


def selected_small_cohorts(plan,context):
    need(plan['cohort_selection']==small_cohort_manifest(plan['metadata']),'FIT_FROZEN_SOURCE_SELECTION')
    selected={};report={}
    for name,item in plan['cohort_selection']['cohorts'].items():
        pool=context['cohorts'][item['pool']]['rows']
        issues=context['cohorts'][item['pool']]['truth']['issue_ns']
        need(len(pool)==item['pool_rows']and digest(pool)==item['pool_source_rows_sha256']
            and digest(issues)==item['pool_issue_ns_sha256'],'FIT_RUNTIME_SOURCE_POOL')
        ranks=spread_rank_blocks(len(pool),item['blocks'],item['block_rows'])
        need(digest(ranks)==item['selected_rank_sha256'],'FIT_RUNTIME_SOURCE_RANKS')
        rows=pool[ranks];selected[name]=rows
        report[name]={'rows':len(rows),'source_rows_sha256':digest(rows),'issue_ns_sha256':digest(issues[ranks]),
            'first_issue_ns':int(issues[ranks[0]]),'last_issue_ns':int(issues[ranks[-1]]),
            'first_and_last_pool_rows_included':bool(rows[0]==pool[0]and rows[-1]==pool[-1]),
            'frozen_selection_identity_sha256':item['selection_identity_sha256']}
    return selected,report


def prepare_matched_prior(context,fit_rows,plan,root):
    from gx1.contracts.market_state_ta_integrated_objective_v1 import fit_native_bar_train_prior
    prior=fit_native_bar_train_prior(
        truth=slice_truth(context['cohorts']['TRAIN'],fit_rows,fit=True),
        fold_upper_ns=FIT_END_NS,target_contract_sha256=plan['metadata']['target_contract']['contract_sha256'],**noise_target_kwargs(plan))
    context['prior']=prior
    context['prior_ref']=publish(root/'MATCHED_FIT_PRIOR.json',prior)
    return context['prior_ref']


def fit_limits(recipe,mode,metadata=None):
    if mode in LARGER_FIT_MODES:return larger_limits(recipe,mode,metadata)
    if mode in ('ta_mechanics','ta_main','ta_shuffle','ta_primary_mechanics','ta_primary_main'):
        score=sum((item['rows']+recipe['score_batch_rows']-1)//recipe['score_batch_rows']
            for item in small_cohort_manifest(metadata)['cohorts'].values())
        mechanics=mode in('ta_mechanics','ta_primary_mechanics','ta_noise_mechanics')
        updates=3 if mechanics else recipe['fit_batches']
        return {'model_constructions':2 if mechanics else 1,
            'model_forwards':updates+score*(2 if mechanics else recipe['passes']+1),
            'backward_calls':updates,'optimizer_steps':updates,
            **({'autograd_calls':6 if mode=='ta_primary_mechanics'else 0}if mode in('ta_primary_mechanics','ta_primary_main')else{})}
    updates=3 if mode=='mechanics'else recipe['fit_batches']
    return {'model_constructions':2 if mode=='mechanics'else 1,
        'model_forwards':updates+(4*recipe['score_batches_per_split']if mode=='mechanics'else
            2*sum((item['rows']+recipe['score_batch_rows']-1)//recipe['score_batch_rows']
                for item in small_cohort_manifest(metadata)['cohorts'].values())),
        'backward_calls':updates,'optimizer_steps':updates}


def build_fit_plan(*,inputs,recipe,mode,source_base_commit=None):
    if _larger_fit({'mode':mode}):return build_larger_fit_plan(inputs=inputs,recipe=recipe,mode=mode,source_base_commit=source_base_commit)
    if _noise_fit({'mode':mode}):return build_noise_fit_plan(inputs=inputs,recipe=recipe,mode=mode,source_base_commit=source_base_commit)
    need(mode in FIT_SCOPES,'FIT_MODE');recipe=fit_recipe(recipe,mode)
    stage,path,root=FIT_SCOPES[mode]
    metadata=fit_metadata(inputs,mode)
    head=source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    ta_only=mode in ('ta_mechanics','ta_main','ta_shuffle','ta_primary_mechanics','ta_primary_main')
    paths=PRIMARY_FIT_SOURCE_PATHS if mode in('ta_primary_mechanics','ta_primary_main')else TA_FIT_SOURCE_PATHS if ta_only else FIT_SOURCE_PATHS
    return {'schema_version':'gx1_native_m5_ta_prior_fit_plan_v1'if ta_only else'gx1_native_m5_full_gc_fit_plan_v1','stage':stage,'mode':mode,
        'status':'PREPARED_NOT_ADMITTED','output_root':str(root),'source_base_commit':head,
        'source_owners':{p:sha(REPO/p)for p in paths},'inputs':inputs,'metadata':metadata,
        'recipe':recipe,**({'cohort_selection':small_cohort_manifest(metadata)}if mode=='main'or ta_only else{}),
        **({'training_schedule':metadata['parent_fit_plan']['training_schedule']}if mode=='ta_primary_main'else
            {'training_schedule':ta_block_schedule(metadata,recipe)}if mode=='ta_shuffle'else{}),
        **({'backbone_gradient_policy':PRIMARY_BACKBONE_GRADIENT_POLICY}if mode in('ta_primary_mechanics','ta_primary_main')else{}),
        'execution':{'class':'trainer','device':'cuda','memory':'20G','swap':'512M',
            'tasks_max':128,'numerical_threads':8,'outer_seconds':recipe['outer_seconds'],
            'internal_seconds':recipe['internal_seconds'],'kill_after_seconds':10},
        'fit_order':('whole_fixed_B16_blocks_permuted_per_pass_by_local_PCG64_seed'if mode in('ta_shuffle','ta_primary_main','ta_noise_main')else
            'declared_passes_of_same_source_only_nonoverlapping_FIT_batches_no_CONTROL_selection'if ta_only
            else'source_only_evenly_spaced_nonoverlapping_contiguous_batches'),
        'score_order':'fixed_source_only_evenly_spaced_nonoverlapping_contiguous_batches',
        'normalization_axis':'all_TRAIN_input_ready_issues_with_any_mature_supervised_cell_before_sampling',
        **({'GC_used':False,'statistics_fits':{'normalization':1 if mode=='ta_mechanics'else 0,
            'matched_TRAIN_prior':1 if mode=='ta_mechanics'else 0}}if ta_only else{'GC_observedness_row_filter':False}),
        'scheduler_state':None,'EMA_state':None,
        'limits':fit_limits(recipe,mode,metadata),'qualification_weights_loaded':False,'training_authorized':True,'test_data_used':False,
        'economic_edge_proven':False}


def fit_source_clean(plan):
    if _larger_fit(plan):return larger_source_clean(plan)
    if _noise_fit(plan):return noise_source_clean(plan)
    if _primary_fit(plan):
        need(set(plan['source_owners'])==set(PRIMARY_FIT_SOURCE_PATHS),'PRIMARY_SOURCE_SET')
        adapted={**plan,'source_owners':{p:plan['source_owners'][p]for p in ta_audit.SOURCE_PATHS}}
        return ta_audit.source_clean(adapted)and all(sha(REPO/p)==value for p,value in plan['source_owners'].items())
    if _ta_fit(plan):
        need(set(plan['source_owners'])==set(TA_FIT_SOURCE_PATHS),'TA_FIT_SOURCE_SET')
        adapted={**plan,'source_owners':{p:plan['source_owners'][p]for p in ta_audit.SOURCE_PATHS}}
        return ta_audit.source_clean(adapted)and all(sha(REPO/p)==value for p,value in plan['source_owners'].items())
    # Reuse the same source ancestry/body guard with this scope's explicit owners.
    original=plan['source_owners']
    need(set(original)==set(FIT_SOURCE_PATHS),'FIT_SOURCE_SET')
    adapted={**plan,'source_owners':{p:original[p]for p in SOURCE_PATHS}}
    return source_clean(adapted)and all(sha(REPO/p)==original[p]for p in FIT_SOURCE_PATHS)


def checked_fit_inputs(plan):
    checked_inputs(plan)
    for value in plan['metadata'].get('mechanics',plan['metadata'].get('statistics_parent',{})).values():
        if type(value)is dict and set(value)=={'path','sha256'}:need(ref(value['path'])==value,'FIT_MECHANICS_ARTIFACT')


def inspect_fit(plan_sha,mode,*,admitted):
    stage,path,root=FIT_SCOPES[mode]
    need(Path.cwd().resolve()==REPO and not root.exists(),'FIT_CWD_OUTPUT')
    plan_ref={'path':str(path),'sha256':plan_sha};plan=read(plan_ref)
    need(plan==build_fit_plan(inputs=plan['inputs'],recipe=plan['recipe'],mode=mode,
        source_base_commit=plan['source_base_commit'])and fit_source_clean(plan),'FIT_PLAN_SOURCE')
    policy_ref=ref(POLICY);policy=read(policy_ref);gate=policy.get(stage,{})
    other_flags=set(ta_audit.data.CLOSED_FLAGS)-{'training_enabled','native_training_enabled'}
    need(all(policy.get(key)is False for key in other_flags)
        and policy.get('training_enabled')is admitted and policy.get('native_training_enabled')is admitted,
        'FIT_GLOBAL_TRAINING_GATE')
    need(gate.get('plan')==plan_ref and gate.get('output_root')==str(root)
        and gate.get('max_invocations')==1 and gate.get('claimed_invocations')==gate.get('completed_invocations')==0
        and gate.get('relaunch_allowed')is False and gate.get('training_authorized')is admitted
        and gate.get('test_authorized')is False and all(gate.get(key)==value for key,value in plan['limits'].items()),'FIT_EXACT_GATE')
    expected=('ADMITTED_ONE_USE',True,1)if admitted else('PREPARED_NOT_ADMITTED',False,0)
    need((gate.get('status'),gate.get('execution_authorized'),gate.get('remaining_invocations'))==expected,'FIT_ONE_USE')
    checked_fit_inputs(plan)
    if admitted:fit_boot_state(plan,admitted=True)
    return plan,plan_ref,policy_ref


def physical_windows_boot_id():
    executable='/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    command=r"(Get-ItemProperty -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management\PrefetchParameters' -Name BootId).BootId"
    value=subprocess.check_output([executable,'-NoProfile','-Command',command],text=True,timeout=20).strip()
    need(value.isdecimal(),'FIT_WINDOWS_BOOT_ID')
    return int(value)


def fit_boot_state(plan,*,admitted):
    actual=physical_windows_boot_id();threshold=plan['recipe']['required_windows_boot_id_greater_than']
    fresh=actual>threshold
    if admitted:
        need(fresh,'FIT_FRESH_PHYSICAL_BOOT')
        if plan['mode']in('main','ta_main','ta_shuffle','ta_primary_main','ta_noise_main','ta_larger_main','ta_five_year_main'):
            need(actual>=plan['metadata']['mechanics']['physical_windows_boot_id'],'FIT_MAIN_BOOT_LINEAGE')
    return {'actual_windows_boot_id':actual,'required_greater_than':threshold,'fresh':fresh}


def fit_axes(provider,target_ref):
    keys=('issue_ns','target_row','validity_ns','target_present','observed_previous',
        'mature','eligible','open_known_at_issue','open_sign','close_sign')
    wanted=np.flatnonzero((provider.issue_ns>=FIT_START_NS)&(provider.issue_ns<CONTROL_END_NS))
    with np.load(target_ref['path'],allow_pickle=False)as archive:
        source=archive['source_row'];at=np.searchsorted(source,provider.source_row[wanted])
        need(np.all(at<len(source))and np.array_equal(source[at],provider.source_row[wanted])
            and np.array_equal(archive['issue_ns'][at],provider.issue_ns[wanted]),'FIT_TARGET_AXIS')
        truth={key:archive['future_'+key][at]for key in keys}
    cohorts={};report={}
    for name,start,end in [('TRAIN',FIT_START_NS,FIT_END_NS),('CONTROL',FIT_END_NS,CONTROL_END_NS)]:
        selected=(provider.issue_ns[wanted]>=start)&(provider.issue_ns[wanted]<end)
        positions=wanted[selected];base_rows=provider.source_row[positions]
        ready=provider.input_ready[positions]
        full=ta_audit.fold_truth({key:value[selected]for key,value in truth.items()},end)
        rows=base_rows[ready];outcomes={key:value[ready]for key,value in full.items()}
        supervised=outcomes['eligible'].any(1)
        fit_rows=rows[supervised]if name=='TRAIN'else rows
        fit_truth={key:value[supervised]for key,value in outcomes.items()}if name=='TRAIN'else outcomes
        need(len(rows)>=16 and len(fit_rows)>=16 and np.all(np.diff(rows)>0),'FIT_COHORT:'+name)
        cohorts[name]={'rows':rows,'truth':outcomes,'fit_rows':fit_rows,'fit_truth':fit_truth,'upper_ns':end}
        report[name]={'start_ns':start,'end_exclusive_ns':end,'base_issue_rows':len(base_rows),
            'input_ready_rows':len(rows),'input_not_ready_rows':int((~ready).sum()),
            'no_mature_supervision_rows':int((~supervised).sum()),'fit_rows':len(fit_rows),
            'source_rows_sha256':digest(rows),'fit_source_rows_sha256':digest(fit_rows),
            'issue_ns_sha256':digest(outcomes['issue_ns']),'fit_issue_ns_sha256':digest(fit_truth['issue_ns']),
            'per_tf_mature':outcomes['mature'].sum(0).tolist(),
            'per_tf_eligible':outcomes['eligible'].sum(0).tolist(),'unknown_GC_rows_removed':0}
    return cohorts,report


def selected_batches(rows,count,batch=16):
    need(type(count)is int and count>0 and len(rows)//batch>=count,'FIT_BATCH_POPULATION')
    blocks=np.linspace(0,len(rows)//batch-1,count,dtype=np.int64)
    need(len(np.unique(blocks))==count,'FIT_BATCH_REPETITION')
    return [np.asarray(rows[i*batch:(i+1)*batch],np.int64)for i in blocks]


def all_score_batches(rows,batch):
    need(len(rows)>0 and type(batch)is int and batch>0,'FIT_SCORE_BATCH_POPULATION')
    return [np.asarray(rows[left:left+batch],np.int64)for left in range(0,len(rows),batch)]


def slice_truth(cohort,rows,*,fit=False):
    axis=cohort['fit_rows']if fit else cohort['rows'];truth=cohort['fit_truth']if fit else cohort['truth']
    positions=np.searchsorted(axis,rows)
    need(np.all(positions<len(axis))and np.array_equal(axis[positions],rows),'FIT_TRUTH_JOIN')
    return {key:value[positions]for key,value in truth.items()}


def training_rng():
    import random,torch
    name,keys,position,gauss,cached=np.random.get_state()
    need(name=='MT19937','FIT_NUMPY_RNG')
    return {'python':random.getstate(),'numpy_name':name,'numpy_keys':torch.as_tensor(keys.copy(),dtype=torch.uint32),
        'numpy_position':int(position),'numpy_has_gauss':int(gauss),'numpy_cached':float(cached),
        'torch_cpu':torch.get_rng_state().cpu(),'torch_cuda':[s.cpu()for s in torch.cuda.get_rng_state_all()]}


def restore_training_rng(state):
    import random,torch
    need(set(state)=={'python','numpy_name','numpy_keys','numpy_position','numpy_has_gauss','numpy_cached',
        'torch_cpu','torch_cuda'}and state['numpy_name']=='MT19937'
        and state['numpy_keys'].dtype==torch.uint32 and state['torch_cpu'].dtype==torch.uint8
        and len(state['torch_cuda'])==torch.cuda.device_count(),'FIT_RNG_SCHEMA')
    random.setstate(state['python'])
    np.random.set_state(('MT19937',state['numpy_keys'].cpu().numpy(),state['numpy_position'],
        state['numpy_has_gauss'],state['numpy_cached']))
    torch.set_rng_state(state['torch_cpu'].cpu());torch.cuda.set_rng_state_all(state['torch_cuda'])


def state_sha(value):
    import torch
    h=hashlib.sha256()
    def visit(item):
        if isinstance(item,torch.Tensor):
            array=item.detach().cpu().contiguous().numpy()
            h.update(b'T'+str(array.dtype).encode()+repr(array.shape).encode()+array.tobytes())
        elif isinstance(item,np.ndarray):
            h.update(b'N'+str(item.dtype).encode()+repr(item.shape).encode()+np.ascontiguousarray(item).tobytes())
        elif isinstance(item,dict):
            h.update(b'D')
            for key in sorted(item,key=lambda x:(type(x).__name__,repr(x))):visit(key);visit(item[key])
        elif isinstance(item,(tuple,list)):
            h.update(type(item).__name__.encode())
            for child in item:visit(child)
        else:h.update((type(item).__name__+':'+repr(item)).encode())
    visit(value);return h.hexdigest()


def seed_native_fit(seed):
    import os,random,torch
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_num_threads(8);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def fresh_fit_model(provider,normalization,plan,*,train_prior=None):
    import torch
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_bar_direction_model
    seed_native_fit(plan['recipe']['seed'])
    if _larger_fit(plan):
        torch.cuda.set_per_process_memory_fraction(LARGER_VRAM_MAX/torch.cuda.get_device_properties(0).total_memory,0)
    kwargs={'train_prior':train_prior}if _ta_fit(plan)else{'gc_mtf_history':normalization['gc_mtf_history']}
    need(not _ta_fit(plan)or train_prior is not None,'TA_FIT_INITIAL_PRIOR')
    if _primary_fit(plan):
        need(plan['backbone_gradient_policy']==PRIMARY_BACKBONE_GRADIENT_POLICY,'PRIMARY_FACTORY_POLICY')
        kwargs['backbone_gradient_policy']=PRIMARY_BACKBONE_GRADIENT_POLICY
    kwargs.update(noise_target_kwargs(plan))
    model=build_native_bar_direction_model(input_normalization=normalization,
        input_binding=provider.input_binding(normalization),architecture=plan['metadata']['architecture'],
        device=torch.device('cuda',0),**kwargs)
    options={**plan['recipe']['optimizer'],'betas':tuple(plan['recipe']['optimizer']['betas'])}
    optimizer=torch.optim.AdamW(model.parameters(),**options)
    need(not optimizer.state,'FIT_FRESH_OPTIMIZER')
    model._native_fit_counts={'model_constructions':1,'model_forwards':0,'backward_calls':0,'optimizer_steps':0}
    if _primary_fit(plan):model._native_fit_counts['autograd_calls']=0
    def count_forward(module,args):
        module._native_fit_counts['model_forwards']+=1
        need(module._native_fit_counts['model_forwards']<=plan['limits']['model_forwards'],'FIT_FORWARD_LIMIT')
    model.register_forward_pre_hook(count_forward)
    return model,optimizer


def _ta_training_batches(selected,plan):
    # Mechanics crosses a pass boundary on one fixed source-only B16 panel.
    # The admitted main consumes all 96 frozen FIT batches once per declared pass.
    if _larger_fit(plan):return larger_training_batches(selected,plan)
    base=all_score_batches(selected['FIT'],plan['recipe']['batch_rows'])
    if plan['mode']in('ta_mechanics','ta_primary_mechanics','ta_noise_mechanics'):base=base[:1]
    if plan['mode']in('ta_shuffle','ta_primary_main','ta_noise_main'):
        expected=ta_block_schedule_from_rows(selected['FIT'],plan['recipe'],
            plan['cohort_selection']['cohorts']['FIT']['selection_identity_sha256'])
        need(plan['training_schedule']==expected,'TA_BLOCK_EXACT_SCHEDULE')
        return [base[index]for order in expected['block_orders']for index in order]
    return base*plan['recipe']['passes']


def _ta_schedule_position(plan,cursor):
    if _larger_fit(plan):return larger_schedule_position(plan,cursor)
    per_pass=1 if plan['mode']in('ta_mechanics','ta_primary_mechanics','ta_noise_mechanics')else 96
    need(type(cursor)is int and 0<=cursor<=plan['recipe']['fit_batches'],'TA_FIT_SCHEDULE_CURSOR')
    position={'completed_updates':cursor,'next_pass':cursor//per_pass,'next_batch_in_pass':cursor%per_pass,
        'batches_per_pass':per_pass,'declared_passes':plan['recipe']['passes']}
    if plan['mode']in('ta_shuffle','ta_primary_main','ta_noise_main'):
        schedule=plan['training_schedule']
        body={key:value for key,value in schedule.items()if key!='contract_sha256'}
        need(schedule['contract_sha256']==hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
            and schedule['passes']==plan['recipe']['passes'],'TA_BLOCK_CHECKPOINT_SCHEDULE_IDENTITY')
        block=schedule['block_orders'][cursor//per_pass][cursor%per_pass]if cursor<plan['recipe']['fit_batches']else None
        position.update(training_schedule_sha256=schedule['contract_sha256'],next_source_block=block,
            next_source_batch_sha256=schedule['base_block_source_rows_sha256'][block]if block is not None else None)
    return position


def _fit_source_binding(provider,plan):
    return provider.surfaces['source_binding']if _ta_fit(plan)else provider.source_binding


def prepare_ta_native_fit(plan,root,*,reload_normalization=None,reload_prior=None):
    from gx1.contracts import market_state_native_m5_feature_inputs_v1 as features
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    meta=plan['metadata'];provider,admission=ta_audit.load_candidate_provider(meta['ta_plan'])
    cohorts,axis=fit_axes(provider,meta['dependencies']['target_tape'])
    for name,expected in meta['source_pool']['axis'].items():
        need(all(axis[name][key]==value for key,value in expected.items()),'TA_FIT_SOURCE_POOL:'+name)
    need(np.array_equal(cohorts['TRAIN']['rows'],cohorts['TRAIN']['fit_rows']),'TA_FIT_TRAIN_MATURITY')
    context={'provider':provider,'cohorts':cohorts,'axis':axis,'feature_requalification':admission,
        'normalization_fits':0,'prior_fits':0}
    selected,_=selected_small_cohorts(plan,context)
    if plan['mode']=='ta_main':
        parent=meta['mechanics'];normalization_ref=parent['normalization'];prior_ref=parent['prior']
        need(axis==parent['axis'],'TA_FIT_MECHANICS_AXIS')
    elif reload_normalization is not None or reload_prior is not None:
        root_authority=FIT_SCOPES['ta_mechanics'][2]
        need(reload_normalization is not None and reload_prior is not None
            and reload_normalization['path']==str(root_authority/'NORMALIZATION.json')
            and reload_prior['path']==str(root_authority/'MATCHED_FIT_PRIOR.json'),'TA_FIT_COLD_STATISTICS')
        normalization_ref,prior_ref=reload_normalization,reload_prior
    else:
        normalization=features.fit_native_m5_feature_normalization(provider,
            fit_source_rows=cohorts['TRAIN']['fit_rows'],cutoff_ns=FIT_END_NS)
        normalization_ref=publish(root/'NORMALIZATION.json',normalization)
        context.update(normalization=normalization,normalization_ref=normalization_ref)
        prior_ref=prepare_matched_prior(context,selected['FIT'],plan,root)
        context.update(normalization_fits=1,prior_fits=1)
    need(reload_normalization in(None,normalization_ref)and reload_prior in(None,prior_ref),'TA_FIT_EXACT_STATISTICS')
    normalization=read(normalization_ref);provider.input_binding(normalization)
    population=normalization['fit_population']
    need(population['cutoff_ns']==FIT_END_NS and population['issue_rows']==len(cohorts['TRAIN']['fit_rows'])
        and population['selected_source_rows_sha256']==digest(cohorts['TRAIN']['fit_rows']),'TA_FIT_TRAIN_ONLY_NORMALIZATION')
    if plan['mode']=='ta_main':need(population==meta['mechanics']['normalization_population'],'TA_FIT_IMMUTABLE_STATS')
    prior=objective.require_native_bar_train_prior(read(prior_ref))
    reference=objective.require_native_bar_train_prior(read(plan['inputs']['matched_prior_reference']))
    need(prior==reference and prior_ref['sha256']==plan['inputs']['matched_prior_reference']['sha256'],
        'TA_FIT_LABEL_ONLY_PRIOR_BYTE_PARITY')
    need(prior['fit_population']['issue_rows']==len(selected['FIT'])
        and prior['fit_population']['issue_ns_sha256']==digest(provider.issue_ns[selected['FIT']]),'TA_FIT_PRIOR_AXIS')
    context.update(normalization=normalization,normalization_ref=normalization_ref,prior=prior,prior_ref=prior_ref)
    return context


def prepare_native_fit(plan,root,*,reload_normalization=None,reload_prior=None):
    if _larger_fit(plan):return prepare_larger_native_fit(plan,root,reload_normalization=reload_normalization,reload_prior=reload_prior)
    if _noise_fit(plan):return prepare_noise_native_fit(plan,root,reload_normalization=reload_normalization,reload_prior=reload_prior)
    if _primary_fit(plan):return prepare_primary_native_fit(plan,root,reload_normalization=reload_normalization,reload_prior=reload_prior)
    if _ta_fit(plan):
        return prepare_ta_native_fit({**plan,'mode':'ta_main'}if plan['mode']=='ta_shuffle'else plan,root,
            reload_normalization=reload_normalization,reload_prior=reload_prior)
    from gx1.contracts import market_state_native_m5_gc_feature_inputs_v1 as gc
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    meta=plan['metadata'];ta,_=ta_audit.load_candidate_provider(meta['ta_plan'])
    packet=gc.load_native_m5_gc_feature_packet(meta['native_gc_source_binding'])
    provider=gc.NativeM5GCFeatureProvider(ta,packet,meta['native_gc_source_binding'])
    need(np.array_equal(provider.input_ready,ta.input_ready),'FIT_GC_ROW_FILTER')
    cohorts,axis=fit_axes(provider,meta['dependencies']['target_tape'])
    parent=meta['mechanics']if plan['mode']=='main'else meta['statistics_parent']
    need(axis==parent['axis'],'FIT_MECHANICS_AXIS')
    normalization_ref=parent['normalization'];prior_ref=parent['prior']
    need(reload_normalization in(None,normalization_ref)and reload_prior in(None,prior_ref),
        'FIT_EXACT_REUSED_STATISTICS')
    normalization=read(normalization_ref)
    provider.input_binding(normalization)
    need(normalization['fit_population']['cutoff_ns']==FIT_END_NS
        and normalization['fit_population']['issue_rows']==len(cohorts['TRAIN']['fit_rows'])
        and normalization['fit_population']['selected_source_rows_sha256']==digest(cohorts['TRAIN']['fit_rows'])
        and normalization['fit_population']==parent['normalization_population'],'FIT_TRAIN_ONLY_NORMALIZATION')
    prior=objective.require_native_bar_train_prior(read(prior_ref))
    return {'provider':provider,'cohorts':cohorts,'axis':axis,'normalization':normalization,
        'normalization_ref':normalization_ref,'prior':prior,'prior_ref':prior_ref}


def finite_all_per_device(tensors):
    """Same finite predicate; at most one bool synchronization per participating device."""
    import torch
    flags_by_device = {}
    for tensor in tensors:
        flag = torch.isfinite(tensor).all()
        flags_by_device.setdefault(flag.device, []).append(flag)
    return all((bool(torch.stack(flags).all()) for flags in flags_by_device.values()))

def native_fit_forward(model,features,*,grouped=True):
    """Complete every finite predicate before exposing the model output."""
    from contextlib import nullcontext
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import model_finite_check_scope
    need(type(grouped)is bool,'FIT_FORWARD_FINITE_SCOPE')
    with model_finite_check_scope('grouped')if grouped else nullcontext():
        return model(features)


def fit_step(model,optimizer,provider,cohort,rows,plan):
    import torch
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_direction_loss
    model.train();optimizer.zero_grad(set_to_none=True)
    raw=provider.gather(rows,max_rows=plan['recipe']['batch_rows'])
    features={key:torch.from_numpy(value).to(next(model.parameters()).device)for key,value in raw.items()}
    if plan.get('mode')in('ta_primary_mechanics','ta_noise_mechanics','ta_larger_mechanics','ta_five_year_mechanics'):
        for value in features.values():
            if value.is_floating_point():value.requires_grad_(True)
    truth=slice_truth(cohort,rows,fit=True)
    output=native_fit_forward(model,features)
    loss=native_bar_direction_loss(model_output=output,truth=truth,fold_upper_ns=plan['metadata']['fit_end_ns']if _larger_fit(plan)else FIT_END_NS,
        target_contract_sha256=plan['metadata']['target_contract']['contract_sha256'],**noise_target_kwargs(plan))
    need(not loss['skip_update']and bool(torch.isfinite(loss['loss'])),'FIT_SUPERVISED_FINITE_LOSS')
    primary_probe=primary_backbone_probe(model,loss,features)if plan.get('mode')in('ta_primary_mechanics','ta_noise_mechanics','ta_larger_mechanics','ta_five_year_mechanics')else{}
    loss['loss'].backward()
    model._native_fit_counts['backward_calls']+=1
    gradients=[p for p in model.parameters()if p.grad is not None]
    need(gradients and finite_all_per_device(p.grad for p in gradients),'FIT_FINITE_GRADIENTS')
    gradient_proof={}
    if _ta_fit(plan)and model._native_fit_counts['optimizer_steps']<2:
        for group,prefix in [('backbone','backbone.'),('heads',None)]:
            tensors=[p.grad for name,p in model.named_parameters()if p.grad is not None
                and(name.startswith(prefix)if prefix is not None else not name.startswith('backbone.'))]
            gradient_proof[group+'_gradient_l2']=float(torch.sqrt(sum(
                (value.detach().double().square().sum()for value in tensors),torch.zeros((),device=gradients[0].device))))
        gradient_proof['role']='actual_loss_gradient_before_clip_not_AdamW_weight_decay'
    norm=torch.nn.utils.clip_grad_norm_(gradients,plan['recipe']['gradient_clip_norm'],error_if_nonfinite=True)
    optimizer.step();model._native_fit_counts['optimizer_steps']+=1;optimizer.zero_grad(set_to_none=True)
    need(finite_all_per_device(model.parameters()),'FIT_FINITE_PARAMETERS')
    return {'loss':float(loss['loss'].detach()),'gradient_norm':float(norm),'source_rows_sha256':digest(rows),
        **gradient_proof,**primary_probe,
        'predictive_counts':{key:value.detach().cpu().tolist()for key,value in loss['counts'].items()},
        'output_sha256':{key:digest(value.detach().cpu().numpy())for key,value in output.items()}}


def native_fit_score_forward(model,features,plan):
    from contextlib import nullcontext
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import batch_equal_length_mtf_eval_scope
    grouped_tf=_larger_main(plan)and plan['recipe']['score_batch_rows']==256
    with batch_equal_length_mtf_eval_scope()if grouped_tf else nullcontext():
        return native_fit_forward(model,features)


def fit_score(model,context,panels,plan):
    import torch
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_direction_score_sums,native_bar_train_prior_logits
    before=state_sha(model.state_dict());rng=training_rng();training=model.training;reports={};captures={}
    model.eval()
    try:
        with torch.no_grad():
            for name,batches in panels.items():
                outputs={'open_logits':[],'close_logits':[]};rows=np.concatenate(batches)
                for batch in batches:
                    raw=context['provider'].gather(batch,max_rows=plan['recipe']['score_batch_rows'])
                    output=native_fit_score_forward(model,{key:torch.from_numpy(value).to(next(model.parameters()).device)for key,value in raw.items()},plan)
                    for key,value in output.items():outputs[key].append(value.detach().cpu().numpy().copy())
                outputs={key:np.concatenate(value)for key,value in outputs.items()}
                cohort=context['cohorts'][name];truth=slice_truth(cohort,rows)
                kwargs={'truth':truth,'fold_upper_ns':cohort['upper_ns'],
                    'target_contract_sha256':plan['metadata']['target_contract']['contract_sha256'],**noise_target_kwargs(plan)}
                reports[name]={'rows':len(rows),'source_rows_sha256':digest(rows),
                    'model':native_bar_direction_score_sums(model_output=outputs,**kwargs),
                    'fixed_TRAIN_prior':native_bar_direction_score_sums(
                        model_output=native_bar_train_prior_logits(context['prior'],rows=len(rows),**noise_target_kwargs(plan)),**kwargs)}
                captures.update({name+'__source_row':rows,**{name+'__'+key:value for key,value in outputs.items()},
                    **{name+'__truth_'+key:value for key,value in truth.items()}})
    finally:model.train(training);restore_training_rng(rng)
    need(state_sha(model.state_dict())==before,'FIT_SCORE_STATE_MUTATION')
    return reports,captures


def fit_checkpoint(model,optimizer,context,plan,plan_ref,cursor):
    ta_only=_ta_fit(plan)
    return {'schema_version':'gx1_native_m5_ta_prior_training_state_v1'if ta_only else'gx1_native_m5_full_gc_training_state_v1','plan':plan_ref,
        **({'schedule_position':_ta_schedule_position(plan,cursor)}if ta_only else{}),
        'cursor':cursor,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
        'scheduler_state':None,'EMA_state':None,'rng_state':training_rng(),'training':model.training,
        'optimizer_parameter_names':[name for name,_ in model.named_parameters()],
        'bindings':{'input':context['provider'].input_binding(context['normalization']),
            'normalization':context['normalization_ref'],'prior':context['prior_ref'],'target':plan['metadata']['target_contract'],
            'architecture':plan['metadata']['architecture'],'recipe':plan['recipe'],
            **({'model_contract':model.research_objective}if ta_only else{})}}


def publish_checkpoint(path,checkpoint):
    import torch
    rng_before=state_sha(training_rng())
    stream=io.BytesIO();torch.save(checkpoint,stream);payload=stream.getvalue();stream.close()
    binding=publish(path,payload)
    # Strict structural parity on serialized state; this does not construct a model.
    reloaded=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
    need(state_sha(reloaded)==state_sha(checkpoint),'FIT_CHECKPOINT_ROUNDTRIP')
    need(state_sha(training_rng())==rng_before,'FIT_CHECKPOINT_PUBLICATION_RNG_MUTATION')
    return binding


def restore_fit_checkpoint(model,optimizer,context,plan,plan_ref,checkpoint):
    import torch
    fields={'schema_version','plan','cursor','model_state','optimizer_state','scheduler_state','EMA_state',
        'rng_state','training','optimizer_parameter_names','bindings'}
    if _ta_fit(plan):fields.add('schedule_position')
    schema='gx1_native_m5_ta_prior_training_state_v1'if _ta_fit(plan)else'gx1_native_m5_full_gc_training_state_v1'
    need(type(checkpoint)is dict and set(checkpoint)==fields
        and checkpoint['schema_version']==schema
        and checkpoint['plan']==plan_ref and checkpoint['scheduler_state']is checkpoint['EMA_state']is None
        and checkpoint['optimizer_parameter_names']==[name for name,_ in model.named_parameters()]
        and checkpoint['bindings']==fit_checkpoint(model,optimizer,context,plan,plan_ref,checkpoint['cursor'])['bindings'],
        'FIT_CHECKPOINT_BINDINGS')
    need(type(checkpoint['cursor'])is int and 0<=checkpoint['cursor']<=plan['recipe']['fit_batches']
        and type(checkpoint['training'])is bool,'FIT_CHECKPOINT_CURSOR')
    if _ta_fit(plan):need(checkpoint['schedule_position']==_ta_schedule_position(plan,checkpoint['cursor']),'TA_FIT_CHECKPOINT_PASS_CURSOR')
    opt=checkpoint['optimizer_state'];fresh=optimizer.state_dict()
    need(set(opt)=={'state','param_groups'}and len(opt['param_groups'])==len(fresh['param_groups'])==1
        and opt['param_groups'][0]==fresh['param_groups'][0],'FIT_OPTIMIZER_GROUPS')
    ids=opt['param_groups'][0]['params'];parameters=list(model.parameters())
    need(len(ids)==len(parameters)and set(opt['state'])<=set(ids),'FIT_OPTIMIZER_PARAMETER_IDS')
    for index,item in opt['state'].items():
        parameter=parameters[ids.index(index)]
        need(set(item)=={'step','exp_avg','exp_avg_sq'}and all(isinstance(v,torch.Tensor)
            and bool(torch.isfinite(v).all())for v in item.values())
            and item['step'].numel()==1 and float(item['step'])>=1
            and all(item[key].shape==parameter.shape and item[key].dtype==parameter.dtype
                for key in ('exp_avg','exp_avg_sq')),'FIT_OPTIMIZER_STATE')
    model.load_state_dict(checkpoint['model_state'],strict=True)
    optimizer.load_state_dict(checkpoint['optimizer_state']);model.train(checkpoint['training'])
    restore_training_rng(checkpoint['rng_state'])
    need(state_sha(model.state_dict())==state_sha(checkpoint['model_state'])
        and state_sha(optimizer.state_dict())==state_sha(checkpoint['optimizer_state'])
        and state_sha(training_rng())==state_sha(checkpoint['rng_state']),'FIT_COLD_LOAD_PARITY')
    return checkpoint['cursor']


def fit_state_proof(model,optimizer,cursor,step):
    return {'model_state_sha256':state_sha(model.state_dict()),'optimizer_state_sha256':state_sha(optimizer.state_dict()),
        'scheduler_state':None,'EMA_state':None,'rng_sha256':state_sha(training_rng()),'cursor':cursor,'step':step}


def _ta_initial_prior_proof(model,context,captures,plan=None):
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_train_prior_logits
    counts={}
    for name in ('TRAIN','CONTROL'):
        rows=captures[name+'__source_row'];expected=native_bar_train_prior_logits(context['prior'],rows=len(rows),**noise_target_kwargs(plan or {}))
        for key,value in expected.items():
            need(np.array_equal(captures[name+'__'+key],value.astype(np.float32)),'TA_FIT_INITIAL_PRIOR_LOGITS:'+name+':'+key)
        counts[name]=len(rows)
    expected_counts=({'TRAIN':plan['metadata']['selected_cohorts']['FIT']['rows'],
        'CONTROL':plan['metadata']['selected_cohorts']['CONTROL']['rows']}if _larger_fit(plan or{})else{'TRAIN':1536,'CONTROL':512})
    need(counts==expected_counts,'TA_FIT_INITIAL_SCORE_POPULATION')
    return {'all_selected_initial_logits_equal_prior':True,'rows':counts,
        'prior_sha256':context['prior']['contract_sha256'],'model_initialization':model.research_objective['initialization'],
        'reference_role':('whole_input_TRAIN_predictive_cell_Laplace1_no_CONTROL_fit'if _larger_fit(plan or{})else'matched_1536_FIT_predictive_cell_Laplace1_no_CONTROL_fit')}


def fit_physical(plan,plan_ref,root):
    import torch
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    prepare_started=time.monotonic()
    context=prepare_native_fit(plan,root);prepare_seconds=time.monotonic()-prepare_started
    provider=context['provider'];recipe=plan['recipe']
    normalization_body=state_sha(context['normalization'])
    source_body=state_sha(_fit_source_binding(provider,plan))
    ta_only=_ta_fit(plan)
    selected_report=None;pool_prior_ref=context['prior_ref']
    if ta_only:
        selected,selected_report=larger_selected(context,plan)if _larger_fit(plan)else selected_small_cohorts(plan,context)
        batches=_ta_training_batches(selected,plan)
        score_selected=({k:v[:64]for k,v in selected.items()}if _larger_fit(plan)and _mechanics(plan['mode'])else selected)
        panels={'TRAIN':all_score_batches(score_selected['FIT'],recipe['score_batch_rows']),
            'CONTROL':all_score_batches(score_selected['CONTROL'],recipe['score_batch_rows'])}
    elif plan['mode']=='main':
        selected,selected_report=selected_small_cohorts(plan,context)
        batches=all_score_batches(selected['FIT'],recipe['batch_rows'])
        panels={'TRAIN':all_score_batches(selected['FIT'],recipe['score_batch_rows']),
            'CONTROL':all_score_batches(selected['CONTROL'],recipe['score_batch_rows'])}
        # Only the frozen selected FIT cells fit this primary reference.
        # The original full-pool TRAIN normalization remains unchanged.
        prepare_matched_prior(context,selected['FIT'],plan,root)
    else:
        batches=selected_batches(context['cohorts']['TRAIN']['fit_rows'],recipe['fit_batches'])
        panels={name:selected_batches(cohort['rows'],recipe['score_batches_per_split'],recipe['score_batch_rows'])
            for name,cohort in context['cohorts'].items()}
    model,optimizer=fresh_fit_model(provider,context['normalization'],plan,**({'train_prior':context['prior']}if ta_only else{}))
    if ta_only:model.require_prior_initialization_state()
    initial_sha=canonical_model_state_sha256(model.state_dict())
    if _noise_fit(plan):need(initial_sha==plan['metadata']['prepared']['initial_state_sha256'],'NOISE_FRESH_CUDA_MATCHES_CPU_INITIAL')
    statistics_parent=(plan['metadata'].get('mechanics')if ta_only else
        plan['metadata']['mechanics']if plan['mode']=='main'else plan['metadata']['statistics_parent'])
    if statistics_parent is not None and plan['mode']not in('ta_primary_mechanics','ta_noise_mechanics','ta_larger_mechanics','ta_five_year_mechanics'):
        need(initial_sha==statistics_parent['initial_state_sha256'],'FIT_FRESH_MAIN_INITIAL')
    score_started=time.monotonic()
    if _larger_main(plan):
        initial,capture_initial,prior_initialization=load_larger_initial(model,context,plan)
        initial_score_ref=plan['metadata']['prepared']['initial_score_artifact']
        capture_initial_ref=plan['metadata']['prepared']['initial_capture'];initial_score_seconds=0.
    else:
        initial,capture_initial=fit_score(model,context,panels,plan)
        torch.cuda.synchronize();initial_score_seconds=time.monotonic()-score_started
        prior_initialization=_ta_initial_prior_proof(model,context,capture_initial,plan)if ta_only else None
        initial_score_ref=publish(root/'INITIAL_SCORE.json',{'axis':context['axis'],
            'axis_role':'original_source_and_normalization_pools','selected_cohorts':selected_report,'scores':initial,
            'model_state_sha256':initial_sha,'normalization':context['normalization_ref'],'prior':context['prior_ref']})
        stream=io.BytesIO();np.savez_compressed(stream,**capture_initial)
        capture_initial_ref=publish(root/'INITIAL_CAPTURE.npz',stream.getvalue());stream.close()
    del capture_initial
    initial_checkpoint=publish_checkpoint(root/'INITIAL_STATE.pt',fit_checkpoint(model,optimizer,context,plan,plan_ref,0))
    steps=[];checkpoint_refs=[];mechanics_checkpoint=None;curve=[]
    epoch_batches=plan['metadata']['epoch_batches']if _larger_fit(plan)else 96
    for cursor,rows in enumerate(batches):
        started=time.monotonic();step=fit_step(model,optimizer,provider,context['cohorts']['TRAIN'],rows,plan)
        torch.cuda.synchronize();step['elapsed_seconds']=time.monotonic()-started
        steps.append(step)
        if ta_only and cursor<2:
            need(step['heads_gradient_l2']>0 and(step['backbone_gradient_l2']==0 if cursor==0
                else step['backbone_gradient_l2']>0),'TA_FIT_ZERO_HEAD_GRADIENT_TRANSITION')
        if plan['mode']in('ta_primary_mechanics','ta_noise_mechanics','ta_larger_mechanics','ta_five_year_mechanics'):
            proof=step['primary_gradient_routing']
            need(proof['M5_head_gradient_l2']>0 and(proof['M5_shared_gradient_l2']==0 if cursor==0
                else proof['M5_shared_gradient_l2']>0),'PRIMARY_M5_GRADIENT_TRANSITION')
        if _mechanics(plan['mode'])and cursor==0:
            mechanics_checkpoint=publish_checkpoint(root/'RESUME_STATE.pt',fit_checkpoint(
                model,optimizer,context,plan,plan_ref,1))
        if (cursor+1)%recipe['checkpoint_every_updates']==0 or cursor+1==len(batches):
            checkpoint_refs.append(publish_checkpoint(root/('CHECKPOINT_'+str(cursor+1).zfill(6)+'.pt'),
                fit_checkpoint(model,optimizer,context,plan,plan_ref,cursor+1)))
        if _ta_main(plan['mode'])and(cursor+1)%epoch_batches==0:
            score_started=time.monotonic();pass_scores,pass_capture=fit_score(model,context,panels,plan)
            torch.cuda.synchronize();elapsed=time.monotonic()-score_started
            number=(cursor+1)//epoch_batches;score_ref=publish(root/('SCORE_PASS_'+str(number).zfill(3)+'.json'),
                {'pass':number,'completed_updates':cursor+1,'scores':pass_scores,'elapsed_seconds':elapsed})
            stream=io.BytesIO();np.savez_compressed(stream,**pass_capture)
            capture_ref=publish(root/('CAPTURE_PASS_'+str(number).zfill(3)+'.npz'),stream.getvalue());stream.close()
            curve.append({'pass':number,'completed_updates':cursor+1,'scores':score_ref,'capture':capture_ref,
                'checkpoint':checkpoint_refs[-1],'model_state_sha256':canonical_model_state_sha256(model.state_dict())})
    if plan['mode']in('ta_shuffle','ta_primary_main','ta_noise_main'):need(digest(np.concatenate(batches))==plan['training_schedule']['source_row_schedule_sha256'],
        'TA_BLOCK_ACTUAL_SOURCE_ROW_SCHEDULE')
    score_started=time.monotonic();final,capture_final=(pass_scores,pass_capture)if _ta_main(plan['mode'])else fit_score(model,context,panels,plan)
    torch.cuda.synchronize();final_score_seconds=elapsed if _ta_main(plan['mode'])else time.monotonic()-score_started
    need(state_sha(context['normalization'])==normalization_body
        and state_sha(_fit_source_binding(provider,plan))==source_body
        and ref(context['normalization_ref']['path'])==context['normalization_ref'],'FIT_NORMALIZATION_SOURCE_BOOKEND')
    final_sha=canonical_model_state_sha256(model.state_dict());need(final_sha!=initial_sha,'FIT_WEIGHTS_UNCHANGED')
    final_checkpoint=publish_checkpoint(root/'FINAL_STATE.pt',fit_checkpoint(
        model,optimizer,context,plan,plan_ref,len(batches)))
    stream=io.BytesIO();np.savez_compressed(stream,**capture_final)
    capture_final_ref=publish(root/'FINAL_CAPTURE.npz',stream.getvalue());stream.close()
    report={'axis':context['axis'],'axis_role':'original_source_and_normalization_pools',
        'normalization':context['normalization_ref'],
        'normalization_population':context['normalization']['fit_population'],'prior':context['prior_ref'],
        'normalization_fits':context.get('normalization_fits',0),'prior_fits':context.get('prior_fits',1 if plan['mode']=='main'else 0),
        **({'label_only_prior_reference':context['prior_ref']if _larger_fit(plan)else plan['inputs']['matched_prior_reference']}if ta_only else
            {'original_TRAIN_pool_prior':pool_prior_ref}),'prior_scope':('whole_input_TRAIN_predictive_cells'if _larger_fit(plan)else
                'matched_1536_FIT_predictive_cells'if plan['mode']=='main'or ta_only else'original_TRAIN_pool'),
        'selected_cohorts':selected_report,'statistics_parent':statistics_parent['result']if statistics_parent else None,
        **({'prior_initialization':prior_initialization,'feature_requalification':context['feature_requalification'],
            'declared_passes':recipe['passes'],'learning_curve':curve,
            **({'training_schedule':plan['training_schedule'],
                'actual_source_row_schedule_sha256':digest(np.concatenate(batches)),
                'chronological_reference':plan['metadata']['chronological_reference']}if plan['mode']in('ta_shuffle','ta_primary_main','ta_noise_main')else{}),
            **({'backbone_gradient_policy':PRIMARY_BACKBONE_GRADIENT_POLICY,
                'raw_gradient_design_evidence':plan['metadata']['raw_gradient_design_evidence']}if _primary_fit(plan)else{}),
            **({'primary_backbone_qualification':{
                'auxiliary_shared_zero_all_steps':all(x['primary_gradient_routing']['auxiliary_shared_gradient_l2']==0 for x in steps),
                'M5_shared_positive_after_first_head_update':steps[1]['primary_gradient_routing']['M5_shared_gradient_l2']>0,
                'supported_auxiliary_heads_positive_all_steps':all(x['primary_gradient_routing']['auxiliary_head_gradient_l2']>0
                    for x in steps if x['primary_gradient_routing']['auxiliary_predictive_cells']>0)}}if plan['mode']in('ta_primary_mechanics','ta_noise_mechanics','ta_larger_mechanics','ta_five_year_mechanics')else{}),
            'gradient_transition':{'first_backbone_gradient_l2':steps[0]['backbone_gradient_l2'],
                'second_backbone_gradient_l2':steps[1]['backbone_gradient_l2'],
                'zero_first_gradient_is_expected_from_zero_readout_weights':True,
                'AdamW_weight_decay_is_not_counted_as_learning_gradient':True}}if ta_only else{}),
        'train_batch_rows':recipe['batch_rows'],'score_batch_rows':recipe['score_batch_rows'],
        'initial_state_sha256':initial_sha,'final_state_sha256':final_sha,'initial':initial,'final':final,
        'initial_checkpoint':initial_checkpoint,'final_checkpoint':final_checkpoint,'checkpoints':checkpoint_refs,
        'initial_score_artifact':initial_score_ref,'captures':{'INITIAL':capture_initial_ref,'FINAL':capture_final_ref},
        'prepare_seconds':prepare_seconds,'initial_score_seconds':initial_score_seconds,'final_score_seconds':final_score_seconds,
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'steps':steps,
        'canonical_optimizer_updates':len(batches),'evaluation_batches_per_phase':sum(len(value)for value in panels.values()),
        'actual_counts':dict(model._native_fit_counts),
        **({'initial_baseline_reused':_larger_main(plan),'prepared_initial_parent':plan['metadata']['prepared']['result'],
            'training_schedule':plan['training_schedule'],'actual_source_row_schedule_sha256':digest(np.concatenate(batches))}if _larger_fit(plan)else{}),
        'score_scope':('five_year_TRAIN_and_Jan_Jun2026_development_VAL_all_rows_including_unknown_tails'if plan['mode']=='ta_five_year_main'else
            'whole_year_TRAIN_and_later_CONTROL_all_rows_including_unknown_tails'if _larger_main(plan)else
            'first_64_TRAIN_and_first_64_CONTROL_technical_panels'if _larger_fit(plan)and _mechanics(plan['mode'])else'fixed_source_only_mechanics_panels_not_full_population'if plan['mode']=='mechanics'
            else'all_source_selected_1536_FIT_and_512_CONTROL_rows_including_unknown_last_CONTROL_row'),
        **({'later_split_role':plan['metadata']['later_split_role']}if plan['mode']in('ta_five_year_mechanics','ta_five_year_main')else{}),
        'score_lane':'eval_no_grad_same_batch_geometry_and_frozen_TRAIN_stats',
        'qualification_weights_loaded':False,'test_data_used':False,'economic_edge_proven':False}
    if _larger_fit(plan):need(torch.cuda.max_memory_reserved()<=LARGER_VRAM_MAX,'LARGER_VRAM_LIMIT')
    if _mechanics(plan['mode']):
        expected=publish(root/'UNINTERRUPTED_PROOF.json',fit_state_proof(model,optimizer,2,steps[-1]))
        report['_cold_authority']={'checkpoint':mechanics_checkpoint,'expected':expected,
            'normalization':context['normalization_ref'],'prior':context['prior_ref']}
    del model,optimizer,provider,context
    import gc
    gc.collect();torch.cuda.empty_cache()
    return report


def cold_resume_fit(plan_ref):
    import os,torch
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    guard=require_guarded_cuda_trainer_execution();plan=read(plan_ref)
    need(_mechanics(plan['mode']),'FIT_COLD_MECHANICS_ONLY')
    stage,path,root=FIT_SCOPES[plan['mode']]
    authority=read(ref(root/'COLD_RESUME_AUTHORITY.json'))
    need(plan_ref['path']==str(path)and plan['stage']==stage and authority['plan']==plan_ref
        and authority['parent_pid']==os.getppid()and authority['claim']==ref(root/'CLAIM.json')
        and authority['policy']==ref(POLICY)and fit_source_clean(plan),'FIT_COLD_PARENT')
    checked_fit_inputs(plan);boot=fit_boot_state(plan,admitted=True)
    need(boot['actual_windows_boot_id']==authority['physical_windows_boot_id'],'FIT_COLD_BOOT')
    out=root/'COLD_RESUME';out.mkdir(exist_ok=False)
    context=prepare_native_fit(plan,out,reload_normalization=authority['normalization'],reload_prior=authority['prior'])
    model,optimizer=fresh_fit_model(context['provider'],context['normalization'],plan,
        **({'train_prior':context['prior']}if _ta_fit(plan)else{}))
    need(ref(authority['checkpoint']['path'])==authority['checkpoint'],'FIT_COLD_CHECKPOINT_HASH')
    checkpoint=torch.load(authority['checkpoint']['path'],map_location='cpu',weights_only=True)
    cursor=restore_fit_checkpoint(model,optimizer,context,plan,plan_ref,checkpoint)
    batches=(_ta_training_batches((larger_selected(context,plan)if _larger_fit(plan)else selected_small_cohorts(plan,context))[0],plan)if _ta_fit(plan)else
        selected_batches(context['cohorts']['TRAIN']['fit_rows'],plan['recipe']['fit_batches']))
    need(cursor==1 and len(batches)==2,'FIT_COLD_CURSOR')
    if _larger_fit(plan):need(checkpoint['schedule_position']==larger_schedule_position(plan,1),'LARGER_COLD_EXACT_CURSOR')
    if _ta_fit(plan)and not _larger_fit(plan):need(checkpoint['schedule_position']['next_pass']==1
        and checkpoint['schedule_position']['next_batch_in_pass']==0,'TA_FIT_COLD_PASS_BOUNDARY')
    step=fit_step(model,optimizer,context['provider'],context['cohorts']['TRAIN'],batches[cursor],plan)
    actual=fit_state_proof(model,optimizer,cursor+1,step)
    expected=read(authority['expected'])
    # Throughput is a process observation; it is not deterministic model state.
    expected['step'].pop('elapsed_seconds',None)
    need(actual==expected,'FIT_COLD_CONTINUATION_BITEXACT')
    need(fit_source_clean(plan)and ref(POLICY)==authority['policy'],'FIT_COLD_BOOKEND')
    publish(out/'PROOF.json',{'status':'COMPLETE_COLD_NATIVE_RESUME','bitexact':True,'plan':plan_ref,
        'parent_claim':authority['claim'],'process_id':os.getpid(),'parent_pid':os.getppid(),
        'guard':guard,'physical_windows_boot_id':boot['actual_windows_boot_id'],'state':actual,
        **({'restored_schedule_position':checkpoint['schedule_position']}if _ta_fit(plan)else{}),
        'actual_counts':dict(model._native_fit_counts),'clean_process_io_firewall':common.firewall_state(),
        'test_data_used':False})
    return actual


def require_cold_resume_proof(proof,*,parent_pid):
    firewall=proof.get('clean_process_io_firewall')
    need(proof.get('bitexact')is True and proof.get('parent_pid')==parent_pid
        and type(firewall)is dict and firewall.get('denied_attempts')==0
        and firewall.get('denied_operations')=={},'FIT_COLD_PROOF')
    return proof


def runtime_mechanics_trajectory(report,plan):
    import torch
    qualification=plan['metadata']['runtime_qualification']
    old=read(qualification['reference_mechanics_result'])['report']
    need(old['initial_state_sha256']==report['initial_state_sha256']
        and old['final_state_sha256']==report['final_state_sha256'],'RUNTIME_MECHANICS_MODEL_TRAJECTORY')
    refs=[old['final_checkpoint'],report['final_checkpoint']]
    for binding in refs:need(ref(binding['path'])==binding,'RUNTIME_MECHANICS_CHECKPOINT_HASH')
    checkpoints=[torch.load(binding['path'],map_location='cpu',weights_only=True)for binding in refs]
    fields=('model_state','optimizer_state','rng_state','cursor','training','optimizer_parameter_names','schedule_position')
    identities={key:[state_sha(checkpoint[key])for checkpoint in checkpoints]for key in fields}
    need(all(a==b for a,b in identities.values()),'RUNTIME_MECHANICS_OPTIMIZER_RNG_TRAJECTORY')
    steps=[{k:v for k,v in step.items()if k!='elapsed_seconds'}for step in report['steps']]
    previous=[{k:v for k,v in step.items()if k!='elapsed_seconds'}for step in old['steps']]
    need(steps==previous,'RUNTIME_MECHANICS_LOSS_GRAD_OUTPUT_TRAJECTORY')
    return {'bitexact':True,'reference_mechanics_result':qualification['reference_mechanics_result'],
        'reference_checkpoint':refs[0],'current_checkpoint':refs[1],
        'state_sha256':{key:pair[0]for key,pair in identities.items()},'actual_step_reports_bitexact':True}


def run_native_fit(plan_sha,mode):
    import os,sys
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    began=time.monotonic();common.firewall_state();guard=require_guarded_cuda_trainer_execution()
    plan,plan_ref,policy_ref=inspect_fit(plan_sha,mode,admitted=True)
    boot=fit_boot_state(plan,admitted=True)
    stage,path,root=FIT_SCOPES[mode];root.mkdir(exist_ok=False)
    claim=publish(root/'CLAIM.json',{'stage':stage,'status':'CLAIMED_ONE_USE','plan':plan_ref,
        'policy_at_claim':policy_ref,'physical_windows_boot_id':boot['actual_windows_boot_id'],
        'process_identity':checkpoint_process_identity(),
        'max_invocations':1,'remaining_invocations':0,'test_data_used':False})
    old=signal.getsignal(signal.SIGALRM);result=None
    def expired(*args):raise TimeoutError('NATIVE_M5_FIT_INTERNAL_LIMIT')
    try:
        remaining=plan['recipe']['internal_seconds']-(time.monotonic()-began);need(remaining>0,'FIT_TIME_BEFORE_INPUTS')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
        if mode=='ta_five_year_main'and plan['recipe']['checkpoint_every_updates']==256:
            report=checkpoint_window_physical(plan,plan_ref,root,execution_ref=plan_ref,claim=claim,
                limits=plan['limits'],stop_after_cursor=plan['recipe']['fit_batches'],
                deadline=began+plan['recipe']['internal_seconds'])
        else:report=fit_physical(plan,plan_ref,root)
        if _mechanics(mode):
            authority=report.pop('_cold_authority')
            publish(root/'COLD_RESUME_AUTHORITY.json',{**authority,'plan':plan_ref,'claim':claim,
                'policy':policy_ref,'parent_pid':os.getpid(),'physical_windows_boot_id':boot['actual_windows_boot_id']})
            command=[sys.executable,'-m','gx1.audit.qualify_native_m5_full_gc_direction_v1',
                '--plan',str(path),'--plan-sha256',plan_sha,'--device','cuda','--out-dir',str(root/'COLD_RESUME')]
            with(root/'COLD_RESUME.log').open('wb')as log:
                completed=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,
                    timeout=max(1,plan['recipe']['internal_seconds']-(time.monotonic()-began)))
            need(completed.returncode==0,'FIT_COLD_PROCESS_FAILED')
            proof_ref=ref(root/'COLD_RESUME/PROOF.json');proof=read(proof_ref)
            require_cold_resume_proof(proof,parent_pid=os.getpid())
            for key,value in proof['actual_counts'].items():report['actual_counts'][key]+=value
            report['cold_resume']={'proof':proof_ref,'bitexact':True,'new_process':True}
        if mode=='ta_five_year_mechanics':
            report['runtime_reference_trajectory']=runtime_mechanics_trajectory(report,plan)
        checked_fit_inputs(plan)
        if report.get('status')=='RESUMABLE':
            need(all(report['actual_counts'][k]<=v for k,v in plan['limits'].items()),'FIT_PAUSED_COUNT_LIMITS')
        else:need(report['actual_counts']==plan['limits'],'FIT_ACTUAL_COUNT_LIMITS')
        need(fit_source_clean(plan)and ref(POLICY)==policy_ref and ref(path)==plan_ref,'FIT_SOURCE_POLICY_BOOKEND')
        updates=report['actual_counts']['optimizer_steps']if report.get('status')=='RESUMABLE'else 3 if _mechanics(mode)else plan['recipe']['fit_batches']
        result=publish(root/'RESULT.json',{'stage':stage,
            'status':('RESUMABLE_NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT'if report.get('status')=='RESUMABLE'else
                'COMPLETE_NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS'if mode=='ta_five_year_mechanics'else
                'COMPLETE_NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT'if mode=='ta_five_year_main'else
                'COMPLETE_NATIVE_M5_TA_ATR03_YEAR_MECHANICS'if mode=='ta_larger_mechanics'else
                'COMPLETE_NATIVE_M5_TA_ATR03_YEAR_FIT'if mode=='ta_larger_main'else
                'COMPLETE_NATIVE_M5_TA_ATR03_MECHANICS'if mode=='ta_noise_mechanics'else
                'COMPLETE_NATIVE_M5_TA_ATR03_FIT'if mode=='ta_noise_main'else
                'COMPLETE_NATIVE_M5_TA_PRIMARY_BACKBONE_MECHANICS'if mode=='ta_primary_mechanics'else
                'COMPLETE_NATIVE_M5_TA_PRIMARY_BACKBONE_FIT'if mode=='ta_primary_main'else
                'COMPLETE_NATIVE_M5_TA_PRIOR_MECHANICS'if mode=='ta_mechanics'else
                'COMPLETE_NATIVE_M5_TA_PRIOR_BLOCK_ORDER'if mode=='ta_shuffle'else
                'COMPLETE_NATIVE_M5_TA_PRIOR_LEARNING_CURVE'if mode=='ta_main'else
                'COMPLETE_NATIVE_M5_FULL_GC_FIT_MECHANICS'if mode=='mechanics'else'COMPLETE_NATIVE_M5_FULL_GC_SHORT_FIT'),
            'plan':plan_ref,'claim':claim,'report':report,'guard':guard,'source_unchanged':True,
            'physical_windows_boot_id':boot['actual_windows_boot_id'],
            'optimizer_steps':updates,'training_authorized':True,'test_data_used':False,
            'economic_edge_proven':False,'clean_process_io_firewall':common.firewall_state(),
            'elapsed_seconds':time.monotonic()-began})
        terminal=publish(root/'TERMINAL.json',{'stage':stage,'status':'RESUMABLE'if report.get('status')=='RESUMABLE'else'COMPLETE','exit_code':0,
            'plan':plan_ref,'claim':claim,'result':result,'source_unchanged':True,'remaining_invocations':0,
            **({'checkpoint_receipt':report['checkpoint_receipt']}if report.get('status')=='RESUMABLE'else{}),
            'relaunch_allowed':False,'test_data_used':False})
        return {'result':result,'terminal':terminal}
    except BaseException as error:
        publish(root/'TERMINAL.json',{'stage':stage,'status':'FAILED_CONSUMED','exit_code':1,
            'plan':plan_ref,'claim':claim,'result':result,'error':type(error).__name__+':'+str(error)[:2000],
            'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


def fit_cli(args):
    mode=next((key for key,(_,path,_)in FIT_SCOPES.items()if args.plan==str(path)),None)
    need(mode is not None and args.device=='cuda','FIT_CLI_SCOPE')
    stage,path,root=FIT_SCOPES[mode]
    if args.out_dir==str(root/'COLD_RESUME')and _mechanics(mode)and not args.preflight:
        cold_resume_fit({'path':str(path),'sha256':args.plan_sha256});return
    need(args.out_dir==str(root),'FIT_CLI_OUTPUT')
    if args.preflight:
        plan,plan_ref,policy_ref=inspect_fit(args.plan_sha256,mode,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_NATIVE_M5_FULL_GC_FIT_METADATA_ONLY','plan':plan_ref,
            'policy':policy_ref,'physical_boot':fit_boot_state(plan,admitted=False),
            'claim_created':False,'raw_values_decoded':0,'feature_values_decoded':0,
            'model_constructions':0,'optimizer_steps':0}))
    else:print(json.dumps(run_native_fit(args.plan_sha256,mode),sort_keys=True))


SHARED_GRADIENT_STAGE='native_m5_ta_shared_gradient_20261119_001'
SHARED_GRADIENT_PLAN=REPO/'configs/research/NATIVE_M5_TA_SHARED_GRADIENT_20261119_001.json'
SHARED_GRADIENT_ROOT=Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_SHARED_GRADIENT_20261119_001')
SHARED_GRADIENT_PARENT={
    'block_result':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_TA_PRIOR_BLOCK_ORDER_20261119_001/RESULT.json',
        'sha256':'396959706d0ad0676b407fc008d2db40d7e22a2ed73d36c17fc21a7f766ce0f5'},
    'block_terminal':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_TA_PRIOR_BLOCK_ORDER_20261119_001/TERMINAL.json',
        'sha256':'2248843836fc5c2bd711c78e38a695beaa89561921ad9c4fdcea2e462fc18e40'},
    'block_review':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_TA_PRIOR_BLOCK_ORDER_20261119_001/MATCHED_ORDER_RESULT_REVIEW.json',
        'sha256':'09e03b5418ac6ef2ddd3ca4188d0f50f0fce3dcc6fb2c901760ad7cadc0b6d5d'}}
SHARED_GRADIENT_BLOCKS=(0,31,63,95)
SHARED_GRADIENT_LIMITS={'model_constructions':1,'checkpoint_loads':1,'model_forwards':4,'autograd_calls':8,
    'backward_calls':0,'optimizer_constructions':0,'optimizer_steps':0}
SHARED_GRADIENT_EXECUTION={'class':'producer','device':'cpu','memory':'12G','swap':'512M','tasks_max':64,
    'numerical_threads':1,'internal_seconds':870,'external_wall_timeout_enforced':False}


def shared_gradient_source_identity(source):
    import ast
    tree=ast.parse(source)
    diagnostic={'shared_gradient_source_identity','shared_gradient_source_bridge','shared_gradient_metadata',
        'build_shared_gradient_plan','shared_gradient_checked_inputs','shared_gradient_source_clean',
        'inspect_shared_gradient','shared_gradient_rng','shared_gradient_restore_rng','shared_gradient_measures',
        'shared_gradient_load_model','qualify_shared_gradient','run_shared_gradient','shared_gradient_cli','main'}
    constants={'SHARED_GRADIENT_STAGE','SHARED_GRADIENT_PLAN','SHARED_GRADIENT_ROOT','SHARED_GRADIENT_PARENT',
        'SHARED_GRADIENT_BLOCKS','SHARED_GRADIENT_LIMITS','SHARED_GRADIENT_EXECUTION'}
    tree.body=[node for node in tree.body if not(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))
        and node.name in diagnostic)and not(isinstance(node,ast.Assign)and any(
            isinstance(target,ast.Name)and target.id in constants for target in node.targets))]
    return {'existing_module_ast_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
        'numeric_identity':ta_block_order_numeric_identity(ast.unparse(tree))}


def shared_gradient_source_bridge(parent):
    owner='gx1/audit/qualify_native_m5_full_gc_direction_v1.py'
    allowed={owner,'tests/test_qualify_native_m5_full_gc_direction_v1.py'}
    need(parent['mode']=='ta_shuffle'and set(parent['source_owners'])==set(TA_FIT_SOURCE_PATHS),'SHARED_PARENT_SOURCE_SET')
    current={name:sha(REPO/name)for name in TA_FIT_SOURCE_PATHS}
    need(all(current[name]==value for name,value in parent['source_owners'].items()if name not in allowed),
        'SHARED_UNCHANGED_MODEL_LOSS_STATS_OWNERS')
    original=subprocess.check_output(['git','show',parent['source_base_commit']+':'+owner],cwd=REPO)
    need(hashlib.sha256(original).hexdigest()==parent['source_owners'][owner],'SHARED_ORIGINAL_DRIVER_BLOB')
    identity=shared_gradient_source_identity(original.decode())
    need(identity==shared_gradient_source_identity((REPO/owner).read_text()),'SHARED_EXISTING_DRIVER_AST')
    return {'schema_version':'gx1_native_m5_ta_shared_gradient_source_bridge_v1',**identity,
        'original_block_order_source_base_commit':parent['source_base_commit'],
        'allowed_changed_source_paths':sorted(allowed),'all_other_source_owners_exact':True,
        'source_changes':{name:{'original_sha256':parent['source_owners'][name],'current_sha256':current[name]}
            for name in sorted(allowed)if current[name]!=parent['source_owners'][name]},
        'scope':'new_zero_update_diagnostic_and_exact_CPU_CLI_dispatch_only'}


def shared_gradient_metadata(inputs):
    need(type(inputs)is dict and inputs==SHARED_GRADIENT_PARENT,'SHARED_EXACT_PARENT_REFS')
    result,terminal,review=(read(inputs[key])for key in SHARED_GRADIENT_PARENT)
    parent,claim=read(result['plan']),read(result['claim']);report=result['report']
    need(result['status']=='COMPLETE_NATIVE_M5_TA_PRIOR_BLOCK_ORDER'
        and result['stage']==terminal['stage']==parent['stage']==claim['stage']==FIT_SCOPES['ta_shuffle'][0]
        and terminal['status']=='COMPLETE'and terminal['exit_code']==0
        and terminal['result']==inputs['block_result']and terminal['plan']==result['plan']==claim['plan']
        and terminal['claim']==result['claim']and result['source_unchanged']is True
        and result['test_data_used']is terminal['test_data_used']is False
        and parent['mode']=='ta_shuffle'and parent['recipe']==fit_recipe(parent['recipe'],'ta_shuffle')
        and result['optimizer_steps']==report['canonical_optimizer_updates']==384
        and report['actual_counts']=={'model_constructions':1,'model_forwards':544,'backward_calls':384,'optimizer_steps':384}
        and report['declared_passes']==4 and report['normalization_fits']==report['prior_fits']==0
        and result['clean_process_io_firewall']['denied_attempts']==0,'SHARED_COMPLETE_BLOCK_PARENT')
    need(review['schema_version']=='gx1_native_m5_whole_block_order_result_review_v1'
        and review['result']==inputs['block_result']and review['terminal']==inputs['block_terminal']
        and review['plan']==result['plan']and review['normalization']==report['normalization']
        and review['prior']==report['prior']and review['final_state_sha256']==report['final_state_sha256']
        and review['actual_counts']==report['actual_counts']and review['training_schedule']==report['training_schedule']
        and review['actual_source_row_schedule_sha256']==report['actual_source_row_schedule_sha256']
        and review['GC_used']is review['test_data_used']is False and review['M1_attempts']==0,'SHARED_MATCHED_REVIEW')
    dependencies={**parent['metadata']['dependencies'],'parent_plan':result['plan'],'parent_claim':result['claim'],
        'final_checkpoint':report['final_checkpoint'],'normalization':report['normalization'],'prior':report['prior']}
    return {'dependencies':dependencies,'parent_fit_plan':parent,'normalization':report['normalization'],
        'prior':report['prior'],'final_checkpoint':report['final_checkpoint'],
        'final_state_sha256':report['final_state_sha256'],'selected_cohorts':report['selected_cohorts'],
        'source_bridge':shared_gradient_source_bridge(parent)}


def build_shared_gradient_plan(*,inputs,source_base_commit=None):
    metadata=shared_gradient_metadata(inputs)
    head=source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    return {'schema_version':'gx1_native_m5_ta_shared_gradient_plan_v1','stage':SHARED_GRADIENT_STAGE,
        'status':'PREPARED_NOT_ADMITTED','output_root':str(SHARED_GRADIENT_ROOT),'source_base_commit':head,
        'source_owners':{name:sha(REPO/name)for name in TA_FIT_SOURCE_PATHS},'inputs':inputs,'metadata':metadata,
        'execution':SHARED_GRADIENT_EXECUTION,'limits':SHARED_GRADIENT_LIMITS,
        'canonical_TRAIN_block_indices':list(SHARED_GRADIENT_BLOCKS),'rows_per_block':16,
        'gradient_partition':'M5_close_sum/Ntotal_and_all_remaining_predictive_loss_sums/Ntotal',
        'panel_aggregation':'sum_predictive_cell_losses_across_four_batches_divided_by_sum_Ntotal',
        'analysis_mode':'eval_raw_parameter_gradients_not_AdamW_or_historical_dropout',
        'CONTROL_analysis_rows':0,'shared_loader_may_decode_qualified_CONTROL':True,
        'training_authorized':False,'test_data_used':False,'economic_edge_proven':False}


def shared_gradient_checked_inputs(plan):
    for binding in [*plan['inputs'].values(),*plan['metadata']['dependencies'].values()]:
        need(ref(binding['path'])==binding,'SHARED_INPUT_HASH_BOOKEND')


def shared_gradient_source_clean(plan):
    return fit_source_clean({**plan,'mode':'ta_shuffle'})


def inspect_shared_gradient(plan_sha,*,admitted):
    need(Path.cwd().resolve()==REPO and not SHARED_GRADIENT_ROOT.exists(),'SHARED_CWD_OUTPUT')
    plan_ref={'path':str(SHARED_GRADIENT_PLAN),'sha256':plan_sha};plan=read(plan_ref)
    need(plan==build_shared_gradient_plan(inputs=plan['inputs'],source_base_commit=plan['source_base_commit'])
        and shared_gradient_source_clean(plan),'SHARED_PLAN_SOURCE')
    policy_ref=ref(POLICY);policy=read(policy_ref);gate=policy.get(SHARED_GRADIENT_STAGE,{})
    need(all(policy.get(key)is False for key in ta_audit.data.CLOSED_FLAGS)
        and gate.get('plan')==plan_ref and gate.get('output_root')==str(SHARED_GRADIENT_ROOT)
        and gate.get('max_invocations')==1 and gate.get('claimed_invocations')==gate.get('completed_invocations')==0
        and gate.get('relaunch_allowed')is False and gate.get('training_authorized')is False
        and gate.get('test_authorized')is False and gate.get('execution')==plan['execution']
        and all(gate.get(key)==value for key,value in SHARED_GRADIENT_LIMITS.items()),'SHARED_CLOSED_POLICY')
    expected=('ADMITTED_ONE_USE',True,1)if admitted else('PREPARED_NOT_ADMITTED',False,0)
    need((gate.get('status'),gate.get('execution_authorized'),gate.get('remaining_invocations'))==expected,'SHARED_ONE_USE')
    shared_gradient_checked_inputs(plan);return plan,plan_ref,policy_ref


def shared_gradient_rng():
    import random,torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state().clone()}


def shared_gradient_restore_rng(value):
    import random,torch
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch_cpu'])


def shared_gradient_measures(g5,gaux):
    import math,torch
    need(len(g5)==len(gaux),'SHARED_GRADIENT_PARAMETER_AXIS')
    norm5=normaux=dot=0.;unused5=unusedaux=0
    for left,right in zip(g5,gaux):
        unused5+=left is None;unusedaux+=right is None
        for value in (left,right):
            need(value is None or bool(torch.isfinite(value).all()),'SHARED_NONFINITE_GRADIENT')
        if left is not None:norm5+=float(left.detach().double().square().sum())
        if right is not None:normaux+=float(right.detach().double().square().sum())
        if left is not None and right is not None:
            need(left.shape==right.shape,'SHARED_GRADIENT_SHAPE')
            dot+=float((left.detach().double()*right.detach().double()).sum())
    a,b=math.sqrt(norm5),math.sqrt(normaux);joint=norm5+dot
    return {'g5_l2':a,'gaux_l2':b,'cross_dot':dot,'cosine':dot/(a*b)if a>0 and b>0 else None,
        'g5_dot_joint':joint,'raw_joint_opposes_M5':joint<0,
        'unused_M5_parameter_tensors':unused5,'unused_aux_parameter_tensors':unusedaux}


def shared_gradient_load_model(plan,context):
    import torch
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_bar_direction_model
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    meta=plan['metadata'];parent=meta['parent_fit_plan'];checkpoint_ref=meta['final_checkpoint']
    need(ref(checkpoint_ref['path'])==checkpoint_ref,'SHARED_FINAL_BEFORE_DECODE')
    model=build_native_bar_direction_model(input_normalization=context['normalization'],
        input_binding=context['provider'].input_binding(context['normalization']),
        architecture=parent['metadata']['architecture'],device=torch.device('cpu'),train_prior=context['prior'])
    checkpoint=torch.load(checkpoint_ref['path'],map_location='cpu',weights_only=True)
    need(checkpoint['schema_version']=='gx1_native_m5_ta_prior_training_state_v1'
        and checkpoint['plan']==meta['dependencies']['parent_plan']and checkpoint['cursor']==384
        and checkpoint['schedule_position']==_ta_schedule_position(parent,384)
        and checkpoint['bindings']=={'input':context['provider'].input_binding(context['normalization']),
            'normalization':context['normalization_ref'],'prior':context['prior_ref'],
            'target':parent['metadata']['target_contract'],'architecture':parent['metadata']['architecture'],
            'recipe':parent['recipe'],'model_contract':model.research_objective}
        and checkpoint['optimizer_parameter_names']==[name for name,_ in model.named_parameters()]
        and canonical_model_state_sha256(checkpoint['model_state'])==meta['final_state_sha256'],
        'SHARED_FINAL_CHECKPOINT_BINDINGS')
    # Only trained model state is applied. No optimizer is constructed or restored.
    model.load_state_dict(checkpoint['model_state'],strict=True);model.eval()
    need(canonical_model_state_sha256(model.state_dict())==meta['final_state_sha256']
        and ref(checkpoint_ref['path'])==checkpoint_ref,'SHARED_FINAL_STRICT_LOAD')
    return model


def qualify_shared_gradient(plan):
    import torch
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_direction_loss
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    meta=plan['metadata'];parent=meta['parent_fit_plan'];torch.set_num_threads(1)
    context=prepare_native_fit(parent,SHARED_GRADIENT_ROOT)
    need(context['normalization_ref']==meta['normalization']and context['prior_ref']==meta['prior'],
        'SHARED_FROZEN_STATISTICS')
    selected,selection=selected_small_cohorts(parent,context)
    need(selection==meta['selected_cohorts'],'SHARED_MATCHED_SOURCE_COHORTS')
    blocks=all_score_batches(selected['FIT'],16)
    norm_before=state_sha(context['normalization']);prior_before=state_sha(context['prior'])
    source_before=state_sha(_fit_source_binding(context['provider'],parent));rng_before=shared_gradient_rng()
    model=shared_gradient_load_model(plan,context)
    # Construction draws no decision information; restore its temporary CPU RNG consumption.
    shared_gradient_restore_rng(rng_before)
    named=[(name,parameter)for name,parameter in model.backbone.named_parameters()if parameter.requires_grad]
    parameters=tuple(parameter for _,parameter in named)
    need(parameters and all(parameter.grad is None for parameter in model.parameters()),'SHARED_INITIAL_GRAD_NONE')
    before=canonical_model_state_sha256(model.state_dict());counts=dict.fromkeys(SHARED_GRADIENT_LIMITS,0)
    counts.update(model_constructions=1,checkpoint_loads=1);aggregate5=[None]*len(parameters);aggregateaux=[None]*len(parameters)
    total=0;reports=[];count_sums={'open':np.zeros(5,np.int64),'close':np.zeros(5,np.int64)}
    for index in SHARED_GRADIENT_BLOCKS:
        rows=blocks[index];raw=context['provider'].gather(rows,max_rows=16)
        output=model({key:torch.from_numpy(value)for key,value in raw.items()});counts['model_forwards']+=1
        loss=native_bar_direction_loss(model_output=output,truth=slice_truth(context['cohorts']['TRAIN'],rows,fit=True),
            fold_upper_ns=FIT_END_NS,target_contract_sha256=parent['metadata']['target_contract']['contract_sha256'])
        n=int(loss['counts']['open'].sum()+loss['counts']['close'].sum())
        need(n>0 and not loss['skip_update'],'SHARED_PREDICTIVE_SUPPORT')
        l5=loss['loss_sums']['close'][0]/n
        laux=(loss['loss_sums']['open'].sum()+loss['loss_sums']['close'][1:].sum())/n
        g5=torch.autograd.grad(l5,parameters,retain_graph=True,allow_unused=True);counts['autograd_calls']+=1
        gaux=torch.autograd.grad(laux,parameters,allow_unused=True);counts['autograd_calls']+=1
        measures=shared_gradient_measures(g5,gaux)
        for accum,gradients in ((aggregate5,g5),(aggregateaux,gaux)):
            for j,value in enumerate(gradients):
                if value is None:continue
                if accum[j]is None:accum[j]=value.detach().double()*n
                else:accum[j].add_(value.detach().double(),alpha=n)
        support={key:value.detach().cpu().tolist()for key,value in loss['counts'].items()}
        for key,value in support.items():count_sums[key]+=np.asarray(value,np.int64)
        total+=n
        reports.append({'canonical_TRAIN_block':index,'rows':len(rows),'source_rows_sha256':digest(rows),
            'issue_ns_sha256':digest(context['provider'].issue_ns[rows]),'Ntotal':n,'predictive_counts':support,
            'M5_close_predictive_cells':support['close'][0],'aux_predictive_cells':n-support['close'][0],
            'L5_joint_denominator':float(l5.detach()),'Laux_joint_denominator':float(laux.detach()),**measures})
        need(all(parameter.grad is None for parameter in model.parameters()),'SHARED_AUTOGRAD_LEFT_PARAMETER_GRAD')
    for gradients in (aggregate5,aggregateaux):
        for value in gradients:
            if value is not None:value.div_(total)
    panel=shared_gradient_measures(aggregate5,aggregateaux)
    need(counts==SHARED_GRADIENT_LIMITS and canonical_model_state_sha256(model.state_dict())==before==meta['final_state_sha256']
        and state_sha(context['normalization'])==norm_before and state_sha(context['prior'])==prior_before
        and state_sha(_fit_source_binding(context['provider'],parent))==source_before
        and state_sha(shared_gradient_rng())==state_sha(rng_before)
        and all(parameter.grad is None for parameter in model.parameters()),'SHARED_IMMUTABLE_BOOKEND')
    need(ref(meta['normalization']['path'])==meta['normalization']and ref(meta['prior']['path'])==meta['prior'],
        'SHARED_STATISTICS_FILE_BOOKEND')
    return {'actual_counts':counts,'batches':reports,'panel':{**panel,'Ntotal':total,
        'predictive_counts':{key:value.tolist()for key,value in count_sums.items()},
        'aggregation':'sum_cell_losses_all_four_batches_divided_by_sum_Ntotal; gradients_weighted_by_batch_Ntotal'},
        'shared_parameter_tensors':len(parameters),'shared_parameter_elements':sum(p.numel()for p in parameters),
        'shared_parameter_names_sha256':hashlib.sha256(json.dumps([name for name,_ in named]).encode()).hexdigest(),
        'model_state_sha256_before':before,'model_state_sha256_after':before,'model_unchanged':True,
        'normalization_unchanged':True,'prior_unchanged':True,'CPU_python_numpy_torch_RNG_unchanged':True,
        'CUDA_RNG_not_accessed':True,'parameter_grad_None_before_and_after':True,
        'normalization':meta['normalization'],'prior':meta['prior'],'checkpoint':meta['final_checkpoint'],
        'CONTROL_analysis_rows':0,'shared_loader_may_decode_qualified_CONTROL':True,
        'role':'raw_eval_gradients_at_frozen_FINAL_not_actual_AdamW_or_historical_training_causality'}


def run_shared_gradient(plan_sha):
    began=time.monotonic();common.firewall_state()
    cap=common.require_capped_cpu_research_producer_execution();lock=common.require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==12*1024**3 and cap['swap']==512*1024**2
        and cap['pids']==64 and cap['numerical_threads']==1
        and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):]
        and signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'SHARED_CAP_LIMITS')
    plan,plan_ref,policy_ref=inspect_shared_gradient(plan_sha,admitted=True)
    SHARED_GRADIENT_ROOT.mkdir(exist_ok=False)
    claim=publish(SHARED_GRADIENT_ROOT/'CLAIM.json',{'stage':SHARED_GRADIENT_STAGE,'status':'CLAIMED_ONE_USE',
        'plan':plan_ref,'policy_at_claim':policy_ref,'max_invocations':1,'remaining_invocations':0,'test_data_used':False})
    old=signal.getsignal(signal.SIGALRM);result=None
    def expired(*args):raise TimeoutError('NATIVE_M5_SHARED_GRADIENT_INTERNAL_870S')
    try:
        remaining=870-(time.monotonic()-began);need(remaining>0,'SHARED_TIME_BEFORE_DECODE')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
        report=qualify_shared_gradient(plan);shared_gradient_checked_inputs(plan)
        need(shared_gradient_source_clean(plan)and ref(POLICY)==policy_ref
            and ref(SHARED_GRADIENT_PLAN)==plan_ref,'SHARED_SOURCE_POLICY_BOOKEND')
        result=publish(SHARED_GRADIENT_ROOT/'RESULT.json',{'stage':SHARED_GRADIENT_STAGE,
            'status':'COMPLETE_NATIVE_M5_TA_SHARED_GRADIENT','plan':plan_ref,'claim':claim,'report':report,
            'source_unchanged':True,'training_authorized':False,'test_data_used':False,'economic_edge_proven':False,
            'optimizer_steps':0,'model_forwards':4,'autograd_calls':8,
            'clean_process_io_firewall':common.firewall_state(),'elapsed_seconds':time.monotonic()-began})
        terminal=publish(SHARED_GRADIENT_ROOT/'TERMINAL.json',{'stage':SHARED_GRADIENT_STAGE,'status':'COMPLETE',
            'exit_code':0,'plan':plan_ref,'claim':claim,'result':result,'remaining_invocations':0,
            'relaunch_allowed':False,'test_data_used':False})
        return {'result':result,'terminal':terminal}
    except BaseException as error:
        publish(SHARED_GRADIENT_ROOT/'TERMINAL.json',{'stage':SHARED_GRADIENT_STAGE,'status':'FAILED_CONSUMED',
            'exit_code':1,'plan':plan_ref,'claim':claim,'result':result,'error':type(error).__name__+':'+str(error)[:2000],
            'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


def shared_gradient_cli(args):
    need(args.plan==str(SHARED_GRADIENT_PLAN)and args.out_dir==str(SHARED_GRADIENT_ROOT)
        and args.device=='cpu','SHARED_CLI_SCOPE')
    if args.preflight:
        _,plan_ref,policy_ref=inspect_shared_gradient(args.plan_sha256,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_NATIVE_M5_TA_SHARED_GRADIENT_METADATA_ONLY','plan':plan_ref,
            'policy':policy_ref,'claim_created':False,'feature_values_decoded':0,'target_values_decoded':0,
            'checkpoint_decodes':0,'model_constructions':0,'optimizer_constructions':0,'optimizer_steps':0}))
    else:print(json.dumps(run_shared_gradient(args.plan_sha256),sort_keys=True))


PRIMARY_BACKBONE_GRADIENT_POLICY='m5_close_only_backbone_v1'
PRIMARY_GRADIENT_DESIGN_REFERENCE={
    'gradient_result':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_TA_SHARED_GRADIENT_20261119_001/RESULT.json',
        'sha256':'c9f90357f73ca1d4ac1bb214ad885100f806d06a0b53e444657daa23a1cdeb76'},
    'gradient_terminal':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_TA_SHARED_GRADIENT_20261119_001/TERMINAL.json',
        'sha256':'20546b6e7010f8127b13f5d16584b76725d86d99912377be2464ece01b84be87'},
    'gradient_review':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_TA_SHARED_GRADIENT_20261119_001/MATCHED_GRADIENT_REVIEW.json',
        'sha256':'ac838e47c54cfe40b767714937f99920582a599bbc79b690d3927c548c608e02'}}
PRIMARY_FIT_SOURCE_PATHS=tuple(dict.fromkeys((*TA_FIT_SOURCE_PATHS,'tests/test_market_state_ta_integrated_model_v1.py')))


def _primary_fit(plan):
    return plan.get('mode')in('ta_primary_mechanics','ta_primary_main','ta_noise_mechanics','ta_noise_main','ta_larger_mechanics','ta_larger_main','ta_five_year_mechanics','ta_five_year_main')


def primary_model_source_identity(source):
    import ast
    tree=ast.parse(source)
    top={'native_bar_backbone_gradient_policy_contract','_require_backbone_gradient_policy',
        'native_bar_model_contract','build_native_bar_direction_model'}
    methods={'__init__','research_objective','_require_binding','load_state_dict','forward'}
    changed={}
    retained=[]
    for node in tree.body:
        if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='M5_CLOSE_ONLY_BACKBONE_GRADIENT_POLICY'for t in node.targets):continue
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))and node.name in top:
            changed[node.name]=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest();continue
        if isinstance(node,ast.ClassDef)and node.name=='NativeM5BarDirectionModel':
            body=[]
            for method in node.body:
                if isinstance(method,(ast.FunctionDef,ast.AsyncFunctionDef))and method.name in methods:
                    changed[node.name+'.'+method.name]=hashlib.sha256(ast.dump(method,include_attributes=False).encode()).hexdigest()
                else:body.append(method)
            node.body=body
        retained.append(node)
    tree.body=retained
    return {'unchanged_model_ast_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
        'reviewed_policy_definition_sha256':changed}


def primary_driver_source_identity(source):
    import ast
    tree=ast.parse(source)
    # Strip only the exact opt-in factory/count instrumentation and the two
    # non-mutating autograd probes; the existing optimizer/clip/loss math remains.
    for node in tree.body:
        if not isinstance(node,ast.FunctionDef):continue
        if node.name=='fresh_fit_model':
            node.body=[item for item in node.body if not(isinstance(item,ast.If)
                and isinstance(item.test,ast.Call)and isinstance(item.test.func,ast.Name)and item.test.func.id=='_primary_fit')]
        if node.name=='fit_step':
            node.body=[item for item in node.body if not(isinstance(item,ast.Assign)
                and any(isinstance(target,ast.Name)and target.id=='primary_probe'for target in item.targets))
                and not(isinstance(item,ast.If)and isinstance(item.test,ast.Compare)
                    and any(isinstance(value,ast.Constant)and value.value=='ta_primary_mechanics'for value in item.test.comparators))]
            for item in node.body:
                if isinstance(item,ast.Return)and isinstance(item.value,ast.Dict):
                    pairs=[(key,value)for key,value in zip(item.value.keys,item.value.values)
                        if not(key is None and isinstance(value,ast.Name)and value.id=='primary_probe')]
                    item.value.keys=[key for key,_ in pairs];item.value.values=[value for _,value in pairs]
    orchestration={'_ta_fit','_ta_main','_mechanics','fit_recipe','fit_metadata','fit_limits','build_fit_plan',
        'fit_source_clean','fit_boot_state','_ta_training_batches','_ta_schedule_position','prepare_native_fit',
        'fit_physical','run_native_fit','_primary_fit','primary_model_source_identity','primary_driver_source_identity',
        'primary_source_bridge','primary_backbone_metadata','prepare_primary_native_fit','primary_backbone_probe'}
    tree.body=[node for node in tree.body if not(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))and node.name in orchestration)
        and not(isinstance(node,ast.Assign)and any(isinstance(target,ast.Name)and target.id in(
            'FIT_SCOPES','PRIMARY_BACKBONE_GRADIENT_POLICY','PRIMARY_GRADIENT_DESIGN_REFERENCE','PRIMARY_FIT_SOURCE_PATHS')for target in node.targets))]
    leaves={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    names=('fresh_fit_model','fit_step','fit_score','prepare_ta_native_fit','fit_axes','seed_native_fit',
        'training_rng','restore_training_rng','state_sha','slice_truth','fit_checkpoint','publish_checkpoint',
        'restore_fit_checkpoint','fit_state_proof','_ta_initial_prior_proof','cold_resume_fit','require_cold_resume_proof')
    return {'unchanged_driver_ast_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
        'unchanged_numerical_leaf_sha256':{name:hashlib.sha256(ast.dump(leaves[name],include_attributes=False).encode()).hexdigest()for name in names}}


def primary_source_bridge(parent):
    driver='gx1/audit/qualify_native_m5_full_gc_direction_v1.py';model='gx1/contracts/market_state_ta_integrated_model_v1.py'
    model_test='tests/test_market_state_ta_integrated_model_v1.py'
    allowed={driver,model,'tests/test_qualify_native_m5_full_gc_direction_v1.py',model_test,
        'scripts/gx1_capped_run.sh','tests/test_gx1_capped_run_contract.py'}
    need(parent['stage']==SHARED_GRADIENT_STAGE and set(parent['source_owners'])==set(TA_FIT_SOURCE_PATHS),
        'PRIMARY_DESIGN_SOURCE_SET')
    current={name:sha(REPO/name)for name in PRIMARY_FIT_SOURCE_PATHS}
    need(all(current[name]==value for name,value in parent['source_owners'].items()if name not in allowed),
        'PRIMARY_UNCHANGED_OTHER_NUMERICAL_OWNERS')
    originals={}
    for name in (driver,model,model_test):
        raw=subprocess.check_output(['git','show',parent['source_base_commit']+':'+name],cwd=REPO)
        if name in parent['source_owners']:need(hashlib.sha256(raw).hexdigest()==parent['source_owners'][name],'PRIMARY_ORIGINAL_BLOB:'+name)
        originals[name]=raw
    driver_before=primary_driver_source_identity(originals[driver].decode())
    need(driver_before==primary_driver_source_identity((REPO/driver).read_text()),'PRIMARY_UNCHANGED_DRIVER_MATH')
    old_model=primary_model_source_identity(originals[model].decode());new_model=primary_model_source_identity((REPO/model).read_text())
    need(old_model['unchanged_model_ast_sha256']==new_model['unchanged_model_ast_sha256'],'PRIMARY_MODEL_POLICY_SCOPE')
    previous={**parent['source_owners'],model_test:hashlib.sha256(originals[model_test]).hexdigest()}
    return {'schema_version':'gx1_native_m5_primary_backbone_source_bridge_v1','basis_plan_stage':parent['stage'],
        'original_source_base_commit':parent['source_base_commit'],'allowed_changed_source_paths':sorted(allowed),
        'all_other_source_owners_exact':True,**driver_before,
        'unchanged_model_ast_sha256':old_model['unchanged_model_ast_sha256'],
        'reviewed_model_policy_definitions':{'original':old_model['reviewed_policy_definition_sha256'],
            'current':new_model['reviewed_policy_definition_sha256']},
        'source_changes':{name:{'original_sha256':previous[name],'current_sha256':current[name]}
            for name in allowed if previous[name]!=current[name]},
        'scope':'one_explicit_detach_at_auxiliary_readout_policy_and_driver_orchestration_probes_only'}


def primary_backbone_metadata(inputs,mode):
    base={**SHARED_GRADIENT_PARENT,**PRIMARY_GRADIENT_DESIGN_REFERENCE,'matched_prior_reference':TA_MATCHED_PRIOR_REFERENCE}
    fields=set(base)|({'primary_mechanics_result','primary_mechanics_terminal'}if mode=='ta_primary_main'else set())
    need(type(inputs)is dict and set(inputs)==fields and all(inputs[key]==value for key,value in base.items()),
        'PRIMARY_EXACT_PARENT_REFS')
    block,terminal,review=(read(inputs[key])for key in SHARED_GRADIENT_PARENT)
    parent,claim=read(block['plan']),read(block['claim']);report=block['report']
    need(block['status']=='COMPLETE_NATIVE_M5_TA_PRIOR_BLOCK_ORDER'and terminal['status']=='COMPLETE'
        and terminal['exit_code']==0 and terminal['result']==inputs['block_result']
        and terminal['plan']==block['plan']==claim['plan']and terminal['claim']==block['claim']
        and block['stage']==terminal['stage']==parent['stage']==claim['stage']==FIT_SCOPES['ta_shuffle'][0]
        and block['source_unchanged']is terminal['source_unchanged']is True
        and block['test_data_used']is terminal['test_data_used']is False
        and report['declared_passes']==4 and report['actual_counts']=={
            'model_constructions':1,'model_forwards':544,'backward_calls':384,'optimizer_steps':384}
        and report['normalization_fits']==report['prior_fits']==0
        and review['result']==inputs['block_result']and review['terminal']==inputs['block_terminal']
        and review['normalization']==report['normalization']and review['prior']==report['prior'],'PRIMARY_BLOCK_PARENT')
    result,done,audit=(read(inputs[key])for key in PRIMARY_GRADIENT_DESIGN_REFERENCE)
    design,design_claim=read(result['plan']),read(result['claim'])
    need(result['status']=='COMPLETE_NATIVE_M5_TA_SHARED_GRADIENT'and done['status']=='COMPLETE'and done['exit_code']==0
        and done['result']==inputs['gradient_result']and done['plan']==result['plan']==design_claim['plan']
        and done['claim']==result['claim']and result['stage']==done['stage']==design['stage']==design_claim['stage']==SHARED_GRADIENT_STAGE
        and result['source_unchanged']is True and result['test_data_used']is done['test_data_used']is False
        and result['optimizer_steps']==0 and result['report']['actual_counts']==SHARED_GRADIENT_LIMITS
        and design['inputs']==SHARED_GRADIENT_PARENT
        and result['report']['model_state_sha256_before']==result['report']['model_state_sha256_after']==report['final_state_sha256']
        and audit['schema_version']=='gx1_native_m5_ta_shared_gradient_review_v1'
        and audit['result']==inputs['gradient_result']and audit['terminal']==inputs['gradient_terminal']
        and audit['metadata_and_scalar_arithmetic_independently_checked']is True
        and audit['model_statistics_RNG_grad_fields_unchanged']is True and audit['test_data_used']is False,
        'PRIMARY_RAW_GRADIENT_DESIGN_EVIDENCE')
    dependencies={**parent['metadata']['dependencies'],'block_plan':block['plan'],'block_claim':block['claim'],
        'gradient_plan':result['plan'],'gradient_claim':result['claim'],
        'train_normalization':report['normalization'],'train_prior':report['prior']}
    metadata={**parent['metadata'],'dependencies':dependencies,'parent_fit_plan':parent,
        'primary_backbone_gradient_policy':PRIMARY_BACKBONE_GRADIENT_POLICY,
        'primary_source_compatibility':primary_source_bridge(design),
        'block_order_reference':{key:inputs[key]for key in SHARED_GRADIENT_PARENT},
        'raw_gradient_design_evidence':{key:inputs[key]for key in PRIMARY_GRADIENT_DESIGN_REFERENCE}}
    if mode=='ta_primary_main':
        stage,_,root=FIT_SCOPES['ta_primary_mechanics']
        need(inputs['primary_mechanics_result']['path']==str(root/'RESULT.json')
            and inputs['primary_mechanics_terminal']['path']==str(root/'TERMINAL.json'),'PRIMARY_MECHANICS_PATH')
        trained,last=read(inputs['primary_mechanics_result']),read(inputs['primary_mechanics_terminal'])
        mechanics,mc=read(trained['plan']),read(trained['claim']);proof=trained['report']
        need(trained['status']=='COMPLETE_NATIVE_M5_TA_PRIMARY_BACKBONE_MECHANICS'
            and trained['stage']==last['stage']==mechanics['stage']==mc['stage']==stage
            and last['status']=='COMPLETE'and last['exit_code']==0 and last['result']==inputs['primary_mechanics_result']
            and last['plan']==trained['plan']==mc['plan']and last['claim']==trained['claim']
            and trained['source_unchanged']is last['source_unchanged']is True
            and trained['test_data_used']is last['test_data_used']is False
            and proof['cold_resume']['bitexact']is True and proof['actual_counts']['autograd_calls']==6
            and proof['primary_backbone_qualification']['auxiliary_shared_zero_all_steps']is True
            and proof['primary_backbone_qualification']['M5_shared_positive_after_first_head_update']is True
            and proof['primary_backbone_qualification']['supported_auxiliary_heads_positive_all_steps']is True
            and proof['normalization_fits']==proof['prior_fits']==0
            and proof['normalization']==report['normalization']and proof['prior']==report['prior']
            and mechanics['metadata']==metadata and all(mechanics['inputs'][key]==value for key,value in base.items())
            and set(mechanics['source_owners'])==set(PRIMARY_FIT_SOURCE_PATHS)
            and all(sha(REPO/name)==value for name,value in mechanics['source_owners'].items()),'PRIMARY_COMPLETED_MECHANICS')
        metadata['mechanics']={'result':inputs['primary_mechanics_result'],'terminal':inputs['primary_mechanics_terminal'],
            'plan':trained['plan'],'claim':trained['claim'],'normalization':proof['normalization'],'prior':proof['prior'],
            'axis':proof['axis'],'normalization_population':proof['normalization_population'],
            'initial_state_sha256':proof['initial_state_sha256'],'physical_windows_boot_id':trained['physical_windows_boot_id']}
        dependencies.update(primary_mechanics_plan=trained['plan'],primary_mechanics_claim=trained['claim'])
    return metadata


def prepare_primary_native_fit(plan,root,*,reload_normalization=None,reload_prior=None):
    context=prepare_native_fit(plan['metadata']['parent_fit_plan'],root,
        reload_normalization=reload_normalization,reload_prior=reload_prior)
    need(context.get('normalization_fits',0)==context.get('prior_fits',0)==0,'PRIMARY_NO_STATISTICS_REFIT')
    if plan['mode']=='ta_primary_main':
        parent=plan['metadata']['mechanics']
        need(context['normalization_ref']==parent['normalization']and context['prior_ref']==parent['prior']
            and context['axis']==parent['axis'],'PRIMARY_IMMUTABLE_MECHANICS_INPUTS')
    return context


def primary_backbone_probe(model,loss,features):
    import torch
    from gx1.contracts.market_state_native_m5_feature_inputs_v1 import TIMEFRAMES
    shared=tuple(p for p in model.backbone.parameters()if p.requires_grad)
    auxiliary=tuple(p for task,heads in (('open',model.open_heads),('close',model.close_heads))
        for tf in TIMEFRAMES if not(task=='close'and tf=='M5')for p in heads[tf].parameters()if p.requires_grad)
    primary=tuple(p for p in model.close_heads['M5'].parameters()if p.requires_grad)
    n=int(loss['counts']['open'].sum()+loss['counts']['close'].sum())
    need(n>0 and shared and auxiliary and primary,'PRIMARY_PROBE_PARAMETER_SUPPORT')
    laux=(loss['loss_sums']['open'].sum()+loss['loss_sums']['close'][1:].sum())/n
    l5=loss['loss_sums']['close'][0]/n
    aux=torch.autograd.grad(laux,shared+auxiliary,retain_graph=True,allow_unused=True)
    model._native_fit_counts['autograd_calls']+=1
    input_names=tuple(sorted(key for key,value in features.items()if value.is_floating_point()))
    inputs=tuple(features[key]for key in input_names)
    m5=torch.autograd.grad(l5,shared+primary+inputs,retain_graph=True,allow_unused=True)
    model._native_fit_counts['autograd_calls']+=1
    def l2(values):
        need(all(v is None or bool(torch.isfinite(v).all())for v in values),'PRIMARY_PROBE_FINITE')
        return float(torch.sqrt(sum((v.detach().double().square().sum()for v in values if v is not None),
            torch.zeros((),device=primary[0].device))))
    result={'auxiliary_shared_gradient_l2':l2(aux[:len(shared)]),'auxiliary_head_gradient_l2':l2(aux[len(shared):]),
        'M5_shared_gradient_l2':l2(m5[:len(shared)]),'M5_head_gradient_l2':l2(m5[len(shared):len(shared)+len(primary)]),
        'input_key_gradient_l2':{key:l2((value,))for key,value in zip(input_names,m5[len(shared)+len(primary):])},
        'auxiliary_predictive_cells':n-int(loss['counts']['close'][0]),
        'M5_predictive_cells':int(loss['counts']['close'][0]),'autograd_probe_calls':2}
    need(all(value is None or int(torch.count_nonzero(value))==0 for value in aux[:len(shared)])
        and result['auxiliary_shared_gradient_l2']==0
        and(result['auxiliary_head_gradient_l2']>0 if result['auxiliary_predictive_cells']>0 else result['auxiliary_head_gradient_l2']==0)
        and all(p.grad is None for p in model.parameters()),'PRIMARY_AUXILIARY_ROUTE_OR_GRAD_ACCUMULATION')
    return {'primary_gradient_routing':result}




# Explicit net-change noise target route. Raw and market-range routes stay distinct.
NOISE_PREPARE_STAGE='native_m5_ta_atr03_prepare_20261120_002'
NOISE_PREPARE_PLAN=REPO/'configs/research/NATIVE_M5_TA_ATR03_PREPARE_20261120_002.json'
NOISE_PREPARE_ROOT=Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_PREPARE_20261120_002')
NOISE_SOURCE_PATHS=tuple(dict.fromkeys((*TA_FIT_SOURCE_PATHS,
    'tests/test_market_state_native_m5_targets_v1.py',
    'tests/test_market_state_ta_integrated_objective_v1.py',
    'tests/test_market_state_ta_integrated_model_v1.py')))
NOISE_PREPARE_EXECUTION={'class':'producer','device':'cpu','memory':'12G','swap':'512M',
    'outer_seconds':1800,'internal_seconds':1770,'kill_after_seconds':10,'numerical_threads':1,
    'model_constructions':1,'model_forwards':32,'optimizer_steps':0,'prior_fits':1,'normalization_fits':0}
FIT_SCOPES.update({
    'ta_noise_mechanics':('native_m5_ta_atr03_mechanics_20261120_001',
        REPO/'configs/research/NATIVE_M5_TA_ATR03_MECHANICS_20261120_001.json',
        Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_MECHANICS_20261120_001')),
    'ta_noise_main':('native_m5_ta_atr03_fit_20261120_001',
        REPO/'configs/research/NATIVE_M5_TA_ATR03_FIT_20261120_001.json',
        Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_FIT_20261120_001'))})


def _noise_fit(plan):
    return plan.get('mode')in('ta_noise_mechanics','ta_noise_main','ta_larger_mechanics','ta_larger_main','ta_larger_prepare','ta_five_year_prepare','ta_five_year_mechanics','ta_five_year_main')


def noise_target_kwargs(plan):
    return {'target_contract':plan['metadata']['target_contract']}if _noise_fit(plan)else{}


def build_noise_prepare_plan(*,inputs,margin_atr,source_base_commit=None):
    from gx1.contracts.market_state_native_m5_targets_v1 import noise_margin_dataset_contract
    fields={'reference_fit_plan','normalization','source_qualification','comparison_review','raw_target_result','raw_target_tape','reused_noise_dataset','previous_prepare_terminal'}
    need(type(inputs)is dict and set(inputs)==fields,'NOISE_PREPARE_INPUT_ROLES')
    base=read(inputs['reference_fit_plan']);normalization=read(inputs['normalization'])
    qualification=read(inputs['source_qualification']);comparison=read(inputs['comparison_review'])
    raw=read(inputs['raw_target_result']);proof=read(qualification['genuine_TRAIN_proof']['result'])
    need(proof['status']=='PASS_GENUINE_PRODUCER_API'and proof['raw_target_tape']==inputs['raw_target_tape']
        and proof['raw_target_result']==inputs['raw_target_result'],'NOISE_QUALIFIED_RAW_SOURCE')
    from gx1.contracts.entry_model_native_input_normalization_v1 import require_native_m5_input_normalization_contract
    require_native_m5_input_normalization_contract(normalization)
    snapshot=read(base['metadata']['ta_plan']['inputs']['candidate_snapshot'])
    need(normalization['source_axes_sha256']==snapshot['surfaces_metadata']['axes']['contract_sha256'],
        'NOISE_PHYSICAL_FEATURE_NORMALIZATION_IDENTITY')
    need(base['mode']=='ta_primary_main'and base['GC_used']is False
        and base['metadata']['dependencies']['target_tape']==inputs['raw_target_tape']
        and base['metadata']['dependencies']['train_normalization']==inputs['normalization']
        and qualification['genuine_TRAIN_proof']['result']['path'].endswith('/NATIVE_BAR_NOISE_MARGIN_TRAIN_PROOF.json')
        and comparison['status']=='PASS_TRAIN_DESCRIPTOR_ONLY_NOT_MODEL_BOUND'
        and margin_atr==comparison['first_candidate_margin_atr']==0.3,'NOISE_PREPARE_BOUND_CANDIDATE')
    need(normalization['fit_population']['issue_rows']==24175
        and normalization['fit_population']['cutoff_ns']==FIT_END_NS
        and raw['target_contract']==base['metadata']['target_contract']
        and raw['target_tape']==inputs['raw_target_tape'],'NOISE_PREPARE_RAW_NORM_SCOPE')
    ta_plan=base['metadata']['ta_plan']
    need(ta_audit.candidate_metadata(ta_plan['inputs'],ta_plan['feature_contract'])==ta_plan['candidate_requalification'],
        'NOISE_PROVIDER_RELOAD_METADATA_GATE_BEFORE_CLAIM')
    previous=read(inputs['previous_prepare_terminal']);old_plan=read(previous['plan']);old_claim=read(previous['claim'])
    dataset=read(inputs['reused_noise_dataset'])
    need(previous['stage']==old_plan['stage']==old_claim['stage']=='native_m5_ta_atr03_prepare_20261120_001'
        and previous['status']=='FAILED_CONSUMED'and previous['exit_code']==1 and previous['result']is None
        and previous['remaining_invocations']==0 and previous['relaunch_allowed']is False
        and previous['error']=='ValueError:NATIVE_M5_PIPELINE_CANDIDATE_NUMERIC_OWNER_CHANGED:gx1/contracts/market_state_native_m5_feature_inputs_v1.py'
        and old_claim['plan']==previous['plan']and old_plan['inputs']['raw_target_tape']==inputs['raw_target_tape']
        and inputs['reused_noise_dataset']['path']==str(Path(old_plan['output_root'])/'NOISE_TARGETS.json')
        and dataset['raw_target_tape']==inputs['raw_target_tape']and dataset['raw_fields_byte_identical']is True
        and dataset['contract']==noise_margin_dataset_contract(margin_atr)
        and all(old_plan['source_owners'][name]==sha(REPO/name)for name in (
            'gx1/contracts/market_state_native_m5_targets_v1.py','gx1/features/technical_indicators_v1.py','gx1/features/htf_features.py'))
        and dataset['cutoff_ns']==raw['source_spec']['cutoff_ns']and dataset['issue_rows']==raw['report']['selected_issue_rows']
        and set(dataset['headers'])==set(dataset['contract']['required_array_fields'])|{'source_row'}
        and ref(dataset['arrays']['path'])==dataset['arrays'],'NOISE_REUSE_COMPLETE_TARGETS_FROM_PRE_MODEL_FAILURE')
    metadata={'ta_plan':base['metadata']['ta_plan'],'architecture':base['metadata']['architecture'],
        'seed':base['recipe']['seed'],'target_contract':noise_margin_dataset_contract(margin_atr),
        'raw_target_contract':raw['target_contract'],'raw_source_spec':raw['source_spec'],
        'source_pool':base['metadata']['source_pool'],
        'fit_start_ns':FIT_START_NS,'fit_end_ns':FIT_END_NS,'control_end_ns':CONTROL_END_NS,
        'normalization_population':normalization['fit_population'],
        'dependencies':dict(base['metadata']['ta_plan']['inputs']),
        'scope':'all_preTEST_raw_issue_noise_projection_plus_fixed_1536FIT_512CONTROL_fresh_initialbaseline',
        'full_market_path_FLAT_qualified':False,'six_year_model_or_norm_qualified':False}
    return {'schema_version':'gx1_native_m5_noise_prepare_plan_v1','stage':NOISE_PREPARE_STAGE,
        'status':'PREPARED_NOT_ADMITTED','output_root':str(NOISE_PREPARE_ROOT),
        'source_base_commit':source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'source_owners':{name:sha(REPO/name)for name in NOISE_SOURCE_PATHS},
        'inputs':inputs,'metadata':metadata,'margin_atr':margin_atr,
        'cohort_selection':small_cohort_manifest(metadata),
        'execution':dict(NOISE_PREPARE_EXECUTION),'training_authorized':False,'test_data_used':False,
        'economic_edge_proven':False}


def noise_source_clean(plan):
    need(set(plan['source_owners'])==set(NOISE_SOURCE_PATHS),'NOISE_SOURCE_SET')
    adapted={**plan,'source_owners':{name:plan['source_owners'][name]for name in ta_audit.SOURCE_PATHS}}
    return ta_audit.source_clean(adapted)and all(sha(REPO/name)==value for name,value in plan['source_owners'].items())


def noise_fold_truth(truth,upper):
    value=ta_audit.fold_truth(truth,upper);mature=value['mature']
    value['m5_count'][~mature]=0;value['source_gap'][~mature]=False
    for task in ('open','close'):
        value['noise_'+task+'_known']&=value['eligible']
        value['noise_'+task+'_eligible']=value['noise_'+task+'_known']&(
            ~value['open_known_at_issue']if task=='open'else True)
        value['noise_'+task+'_sign'][~value['noise_'+task+'_known']]=0
        value['noise_'+task+'_delta'][~value['eligible']]=np.nan
    return value


def noise_fit_axes(provider,dataset,dataset_contract,*,fold_bounds=None):
    from gx1.contracts.market_state_ta_integrated_objective_v1 import _native_truth
    bounds=(FIT_START_NS,FIT_END_NS,CONTROL_END_NS)if fold_bounds is None else fold_bounds
    need(type(bounds)is tuple and len(bounds)==3 and all(type(x)is int and x>0 and x%300_000_000_000==0 for x in bounds)
        and bounds[0]<bounds[1]<bounds[2],'NOISE_FOLD_BOUNDS')
    start,cut,stop=bounds
    need(ref(dataset['arrays']['path'])==dataset['arrays'],'NOISE_DATASET_HASH')
    wanted=np.flatnonzero((provider.issue_ns>=start)&(provider.issue_ns<stop))
    with np.load(dataset['arrays']['path'],allow_pickle=False)as archive:
        need(set(archive.files)==set(dataset_contract['required_array_fields'])|{'source_row'},'NOISE_ARRAY_INVENTORY')
        times=archive['issue_ns'];sources=archive['source_row'];at=np.searchsorted(times,provider.issue_ns[wanted])
        source_header=dataset['headers']['source_row']
        need(sources.dtype==np.int64 and sources.shape==(dataset['issue_rows'],)
            and sources.dtype.str==source_header['dtype']and list(sources.shape)==source_header['shape']
            and digest(sources)==source_header['sha256']and np.array_equal(sources,np.arange(len(sources),dtype=np.int64)),
            'NOISE_FULL_NATIVE_SOURCE_ROW_AXIS')
        need(np.all(at<len(times))and np.array_equal(times[at],provider.issue_ns[wanted])
            and np.array_equal(sources[at],provider.source_row[wanted]),'NOISE_DATASET_SOURCE_JOIN')
        truth={}
        for key in dataset_contract['required_array_fields']:
            array=archive[key];header=dataset['headers'][key]
            need(array.dtype.str==header['dtype']and list(array.shape)==header['shape']
                and digest(array)==header['sha256'],'NOISE_ARRAY_BYTES:'+key)
            truth[key]=array
    _native_truth(truth,rows=dataset['issue_rows'],fold_upper_ns=dataset['cutoff_ns'],
        target_contract_sha256=dataset_contract['contract_sha256'],target_contract=dataset_contract)
    truth={key:value[at]for key,value in truth.items()}
    cohorts={};report={}
    for name,start,end in [('TRAIN',start,cut),('CONTROL',cut,stop)]:
        chosen=(provider.issue_ns[wanted]>=start)&(provider.issue_ns[wanted]<end)
        positions=wanted[chosen];ready=provider.input_ready[positions]
        full=noise_fold_truth({key:value[chosen]for key,value in truth.items()},end)
        outcomes={key:value[ready]for key,value in full.items()};rows=provider.source_row[positions][ready]
        _native_truth(outcomes,rows=len(rows),fold_upper_ns=end,
            target_contract_sha256=dataset_contract['contract_sha256'],target_contract=dataset_contract)
        supervised=(outcomes['noise_open_eligible']|outcomes['noise_close_eligible']).any(1)
        take=supervised if name=='TRAIN'else np.ones(len(rows),bool)
        fit_rows=rows[take];fit_truth={key:value[take]for key,value in outcomes.items()}
        need(len(rows)>=16 and np.all(np.diff(rows)>0),'NOISE_FIT_AXIS')
        cohorts[name]={'rows':rows,'truth':outcomes,'fit_rows':fit_rows,'fit_truth':fit_truth,'upper_ns':end}
        report[name]={'start_ns':start,'end_exclusive_ns':end,'base_issue_rows':len(positions),
            'input_ready_rows':len(rows),'input_not_ready_rows':int((~ready).sum()),
            'no_mature_supervision_rows':int((~supervised).sum()),'fit_rows':len(fit_rows),
            'source_rows_sha256':digest(rows),'fit_source_rows_sha256':digest(fit_rows),
            'issue_ns_sha256':digest(outcomes['issue_ns']),'fit_issue_ns_sha256':digest(fit_truth['issue_ns']),
            'per_tf_mature':outcomes['mature'].sum(0).tolist(),
            'per_tf_eligible':outcomes['eligible'].sum(0).tolist(),'unknown_GC_rows_removed':0,
            'per_tf_noise_open_eligible':outcomes['noise_open_eligible'].sum(0).tolist(),
            'per_tf_noise_close_eligible':outcomes['noise_close_eligible'].sum(0).tolist()}
    return cohorts,report


def noise_prepare_physical(plan,root):
    import dataclasses
    from types import MappingProxyType
    import torch
    from gx1.contracts import market_state_native_m5_targets_v1 as targets
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_bar_direction_model
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    meta=plan['metadata'];target_contract=meta['target_contract'];began=time.monotonic()
    if 'reused_noise_dataset'in plan['inputs']:
        dataset_ref=plan['inputs']['reused_noise_dataset'];dataset=read(dataset_ref)
        need(dataset['contract']==target_contract and dataset['raw_target_tape']==plan['inputs']['raw_target_tape']
            and ref(dataset['arrays']['path'])==dataset['arrays'],'NOISE_REUSED_DATASET_BINDING')
    else:
        with np.load(plan['inputs']['raw_target_tape']['path'],allow_pickle=False)as z:
            def owned(key):
                a=z[key];a.flags.writeable=False;return a
            bars={tf:targets.NativeTFBarDirections(**{f.name:owned('bars_'+tf+'_'+f.name)
                  for f in dataclasses.fields(targets.NativeTFBarDirections)})for tf in targets.TIMEFRAMES}
            tape=targets.NativeM5Targets(issue_ns=owned('issue_ns'),source_row=owned('source_row'),
                ohlc=owned('ohlc'),source_gap=owned('source_gap'),bars_by_timeframe=MappingProxyType(bars),
                source_ref=MappingProxyType(meta['raw_source_spec']['manifest']),cutoff_ns=meta['raw_source_spec']['cutoff_ns'])
            truth=tape.gather_next_bar_noise_margin(tape.issue_ns,fold_upper_ns=tape.cutoff_ns,margin_atr=plan['margin_atr'])
            need(truth.pop('noise_margin_contract')==target_contract['noise_margin_contract'],'NOISE_PROJECTION_CONTRACT')
            rawkeys=[key for key in target_contract['required_array_fields']if not key.startswith('noise_')]
            for key in rawkeys:
                a=z['future_'+key];need(a.dtype==truth[key].dtype and a.shape==truth[key].shape
                    and digest(a)==digest(truth[key]),'NOISE_RAW_FIELD_PRESERVED:'+key)
            need(set(truth)==set(target_contract['required_array_fields']),'NOISE_PROJECTION_INVENTORY')
            projected={**truth,'source_row':tape.source_row}
            headers={key:{'dtype':a.dtype.str,'shape':list(a.shape),'sha256':digest(a)}for key,a in projected.items()}
            stream=io.BytesIO();np.savez_compressed(stream,**projected)
            array_ref=publish(root/'NOISE_TARGETS.npz',stream.getvalue());stream.close()
            dataset={'schema_version':'gx1_native_m5_noise_dataset_artifact_v1','contract':target_contract,
                'raw_target_tape':plan['inputs']['raw_target_tape'],'arrays':array_ref,'headers':headers,
                'cutoff_ns':tape.cutoff_ns,'issue_rows':len(tape.issue_ns),
                'raw_fields_byte_identical':True,'model_input_fields_added':0,'test_data_used':False,
                'per_TF':{tf:{task:{
                    'raw_eligible_endpoints':int(truth['eligible'][:,j].sum()),
                    'noise_known':int(truth['noise_'+task+'_known'][:,j].sum()),
                    'noise_predictive':int(truth['noise_'+task+'_eligible'][:,j].sum()),
                    'raw_eligible_without_known_noise_target':int((truth['eligible'][:,j]&~truth['noise_'+task+'_known'][:,j]).sum()),
                    'predictive_counts_down_neutral_up':np.bincount(
                        truth['noise_'+task+'_sign'][truth['noise_'+task+'_eligible'][:,j],j].astype(np.int64)+1,minlength=3).tolist()
                }for task in('open','close')}for j,tf in enumerate(targets.TIMEFRAMES)}}
            dataset_ref=publish(root/'NOISE_TARGETS.json',dataset)
        del tape,bars,truth,projected
    projection_seconds=time.monotonic()-began
    provider,admission=ta_audit.load_candidate_provider(meta['ta_plan'])
    cohorts,axis=noise_fit_axes(provider,dataset,target_contract)
    need(np.array_equal(cohorts['TRAIN']['rows'],cohorts['TRAIN']['fit_rows']),'NOISE_FULL_TRAIN_NORM_AXIS')
    for name,item in meta['source_pool']['axis'].items():
        need(all(axis[name][key]==value for key,value in item.items()),'NOISE_ORIGINAL_SOURCE_POOL:'+name)
    normalization=read(plan['inputs']['normalization']);provider.input_binding(normalization)
    need(normalization['fit_population']['selected_source_rows_sha256']==digest(cohorts['TRAIN']['fit_rows']),
        'NOISE_REUSED_EXACT_TRAIN_NORMALIZATION')
    context={'provider':provider,'cohorts':cohorts,'axis':axis}
    selected,selection=selected_small_cohorts(plan,context)
    prior=objective.fit_native_bar_train_prior(truth=slice_truth(cohorts['TRAIN'],selected['FIT'],fit=True),
        fold_upper_ns=FIT_END_NS,target_contract_sha256=target_contract['contract_sha256'],target_contract=target_contract)
    prior_ref=publish(root/'MATCHED_FIT_PRIOR.json',prior)
    torch.set_num_threads(1);torch.manual_seed(meta['seed'])
    model=build_native_bar_direction_model(input_normalization=normalization,
        input_binding=provider.input_binding(normalization),architecture=meta['architecture'],device=torch.device('cpu'),
        train_prior=prior,backbone_gradient_policy=PRIMARY_BACKBONE_GRADIENT_POLICY,target_contract=target_contract)
    model.require_prior_initialization_state();model.eval();initial_sha=canonical_model_state_sha256(model.state_dict())
    captures={};scores={};forwards=0
    with torch.no_grad():
        for name,rows in [('TRAIN',selected['FIT']),('CONTROL',selected['CONTROL'])]:
            outputs={'open_logits':[],'close_logits':[]}
            for batch in all_score_batches(rows,64):
                raw=provider.gather(batch,max_rows=64)
                output=model({key:torch.from_numpy(value)for key,value in raw.items()});forwards+=1
                for key,value in output.items():outputs[key].append(value.detach().numpy().copy())
            outputs={key:np.concatenate(value)for key,value in outputs.items()}
            expected=objective.native_bar_train_prior_logits(prior,rows=len(rows),target_contract=target_contract)
            need(all(np.array_equal(value,expected[key].astype(np.float32))for key,value in outputs.items()),'NOISE_INITIAL_PRIOR_PARITY')
            truth=slice_truth(cohorts[name],rows)
            scores[name]=objective.native_bar_direction_score_sums(model_output=outputs,truth=truth,
                fold_upper_ns=cohorts[name]['upper_ns'],target_contract_sha256=target_contract['contract_sha256'],target_contract=target_contract)
            captures.update({name+'__source_row':rows,**{name+'__'+key:value for key,value in outputs.items()},
                **{name+'__truth_'+key:value for key,value in truth.items()}})
    need(forwards==32 and canonical_model_state_sha256(model.state_dict())==initial_sha,'NOISE_INITIAL_STATE_AND_FORWARDS')
    model.load_state_dict(model.state_dict(),strict=True)
    stream=io.BytesIO();torch.save(model.state_dict(),stream)
    checkpoint=publish(root/'INITIAL_MODEL.pt',stream.getvalue());stream.close()
    stream=io.BytesIO();np.savez_compressed(stream,**captures)
    capture=publish(root/'INITIAL_CAPTURE.npz',stream.getvalue());stream.close()
    return {'dataset':dataset_ref,'axis':axis,'selected_cohorts':selection,
        'normalization':plan['inputs']['normalization'],'normalization_population':normalization['fit_population'],
        'prior':prior_ref,'prior_fits':1,'normalization_fits':0,'model_constructions':1,'model_forwards':forwards,
        'optimizer_steps':0,'initial_state_sha256':initial_sha,'initial_checkpoint':checkpoint,
        'initial_capture':capture,'initial_scores':scores,'all_selected_initial_logits_equal_new_noise_prior':True,
        'input_binding':provider.input_binding(normalization),'model_contract':model.research_objective,
        'feature_requalification':admission,'target_dataset_acquisition_seconds':projection_seconds,
        'noise_target_projection_recomputed':'reused_noise_dataset'not in plan['inputs'],'reused_noise_dataset':dataset_ref,
        'scope':'all_preTEST_noise_targets_and_selected1536FIT_512CONTROL_initialbaseline_only',
        'six_year_model_baseline_measured':False,'whole_market_FLAT_qualified':False,'economic_edge_proven':False}


def inspect_noise_prepare(plan_sha,*,admitted):
    need(Path.cwd().resolve()==REPO and not NOISE_PREPARE_ROOT.exists(),'NOISE_PREPARE_CWD_OUTPUT')
    plan_ref={'path':str(NOISE_PREPARE_PLAN),'sha256':plan_sha};plan=read(plan_ref)
    need(plan==build_noise_prepare_plan(inputs=plan['inputs'],margin_atr=plan['margin_atr'],
        source_base_commit=plan['source_base_commit'])and noise_source_clean(plan),'NOISE_PREPARE_PLAN_SOURCE')
    policy_ref=ref(POLICY);p=read(policy_ref);gate=p.get(NOISE_PREPARE_STAGE,{})
    need(all(p.get(key)is False for key in ta_audit.data.CLOSED_FLAGS),'NOISE_PREPARE_CLOSED_FLAGS')
    need(gate.get('plan')==plan_ref and gate.get('output_root')==str(NOISE_PREPARE_ROOT)
        and gate.get('max_invocations')==1 and gate.get('claimed_invocations')==gate.get('completed_invocations')==0
        and gate.get('remaining_invocations')==(1 if admitted else 0)
        and gate.get('status')==('ADMITTED_ONE_USE'if admitted else'PREPARED_NOT_ADMITTED')
        and gate.get('execution_authorized')is admitted and gate.get('relaunch_allowed')is False
        and gate.get('training_authorized')is False and gate.get('test_authorized')is False,'NOISE_PREPARE_GATE')
    checked_inputs(plan)
    return plan,plan_ref,policy_ref


def noise_prepare_cli(args):
    need(args.device in(None,'cpu')and args.out_dir==str(NOISE_PREPARE_ROOT),'NOISE_PREPARE_CLI')
    if args.preflight:
        plan,plan_ref,policy_ref=inspect_noise_prepare(args.plan_sha256,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_NOISE_PREPARE_METADATA_ONLY','plan':plan_ref,'policy':policy_ref,
            'claim_created':False,'model_forwards':0,'raw_values_decoded':0}));return
    began=time.monotonic();cap=common.require_capped_cpu_research_producer_execution();lock=common.require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==12*1024**3 and cap['swap']==512*1024**2
        and cap['numerical_threads']==1 and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):]
        and signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'NOISE_PREPARE_CAP')
    plan,plan_ref,policy_ref=inspect_noise_prepare(args.plan_sha256,admitted=True)
    root=NOISE_PREPARE_ROOT;root.mkdir(exist_ok=False)
    claim=publish(root/'CLAIM.json',{'stage':plan['stage'],'plan':plan_ref,'policy':policy_ref,
        'status':'CLAIMED_ONE_USE','remaining_invocations':0,'cap':cap,'lock':lock})
    result=None;old=signal.getsignal(signal.SIGALRM)
    def expired(*args):raise TimeoutError('NOISE_PREPARE_INTERNAL_LIMIT')
    try:
        remaining=plan['execution']['internal_seconds']-(time.monotonic()-began);need(remaining>0,'NOISE_TIME_BEFORE_INPUTS')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
        report=noise_prepare_physical(plan,root);checked_inputs(plan)
        need(noise_source_clean(plan)and ref(POLICY)==policy_ref,'NOISE_PREPARE_BOOKEND')
        result=publish(root/'RESULT.json',{'stage':plan['stage'],'status':'COMPLETE_NATIVE_M5_ATR03_PREPARATION',
            'plan':plan_ref,'claim':claim,'report':report,'source_unchanged':True,'optimizer_steps':0,
            'test_data_used':False,'clean_process_io_firewall':common.firewall_state(),'elapsed_seconds':time.monotonic()-began})
        terminal=publish(root/'TERMINAL.json',{'stage':plan['stage'],'status':'COMPLETE','exit_code':0,
            'plan':plan_ref,'claim':claim,'result':result,'source_unchanged':True,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        print(json.dumps({'result':result,'terminal':terminal}))
    except BaseException as e:
        publish(root/'TERMINAL.json',{'stage':plan['stage'],'status':'FAILED_CONSUMED','exit_code':1,
            'plan':plan_ref,'claim':claim,'result':result,'error':type(e).__name__+':'+str(e)[:2000],
            'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


def noise_parent_source_compatible(parent):
    """Reuse numeric qualification across this plan-metadata-only repair."""
    driver='gx1/audit/qualify_native_m5_full_gc_direction_v1.py'
    need(set(parent['source_owners'])==set(NOISE_SOURCE_PATHS)
        and all(sha(REPO/name)==value for name,value in parent['source_owners'].items()if name!=driver),
        'NOISE_PARENT_UNCHANGED_NUMERIC_OWNERS')
    original=subprocess.check_output(['git','show',parent['source_base_commit']+':'+driver],cwd=REPO)
    need(hashlib.sha256(original).hexdigest()==parent['source_owners'][driver],'NOISE_PARENT_ORIGINAL_DRIVER')
    import ast
    excluded=('build_noise_fit_plan','noise_parent_source_compatible')
    def identity(source):
        tree=ast.parse(source)
        tree.body=[node for node in tree.body if not(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))and node.name in excluded)]
        return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
    before=identity(original.decode());after=identity((REPO/driver).read_text())
    need(before==after,'NOISE_PARENT_UNCHANGED_EXECUTABLE_DRIVER')
    return {'schema_version':'gx1_native_m5_noise_plan_metadata_source_bridge_v1',
        'original_source_base_commit':parent['source_base_commit'],
        'original_driver_sha256':parent['source_owners'][driver],'current_driver_sha256':sha(REPO/driver),
        'unchanged_executable_driver_ast_sha256':before,'all_other_source_owners_exact':True,
        'only_excluded_orchestration_functions':list(excluded),
        'scope':'selected_cohort_reference_adapter_and_its_source_validation_only_no_model_data_loss_optimizer_score_resume_change'}


def build_noise_fit_plan(*,inputs,recipe,mode,source_base_commit=None):
    need(mode in('ta_noise_mechanics','ta_noise_main'),'NOISE_FIT_MODE')
    proxy='ta_primary_mechanics'if mode=='ta_noise_mechanics'else'ta_primary_main'
    recipe=fit_recipe(recipe,proxy)
    fields={'prepare_result','prepare_terminal','matched_prior_reference'}|({'noise_mechanics_result','noise_mechanics_terminal'}if mode=='ta_noise_main'else set())
    need(type(inputs)is dict and set(inputs)==fields,'NOISE_FIT_INPUT_ROLES')
    done,last=read(inputs['prepare_result']),read(inputs['prepare_terminal'])
    prepared=read(done['plan']);report=done['report']
    need(done['stage']==last['stage']==NOISE_PREPARE_STAGE and done['status']=='COMPLETE_NATIVE_M5_ATR03_PREPARATION'
        and last['status']=='COMPLETE'and last['exit_code']==0 and last['result']==inputs['prepare_result']
        and last['claim']==done['claim']and last['plan']==done['plan']
        and done['source_unchanged']is True and inputs['matched_prior_reference']==report['prior']
        and report['all_selected_initial_logits_equal_new_noise_prior']is True
        and report['optimizer_steps']==0 and noise_parent_source_compatible(prepared),'NOISE_FIT_PREPARED')
    from gx1.contracts.market_state_ta_integrated_model_v1 import native_bar_model_contract
    from gx1.contracts.market_state_ta_integrated_objective_v1 import require_native_bar_train_prior
    prior=require_native_bar_train_prior(read(report['prior']),target_contract=prepared['metadata']['target_contract'])
    normalization=read(report['normalization']);binding=report['input_binding']
    need(binding['source_axes_sha256']==normalization['source_axes_sha256']
        and binding['input_normalization_sha256']==normalization['contract_sha256']
        and report['model_contract']==native_bar_model_contract(binding,train_prior=prior,
            backbone_gradient_policy=PRIMARY_BACKBONE_GRADIENT_POLICY,target_contract=prepared['metadata']['target_contract']),
        'NOISE_PHYSICAL_INPUT_CHECKPOINT_CONTRACT')
    meta={**prepared['metadata'],'dependencies':{**prepared['metadata']['dependencies'],
        'prepared_plan':done['plan'],'prepared_claim':done['claim'],'noise_dataset':report['dataset'],
        'noise_arrays':read(report['dataset'])['arrays'],'train_prior':report['prior'],'train_normalization':report['normalization']},
        'prepared':{'result':inputs['prepare_result'],'terminal':inputs['prepare_terminal'],
            'dataset':report['dataset'],'normalization':report['normalization'],'prior':report['prior'],
            'axis':report['axis'],'normalization_population':report['normalization_population'],
            'initial_state_sha256':report['initial_state_sha256']},
        'chronological_reference':{'role':'historical_raw_target_context_only_not_same_label_function','plan':prepared['inputs']['reference_fit_plan']},
        'raw_gradient_design_evidence':{'role':'retained_existing_primary_backbone_policy_new_noise_mechanics_required'}}
    if mode=='ta_noise_main':
        r,t=read(inputs['noise_mechanics_result']),read(inputs['noise_mechanics_terminal'])
        parent=read(r['plan']);proof=r['report']
        need(r['stage']==t['stage']==FIT_SCOPES['ta_noise_mechanics'][0]
            and r['status']=='COMPLETE_NATIVE_M5_TA_ATR03_MECHANICS'and t['status']=='COMPLETE'and t['exit_code']==0
            and t['result']==inputs['noise_mechanics_result']and t['plan']==r['plan']and t['claim']==r['claim']
            and proof['cold_resume']['bitexact']is True and proof['actual_counts']['autograd_calls']==6
            and proof['normalization_fits']==proof['prior_fits']==0
            and proof['primary_backbone_qualification']['auxiliary_shared_zero_all_steps']is True
            and proof['primary_backbone_qualification']['M5_shared_positive_after_first_head_update']is True
            and parent['metadata']==meta and noise_parent_source_compatible(parent),'NOISE_MAIN_MECHANICS')
        meta['mechanics']={'result':inputs['noise_mechanics_result'],'terminal':inputs['noise_mechanics_terminal'],
            'plan':r['plan'],'claim':r['claim'],'normalization':proof['normalization'],'prior':proof['prior'],
            'axis':proof['axis'],'initial_state_sha256':proof['initial_state_sha256'],
            'physical_windows_boot_id':r['physical_windows_boot_id']}
    stage,path,root=FIT_SCOPES[mode]
    selection=small_cohort_manifest(meta)
    return {'schema_version':'gx1_native_m5_ta_atr03_fit_plan_v1','stage':stage,'mode':mode,
        'status':'PREPARED_NOT_ADMITTED','output_root':str(root),
        'source_base_commit':source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'source_owners':{p:sha(REPO/p)for p in NOISE_SOURCE_PATHS},'inputs':inputs,'metadata':meta,
        'recipe':recipe,'cohort_selection':selection,
        'source_compatibility':{'initial_preparation':noise_parent_source_compatible(prepared),
            **({'optimizer_mechanics':noise_parent_source_compatible(parent)}if mode=='ta_noise_main'else{})},
        **({'training_schedule':ta_block_schedule({**meta,'chronological_reference':{**meta['chronological_reference'],
            'selected_cohorts':report['selected_cohorts']}},recipe)}if mode=='ta_noise_main'else{}),
        'backbone_gradient_policy':PRIMARY_BACKBONE_GRADIENT_POLICY,
        'execution':{'class':'trainer','device':'cuda','memory':'20G','swap':'512M','tasks_max':128,
            'numerical_threads':8,'outer_seconds':recipe['outer_seconds'],'internal_seconds':recipe['internal_seconds'],'kill_after_seconds':10},
        'GC_used':False,'statistics_fits':{'normalization':0,'matched_TRAIN_prior':0},
        'limits':fit_limits(recipe,proxy,meta),'qualification_weights_loaded':False,
        'scheduler_state':None,'EMA_state':None,'training_authorized':True,'test_data_used':False,'economic_edge_proven':False}


def prepare_noise_native_fit(plan,root,*,reload_normalization=None,reload_prior=None):
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    meta=plan['metadata'];saved=meta['prepared'];target_contract=meta['target_contract']
    provider,admission=ta_audit.load_candidate_provider(meta['ta_plan'])
    dataset=read(saved['dataset']);need(dataset['contract']==target_contract,'NOISE_BOUND_DATASET_CONTRACT')
    cohorts,axis=noise_fit_axes(provider,dataset,target_contract)
    need(axis==saved['axis'],'NOISE_REUSED_PREPARED_AXIS')
    normalization=read(saved['normalization']);provider.input_binding(normalization)
    need(normalization['fit_population']==saved['normalization_population']
        and digest(cohorts['TRAIN']['fit_rows'])==normalization['fit_population']['selected_source_rows_sha256'],'NOISE_EXACT_TRAIN_NORMALIZATION')
    prior=objective.require_native_bar_train_prior(read(saved['prior']),target_contract=target_contract)
    selected,_=selected_small_cohorts(plan,{'cohorts':cohorts})
    need(prior['fit_population']['issue_rows']==len(selected['FIT'])
        and prior['fit_population']['issue_ns_sha256']==digest(provider.issue_ns[selected['FIT']]),'NOISE_MATCHED_PRIOR_AXIS')
    need(reload_normalization in(None,saved['normalization'])and reload_prior in(None,saved['prior']),'NOISE_COLD_EXACT_STATS')
    return {'provider':provider,'cohorts':cohorts,'axis':axis,'normalization':normalization,
        'normalization_ref':saved['normalization'],'prior':prior,'prior_ref':saved['prior'],
        'normalization_fits':0,'prior_fits':0,'feature_requalification':admission}


# Each throughput scope is one-use; consumed001/002 are not executable aliases.
THROUGHPUT_STAGE='native_m5_ta_atr03_throughput_20261120_004'
THROUGHPUT_PLAN=REPO/'configs/research/NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_004.json'
THROUGHPUT_ROOT=Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_004')
THROUGHPUT_SOURCE_COMPATIBILITY_PATH=Path('/home/andre2/GX1_RUNS/GX1_FULL_PROJECT_AUDIT_20261120_001/CRASH_RECOVERY_QUALIFICATION_001/SOURCE_COMPATIBILITY002.json')
THROUGHPUT_INPUTS={
    'mechanics_result':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_001/RESULT.json',
        'sha256':'7b099711b52f12312d94cefd9fb427a2e26bd1abbdc529dcc1e4f523f1b394e5'},
    'mechanics_terminal':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_001/TERMINAL.json',
        'sha256':'a7a592aee1c0d0afe7c114b5cba2df7960ce56436590767000f4e5ffad096aa9'}}
THROUGHPUT_RUNTIME_SOURCE_PATHS=(
    'gx1/audit/qualify_native_m5_full_gc_direction_v1.py',
    'gx1/contracts/market_state_ta_integrated_model_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_qualify_native_m5_full_gc_direction_v1.py',
    'tests/test_market_state_ta_integrated_model_v1.py',
    'tests/test_gx1_capped_run_native_year_contract.py')
THROUGHPUT_LIMITS={'model_constructions':1,'checkpoint_loads':1,'model_forwards':15,
    'backward_calls':6,'autograd_calls':0,'optimizer_constructions':0,'optimizer_steps':0}
THROUGHPUT_EXECUTION={'class':'trainer','device':'cuda','memory':'20G','swap':'512M','tasks_max':128,
    'numerical_threads':8,'outer_seconds':1800,'internal_seconds':1770,'kill_after_seconds':10,
    'cuda_peak_reserved_bytes_max':12*1024**3}
THROUGHPUT_EVAL_TOLERANCE={'atol':1e-6,'rtol':1e-5,
    'owner':'tests/test_entry_v10_ctx_model_shapes.py::test_equal_length_mtf_batches_only_eval_without_grad'}


def throughput_source_compatibility(binding,parent):
    need(type(binding)is dict and set(binding)=={'path','sha256'}
        and binding['path']==str(THROUGHPUT_SOURCE_COMPATIBILITY_PATH),'THROUGHPUT_SOURCE_REVIEW_PATH')
    review=read(binding)
    fields={'schema_version','status','old_base_commit','source_base_commit','old_source_owners',
        'current_source_owners','allowed_changed_paths','baseline_prepare_result',
        'reference_mechanics_result','reference_model_blob_sha256','unit_proof','source_diff_review'}
    need(type(review)is dict and fields<=set(review)
        and review['schema_version']=='gx1_native_m5_runtime_source_compatibility_v1'
        and review['status']=='QUALIFIED_RUNTIME_ONLY_SOURCE_COMPATIBILITY'
        and review['old_source_owners']==parent['source_owners']
        and set(review['old_source_owners'])==set(LARGER_SOURCE_PATHS)
        and review['current_source_owners']=={name:sha(REPO/name)for name in LARGER_SOURCE_PATHS}
        and review['baseline_prepare_result']==parent['inputs']['prepare_result']
        and review['reference_mechanics_result']==THROUGHPUT_INPUTS['mechanics_result'],
        'THROUGHPUT_SOURCE_REVIEW_BINDINGS')
    changed=sorted(name for name,old in review['old_source_owners'].items()
        if review['current_source_owners'][name]!=old)
    need(changed==review['allowed_changed_paths']
        and set(changed)<=set(THROUGHPUT_RUNTIME_SOURCE_PATHS)
        and set(THROUGHPUT_RUNTIME_SOURCE_PATHS[:2])<=set(changed),
        'THROUGHPUT_ONLY_REVIEWED_RUNTIME_OWNERS')
    source=subprocess.check_output(['git','show',review['old_base_commit']+':'+THROUGHPUT_RUNTIME_SOURCE_PATHS[1]],cwd=REPO)
    need(hashlib.sha256(source).hexdigest()==review['reference_model_blob_sha256']
        ==parent['source_owners'][THROUGHPUT_RUNTIME_SOURCE_PATHS[1]],'THROUGHPUT_ORIGINAL_REFERENCE_BLOB')
    unit,diff=read(review['unit_proof']),read(review['source_diff_review'])
    need(unit.get('status')=='PASS'and unit.get('exit_code')==0
        and unit.get('current_source_owners')==review['current_source_owners']
        and diff.get('status')=='PASS_STATIC_MODEL_AND_WRAPPER_CONTROL_ONLY'
        and diff.get('current_source_owners')==review['current_source_owners']
        and all(diff.get(k)is True for k in('contract_math_and_remaining_module_ast_unchanged',
            'wrapper_inverse_diff_exact','fit_and_score_pipeline_unchanged_after_unwrap')),
        'THROUGHPUT_QUALIFIED_UNIT_AND_SOURCE_REVIEW')
    return review


def build_throughput_plan(*,inputs,source_base_commit=None):
    need(type(inputs)is dict and set(inputs)==set(THROUGHPUT_INPUTS)|{'source_compatibility'}
        and all(inputs[k]==v for k,v in THROUGHPUT_INPUTS.items()),'THROUGHPUT_EXACT_INPUTS')
    r,t=read(inputs['mechanics_result']),read(inputs['mechanics_terminal'])
    parent=read(r['plan']);claim=read(r['claim']);proof=r['report']
    need(r['stage']==t['stage']==parent['stage']==claim['stage']=='native_m5_ta_atr03_five_year_mechanics_20261120_001'
        and parent['mode']=='ta_five_year_mechanics'
        and r['status']=='COMPLETE_NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS'and t['status']=='COMPLETE'
        and t['exit_code']==0 and t['result']==inputs['mechanics_result']and t['plan']==claim['plan']==r['plan']
        and t['claim']==r['claim']and r['source_unchanged']is True
        and r['test_data_used']is t['test_data_used']is False
        and proof['cold_resume']['bitexact']is True and proof['actual_counts']==parent['limits']
        and proof['normalization_fits']==proof['prior_fits']==0,'THROUGHPUT_PARENT')
    review=throughput_source_compatibility(inputs['source_compatibility'],parent)
    from gx1.contracts.market_state_ta_integrated_model_v1 import native_bar_model_contract
    from gx1.contracts.market_state_ta_integrated_objective_v1 import require_native_bar_train_prior
    prepared=parent['metadata']['prepared'];spec=parent['metadata']['target_contract']
    prior=require_native_bar_train_prior(read(proof['prior']),target_contract=spec)
    normalization=read(proof['normalization']);input_binding=prepared['input_binding']
    need(input_binding['source_axes_sha256']==normalization['source_axes_sha256']
        and input_binding['input_normalization_sha256']==normalization['contract_sha256']
        and prepared['model_contract']==native_bar_model_contract(input_binding,train_prior=prior,
            backbone_gradient_policy=parent['backbone_gradient_policy'],target_contract=spec),
        'THROUGHPUT_BOUND_INPUT_NORM_MODEL_IDENTITY')
    base=source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    need(base==review['source_base_commit'],'THROUGHPUT_REVIEWED_SOURCE_BASE')
    deps={**parent['metadata']['dependencies'],'mechanics_plan':r['plan'],'mechanics_claim':r['claim'],
        'final_checkpoint':proof['final_checkpoint'],'normalization':proof['normalization'],'prior':proof['prior'],
        'source_compatibility':inputs['source_compatibility'],'runtime_unit_proof':review['unit_proof'],
        'runtime_diff_review':review['source_diff_review']}
    return {'schema_version':'gx1_native_m5_atr03_throughput_plan_v2','stage':THROUGHPUT_STAGE,
        'status':'PREPARED_NOT_ADMITTED','output_root':str(THROUGHPUT_ROOT),'inputs':inputs,
        'source_base_commit':base,'source_owners':dict(review['current_source_owners']),
        'metadata':{'dependencies':deps,'analysis_parent':r['plan'],
            'normalization':proof['normalization'],'prior':proof['prior'],
            'reference_base_commit':review['old_base_commit'],
            'reference_model_blob_sha256':review['reference_model_blob_sha256'],
            'initial_state_sha256':proof['initial_state_sha256'],'final_state_sha256':proof['final_state_sha256']},
        'execution':THROUGHPUT_EXECUTION,'limits':THROUGHPUT_LIMITS,
        'panel_rule':'first64_TRAIN_for_train_first256_TRAIN_and_last2_TRAIN_for_eval_no_label_selection',
        'lanes':['warmup_immediate','reference_immediate','grouped_uninstrumented','reference_repeat','grouped_repeat','grouped_profiled'],
        'reference_finite_guard':'old_exact_binding_method_immediate_forward_current_main_per_device_grad_parameter_guards',
        'candidate_finite_guard':'current_binding_method_grouped_forward_same_per_device_grad_parameter_guards',
        'evaluation':{'rows':256,'reference_batch_rows':64,'candidate_batch_rows':256,'tail_rows':2,
            'equal_length_TF_candidate':True,'capture_backbone_timeframe_hidden':True,
            'output_tolerance':THROUGHPUT_EVAL_TOLERANCE,'counts_confusion_ties':'exact',
            'state_rng':'bitexact','memory_failure_retry':False,'candidate_not_main_admission':True},
        'model_mode':'train_and_eval_no_grad','same_rng_per_lane':True,'analysis_weights_only':True,
        'training_authorized':False,'test_data_used':False,'GC_used':False,'economic_edge_proven':False}


def inspect_throughput(plan_sha,*,admitted):
    need(Path.cwd().resolve()==REPO and not THROUGHPUT_ROOT.exists(),'THROUGHPUT_CWD_OUTPUT')
    binding={'path':str(THROUGHPUT_PLAN),'sha256':plan_sha};plan=read(binding)
    need(plan==build_throughput_plan(inputs=plan['inputs'],source_base_commit=plan['source_base_commit'])
        and larger_source_clean(plan),'THROUGHPUT_PLAN_SOURCE')
    policy_ref=ref(POLICY);policy=read(policy_ref);gate=policy.get(THROUGHPUT_STAGE,{})
    need(all(policy.get(k)is False for k in ta_audit.data.CLOSED_FLAGS)
        and gate.get('plan')==binding and gate.get('output_root')==str(THROUGHPUT_ROOT)
        and gate.get('max_invocations')==1 and gate.get('claimed_invocations')==gate.get('completed_invocations')==0
        and gate.get('remaining_invocations')==(1 if admitted else 0)
        and gate.get('status')==('ADMITTED_ONE_USE'if admitted else'PREPARED_NOT_ADMITTED')
        and gate.get('execution_authorized')is admitted and gate.get('training_authorized')is False
        and gate.get('test_authorized')is False and gate.get('relaunch_allowed')is False
        and all(gate.get(k)==v for k,v in plan['limits'].items()),'THROUGHPUT_GATE')
    checked_inputs(plan);return plan,binding,policy_ref


def throughput_profile_summary(profiler):
    import pstats
    stats=pstats.Stats(profiler).stats
    def item(key,value):
        cc,nc,tt,ct,_=value
        return {'file':key[0],'line':key[1],'function':key[2],'primitive_calls':cc,
            'calls':nc,'self_seconds':tt,'cumulative_seconds':ct}
    return {'top12_cumulative':[item(k,v)for k,v in sorted(stats.items(),key=lambda x:x[1][3],reverse=True)[:12]],
        'binding_calls':[item(k,v)for k,v in stats.items()if k[2]in(
            '_require_binding','require_input_normalization_state','require_surface_normalization',
            'require_native_m5_input_binding','compile_feature_profile')]}


def throughput_reference_method(source):
    """Prepare authenticated reference code before any lane is timed."""
    import ast
    from gx1.contracts import market_state_ta_integrated_model_v1 as contract
    tree=ast.parse(source)
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='NativeM5BarDirectionModel']
    need(len(classes)==1,'THROUGHPUT_REFERENCE_CLASS')
    methods=[n for n in classes[0].body if isinstance(n,ast.FunctionDef)and n.name=='_require_binding']
    need(len(methods)==1 and not methods[0].decorator_list,'THROUGHPUT_REFERENCE_METHOD')
    namespace=dict(vars(contract))
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),
        '<verified_old_native_binding>','exec'),namespace)
    return namespace['_require_binding']


def throughput_reference_binding(model,method):
    """Restore the live method even when the reference raises."""
    import types
    from contextlib import contextmanager
    @contextmanager
    def scope():
        need(isinstance(method,types.FunctionType)and method.__name__=='_require_binding',
            'THROUGHPUT_PREPARED_REFERENCE_METHOD')
        had_override='_require_binding'in model.__dict__;original=model.__dict__.get('_require_binding')
        model._require_binding=types.MethodType(method,model)
        try:yield
        finally:
            if had_override:model._require_binding=original
            else:delattr(model,'_require_binding')
    return scope()


def throughput_lane(model,context,rows,parent,*,grouped,profiled,rng,reference_method):
    import cProfile
    from contextlib import nullcontext
    import torch
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_direction_loss
    model.zero_grad(set_to_none=True);restore_training_rng(rng);model.train()
    events=[];phases={}
    def phase(name,fn):
        if not profiled:return fn()
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();began=time.perf_counter()
        with torch.profiler.record_function('GX1_PROFILE_'+name):value=fn()
        b.record();events.append((name,a,b));phases[name]={'cpu_submit_seconds':time.perf_counter()-began}
        return value
    cpu=cProfile.Profile()
    gpu=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,profile_memory=False,with_stack=False)if profiled else nullcontext()
    torch.cuda.synchronize();started=time.perf_counter()
    with gpu as trace:
        if profiled:cpu.enable()
        raw=phase('gather',lambda:context['provider'].gather(rows,max_rows=len(rows)))
        features=phase('host_to_device',lambda:{k:torch.from_numpy(v).to('cuda')for k,v in raw.items()})
        truth=phase('truth_slice',lambda:slice_truth(context['cohorts']['TRAIN'],rows,fit=True))
        def forward():
            with nullcontext()if grouped else throughput_reference_binding(model,reference_method):
                return native_fit_forward(model,features,grouped=grouped)
        output=phase('forward',forward)
        loss=phase('loss',lambda:native_bar_direction_loss(model_output=output,truth=truth,fold_upper_ns=parent['metadata']['fit_end_ns'],
            target_contract_sha256=parent['metadata']['target_contract']['contract_sha256'],**noise_target_kwargs(parent)))
        need(not loss['skip_update']and bool(torch.isfinite(loss['loss'])),'THROUGHPUT_FINITE_LOSS')
        phase('backward',lambda:loss['loss'].backward())
        gradients=[p for p in model.parameters()if p.grad is not None]
        phase('finite_gradients',lambda:need(gradients and finite_all_per_device(p.grad for p in gradients),
            'THROUGHPUT_FINITE_GRADIENTS'))
        norm=phase('clip_grad_norm',lambda:torch.nn.utils.clip_grad_norm_(gradients,
            parent['recipe']['gradient_clip_norm'],error_if_nonfinite=True))
        phase('finite_parameters',lambda:need(finite_all_per_device(model.parameters()),'THROUGHPUT_FINITE_PARAMETERS'))
        if profiled:cpu.disable()
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    for name,a,b in events:phases[name]['cuda_stream_elapsed_ms']=a.elapsed_time(b)
    identity={'output_sha256':state_sha(output),'loss_sha256':state_sha(loss['loss']),
        'clipped_gradient_sha256':state_sha({n:p.grad for n,p in model.named_parameters()}),
        'gradient_norm':float(norm),'rng_after_sha256':state_sha(training_rng())}
    detail={}
    if profiled:
        operations=trace.key_averages()
        detail={'python':throughput_profile_summary(cpu),'phases':phases,
            'operations':[{'key':e.key,'count':e.count,'self_cpu_time_us':e.self_cpu_time_total,
                **({'self_device_time_us':e.self_device_time_total}if hasattr(e,'self_device_time_total')else{})}for e in
                sorted(operations,key=lambda e:e.self_cpu_time_total,reverse=True)[:12]],
            'sync_and_copy_operations':[{'key':e.key,'count':e.count,'self_cpu_time_us':e.self_cpu_time_total}
                for e in operations if any(x in e.key.lower()for x in('item','_local_scalar','memcpy','synchronize','dtoh','_to_copy'))]}
    model.zero_grad(set_to_none=True)
    need(all(p.grad is None for p in model.parameters()),'THROUGHPUT_GRAD_CLEANUP')
    return {'wall_seconds':elapsed,'identity':identity,**detail}


def throughput_eval_lane(model,context,rows,parent,*,batch_rows,grouped,equal_tf,rng,reference_method):
    import torch
    from contextlib import nullcontext
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import batch_equal_length_mtf_eval_scope
    need(batch_rows in(64,256)and type(grouped)is type(equal_tf)is bool,'THROUGHPUT_EVAL_GEOMETRY')
    before=state_sha(model.state_dict());restore_training_rng(rng);rng_sha=state_sha(training_rng())
    training=model.training;model.eval();hidden=[];outputs={'open_logits':[],'close_logits':[]}
    def observe(module,args,output):
        need(isinstance(output,dict)and 'entry_q_timeframe_hidden'in output,'THROUGHPUT_HIDDEN_HOOK')
        hidden.append(output['entry_q_timeframe_hidden'].detach())
    hook=model.backbone.register_forward_hook(observe);torch.cuda.synchronize();started=time.perf_counter()
    try:
        with torch.no_grad(),(batch_equal_length_mtf_eval_scope()if equal_tf else nullcontext()),(
                nullcontext()if grouped else throughput_reference_binding(model,reference_method)):
            for batch in all_score_batches(rows,batch_rows):
                raw=context['provider'].gather(batch,max_rows=batch_rows)
                output=native_fit_forward(model,{k:torch.from_numpy(v).to('cuda')for k,v in raw.items()},grouped=grouped)
                for key,value in output.items():outputs[key].append(value.detach().cpu().numpy().copy())
                need(len(hidden)==1,'THROUGHPUT_ONE_HIDDEN_PER_FORWARD')
                outputs.setdefault('timeframe_hidden',[]).append(hidden.pop().cpu().numpy().copy())
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
    finally:hook.remove();hidden.clear();model.train(training)
    need(state_sha(model.state_dict())==before and state_sha(training_rng())==rng_sha
        and all(p.grad is None for p in model.parameters()),'THROUGHPUT_EVAL_IMMUTABLE')
    arrays={key:np.concatenate(parts)for key,parts in outputs.items()}
    return {'wall_seconds':elapsed,'batch_rows':batch_rows,'grouped_forward':grouped,'equal_length_TF':equal_tf,
        'rows':len(rows),'source_rows_sha256':digest(rows),'output_sha256':{k:digest(v)for k,v in arrays.items()},
        'state_unchanged':True,'rng_unchanged':True,'training_mode_restored':model.training==training},arrays


def throughput_eval_parity(reference,candidate,truth,parent,*,bitexact):
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_direction_score_sums
    tolerance=THROUGHPUT_EVAL_TOLERANCE;differences={};numerical=True
    need(set(reference)==set(candidate)=={'open_logits','close_logits','timeframe_hidden'},'THROUGHPUT_EVAL_OUTPUT_KEYS')
    for key,expected in reference.items():
        value=candidate[key]
        need(value.shape==expected.shape and value.dtype==expected.dtype==np.float32,'THROUGHPUT_EVAL_OUTPUT_DOMAIN')
        exact=np.array_equal(expected,value)
        close=bool(np.allclose(expected,value,atol=tolerance['atol'],rtol=tolerance['rtol']))
        numerical=numerical and(exact if bitexact else close)
        differences[key]={'bitexact':exact,'within_tolerance':close,
            'max_absolute_difference':float(np.max(np.abs(expected.astype(np.float64)-value.astype(np.float64))))}
    kwargs={'truth':truth,'fold_upper_ns':parent['metadata']['fit_end_ns'],
        'target_contract_sha256':parent['metadata']['target_contract']['contract_sha256'],**noise_target_kwargs(parent)}
    scores=[native_bar_direction_score_sums(model_output={k:v for k,v in output.items()if k!='timeframe_hidden'},**kwargs)
        for output in(reference,candidate)]
    counts_exact=True;score_close=True
    for task,by_tf in scores[0]['per_task'].items():
        for tf,expected in by_tf.items():
            value=scores[1]['per_task'][task][tf]
            for key in('count','unique_target_rows','confusion_truth_by_prediction','max_logit_ties'):
                counts_exact=counts_exact and expected[key]==value[key]
            for key in('log_loss_sum','brier_sum'):
                score_close=score_close and(expected[key]==value[key]if bitexact else
                    bool(np.isclose(expected[key],value[key],atol=tolerance['atol'],rtol=tolerance['rtol'])))
    return {'qualified':bool(numerical and counts_exact and score_close),'required_bitexact':bitexact,
        'output_differences':differences,'counts_confusion_ties_exact':bool(counts_exact),
        'score_sums_within_declared_tolerance':bool(score_close),'reference_scores':scores[0],'candidate_scores':scores[1]}


def throughput_physical(plan):
    import torch
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_bar_direction_model
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    parent=read(plan['metadata']['analysis_parent']);started=time.monotonic()
    context=prepare_larger_native_fit(parent,THROUGHPUT_ROOT)
    prep_seconds=time.monotonic()-started;provider=context['provider'];meta=plan['metadata']
    need(context['normalization_ref']==meta['normalization']and context['prior_ref']==meta['prior'],
        'THROUGHPUT_EXACT_STATS')
    source=subprocess.check_output(['git','show',meta['reference_base_commit']+':'+THROUGHPUT_RUNTIME_SOURCE_PATHS[1]],cwd=REPO)
    need(hashlib.sha256(source).hexdigest()==meta['reference_model_blob_sha256'],'THROUGHPUT_REFERENCE_BLOB_BEFORE_EXEC')
    reference_method=throughput_reference_method(source.decode());seed_native_fit(parent['recipe']['seed']);torch.cuda.reset_peak_memory_stats()
    total_memory=torch.cuda.get_device_properties(0).total_memory
    memory_fraction=plan['execution']['cuda_peak_reserved_bytes_max']/total_memory
    need(0<memory_fraction<=1,'THROUGHPUT_ALLOCATOR_LIMIT');torch.cuda.set_per_process_memory_fraction(memory_fraction,0)
    model=build_native_bar_direction_model(input_normalization=context['normalization'],
        input_binding=provider.input_binding(context['normalization']),architecture=parent['metadata']['architecture'],
        device=torch.device('cuda',0),train_prior=context['prior'],
        backbone_gradient_policy=parent['backbone_gradient_policy'],**noise_target_kwargs(parent))
    need(canonical_model_state_sha256(model.state_dict())==meta['initial_state_sha256'],'THROUGHPUT_FRESH_IDENTITY')
    checkpoint_ref=meta['dependencies']['final_checkpoint']
    need(ref(checkpoint_ref['path'])==checkpoint_ref,'THROUGHPUT_CHECKPOINT_PREDECODE_SHA')
    checkpoint=torch.load(checkpoint_ref['path'],map_location='cpu',weights_only=True)
    expected={'input':provider.input_binding(context['normalization']),'normalization':context['normalization_ref'],
        'prior':context['prior_ref'],'target':parent['metadata']['target_contract'],
        'architecture':parent['metadata']['architecture'],'recipe':parent['recipe'],'model_contract':model.research_objective}
    need(checkpoint['plan']==meta['analysis_parent']and checkpoint['bindings']==expected,'THROUGHPUT_PRECOPY_BINDINGS')
    model.load_state_dict(checkpoint['model_state'],strict=True);del checkpoint
    need(canonical_model_state_sha256(model.state_dict())==meta['final_state_sha256'],'THROUGHPUT_TRAINED_IDENTITY')
    nonzero_heads={task:{tf:int(torch.count_nonzero(head.weight))for tf,head in heads.items()}
        for task,heads in(('open',model.open_heads),('close',model.close_heads))}
    need(nonzero_heads['close']['M5']>0,'THROUGHPUT_NONZERO_M5_READOUT')
    before=state_sha(model.state_dict());norm_sha=state_sha(context['normalization']);prior_sha=state_sha(context['prior'])
    rng=training_rng();rng_sha=state_sha(rng);reports={};counts={'forwards':0,'backwards':0}
    def count(module,args):
        counts['forwards']+=1;need(counts['forwards']<=plan['limits']['model_forwards'],'THROUGHPUT_FORWARD_LIMIT')
    hook=model.register_forward_pre_hook(count);all_rows=context['cohorts']['TRAIN']['rows'];rows=all_rows[:64]
    evaluation={};partial={'plan_stage':plan['stage'],'qualification_authority':False,'optimizer_steps':0}
    try:
        lanes={}
        for name,grouped,profiled in [('warmup_immediate',False,False),('reference_immediate',False,False),
            ('grouped_uninstrumented',True,False),('reference_repeat',False,False),
            ('grouped_repeat',True,False),('grouped_profiled',True,True)]:
            lanes[name]=throughput_lane(model,context,rows,parent,grouped=grouped,profiled=profiled,rng=rng,
                reference_method=reference_method);counts['backwards']+=1
        reference=lanes['reference_immediate']['identity']
        need(all(v['identity']==reference for v in lanes.values()),'THROUGHPUT_SAME_RNG_OUTPUT_LOSS_GRAD_PARITY')
        reports['64']={'source_rows_sha256':digest(rows),'issue_ns_sha256':digest(provider.issue_ns[rows]),'lanes':lanes}
        publish(THROUGHPUT_ROOT/'TRAIN_PARITY_PARTIAL.json',{**partial,'train_bitexact':True,'panels':reports})
        eval_rows=all_rows[:256];truth=slice_truth(context['cohorts']['TRAIN'],eval_rows)
        base_report,base=throughput_eval_lane(model,context,eval_rows,parent,batch_rows=64,grouped=False,equal_tf=False,
            rng=rng,reference_method=reference_method)
        b64_report,b64=throughput_eval_lane(model,context,eval_rows[:64],parent,batch_rows=64,grouped=True,equal_tf=False,
            rng=rng,reference_method=reference_method)
        b64_parity=throughput_eval_parity({k:v[:64]for k,v in base.items()},b64,
            slice_truth(context['cohorts']['TRAIN'],eval_rows[:64]),parent,bitexact=True)
        need(b64_parity['qualified'],'THROUGHPUT_B64_EVAL_BITEXACT')
        evaluation.update(reference_b64=base_report,candidate_b64=b64_report,b64_parity=b64_parity)
        publish(THROUGHPUT_ROOT/'B64_EVAL_PARITY_PARTIAL.json',{**partial,'train_bitexact':True,'evaluation':evaluation})
        for name,equal_tf in(('candidate_b256',False),('candidate_b256_equal_tf',True)):
            report,arrays=throughput_eval_lane(model,context,eval_rows,parent,batch_rows=256,grouped=True,equal_tf=equal_tf,
                rng=rng,reference_method=reference_method)
            evaluation[name]={'run':report,'parity':throughput_eval_parity(base,arrays,truth,parent,bitexact=False)}
        tail=all_rows[-2:];tail_truth=slice_truth(context['cohorts']['TRAIN'],tail)
        ref_tail,tail_arrays=throughput_eval_lane(model,context,tail,parent,batch_rows=64,grouped=False,equal_tf=False,
            rng=rng,reference_method=reference_method)
        new_tail,new_arrays=throughput_eval_lane(model,context,tail,parent,batch_rows=256,grouped=True,equal_tf=True,
            rng=rng,reference_method=reference_method)
        evaluation['short_tail']={'reference':ref_tail,'candidate':new_tail,
            'parity':throughput_eval_parity(tail_arrays,new_arrays,tail_truth,parent,bitexact=False)}
        need(torch.cuda.max_memory_reserved()<=plan['execution']['cuda_peak_reserved_bytes_max'],'THROUGHPUT_VRAM_LIMIT')
    finally:hook.remove();model.zero_grad(set_to_none=True);restore_training_rng(rng)
    need(counts=={'forwards':15,'backwards':6}and state_sha(model.state_dict())==before
        and state_sha(context['normalization'])==norm_sha and state_sha(context['prior'])==prior_sha
        and state_sha(training_rng())==rng_sha and all(p.grad is None for p in model.parameters())
        and '_require_binding'not in model.__dict__,'THROUGHPUT_IMMUTABLE_BOOKEND')
    return {'panels':reports,'evaluation':evaluation,'actual_counts':{**plan['limits'],
            'model_forwards':counts['forwards'],'backward_calls':counts['backwards']},'prepare_seconds':prep_seconds,
        'train_bitexact':True,'b64_eval_bitexact':True,
        'b256_candidate_qualified':evaluation['candidate_b256']['parity']['qualified'],
        'b256_equal_tf_candidate_qualified':bool(evaluation['candidate_b256_equal_tf']['parity']['qualified']
            and evaluation['short_tail']['parity']['qualified']),
        'b256_equal_tf_measured_faster':evaluation['candidate_b256_equal_tf']['run']['wall_seconds']
            <evaluation['reference_b64']['wall_seconds'],
        'nonzero_readout_weights':nonzero_heads,'model_state_sha256':meta['final_state_sha256'],
        'normalization':meta['normalization'],'prior':meta['prior'],'normalization_fits':0,'prior_fits':0,
        'weights_unchanged':True,'rng_restored':True,'grad_fields_none':True,
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'scope':'DIAGNOSTIC_TRAIN_EVAL_NO_UPDATES_NO_MAIN_BATCH_ADMISSION_OR_ROBUST_ADAMW_THROUGHPUT'}


def throughput_cli(args):
    need(args.plan==str(THROUGHPUT_PLAN)and args.out_dir==str(THROUGHPUT_ROOT)
        and args.device=='cuda','THROUGHPUT_CLI')
    if args.preflight:
        _,binding,policy_ref=inspect_throughput(args.plan_sha256,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_THROUGHPUT_METADATA_ONLY','plan':binding,'policy':policy_ref,
            'claim_created':False,'model_forwards':0,'feature_values_decoded':0}));return
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    began=time.monotonic();guard=require_guarded_cuda_trainer_execution();common.firewall_state()
    plan,binding,policy_ref=inspect_throughput(args.plan_sha256,admitted=True)
    boot=fit_boot_state(read(plan['metadata']['analysis_parent']),admitted=True)
    THROUGHPUT_ROOT.mkdir(exist_ok=False)
    claim=publish(THROUGHPUT_ROOT/'CLAIM.json',{'stage':THROUGHPUT_STAGE,'status':'CLAIMED_ONE_USE',
        'plan':binding,'policy':policy_ref,'guard':guard,'physical_windows_boot_id':boot['actual_windows_boot_id'],
        'remaining_invocations':0,'test_data_used':False})
    previous=signal.getsignal(signal.SIGALRM);result=None
    def expired(*args):raise TimeoutError('THROUGHPUT_INTERNAL_LIMIT')
    try:
        remaining=1770-(time.monotonic()-began);need(remaining>0,'THROUGHPUT_TIME_BEFORE_INPUTS')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
        report=throughput_physical(plan);checked_inputs(plan)
        need(larger_source_clean(plan)and ref(POLICY)==policy_ref and ref(THROUGHPUT_PLAN)==binding
            and common.firewall_state()['denied_attempts']==0,'THROUGHPUT_SOURCE_POLICY_BOOKEND')
        result=publish(THROUGHPUT_ROOT/'RESULT.json',{'stage':THROUGHPUT_STAGE,'status':'COMPLETE_ATR03_THROUGHPUT_DIAGNOSTIC',
            'plan':binding,'claim':claim,'report':report,'source_unchanged':True,'guard':guard,
            'optimizer_steps':0,'training_authorized':False,'test_data_used':False,
            'clean_process_io_firewall':common.firewall_state(),'elapsed_seconds':time.monotonic()-began})
        terminal=publish(THROUGHPUT_ROOT/'TERMINAL.json',{'stage':THROUGHPUT_STAGE,'status':'COMPLETE','exit_code':0,
            'plan':binding,'claim':claim,'result':result,'source_unchanged':True,'remaining_invocations':0,
            'relaunch_allowed':False,'test_data_used':False})
        print(json.dumps({'result':result,'terminal':terminal}))
    except BaseException as error:
        publish(THROUGHPUT_ROOT/'TERMINAL.json',{'stage':THROUGHPUT_STAGE,'status':'FAILED_CONSUMED','exit_code':1,
            'plan':binding,'claim':claim,'result':result,'error':type(error).__name__+':'+str(error)[:2000],
            'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)




LARGER_FIT_MODES=('ta_larger_mechanics','ta_larger_main','ta_five_year_mechanics','ta_five_year_main')
LARGER_VRAM_MAX=12*1024**3
LARGER_SOURCE_PATHS=tuple(dict.fromkeys((*NOISE_SOURCE_PATHS,'gx1/audit/build_native_m5_ta_feature_cache_v1.py','tests/test_gx1_capped_run_native_year_contract.py')))
def prepare_larger_input_train_fold(*, snapshot_ref, noise_dataset_ref,
                                    fold_bounds, margin_atr):
    """Reload existing 27 arrays; fit fresh stats on the entire input-TRAIN axis.

    fold_bounds: exactly (train_start_ns, train_end_ns, control_end_ns), Python
    ints on the 300s grid. No chosen dates or defaults are present here.
    """
    import numpy as np
    from gx1.audit import build_native_m5_ta_feature_cache_v1 as cache
    from gx1.contracts import market_state_native_m5_feature_inputs_v1 as features
    from gx1.contracts import market_state_native_m5_targets_v1 as targets

    need(type(fold_bounds) is tuple and len(fold_bounds) == 3
         and all(type(t) is int and t > 0 and t % targets.TICK_NS == 0
                 for t in fold_bounds), 'LARGER_EXPLICIT_FOLD_CLOCKS')
    start, cut, stop = fold_bounds
    need(start < cut < stop, 'LARGER_CHRONOLOGICAL_FOLD')
    header = read(snapshot_ref)
    need(len(header['arrays']) == 27, 'LARGER_PHYSICAL_ARRAY_INVENTORY')
    # Owner verifies all 27 dtype/shape/byte hashes, full archive bookends and
    # reconstructed provider axes/readiness, keeping ORIGINAL global source IDs.
    provider = cache.load_snapshot(snapshot_ref)
    axes = provider.surfaces['axes']
    need(axes == header['surfaces_metadata']['axes']
         and provider.source_axes_sha256 == axes['contract_sha256']
         and axes['issue_start_ns'] <= start < stop <= axes['issue_end_ns'],
         'LARGER_RELOADED_SOURCE_AXES')
    spec = targets.noise_margin_dataset_contract(margin_atr)
    dataset = read(noise_dataset_ref)
    need(dataset['schema_version'] == 'gx1_native_m5_noise_dataset_artifact_v1'
         and dataset['contract'] == spec and dataset['raw_fields_byte_identical'] is True
         and dataset['test_data_used'] is False, 'LARGER_NOISE_DATASET_IDENTITY')
    # Requires ONLY the small explicit fold_bounds extension documented above.
    # The same existing reader validates the entire immutable historical payload,
    # joins issue_ns AND source_row, then rematures each TF at the actual fold end.
    cohorts, axis = noise_fit_axes(
        provider, dataset, spec, fold_bounds=fold_bounds)
    positions = np.flatnonzero((provider.issue_ns >= start) & (provider.issue_ns < cut))
    train_rows = cohorts['TRAIN']['rows']
    need(provider.input_ready[positions].all()
         and train_rows.dtype == np.int64
         and np.array_equal(train_rows, provider.source_row[positions]),
         'LARGER_NO_TARGET_FILTER_ON_INPUT_TRAIN_AXIS')
    # Deliberately use every declared input-TRAIN row, NOT fit_rows/eligible.
    # Last labels may be immature: their source inputs remain causal TRAIN data.
    normalization = features.fit_native_m5_feature_normalization(
        provider, fit_source_rows=train_rows, cutoff_ns=cut)
    population = normalization['fit_population']
    need(population['issue_rows'] == len(train_rows)
         and population['context_rows'] == len(train_rows)
         and population['cutoff_ns'] == cut
         and population['selected_source_rows_sha256'] == digest(train_rows),
         'LARGER_WHOLE_INPUT_TRAIN_NORMALIZATION')
    binding = provider.input_binding(normalization)
    need(binding['source_axes_sha256'] == axes['contract_sha256'],
         'LARGER_FRESH_STATS_SOURCE_AXES')
    truth = cohorts['TRAIN']['truth']
    predictive = (truth['noise_open_eligible'] | truth['noise_close_eligible']).any(1)
    report = {
        'snapshot': snapshot_ref, 'archive': header['archive'],
        'noise_dataset': noise_dataset_ref, 'fold_bounds': list(fold_bounds),
        'source_axes_sha256': axes['contract_sha256'],
        'source_binding_sha256': provider.surfaces['source_binding']['contract_sha256'],
        'input_binding': binding, 'fit_population': population,
        'physical_array_count': 27, 'normalization_fits': 1, 'prior_fits': 0,
        'input_TRAIN_rows': len(train_rows),
        'normalization_rows_selected_by_future_masks': False,
        'input_TRAIN_rows_with_predictive_labels': int(predictive.sum()),
        'input_TRAIN_rows_with_only_immature_or_unknown_labels': int((~predictive).sum()),
        'per_tf_mature_counts': truth['mature'].sum(0).tolist(),
        'per_tf_noise_open_predictive': truth['noise_open_eligible'].sum(0).tolist(),
        'per_tf_noise_close_predictive': truth['noise_close_eligible'].sum(0).tolist(),
        'CONTROL_used_for_statistics_or_selection': False,
    }
    need(cache.ref(header['archive']['path']) == header['archive']
         and ref(dataset['arrays']['path']) == dataset['arrays'],
         'LARGER_INPUT_ARCHIVE_BOOKENDS')
    return {'provider': provider, 'cohorts': cohorts, 'axis': axis,
            'normalization': normalization, 'dataset_contract': spec,
            'train_input_rows': train_rows, 'report': report}


def fit_larger_noise_prior(prepared, *, fit_source_rows):
    """Fit the prior on EXACT explicit FIT IDs, after whole-TRAIN input stats.

    Caller supplies a prebound source-only policy (root may choose full TRAIN).
    This function does not sample and does not infer a policy from outcomes.
    """
    import numpy as np
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective

    rows = fit_source_rows
    train = prepared['train_input_rows']
    need(isinstance(rows, np.ndarray) and rows.dtype == np.int64
                and rows.ndim == 1 and len(rows) > 0
                and np.all(np.diff(rows) > 0), 'LARGER_EXPLICIT_FIT_SOURCE_ROWS')
    at = np.searchsorted(train, rows)
    need(np.all(at < len(train)) and np.array_equal(train[at], rows),
                'LARGER_FIT_ROWS_SUBSET_OF_INPUT_TRAIN')
    cohort = prepared['cohorts']['TRAIN']
    truth = slice_truth(cohort, rows)
    spec = prepared['dataset_contract']
    prior = objective.fit_native_bar_train_prior(
        truth=truth, fold_upper_ns=cohort['upper_ns'],
        target_contract_sha256=spec['contract_sha256'], target_contract=spec)
    need(prior['fit_population']['issue_rows'] == len(rows)
                and prior['fit_population']['cutoff_ns'] == cohort['upper_ns'],
                'LARGER_EXACT_FIT_PRIOR_POPULATION')
    return {'prior': prior, 'FIT_source_rows_sha256': digest(rows),
            'FIT_rows': len(rows), 'prior_fits': 1, 'normalization_fits': 0,
            'target_contract_sha256': spec['contract_sha256'],
            'normalization_contract_sha256': prepared['normalization']['contract_sha256']}




def _larger_fit(plan):
    return plan.get('mode') in LARGER_FIT_MODES

def larger_recipe(value, mode, metadata):
    fields = {'optimizer','gradient_clip_norm','scheduler','seed','batch_rows',
        'score_batch_rows','fit_batches','score_batches_per_split',
        'checkpoint_every_updates','outer_seconds','internal_seconds',
        'required_windows_boot_id_greater_than','passes'}
    need(type(value) is dict and set(value) == fields, 'LARGER_RECIPE_FIELDS')
    need(mode in LARGER_FIT_MODES and value['optimizer'] == ADAMW_RECIPE
        and value['gradient_clip_norm'] == 1. and value['scheduler'] is None
        and value['seed'] == 20260911 and value['batch_rows'] == 64
        and value['score_batch_rows'] in (64,256) and value['passes'] == 1
        and value['score_batches_per_split'] is None
        and type(value['required_windows_boot_id_greater_than']) is int
        and value['required_windows_boot_id_greater_than'] == 666,
        'LARGER_UNCHANGED_NUMERIC_RECIPE')
    if value['score_batch_rows']==256:
        qualification=metadata.get('runtime_qualification',{})
        need(mode=='ta_five_year_main'and qualification.get('b256_equal_tf_candidate_qualified')is True
            and qualification.get('b256_equal_tf_measured_faster')is True,
            'LARGER_SCORE_256_EXACT_QUALIFICATION')
    for key in ('fit_batches','checkpoint_every_updates','outer_seconds',
                'internal_seconds'):
        need(type(value[key]) is int and value[key] > 0, 'LARGER_INTEGER:'+key)
    n = metadata['prepared']['full_selected_cohorts']['FIT']['rows']
    mechanics = _mechanics(mode)
    epoch = 2 if mechanics else (n + 63) // 64
    need(type(n) is int and n >= 128 and metadata['epoch_batches'] == epoch,
         'LARGER_WHOLE_INPUT_TRAIN_EPOCH')
    need(value['fit_batches'] == (2 if mechanics else epoch)
        and (value['checkpoint_every_updates'] in (256,1024) if not mechanics and _larger_spec(mode)['scope']=='five_year' else value['checkpoint_every_updates'] == (1 if mechanics else epoch))
        and value['checkpoint_every_updates'] <= value['fit_batches']
        and (value['outer_seconds']==1800 if mechanics else
            value['outer_seconds']==_larger_spec(mode)['main_seconds_max']if _larger_spec(mode)['scope']=='five_year'else value['outer_seconds']==3600)
        and value['internal_seconds'] == value['outer_seconds'] - 30,
        'LARGER_EXPLICIT_ONE_PASS_BUDGET')
    return dict(value)

def larger_batches(rows):
    """Source-only partition; unknown last labels never select/exclude rows."""
    need(isinstance(rows, np.ndarray) and rows.dtype == np.int64
        and rows.ndim == 1 and len(rows) >= 128 and np.all(np.diff(rows) > 0),
        'LARGER_CHRONOLOGICAL_SOURCE_ROWS')
    blocks = all_score_batches(rows, 64)
    if len(rows) % 64 == 1:
        blocks[-2:] = [rows[-65:-2], rows[-2:]]
    need(len(blocks) == (len(rows)+63)//64
        and all(2 <= len(x) <= 64 for x in blocks)
        and np.array_equal(np.concatenate(blocks), rows),
        'LARGER_NO_DROPPED_OR_DUPLICATE_INPUT')
    return blocks

def larger_schedule(rows, plan):
    blocks = larger_batches(rows)
    if _mechanics(plan['mode']): blocks = blocks[:2]
    body = {'schema_version':'gx1_native_m5_ta_larger_source_schedule_v1',
        'selection':'all_chronological_input_TRAIN_no_target_filter',
        'passes':1,'batch_rows':64,'tail_rule':'singleton_tail_63_plus_2',
        'population_source_rows_sha256':digest(rows),'population_rows':len(rows),
        'consumed_rows':sum(map(len,blocks)),
        'batch_sizes':[len(x) for x in blocks],
        'batch_source_rows_sha256':[digest(x) for x in blocks]}
    return {**body, 'contract_sha256':hashlib.sha256(
        json.dumps(body,sort_keys=True).encode()).hexdigest()}

def larger_selected(context, plan):
    train = context['cohorts']['TRAIN']; control = context['cohorts']['CONTROL']
    selected = {'FIT':train['rows'], 'CONTROL':control['rows']}
    mech = _mechanics(plan['mode'])
    def identity(rows, cohort):
        return {'rows':len(rows),'source_rows_sha256':digest(rows),
                'issue_ns_sha256':digest(slice_truth(cohort,rows)['issue_ns'])}
    full = {name:identity(rows, train if name == 'FIT' else control)
            for name,rows in selected.items()}
    scored = {name:identity(rows[:64] if mech else rows,
              train if name == 'FIT' else control) for name,rows in selected.items()}
    need(all(all(v == plan['metadata']['prepared']['full_selected_cohorts'][k][f]
                 for f,v in item.items()) for k,item in full.items())
        and all(all(v == plan['metadata']['selected_cohorts'][k][f]
                    for f,v in item.items()) for k,item in scored.items()),
        'LARGER_SOURCE_AND_SCORE_POPULATIONS')
    need(context['normalization']['fit_population']['issue_rows'] == len(train['rows'])
        and context['prior']['fit_population']['issue_rows'] == len(train['rows'])
        and train['upper_ns'] == plan['metadata']['fit_end_ns']
        and control['upper_ns'] == plan['metadata']['control_end_ns']
        and np.array_equal(train['fit_rows'],train['rows']), 'LARGER_ALL_TRAIN_FOLD')
    schedule = larger_schedule(selected['FIT'],plan)
    need(schedule == plan['training_schedule']
        and len(schedule['batch_sizes']) == plan['metadata']['epoch_batches'],
        'LARGER_EXACT_SOURCE_ONLY_SCHEDULE')
    for batch in larger_training_batches(selected,plan):
        truth = slice_truth(train,batch)
        need(bool((truth['noise_open_eligible'] | truth['noise_close_eligible']).any()),
             'LARGER_DECLARED_BATCH_HAS_NO_PREDICTIVE_CELL')
    return selected, scored

def larger_training_batches(selected, plan):
    batches = larger_batches(selected['FIT'])
    if _mechanics(plan['mode']):
        need(len(batches) >= 2 and all(len(x) == 64 for x in batches[:2]), 'LARGER_TWO_DISTINCT_B64')
        return batches[:2]
    return batches

def larger_schedule_position(plan, cursor):
    need(type(cursor) is int and 0 <= cursor <= plan['recipe']['fit_batches'],
         'LARGER_CURSOR')
    schedule = plan['training_schedule']; epoch = plan['metadata']['epoch_batches']
    body = {k:v for k,v in schedule.items() if k != 'contract_sha256'}
    need(schedule['contract_sha256'] == hashlib.sha256(
        json.dumps(body,sort_keys=True).encode()).hexdigest()
        and len(schedule['batch_sizes']) == epoch, 'LARGER_CURSOR_SCHEDULE')
    block = cursor if cursor < plan['recipe']['fit_batches'] else None
    return {'completed_updates':cursor,'next_pass':cursor//epoch,
        'next_batch_in_pass':cursor%epoch,'batches_per_pass':epoch,
        'declared_passes':1,'training_schedule_sha256':schedule['contract_sha256'],
        'next_source_block':block,'next_source_batch_sha256':
            schedule['batch_source_rows_sha256'][block] if block is not None else None}

def larger_limits(recipe, mode, metadata):
    mech = _mechanics(mode); updates = 3 if mech else recipe['fit_batches']
    score_rows=recipe['score_batch_rows']
    final_score = sum((v['rows']+score_rows-1)//score_rows for v in metadata['selected_cohorts'].values())
    return {'model_constructions':2 if mech else 1,
        'model_forwards':updates+(4 if mech else final_score),
        'backward_calls':updates,'optimizer_steps':updates,
        'autograd_calls':6 if mech else 0}

def larger_initial_reuse(model, context, plan, prepared, score_record, captures):
    """Called only after root verifies immutable COMPLETE prepare/source refs."""
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    need(_larger_main(plan)
        and canonical_model_state_sha256(model.state_dict()) == prepared['initial_state_sha256']
        and prepared['axis'] == context['axis']
        and prepared['normalization'] == context['normalization_ref']
        and prepared['prior'] == context['prior_ref']
        and prepared['input_binding'] == context['provider'].input_binding(context['normalization'])
        and prepared['model_contract'] == model.research_objective
        and prepared['dataset_contract'] == plan['metadata']['target_contract'],
        'LARGER_REUSE_EXACT_FRESH_INITIAL_IDENTITY')
    model.require_prior_initialization_state()
    reports = score_record['scores']
    for split, label in (('TRAIN','FIT'),('CONTROL','CONTROL')):
        expected = plan['metadata']['selected_cohorts'][label]
        rows = captures[split+'__source_row']
        need(reports[split]['rows'] == expected['rows'] == len(rows)
            and reports[split]['source_rows_sha256'] == expected['source_rows_sha256'] == digest(rows)
            and digest(captures[split+'__truth_issue_ns']) == expected['issue_ns_sha256']
            and np.array_equal(rows,context['cohorts'][split]['rows']),
            'LARGER_REUSE_ALLPOOL_INITIAL:'+split)
        truth = slice_truth(context['cohorts'][split],rows)
        need(all(np.array_equal(captures[split+'__truth_'+k],v,equal_nan=True)
                 for k,v in truth.items()), 'LARGER_REUSE_IDENTICAL_TRUTH:'+split)
    # Existing FP32-prior-logit comparison, with expected counts from metadata.
    proof = _ta_initial_prior_proof(model,context,captures,plan)
    return reports, captures, proof


LARGER_PREPARE_STAGE='native_m5_ta_atr03_year_prepare_20261120_001'
LARGER_PREPARE_PLAN=REPO/'configs/research/NATIVE_M5_TA_ATR03_YEAR_PREPARE_20261120_001.json'
LARGER_PREPARE_ROOT=Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_YEAR_PREPARE_20261120_001')
LARGER_FOLD_BOUNDS=(1693526400000000000,1725148800000000000,1727740800000000000)

FIVE_YEAR_PREPARE_STAGE='native_m5_ta_atr03_five_year_prepare_20261120_001'
FIVE_YEAR_PREPARE_PLAN=REPO/'configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_PREPARE_20261120_001.json'
FIVE_YEAR_PREPARE_ROOT=Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_FIVE_YEAR_PREPARE_20261120_001')
FIVE_YEAR_FOLD_BOUNDS=(1609459200000000000,1767225600000000000,1782864000000000000)
LARGER_SCOPE_SPECS={
    'year':{'scope':'year','prepare_mode':'ta_larger_prepare','mechanics_mode':'ta_larger_mechanics','main_mode':'ta_larger_main',
        'prepare_stage':LARGER_PREPARE_STAGE,'prepare_plan':LARGER_PREPARE_PLAN,'prepare_root':LARGER_PREPARE_ROOT,
        'fold_bounds':LARGER_FOLD_BOUNDS,'main_seconds_max':3600,
        'prepare_status':'COMPLETE_LARGER_TRAIN_INITIAL_PREPARATION',
        'mechanics_status':'COMPLETE_NATIVE_M5_TA_ATR03_YEAR_MECHANICS'},
    'five_year':{'scope':'five_year','prepare_mode':'ta_five_year_prepare','mechanics_mode':'ta_five_year_mechanics','main_mode':'ta_five_year_main',
        'prepare_stage':FIVE_YEAR_PREPARE_STAGE,'prepare_plan':FIVE_YEAR_PREPARE_PLAN,'prepare_root':FIVE_YEAR_PREPARE_ROOT,
        'fold_bounds':FIVE_YEAR_FOLD_BOUNDS,'main_seconds_max':10800,
        'prepare_status':'COMPLETE_NATIVE_M5_TA_ATR03_FIVE_YEAR_PREPARATION',
        'mechanics_status':'COMPLETE_NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS'}}


def _larger_spec(value):
    mode=value.get('mode')if type(value)is dict else value
    for key,spec in LARGER_SCOPE_SPECS.items():
        if mode==key or mode in(spec['prepare_mode'],spec['mechanics_mode'],spec['main_mode']):return spec
    need(False,'LARGER_EXPLICIT_SCOPE')


def _larger_main(plan):
    return _larger_fit(plan)and _ta_main(plan['mode'])


def _larger_prepare_execution(spec,outer_seconds):
    if spec['scope']=='year':
        need(outer_seconds in(None,1800),'LARGER_YEAR_FIXED_PREPARE_BUDGET');outer_seconds=1800
    else:
        need(type(outer_seconds)is int and outer_seconds==5400,'FIVE_YEAR_EXPLICIT_PREPARE_BUDGET')
    return {'class':'trainer','device':'cuda','memory':'20G','swap':'512M','numerical_threads':8,'tasks_max':128,
        'outer_seconds':outer_seconds,'internal_seconds':outer_seconds-30,'cuda_peak_reserved_bytes_max':LARGER_VRAM_MAX}


def _require_five_year_cache(inputs,q,result,terminal):
    parent=read(result['plan']);claim=read(result['claim']);snapshot=read(inputs['snapshot'])
    expected='native_m5_ta_five_year_cache_20261120_001'
    need(q['schema_version']=='gx1_native_m5_five_year_TA_input_qualification_v1'
        and result['stage']==terminal['stage']==parent['stage']==claim['stage']==expected
        and terminal['plan']==result['plan']==claim['plan']and terminal['claim']==result['claim']
        and result['test_data_used']is terminal['test_data_used']is False
        and result['training_authorized']is False
        and q['independent_qualified_arrays']==27 and q['source_policy_artifact_bookends_unchanged']is True
        and q['normalization_fits']==q['optimizer_steps']==q['model_constructions_or_forwards']==0
        and snapshot['archive']==q['archive']and ref(snapshot['archive']['path'])==snapshot['archive']
        and snapshot['surfaces_metadata']['axes']['issue_start_ns']==FIVE_YEAR_FOLD_BOUNDS[0]
        and snapshot['surfaces_metadata']['axes']['issue_end_ns']==FIVE_YEAR_FOLD_BOUNDS[2],
        'FIVE_YEAR_INDEPENDENTLY_QUALIFIED_CACHE')




def larger_clock_metadata(snapshot_ref,fold_bounds,*,return_source_rows=False):
    saved=read(snapshot_ref);archive=saved['archive']
    need(ref(archive['path'])==archive,'LARGER_CLOCK_ARCHIVE_HASH')
    with np.load(archive['path'],allow_pickle=False)as z:
        arrays={key:z[key]for key in ('bar_start_ns','retained_issue_rows')}
    for key,a in arrays.items():
        h=saved['arrays'][key];need(a.dtype==np.int64 and a.ndim==1 and a.dtype.str==h['dtype']
            and list(a.shape)==h['shape']and digest(a)==h['sha256'],'LARGER_CLOCK_BYTES:'+key)
    starts=arrays['bar_start_ns'];rows=arrays['retained_issue_rows'];times=starts[rows]+300_000_000_000
    a,b,c=fold_bounds;axis={};selected={}
    for name,low,high in [('TRAIN',a,b),('CONTROL',b,c)]:
        pick=(times>=low)&(times<high);source=rows[pick];issues=times[pick];selected[name]=source
        need(len(source)>128 and np.all(np.diff(source)>0),'LARGER_CLOCK_POPULATION')
        axis[name]={'start_ns':low,'end_exclusive_ns':high,'input_ready_rows':len(source),
            'input_not_ready_rows':0,'source_rows_sha256':digest(source),'issue_ns_sha256':digest(issues)}
    return (axis,selected)if return_source_rows else axis


def build_larger_prepare_plan(*,inputs,fold_bounds,margin_atr,source_base_commit=None,scope='year',outer_seconds=None):
    from gx1.contracts.market_state_native_m5_targets_v1 import noise_margin_dataset_contract
    selected=_larger_spec(scope);execution=_larger_prepare_execution(selected,outer_seconds)
    fields={'cache_qualification','cache_result','cache_terminal','snapshot','noise_dataset','base_fit_plan'}
    need(type(inputs)is dict and set(inputs)==fields and tuple(fold_bounds)==selected['fold_bounds']
        and margin_atr==0.3,'LARGER_PREPARE_INPUT_SCOPE')
    q,r,t=(read(inputs[k])for k in ('cache_qualification','cache_result','cache_terminal'))
    need(q['result']==inputs['cache_result']and q['terminal']==inputs['cache_terminal']and q['snapshot']==inputs['snapshot']
        and q['all_issue_rows_ready']is True and q['all61_macro_fields_observed_on_all_issue_rows']is True
        and t['status']=='COMPLETE'and t['exit_code']==0 and t['result']==inputs['cache_result']
        and r['source_unchanged']is True and r['report']['snapshot']==inputs['snapshot']
        and r['report']['archive']==q['archive'] and q['GC_used']is q['M1_used']is q['test_data_used']is False,'LARGER_QUALIFIED_CACHE')
    if scope=='five_year':_require_five_year_cache(inputs,q,r,t)
    dataset=read(inputs['noise_dataset']);target=noise_margin_dataset_contract(margin_atr)
    need(dataset['contract']==target and dataset['raw_fields_byte_identical']is True,'LARGER_QUALIFIED_NOISE')
    old=read(inputs['base_fit_plan']);saved=read(inputs['snapshot']);axis=larger_clock_metadata(inputs['snapshot'],tuple(fold_bounds))
    meta={'source_snapshot':inputs['snapshot'],'noise_dataset':inputs['noise_dataset'],'fold_bounds':list(fold_bounds),
        'fit_start_ns':fold_bounds[0],'fit_end_ns':fold_bounds[1],'control_end_ns':fold_bounds[2],
        'source_pool':{'result':inputs['cache_result'],'axis':axis},'target_contract':target,
        'source_axes_sha256':saved['surfaces_metadata']['axes']['contract_sha256'],
        'architecture':old['metadata']['architecture'],'seed':old['recipe']['seed'],
        'dependencies':{'cache_archive':saved['archive'],'noise_arrays':dataset['arrays']},
        'scope':'all_ready_input_TRAIN_year_no_target_filter_full_initialbaseline_and_later_reused_development_CONTROL'}
    if scope=='five_year':
        need(saved['surfaces_metadata']['axes']['issue_start_ns']==selected['fold_bounds'][0]
            and saved['surfaces_metadata']['axes']['issue_end_ns']==selected['fold_bounds'][2]
            and q['qualified_issue_rows']==sum(v['input_ready_rows']for v in axis.values()),'FIVE_YEAR_FULL_CACHE_ISSUE_SCOPE')
        meta.update(research_scope='five_year',later_split_role='development_VAL_Jan_Jun2026_reused_not_sealed_TEST',
            scope='whole_2021_2025_input_TRAIN_fresh_stats_prior_fullinitial_then_Jan_Jun2026_development_VAL')
    forwards=sum((v['input_ready_rows']+63)//64 for v in axis.values())
    return {'schema_version':'gx1_native_m5_larger_prepare_plan_v1','stage':selected['prepare_stage'],'mode':selected['prepare_mode'],
        'status':'PREPARED_NOT_ADMITTED','output_root':str(selected['prepare_root']),'inputs':inputs,'metadata':meta,
        'source_base_commit':source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'source_owners':{n:sha(REPO/n)for n in LARGER_SOURCE_PATHS},'margin_atr':margin_atr,
        'recipe':{'score_batch_rows':64,'seed':meta['seed']},
        'execution':execution,
        'limits':{'model_constructions':1,'model_forwards':forwards,'normalization_fits':1,'prior_fits':1,'optimizer_steps':0},
        'training_authorized':False,'test_data_used':False,'economic_edge_proven':False}


def larger_prepare_physical(plan):
    import torch
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_bar_direction_model
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_train_prior_logits
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    meta=plan['metadata'];root=Path(plan['output_root']);began=time.monotonic()
    data=prepare_larger_input_train_fold(snapshot_ref=meta['source_snapshot'],noise_dataset_ref=meta['noise_dataset'],
        fold_bounds=tuple(meta['fold_bounds']),margin_atr=plan['margin_atr'])
    context=larger_standard_context(data)
    prior=fit_larger_noise_prior(data,fit_source_rows=data['train_input_rows'])['prior']
    norm_ref=publish(root/'NORMALIZATION.json',data['normalization']);prior_ref=publish(root/'MATCHED_FIT_PRIOR.json',prior)
    context.update(normalization_ref=norm_ref,prior_ref=prior_ref,prior=prior)
    for name,v in meta['source_pool']['axis'].items():
        need(all(context['axis'][name][k]==x for k,x in v.items()),'LARGER_ACTUAL_SOURCE_AXIS:'+name)
    for rows in larger_batches(context['cohorts']['TRAIN']['rows']):
        truth=slice_truth(context['cohorts']['TRAIN'],rows,fit=True)
        need(bool((truth['noise_open_eligible']|truth['noise_close_eligible']).any()),'LARGER_EACH_UPDATE_HAS_PREDICTIVE_CELLS')
    data_seconds=time.monotonic()-began;provider=context['provider'];seed_native_fit(meta['seed'])
    torch.cuda.set_per_process_memory_fraction(plan['execution']['cuda_peak_reserved_bytes_max']/torch.cuda.get_device_properties(0).total_memory,0)
    model=build_native_bar_direction_model(input_normalization=context['normalization'],input_binding=provider.input_binding(context['normalization']),
        architecture=meta['architecture'],device=torch.device('cuda',0),train_prior=prior,
        backbone_gradient_policy=PRIMARY_BACKBONE_GRADIENT_POLICY,target_contract=meta['target_contract'])
    model.require_prior_initialization_state();initial=canonical_model_state_sha256(model.state_dict());calls=[0]
    def count(module,args):calls[0]+=1;need(calls[0]<=plan['limits']['model_forwards'],'LARGER_INITIAL_FORWARD_LIMIT')
    hook=model.register_forward_pre_hook(count)
    panels={name:all_score_batches(c['rows'],64)for name,c in context['cohorts'].items()}
    rng_before=state_sha(training_rng())
    scores,captures=fit_score(model,context,panels,plan);hook.remove()
    need(state_sha(training_rng())==rng_before,'LARGER_INITIAL_RNG_BOOKEND')
    need(torch.cuda.max_memory_reserved()<=LARGER_VRAM_MAX,'LARGER_PREPARE_VRAM_LIMIT')
    need(calls[0]==plan['limits']['model_forwards']and canonical_model_state_sha256(model.state_dict())==initial,'LARGER_FRESH_STATE_BOOKEND')
    selected={}
    for name,key in [('TRAIN','FIT'),('CONTROL','CONTROL')]:
        rows=captures[name+'__source_row'];expected=native_bar_train_prior_logits(prior,rows=len(rows),target_contract=meta['target_contract'])
        need(all(np.array_equal(captures[name+'__'+k],v)for k,v in expected.items()),'LARGER_FULL_INITIAL_PRIOR_LOGITS')
        selected[key]={'rows':len(rows),'source_rows_sha256':digest(rows),'issue_ns_sha256':digest(provider.issue_ns[rows])}
    score_ref=publish(root/'INITIAL_SCORE.json',{'scores':scores,'selected_cohorts':selected,'model_state_sha256':initial,
        'normalization':norm_ref,'prior':prior_ref,'target_contract':meta['target_contract']})
    stream=io.BytesIO();np.savez_compressed(stream,**captures);capture=publish(root/'INITIAL_CAPTURE.npz',stream.getvalue());stream.close()
    stream=io.BytesIO();torch.save(model.state_dict(),stream);state=publish(root/'INITIAL_MODEL.pt',stream.getvalue());stream.close()
    return {'axis':context['axis'],'data_report':data['report'],'normalization':norm_ref,'prior':prior_ref,
        'normalization_population':context['normalization']['fit_population'],'normalization_fits':1,'prior_fits':1,
        'initial_state_sha256':initial,'initial_score_artifact':score_ref,'initial_capture':capture,'initial_checkpoint':state,
        'input_binding':provider.input_binding(context['normalization']),'model_contract':model.research_objective,
        'full_selected_cohorts':selected,'dataset_contract':meta['target_contract'],
        'initial_score_rng_unchanged':True,'initial_rng_sha256':rng_before,'model_constructions':1,'model_forwards':calls[0],'optimizer_steps':0,
        'all_full_initial_logits_equal_prior':True,'data_prepare_seconds':data_seconds,
        'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'test_data_used':False,'economic_edge_proven':False}


def inspect_larger_prepare(plan_sha,*,admitted,scope='year'):
    selected=_larger_spec(scope);path=selected['prepare_plan'];root=selected['prepare_root']
    need(Path.cwd().resolve()==REPO and not root.exists(),'LARGER_PREPARE_CWD_OUTPUT')
    pr={'path':str(path),'sha256':plan_sha};plan=read(pr)
    need(plan==build_larger_prepare_plan(inputs=plan['inputs'],fold_bounds=tuple(plan['metadata']['fold_bounds']),margin_atr=plan['margin_atr'],
        source_base_commit=plan['source_base_commit'],scope=scope,
        outer_seconds=plan['execution']['outer_seconds']if scope=='five_year'else None)and larger_source_clean(plan),'LARGER_PREPARE_SOURCE')
    policy_ref=ref(POLICY);p=read(policy_ref);g=p.get(selected['prepare_stage'],{})
    need(all(p.get(k)is False for k in ta_audit.data.CLOSED_FLAGS)and g.get('plan')==pr
        and g.get('status')==('ADMITTED_ONE_USE'if admitted else'PREPARED_NOT_ADMITTED')
        and g.get('output_root')==str(root)and g.get('execution_authorized')is admitted
        and g.get('remaining_invocations')==(1 if admitted else 0)and g.get('max_invocations')==1
        and g.get('claimed_invocations')==g.get('completed_invocations')==0 and g.get('relaunch_allowed')is False
        and g.get('training_authorized')is False and g.get('test_authorized')is False,'LARGER_PREPARE_GATE')
    checked_inputs(plan);return plan,pr,policy_ref


def larger_prepare_cli(args):
    selected=_larger_spec('five_year'if args.plan==str(FIVE_YEAR_PREPARE_PLAN)else'year')
    path=selected['prepare_plan'];root=selected['prepare_root']
    need(args.plan==str(path)and args.out_dir==str(root)and args.device=='cuda','LARGER_PREPARE_CLI')
    if args.preflight:
        _,pr,p=inspect_larger_prepare(args.plan_sha256,admitted=False,scope=selected['scope'])
        print(json.dumps({'status':'PREFLIGHT_LARGER_PREPARATION_NO_MODEL','plan':pr,'policy':p,'claim_created':False}));return
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    began=time.monotonic();guard=require_guarded_cuda_trainer_execution();plan,pr,p=inspect_larger_prepare(args.plan_sha256,admitted=True,scope=selected['scope'])
    root.mkdir(exist_ok=False)
    common.firewall_state()
    claim=publish(root/'CLAIM.json',{'stage':selected['prepare_stage'],'plan':pr,'policy':p,'status':'CLAIMED_ONE_USE','guard':guard,'remaining_invocations':0,'max_invocations':1,'test_data_used':False})
    old=signal.getsignal(signal.SIGALRM);result=None
    def expired(*args):raise TimeoutError('LARGER_PREPARE_INTERNAL_LIMIT')
    try:
        remaining=plan['execution']['internal_seconds']-(time.monotonic()-began);need(remaining>0,'LARGER_PREPARE_TIME_BEFORE_INPUTS')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
        report=larger_prepare_physical(plan);checked_inputs(plan)
        need(larger_source_clean(plan)and ref(POLICY)==p and ref(path)==pr
            and common.firewall_state()['denied_attempts']==0,'LARGER_PREPARE_BOOKEND')
        result=publish(root/'RESULT.json',{'stage':selected['prepare_stage'],'status':selected['prepare_status'],
            'plan':pr,'claim':claim,'report':report,'guard':guard,'source_unchanged':True,'optimizer_steps':0,'training_authorized':False,'test_data_used':False,
            'elapsed_seconds':time.monotonic()-began,'clean_process_io_firewall':common.firewall_state()})
        terminal=publish(root/'TERMINAL.json',{'stage':selected['prepare_stage'],'status':'COMPLETE','exit_code':0,'plan':pr,
            'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        print(json.dumps({'result':result,'terminal':terminal}))
    except BaseException as error:
        publish(root/'TERMINAL.json',{'stage':selected['prepare_stage'],'status':'FAILED_CONSUMED','exit_code':1,'plan':pr,'claim':claim,
            'result':result,'error':type(error).__name__+':'+str(error)[:2000],'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


def larger_standard_context(data):
    cohorts=data['cohorts'];axis=data['axis']
    for name,c in cohorts.items():
        c['fit_rows']=c['rows'];c['fit_truth']=c['truth']
        axis[name].update(fit_rows=len(c['rows']),fit_source_rows_sha256=digest(c['rows']),fit_issue_ns_sha256=digest(c['truth']['issue_ns']),fit_row_policy='all_input_TRAIN_no_target_filter'if name=='TRAIN'else'all_input_CONTROL')
    return {**data,'feature_requalification':data['report'],'normalization_fits':1,'prior_fits':0}


def prepare_larger_native_fit(plan,root,*,reload_normalization=None,reload_prior=None):
    from gx1.audit import build_native_m5_ta_feature_cache_v1 as cache
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    meta=plan['metadata'];p=meta['prepared'];provider=cache.load_snapshot(meta['source_snapshot'])
    dataset=read(meta['noise_dataset']);need(dataset['contract']==meta['target_contract'],'LARGER_REUSED_TARGET_CONTRACT')
    cohorts,axis=noise_fit_axes(provider,dataset,meta['target_contract'],fold_bounds=tuple(meta['fold_bounds']))
    context=larger_standard_context({'provider':provider,'cohorts':cohorts,'axis':axis,'report':{}})
    need(axis==p['axis'],'LARGER_REUSED_AXES')
    norm=read(p['normalization']);provider.input_binding(norm)
    need(norm['fit_population']==p['normalization_population']and norm['fit_population']['cutoff_ns']==meta['fit_end_ns']and norm['fit_population']['selected_source_rows_sha256']==digest(cohorts['TRAIN']['rows']),
        'LARGER_EXACT_STATS_POPULATION')
    prior=objective.require_native_bar_train_prior(read(p['prior']),target_contract=meta['target_contract'])
    need(prior['fit_population']['issue_rows']==len(cohorts['TRAIN']['rows'])
        and prior['fit_population']['cutoff_ns']==meta['fit_end_ns']
        and prior['fit_population']['issue_ns_sha256']==digest(cohorts['TRAIN']['truth']['issue_ns']),'LARGER_EXACT_PRIOR_POPULATION')
    need(reload_normalization in(None,p['normalization'])and reload_prior in(None,p['prior']),'LARGER_COLD_STATS')
    context.update(normalization=norm,normalization_ref=p['normalization'],prior=prior,prior_ref=p['prior'],normalization_fits=0,prior_fits=0)
    return context


def larger_source_clean(plan):
    need(set(plan['source_owners'])==set(LARGER_SOURCE_PATHS),'LARGER_SOURCE_SET')
    adapted={**plan,'source_owners':{k:plan['source_owners'][k]for k in NOISE_SOURCE_PATHS}}
    return noise_source_clean(adapted)and all(sha(REPO/k)==v for k,v in plan['source_owners'].items())


def load_larger_initial(model,context,plan):
    saved=plan['metadata']['prepared']
    scores=read(saved['initial_score_artifact'])
    need(scores['model_state_sha256']==saved['initial_state_sha256']
        and scores['normalization']==context['normalization_ref']
        and scores['prior']==context['prior_ref']
        and scores['target_contract']==plan['metadata']['target_contract']
        and scores['selected_cohorts']==saved['full_selected_cohorts'],
        'LARGER_REUSED_SCORE_BINDINGS')
    need(ref(saved['initial_capture']['path'])==saved['initial_capture'],'LARGER_INITIAL_CAPTURE_HASH')
    with np.load(saved['initial_capture']['path'],allow_pickle=False)as archive:
        captures={key:archive[key]for key in archive.files}
    expected={split+'__'+key for split in ('TRAIN','CONTROL')for key in
        ('source_row','open_logits','close_logits',*('truth_'+k for k in context['cohorts'][split]['truth']))}
    need(set(captures)==expected,'LARGER_INITIAL_CAPTURE_FIELDS')
    return larger_initial_reuse(model,context,plan,saved,scores,captures)


def qualified_runtime_throughput(inputs,prepared_parent):
    need({'throughput_result','throughput_terminal'}<=set(inputs),'LARGER_RUNTIME_QUALIFICATION_ROLES')
    need(inputs['throughput_result']['path']==str(THROUGHPUT_ROOT/'RESULT.json')
        and inputs['throughput_terminal']['path']==str(THROUGHPUT_ROOT/'TERMINAL.json'),
        'LARGER_RUNTIME_EXACT_DIAGNOSTIC_SCOPE')
    result,terminal=read(inputs['throughput_result']),read(inputs['throughput_terminal'])
    parent=read(result['plan']);claim=read(result['claim']);report=result['report']
    need(result['stage']==terminal['stage']==parent['stage']==claim['stage']==THROUGHPUT_STAGE
        and result['status']=='COMPLETE_ATR03_THROUGHPUT_DIAGNOSTIC'
        and terminal['status']=='COMPLETE'and terminal['exit_code']==0
        and terminal['result']==inputs['throughput_result']and terminal['plan']==result['plan']==claim['plan']
        and terminal['claim']==result['claim']and result['source_unchanged']is True
        and result['optimizer_steps']==0 and report['actual_counts']==THROUGHPUT_LIMITS
        and report['train_bitexact']is report['b64_eval_bitexact']is True
        and all(report[k]is True for k in('weights_unchanged','rng_restored','grad_fields_none'))
        and report['normalization_fits']==report['prior_fits']==0
        and result['test_data_used']is terminal['test_data_used']is False
        and result['clean_process_io_firewall']['denied_attempts']==0
        and parent==build_throughput_plan(inputs=parent['inputs'],source_base_commit=parent['source_base_commit'])
        and larger_source_clean(parent),'LARGER_RUNTIME_QUALIFIED_THROUGHPUT')
    original_mechanics=read(parent['metadata']['analysis_parent'])
    need(original_mechanics['inputs']['prepare_result']==inputs['prepare_result']
        and original_mechanics['inputs']['prepare_terminal']==inputs['prepare_terminal']
        and original_mechanics['source_owners']==prepared_parent['source_owners']
        and parent['metadata']['initial_state_sha256']==read(inputs['prepare_result'])['report']['initial_state_sha256'],
        'LARGER_RUNTIME_ORIGINAL_PREPARATION')
    return {'result':inputs['throughput_result'],'terminal':inputs['throughput_terminal'],
        'plan':result['plan'],'claim':result['claim'],'source_compatibility':parent['inputs']['source_compatibility'],
        'reference_mechanics_result':parent['inputs']['mechanics_result'],
        'reference_mechanics_terminal':parent['inputs']['mechanics_terminal'],
        'train_bitexact':True,'b64_eval_bitexact':True,
        'b256_equal_tf_candidate_qualified':report['b256_equal_tf_candidate_qualified'],
        'b256_equal_tf_measured_faster':report['b256_equal_tf_measured_faster']}


def larger_prepared_metadata(inputs):
    required={'prepare_result','prepare_terminal'}
    need(required<=set(inputs),'LARGER_PREPARED_INPUTS')
    result,last=read(inputs['prepare_result']),read(inputs['prepare_terminal'])
    parent=read(result['plan']);claim=read(result['claim']);report=result['report'];selected=_larger_spec(parent)
    need(result['stage']==last['stage']==parent['stage']==claim['stage']==selected['prepare_stage']
        and result['status']==selected['prepare_status']
        and last['status']=='COMPLETE'and last['exit_code']==0
        and last['result']==inputs['prepare_result']and last['plan']==result['plan']==claim['plan']
        and last['claim']==result['claim']and result['source_unchanged']is True
        and result['optimizer_steps']==report['optimizer_steps']==0
        and report['normalization_fits']==report['prior_fits']==1
        and report['model_constructions']==1 and report['model_forwards']==parent['limits']['model_forwards']
        and report['all_full_initial_logits_equal_prior']is True
        and report['initial_score_rng_unchanged']is True
        and report['dataset_contract']==parent['metadata']['target_contract']
        and report['cuda_peak_reserved_bytes']<=LARGER_VRAM_MAX
        and result['test_data_used']is last['test_data_used']is False
        and result['clean_process_io_firewall']['denied_attempts']==0,'LARGER_COMPLETE_PREPARATION')
    need(set(parent['source_owners'])==set(LARGER_SOURCE_PATHS),'LARGER_PREPARED_SOURCE_SET')
    qualification=None
    if 'throughput_result'in inputs or 'throughput_terminal'in inputs:
        qualification=qualified_runtime_throughput(inputs,parent)
    else:
        need(all(sha(REPO/k)==v for k,v in parent['source_owners'].items()),'LARGER_PREPARED_IDENTICAL_SOURCE')
    from gx1.contracts.entry_model_native_input_normalization_v1 import require_native_m5_input_normalization_contract
    from gx1.contracts.market_state_ta_integrated_objective_v1 import require_native_bar_train_prior
    from gx1.contracts.market_state_ta_integrated_model_v1 import native_bar_model_contract
    norm=require_native_m5_input_normalization_contract(read(report['normalization']))
    prior=require_native_bar_train_prior(read(report['prior']),target_contract=report['dataset_contract'])
    binding=report['input_binding'];cut=parent['metadata']['fit_end_ns']
    need(norm['fit_population']==report['normalization_population']
        and norm['fit_population']['cutoff_ns']==prior['fit_population']['cutoff_ns']==cut
        and norm['fit_population']['issue_rows']==prior['fit_population']['issue_rows']==report['full_selected_cohorts']['FIT']['rows']
        and prior['fit_population']['issue_ns_sha256']==report['full_selected_cohorts']['FIT']['issue_ns_sha256']
        and norm['fit_population']['selected_source_rows_sha256']==report['full_selected_cohorts']['FIT']['source_rows_sha256']
        and binding['source_axes_sha256']==norm['source_axes_sha256']==parent['metadata']['source_axes_sha256']
        and binding['input_normalization_sha256']==norm['contract_sha256']
        and report['model_contract']==native_bar_model_contract(binding,train_prior=prior,
            backbone_gradient_policy=PRIMARY_BACKBONE_GRADIENT_POLICY,target_contract=report['dataset_contract']),
        'LARGER_PREPARED_INPUT_STATS_MODEL_IDENTITY')
    fields=('axis','normalization','prior','normalization_population','initial_state_sha256',
        'initial_score_artifact','initial_capture','input_binding','model_contract','full_selected_cohorts','dataset_contract')
    prepared={key:report[key]for key in fields}
    prepared.update(result=inputs['prepare_result'],terminal=inputs['prepare_terminal'],plan=result['plan'],claim=result['claim'])
    metadata={**parent['metadata'],'dependencies':{**parent['metadata']['dependencies'],
        'prepared_plan':result['plan'],'prepared_claim':result['claim'],
        'train_normalization':report['normalization'],'train_prior':report['prior'],
        'initial_scores':report['initial_score_artifact'],'initial_capture':report['initial_capture']},
        'prepared':prepared,'raw_gradient_design_evidence':{
            'role':'same_qualified_primary_backbone_policy_fresh_year_mechanics_required'}}
    if qualification is not None:
        metadata['runtime_qualification']=qualification
        metadata['dependencies'].update(runtime_throughput_result=qualification['result'],
            runtime_throughput_terminal=qualification['terminal'],runtime_source_compatibility=qualification['source_compatibility'])
    return metadata


def build_larger_fit_plan(*,inputs,recipe,mode,source_base_commit=None):
    need(mode in LARGER_FIT_MODES,'LARGER_FIT_MODE')
    fields={'prepare_result','prepare_terminal'}|({'throughput_result','throughput_terminal'}if mode in('ta_five_year_mechanics','ta_five_year_main')else set())|({'mechanics_result','mechanics_terminal'}if _larger_main({'mode':mode})else set())
    need(type(inputs)is dict and set(inputs)==fields,'LARGER_FIT_INPUT_ROLES')
    metadata=larger_prepared_metadata(inputs);selected_scope=_larger_spec(mode)
    need(metadata['prepared']['plan']['path']==str(selected_scope['prepare_plan']),'LARGER_SAME_PREPARED_SCOPE')
    clocks,sources=larger_clock_metadata(metadata['source_snapshot'],tuple(metadata['fold_bounds']),return_source_rows=True)
    full=metadata['prepared']['full_selected_cohorts']
    need(all(clocks[split]['source_rows_sha256']==full[label]['source_rows_sha256']
        and clocks[split]['issue_ns_sha256']==full[label]['issue_ns_sha256']
        and clocks[split]['input_ready_rows']==full[label]['rows']
        for split,label in [('TRAIN','FIT'),('CONTROL','CONTROL')]),'LARGER_EXACT_FULL_SOURCE_POPULATIONS')
    mechanics=_mechanics(mode);selected={}
    for split,label in [('TRAIN','FIT'),('CONTROL','CONTROL')]:
        rows=sources[split][:64]if mechanics else sources[split]
        # Source-only issue clock; no labels are loaded to select score cohorts.
        if mechanics:
            snapshot=read(metadata['source_snapshot'])
            with np.load(snapshot['archive']['path'],allow_pickle=False)as archive:
                issues=archive['bar_start_ns'][rows]+300_000_000_000
            item={'rows':len(rows),'source_rows_sha256':digest(rows),'issue_ns_sha256':digest(issues)}
        else:item=dict(full[label])
        selected[label]=item
    metadata.update(selected_cohorts=selected,epoch_batches=2 if mechanics else(len(sources['TRAIN'])+63)//64)
    recipe=larger_recipe(recipe,mode,metadata)
    schedule=larger_schedule(sources['TRAIN'],{'mode':mode})
    if not mechanics:
        done,last=read(inputs['mechanics_result']),read(inputs['mechanics_terminal'])
        parent=read(done['plan']);claim=read(done['claim']);proof=done['report']
        need(done['stage']==last['stage']==parent['stage']==claim['stage']==FIT_SCOPES[selected_scope['mechanics_mode']][0]
            and done['status']==selected_scope['mechanics_status']and parent['mode']==selected_scope['mechanics_mode']
            and last['status']=='COMPLETE'and last['exit_code']==0
            and last['result']==inputs['mechanics_result']and last['plan']==done['plan']==claim['plan']
            and last['claim']==done['claim']and done['source_unchanged']is True
            and done['test_data_used']is last['test_data_used']is False
            and done['clean_process_io_firewall']['denied_attempts']==0
            and done['optimizer_steps']==3 and proof['actual_counts']==parent['limits']
            and proof['actual_counts']['autograd_calls']==6 and proof['cold_resume']['bitexact']is True
            and proof['normalization_fits']==proof['prior_fits']==0
            and proof['primary_backbone_qualification']['auxiliary_shared_zero_all_steps']is True
            and proof['primary_backbone_qualification']['M5_shared_positive_after_first_head_update']is True
            and proof['initial_state_sha256']==metadata['prepared']['initial_state_sha256']
            and all(proof[k]==metadata['prepared'][k]for k in ('axis','normalization','prior','normalization_population'))
            and parent['metadata']['prepared']==metadata['prepared']
            and set(parent['source_owners'])==set(LARGER_SOURCE_PATHS)
            and all(sha(REPO/k)==v for k,v in parent['source_owners'].items()),'LARGER_MAIN_QUALIFIED_MECHANICS')
        if mode=='ta_five_year_main':
            need(proof.get('runtime_reference_trajectory',{}).get('bitexact')is True
                and proof['runtime_reference_trajectory']['reference_mechanics_result']
                    ==metadata['runtime_qualification']['reference_mechanics_result'],
                'LARGER_MAIN_RUNTIME_TRAJECTORY_BITEXACT')
        metadata['mechanics']={'result':inputs['mechanics_result'],'terminal':inputs['mechanics_terminal'],
            'plan':done['plan'],'claim':done['claim'],'initial_state_sha256':proof['initial_state_sha256'],
            'physical_windows_boot_id':done['physical_windows_boot_id']}
        metadata['dependencies'].update(mechanics_plan=done['plan'],mechanics_claim=done['claim'])
    stage,path,root=FIT_SCOPES[mode]
    return {'schema_version':('gx1_native_m5_ta_atr03_five_year_fit_plan_v1'if selected_scope['scope']=='five_year'else'gx1_native_m5_ta_atr03_year_fit_plan_v1'),'stage':stage,'mode':mode,
        'status':'PREPARED_NOT_ADMITTED','output_root':str(root),'inputs':inputs,'metadata':metadata,
        'source_base_commit':source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'source_owners':{name:sha(REPO/name)for name in LARGER_SOURCE_PATHS},'recipe':recipe,
        'training_schedule':schedule,'backbone_gradient_policy':PRIMARY_BACKBONE_GRADIENT_POLICY,
        'execution':{'class':'trainer','device':'cuda','memory':'20G','swap':'512M','tasks_max':128,'numerical_threads':8,
            'outer_seconds':recipe['outer_seconds'],'internal_seconds':recipe['internal_seconds'],'kill_after_seconds':10,
            'cuda_peak_reserved_bytes_max':LARGER_VRAM_MAX},
        'fit_order':'first_two_distinct_source_B64_technical'if mechanics else'all_chronological_input_TRAIN_once_singleton_tail_63_plus_2',
        'score_order':'first64_each_split_technical_only'if mechanics else'all_TRAIN_and_CONTROL_rows_including_unknown_tails',
        'normalization_axis':'all_input_TRAIN_before_prior_and_optimizer_no_target_filter',
        'GC_used':False,'statistics_fits':{'normalization':0,'matched_TRAIN_prior':0},
        'limits':larger_limits(recipe,mode,metadata),'qualification_weights_loaded':False,
        'scheduler_state':None,'EMA_state':None,'training_authorized':True,'test_data_used':False,'economic_edge_proven':False}



FIT_SCOPES.update({
    'ta_larger_mechanics':('native_m5_ta_atr03_year_mechanics_20261120_001',
        REPO/'configs/research/NATIVE_M5_TA_ATR03_YEAR_MECHANICS_20261120_001.json',
        Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_YEAR_MECHANICS_20261120_001')),
    'ta_larger_main':('native_m5_ta_atr03_year_fit_20261120_001',
        REPO/'configs/research/NATIVE_M5_TA_ATR03_YEAR_FIT_20261120_001.json',
        Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_YEAR_FIT_20261120_001'))})


FIT_SCOPES.update({
    'ta_five_year_mechanics':('native_m5_ta_atr03_five_year_mechanics_20261120_003',
        REPO/'configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_003.json',
        Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_003')),
    'ta_five_year_main':('native_m5_ta_atr03_five_year_fit_20261120_003',
        REPO/'configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_003.json',
        Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_003'))})

# Checkpoint windows extend this owner; each launch has a new exact admission.
CHECKPOINT_RECEIPT_SCHEMA='gx1_native_m5_main_checkpoint_receipt_v1'
CHECKPOINT_WINDOW_SCHEMA='gx1_native_m5_main_checkpoint_window_plan_v1'
CHECKPOINT_INTERRUPTION_SCHEMA='gx1_native_m5_execution_interruption_v1'


def checkpoint_process_identity():
    import os
    fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return {'pid':os.getpid(),'start_ticks':int(fields[19]),
        'wsl_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'physical_windows_boot_id':physical_windows_boot_id()}

def require_checkpoint_parent_stopped(identity):
    need(type(identity)is dict and set(identity)=={'pid','start_ticks','wsl_boot_id','physical_windows_boot_id'}
        and type(identity['pid'])is int and identity['pid']>0
        and type(identity['start_ticks'])is int and identity['start_ticks']>0
        and type(identity['wsl_boot_id'])is str and bool(identity['wsl_boot_id'])
        and type(identity['physical_windows_boot_id'])is int and identity['physical_windows_boot_id']>0,
        'CHECKPOINT_PARENT_IDENTITY')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    windows_boot=physical_windows_boot_id()
    need(windows_boot>=identity['physical_windows_boot_id'],'CHECKPOINT_PARENT_BOOT_ROLLBACK')
    if boot!=identity['wsl_boot_id']or windows_boot>identity['physical_windows_boot_id']:return
    path=Path('/proc')/str(identity['pid'])/'stat'
    try:actual=int(path.read_text().rsplit(')',1)[1].split()[19])
    except FileNotFoundError:return
    need(actual!=identity['start_ticks'],'CHECKPOINT_PARENT_STILL_RUNNING')

def checkpoint_state_proof(checkpoint):
    return {'cursor':checkpoint['cursor'],
        'model_state_sha256':state_sha(checkpoint['model_state']),
        'optimizer_state_sha256':state_sha(checkpoint['optimizer_state']),
        'rng_state_sha256':state_sha(checkpoint['rng_state']),
        'schedule_position_sha256':state_sha(checkpoint['schedule_position']),
        'bindings_sha256':state_sha(checkpoint['bindings']),
        'training':checkpoint['training']}


def checkpoint_prefix_counts(cursor):
    return {'model_forwards':cursor,'backward_calls':cursor,'optimizer_steps':cursor,'autograd_calls':0}


def publish_main_checkpoint(path,checkpoint,*,trajectory,trajectory_ref,execution_ref,claim,
                            steps,restored_cursor,current_counts):
    need(trajectory['mode']=='ta_five_year_main'
        and trajectory['recipe']['passes']==1 and len(steps)==checkpoint['cursor'],
        'CHECKPOINT_TRAJECTORY_PREFIX')
    rng_before=state_sha(training_rng())
    checkpoint_ref=publish_checkpoint(path,checkpoint)
    receipt={'schema_version':CHECKPOINT_RECEIPT_SCHEMA,'trajectory_plan':trajectory_ref,
        'execution_plan':execution_ref,'execution_claim':claim,'checkpoint':checkpoint_ref,
        'cursor':checkpoint['cursor'],'schedule_position':checkpoint['schedule_position'],
        'state_proof':checkpoint_state_proof(checkpoint),
        'source_base_commit':trajectory['source_base_commit'],
        'source_owners':trajectory['source_owners'],'source_unchanged':fit_source_clean(trajectory),
        'steps':list(steps),'restored_prefix_updates':restored_cursor,
        'current_window_counts':dict(current_counts),
        'logical_prefix_counts':checkpoint_prefix_counts(checkpoint['cursor']),
        'process_identity':checkpoint_process_identity(),'inherited_checkpoint_receipt':None,
        'phase':'TRAIN_COMPLETE_SCORE_PENDING'if checkpoint['cursor']==trajectory['recipe']['fit_batches']else'TRAIN_PREFIX',
        'test_data_used':False}
    need(receipt['source_unchanged']is True,'CHECKPOINT_SOURCE_CHANGED')
    receipt_path=path.parent/('CHECKPOINT_'+str(checkpoint['cursor']).zfill(6)+'.json')
    binding=publish(receipt_path,receipt)
    need(state_sha(training_rng())==rng_before,'CHECKPOINT_RECEIPT_PUBLICATION_RNG_MUTATION')
    return binding,receipt


def require_main_checkpoint_receipt(binding,trajectory,trajectory_ref):
    receipt=read(binding)
    fields={'schema_version','trajectory_plan','execution_plan','execution_claim','checkpoint',
        'cursor','schedule_position','state_proof','source_base_commit','source_owners',
        'source_unchanged','steps','restored_prefix_updates','current_window_counts',
        'logical_prefix_counts','process_identity','phase','test_data_used','inherited_checkpoint_receipt'}
    need(type(receipt)is dict and set(receipt)==fields
        and receipt['schema_version']==CHECKPOINT_RECEIPT_SCHEMA
        and receipt['trajectory_plan']==trajectory_ref and receipt['source_unchanged']is True
        and receipt['test_data_used']is False
        and receipt['source_base_commit']==trajectory['source_base_commit']
        and receipt['source_owners']==trajectory['source_owners'],'CHECKPOINT_RECEIPT_IDENTITY')
    cursor=receipt['cursor'];total=trajectory['recipe']['fit_batches']
    need(type(cursor)is int and 0<=cursor<=total
        and receipt['schedule_position']==larger_schedule_position(trajectory,cursor)
        and receipt['logical_prefix_counts']==checkpoint_prefix_counts(cursor)
        and type(receipt['restored_prefix_updates'])is int
        and 0<=receipt['restored_prefix_updates']<=cursor
        and type(receipt['steps'])is list and len(receipt['steps'])==cursor,
        'CHECKPOINT_RECEIPT_CURSOR')
    schedule=trajectory['training_schedule']['batch_source_rows_sha256']
    need(all(type(step)is dict and step.get('source_rows_sha256')==schedule[i]
             for i,step in enumerate(receipt['steps'])),'CHECKPOINT_RECEIPT_STEP_PREFIX')
    proof=receipt['state_proof']
    need(type(proof)is dict and set(proof)=={'cursor','model_state_sha256',
        'optimizer_state_sha256','rng_state_sha256','schedule_position_sha256','bindings_sha256','training'}
        and proof['cursor']==cursor and type(proof['training'])is bool
        and proof['schedule_position_sha256']==state_sha(receipt['schedule_position'])
        and all(type(proof[key])is str and len(proof[key])==64
            and all(c in '0123456789abcdef'for c in proof[key])
            for key in proof if key.endswith('_sha256')),'CHECKPOINT_RECEIPT_STATE_PROOF')
    execution=read(receipt['execution_plan']);claim=read(receipt['execution_claim'])
    root=Path(execution['output_root'])
    if receipt['execution_plan']==trajectory_ref:
        need(execution==trajectory and receipt['restored_prefix_updates']==0,'CHECKPOINT_RECEIPT_ORIGINAL_EXECUTION')
    else:
        _,_,expected_root=checkpoint_window_paths(receipt['execution_plan']['path'])
        inp=execution['inputs']
        expected=build_checkpoint_window_plan(plan_path=receipt['execution_plan']['path'],
            trajectory_plan=inp['trajectory_plan'],checkpoint_receipt=inp.get('checkpoint_receipt'),
            parent_end=inp.get('parent_end'),stop_after_cursor=execution['recipe']['stop_after_cursor'],
            outer_seconds=execution['execution']['outer_seconds'],source_base_commit=execution['source_base_commit'])
        need(execution==expected and inp['trajectory_plan']==trajectory_ref and root==expected_root
            and receipt['restored_prefix_updates']==execution['recipe']['start_cursor']
            and execution['recipe']['start_cursor']<=cursor<=execution['recipe']['stop_after_cursor'],
            'CHECKPOINT_RECEIPT_BOUND_EXECUTION')
    checkpoint=receipt['checkpoint'];checkpoint_path=Path(checkpoint['path'])
    inherited=receipt['inherited_checkpoint_receipt']
    if inherited is None:
        need(Path(binding['path'])==root/('CHECKPOINT_'+str(cursor).zfill(6)+'.json')
            and checkpoint_path.parent==root and checkpoint_path.name==(
                'INITIAL_STATE.pt'if cursor==0 else'CHECKPOINT_'+str(cursor).zfill(6)+'.pt'),
            'CHECKPOINT_RECEIPT_OWNED_PATH')
    else:
        original=require_main_checkpoint_receipt(inherited,trajectory,trajectory_ref)
        need(receipt['execution_plan']!=trajectory_ref
            and execution['inputs']['checkpoint_receipt']==inherited
            and cursor==receipt['restored_prefix_updates']==original['cursor']
            and checkpoint==original['checkpoint']and receipt['state_proof']==original['state_proof']
            and receipt['steps']==original['steps']
            and Path(binding['path'])==checkpoint_consumption_path(original),
            'CHECKPOINT_RECEIPT_INHERITED_START')
    need(ref(checkpoint_path)==checkpoint
        and Path(receipt['execution_claim']['path'])==root/'CLAIM.json'
        and claim['plan']==receipt['execution_plan']
        and claim.get('process_identity')==receipt['process_identity']
        and claim['status']=='CLAIMED_ONE_USE'
        and receipt['phase']==('TRAIN_COMPLETE_SCORE_PENDING'if cursor==total else'TRAIN_PREFIX'),
        'CHECKPOINT_RECEIPT_EXECUTION')
    later=[p for p in root.glob('CHECKPOINT_[0-9][0-9][0-9][0-9][0-9][0-9].json')
           if int(p.stem.rsplit('_',1)[1])>cursor]
    need(not later,'CHECKPOINT_RECEIPT_NOT_LAST_PUBLISHED')
    counts=receipt['current_window_counts'];updates=cursor-receipt['restored_prefix_updates']
    need(type(counts)is dict and counts=={'model_constructions':0 if inherited is not None else 1,'model_forwards':updates,
        'backward_calls':updates,'optimizer_steps':updates,'autograd_calls':0},
        'CHECKPOINT_RECEIPT_WINDOW_COUNTS')
    return receipt


def checkpoint_window_paths(plan_path):
    import re
    path=Path(plan_path)
    need(path.is_absolute()and path.parent==REPO/'configs/research'
        and path.resolve()==path and re.fullmatch(
            r'NATIVE_M5_TA_ATR03_FIVE_YEAR_(?:WINDOW|CONTINUE)_[0-9]{8}_[0-9]{3}\.json',path.name)is not None,
        'CHECKPOINT_WINDOW_CANONICAL_PATH')
    stage=path.stem.lower();root=Path('/home/andre2/GX1_RUNS')/path.stem
    return stage,path,root


def require_checkpoint_parent_end(binding,receipt):
    value=read(binding);execution_root=Path(read(receipt['execution_plan'])['output_root'])
    receipt_path=checkpoint_consumption_path(read(receipt['inherited_checkpoint_receipt']))if receipt['inherited_checkpoint_receipt']is not None else execution_root/('CHECKPOINT_'+str(receipt['cursor']).zfill(6)+'.json')
    receipt_ref=ref(receipt_path)
    need(value.get('plan')==receipt['execution_plan']
        and value.get('claim')==receipt['execution_claim']
        and value.get('test_data_used')is False,'CHECKPOINT_PARENT_END_BINDINGS')
    if value.get('schema_version')==CHECKPOINT_INTERRUPTION_SCHEMA:
        terminal_path=execution_root/'TERMINAL.json'
        need(not terminal_path.exists()and not terminal_path.is_symlink(),
            'CHECKPOINT_INTERRUPTION_CANONICAL_TERMINAL_PRESENT')
        need(value.get('status')=='INTERRUPTED'
            and value.get('checkpoint_receipt')==receipt_ref
            and value.get('process_identity')==receipt['process_identity']
            and type(value.get('reason'))is str and bool(value['reason'].strip())
            and value.get('physical_work_after_last_receipt')=='UNKNOWN',
            'CHECKPOINT_INTERRUPTION_EVIDENCE')
    else:
        need(Path(binding['path'])==execution_root/'TERMINAL.json',
            'CHECKPOINT_PARENT_TERMINAL_PATH')
        need(value.get('status')in('RESUMABLE','FAILED_CONSUMED')
            and value.get('remaining_invocations')==0 and value.get('relaunch_allowed')is False,
            'CHECKPOINT_PARENT_TERMINAL')
        if value['status']=='RESUMABLE':
            need(value.get('checkpoint_receipt')==receipt_ref,
                'CHECKPOINT_PARENT_RESUMABLE_RECEIPT')
    require_checkpoint_parent_stopped(receipt['process_identity'])
    return value


def checkpoint_consumption_path(receipt):
    root=Path(read(receipt['execution_plan'])['output_root'])
    return root/('CHECKPOINT_'+str(receipt['cursor']).zfill(6)+'_CONSUMED.json')


def consume_checkpoint_receipt(receipt,binding,*,execution_ref,claim):
    # This one atomic file is both the no-fork claim and the child's safe
    # zero-update start receipt. There is no two-file publication gap.
    need(receipt['logical_prefix_counts']==checkpoint_prefix_counts(receipt['cursor']),
        'CHECKPOINT_CONSUMPTION_PREFIX')
    current_claim=read(claim)
    need(current_claim['plan']==execution_ref,'CHECKPOINT_CONSUMPTION_CLAIM')
    child={**receipt,'execution_plan':execution_ref,'execution_claim':claim,
        'process_identity':current_claim['process_identity'],
        'restored_prefix_updates':receipt['cursor'],
        'current_window_counts':{'model_constructions':0,'model_forwards':0,
            'backward_calls':0,'optimizer_steps':0,'autograd_calls':0},
        'inherited_checkpoint_receipt':binding}
    return publish(checkpoint_consumption_path(receipt),child)


def checkpoint_window_limits(trajectory,start,stop):
    updates=stop-start
    scoring=sum((item['rows']+trajectory['recipe']['score_batch_rows']-1)//trajectory['recipe']['score_batch_rows']
        for item in trajectory['metadata']['selected_cohorts'].values())if stop==trajectory['recipe']['fit_batches']else 0
    return {'model_constructions':1,'model_forwards':updates+scoring,'backward_calls':updates,
        'optimizer_steps':updates,'autograd_calls':0}


def build_checkpoint_window_plan(*,plan_path,trajectory_plan,stop_after_cursor,outer_seconds,
                                 checkpoint_receipt=None,parent_end=None,source_base_commit=None):
    stage,path,root=checkpoint_window_paths(plan_path);trajectory=read(trajectory_plan)
    need(trajectory.get('mode')=='ta_five_year_main'and trajectory['recipe']['passes']==1
        and trajectory['recipe']['checkpoint_every_updates']==256
        and trajectory==build_fit_plan(inputs=trajectory['inputs'],recipe=trajectory['recipe'],
            mode=trajectory['mode'],source_base_commit=trajectory['source_base_commit'])
        and fit_source_clean(trajectory),'CHECKPOINT_WINDOW_ORIGINAL_TRAJECTORY')
    need((checkpoint_receipt is None)==(parent_end is None),'CHECKPOINT_WINDOW_PARENT_ROLES')
    inputs={'trajectory_plan':trajectory_plan};start=0
    if checkpoint_receipt is not None:
        receipt=require_main_checkpoint_receipt(checkpoint_receipt,trajectory,trajectory_plan)
        require_checkpoint_parent_end(parent_end,receipt);start=receipt['cursor']
        inputs.update(checkpoint_receipt=checkpoint_receipt,parent_end=parent_end)
    continuation=checkpoint_receipt is not None
    need(('_CONTINUE_'in path.name)is continuation,'CHECKPOINT_WINDOW_PATH_ROLE')
    total=trajectory['recipe']['fit_batches']
    need(type(stop_after_cursor)is int and start<=stop_after_cursor<=total
        and(stop_after_cursor>start or start==total),'CHECKPOINT_WINDOW_STOP_CURSOR')
    need(type(outer_seconds)is int and outer_seconds in(1800,10800)and outer_seconds<=trajectory['execution']['outer_seconds'],
        'CHECKPOINT_WINDOW_EXPLICIT_DEADLINE')
    execution={**trajectory['execution'],'outer_seconds':outer_seconds,'internal_seconds':outer_seconds-30}
    return {'schema_version':CHECKPOINT_WINDOW_SCHEMA,'stage':stage,
        'mode':'ta_five_year_continuation'if continuation else'ta_five_year_window',
        'status':'PREPARED_NOT_ADMITTED','output_root':str(root),'inputs':inputs,
        'source_base_commit':source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'source_owners':{name:sha(REPO/name)for name in LARGER_SOURCE_PATHS},
        'recipe':{'start_cursor':start,'stop_after_cursor':stop_after_cursor,
            'trajectory_plan':trajectory_plan,'passes':1,'checkpoint_every_updates':256},
        'execution':execution,'limits':checkpoint_window_limits(trajectory,start,stop_after_cursor),
        'normalization_fits':0,'prior_fits':0,'test_data_used':False,'training_authorized':True,
        'automatic_continuation_allowed':False,'relaunch_allowed':False}


def inspect_checkpoint_window(plan_path,plan_sha,*,admitted):
    stage,path,root=checkpoint_window_paths(plan_path)
    need(Path.cwd().resolve()==REPO and not root.exists(),'CHECKPOINT_WINDOW_CWD_OUTPUT')
    plan_ref={'path':str(path),'sha256':plan_sha};plan=read(plan_ref);inputs=plan['inputs']
    expected=build_checkpoint_window_plan(plan_path=path,trajectory_plan=inputs['trajectory_plan'],
        checkpoint_receipt=inputs.get('checkpoint_receipt'),parent_end=inputs.get('parent_end'),
        stop_after_cursor=plan['recipe']['stop_after_cursor'],outer_seconds=plan['execution']['outer_seconds'],
        source_base_commit=plan['source_base_commit'])
    need(plan==expected and larger_source_clean(plan),'CHECKPOINT_WINDOW_PLAN_SOURCE')
    policy_ref=ref(POLICY);policy=read(policy_ref);gate=policy.get(stage,{})
    closed=set(ta_audit.data.CLOSED_FLAGS)-{'training_enabled','native_training_enabled'}
    need(all(policy.get(k)is False for k in closed)
        and policy.get('training_enabled')is admitted and policy.get('native_training_enabled')is admitted,
        'CHECKPOINT_WINDOW_GLOBAL_GATE')
    need(gate.get('plan')==plan_ref and gate.get('output_root')==str(root)
        and gate.get('max_invocations')==1 and gate.get('claimed_invocations')==gate.get('completed_invocations')==0
        and gate.get('training_authorized')is admitted and gate.get('execution_authorized')is admitted
        and gate.get('relaunch_allowed')is False and gate.get('test_authorized')is False
        and gate.get('status')==('ADMITTED_ONE_USE'if admitted else'PREPARED_NOT_ADMITTED')
        and gate.get('remaining_invocations')==(1 if admitted else 0)
        and all(gate.get(k)==v for k,v in plan['limits'].items()),'CHECKPOINT_WINDOW_EXACT_GATE')
    trajectory=read(inputs['trajectory_plan']);checked_fit_inputs(trajectory)
    if 'checkpoint_receipt'in inputs:
        receipt=read(inputs['checkpoint_receipt']);consumed=checkpoint_consumption_path(receipt)
        need(not consumed.exists()and not consumed.is_symlink(),'CHECKPOINT_ALREADY_CONSUMED')
    if admitted:fit_boot_state(trajectory,admitted=True)
    return plan,plan_ref,policy_ref


def checkpoint_window_score(model,context,panels,trajectory,*,deadline,reserve):
    # Stop between eval batches; partial scores are discarded. The ordinary
    # score owner still restores mode/RNG in its finally block.
    deferred=TimeoutError('CHECKPOINT_WINDOW_SCORE_DEFERRED')
    before=state_sha(model.state_dict());rng=state_sha(training_rng());training=model.training
    def safe_batch_boundary(module,args):
        if time.monotonic()>=deadline-reserve:raise deferred
    hook=model.register_forward_pre_hook(safe_batch_boundary,prepend=True)
    try:
        try:return fit_score(model,context,panels,trajectory)
        except TimeoutError as error:
            if error is not deferred:raise
            return None,None
    finally:
        hook.remove()
        need(state_sha(model.state_dict())==before and state_sha(training_rng())==rng
            and model.training is training,'WINDOW_SCORE_STATE_RNG_MUTATION')


def checkpoint_window_physical(trajectory,trajectory_ref,root,*,execution_ref,claim,limits,
                               stop_after_cursor,deadline,checkpoint_receipt=None,execution_start_receipt=None):
    import torch
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    context=prepare_native_fit(trajectory,root);recipe=trajectory['recipe']
    normalization_body=state_sha(context['normalization'])
    source_body=state_sha(_fit_source_binding(context['provider'],trajectory))
    selected,selected_report=larger_selected(context,trajectory)
    batches=larger_training_batches(selected,trajectory)
    panels={'TRAIN':all_score_batches(selected['FIT'],recipe['score_batch_rows']),
        'CONTROL':all_score_batches(selected['CONTROL'],recipe['score_batch_rows'])}
    model,optimizer=fresh_fit_model(context['provider'],context['normalization'],
        {**trajectory,'limits':limits},train_prior=context['prior'])
    model.require_prior_initialization_state()
    initial_sha=canonical_model_state_sha256(model.state_dict())
    need(initial_sha==trajectory['metadata']['prepared']['initial_state_sha256'],'WINDOW_FRESH_INITIAL_IDENTITY')
    initial,_capture,prior_initialization=load_larger_initial(model,context,trajectory);del _capture
    initial_checkpoint=publish_checkpoint(root/'INITIAL_STATE.pt',fit_checkpoint(model,optimizer,context,
        trajectory,trajectory_ref,0))
    restored=0;steps=[];parent_receipt=None
    if checkpoint_receipt is not None:
        parent_receipt=require_main_checkpoint_receipt(checkpoint_receipt,trajectory,trajectory_ref)
        require_checkpoint_parent_stopped(parent_receipt['process_identity'])
        checkpoint=torch.load(parent_receipt['checkpoint']['path'],map_location='cpu',weights_only=True)
        need(checkpoint_state_proof(checkpoint)==parent_receipt['state_proof'],'WINDOW_RESTORE_PUBLISHED_STATE_PROOF')
        restored=restore_fit_checkpoint(model,optimizer,context,trajectory,trajectory_ref,checkpoint)
        steps=list(parent_receipt['steps']);del checkpoint
    need(restored<=stop_after_cursor<=len(batches),'WINDOW_RESTORED_CURSOR')
    saved=[execution_start_receipt]if execution_start_receipt else[]
    latest=execution_start_receipt;latest_body=read(latest)if latest else None;cursor=restored
    def save():
        nonlocal latest,latest_body
        if latest_body is not None and latest_body['cursor']==cursor:return
        need(all(parameter.grad is None for parameter in model.parameters()),'WINDOW_CHECKPOINT_GRADS_NOT_ZERO')
        rng_before=state_sha(training_rng())
        state=fit_checkpoint(model,optimizer,context,trajectory,trajectory_ref,cursor)
        path=root/('INITIAL_STATE.pt'if cursor==0 else'CHECKPOINT_'+str(cursor).zfill(6)+'.pt')
        if cursor==0:
            # The exact fresh state was already published before restoring.
            checkpoint_ref=initial_checkpoint
            need(ref(path)==checkpoint_ref,'WINDOW_INITIAL_CHECKPOINT_BOOKEND')
            proof=checkpoint_state_proof(state)
            receipt={'schema_version':CHECKPOINT_RECEIPT_SCHEMA,'trajectory_plan':trajectory_ref,
                'execution_plan':execution_ref,'execution_claim':claim,'checkpoint':checkpoint_ref,
                'cursor':0,'schedule_position':state['schedule_position'],'state_proof':proof,
                'source_base_commit':trajectory['source_base_commit'],'source_owners':trajectory['source_owners'],
                'source_unchanged':fit_source_clean(trajectory),'steps':[],
                'restored_prefix_updates':0,'current_window_counts':dict(model._native_fit_counts),
                'logical_prefix_counts':checkpoint_prefix_counts(0),'process_identity':checkpoint_process_identity(),
                'phase':'TRAIN_PREFIX','test_data_used':False,'inherited_checkpoint_receipt':None}
            need(receipt['source_unchanged']is True,'WINDOW_INITIAL_SOURCE')
            latest=publish(root/'CHECKPOINT_000000.json',receipt);latest_body=receipt
        else:
            latest,latest_body=publish_main_checkpoint(path,state,trajectory=trajectory,
                trajectory_ref=trajectory_ref,execution_ref=execution_ref,claim=claim,steps=steps,
                restored_cursor=restored,current_counts=model._native_fit_counts)
        saved.append(latest)
        need(state_sha(training_rng())==rng_before,'WINDOW_CHECKPOINT_PUBLICATION_RNG_MUTATION')
    if restored==0:save()
    reserve=trajectory['execution']['outer_seconds']-trajectory['execution']['internal_seconds']
    need(reserve==30,'WINDOW_EXISTING_DEADLINE_RESERVE')
    soft_stop=False
    for index in range(restored,stop_after_cursor):
        if time.monotonic()>=deadline-reserve:
            soft_stop=True;break
        started=time.monotonic()
        step=fit_step(model,optimizer,context['provider'],context['cohorts']['TRAIN'],batches[index],trajectory)
        torch.cuda.synchronize();step['elapsed_seconds']=time.monotonic()-started
        steps.append(step);cursor=index+1
        if cursor<=2:
            need(step['heads_gradient_l2']>0 and(step['backbone_gradient_l2']==0 if cursor==1
                else step['backbone_gradient_l2']>0),'WINDOW_GRADIENT_TRANSITION')
        if cursor%256==0 or cursor==stop_after_cursor:save()
    save()
    complete=cursor==len(batches)and not soft_stop and time.monotonic()<deadline-reserve
    scores=None;capture_ref=None;final_checkpoint=None
    if complete:
        scores,capture=checkpoint_window_score(model,context,panels,trajectory,deadline=deadline,reserve=reserve)
        if scores is None:complete=False;soft_stop=True
    if complete:
        torch.cuda.synchronize()
        publish(root/'SCORE_PASS_001.json',{'pass':1,'completed_updates':cursor,'scores':scores})
        stream=io.BytesIO();np.savez_compressed(stream,**capture)
        capture_ref=publish(root/'FINAL_CAPTURE.npz',stream.getvalue());stream.close();del capture
        final_checkpoint=publish_checkpoint(root/'FINAL_STATE.pt',fit_checkpoint(model,optimizer,context,
            trajectory,trajectory_ref,cursor))
    counts=dict(model._native_fit_counts)
    need(all(counts[k]<=limits[k]for k in limits),'WINDOW_CURRENT_COUNT_LIMITS')
    if complete:need(counts==limits,'WINDOW_COMPLETED_COUNT_LIMITS')
    need(fit_source_clean(trajectory)and len(steps)==cursor
        and state_sha(context['normalization'])==normalization_body
        and state_sha(_fit_source_binding(context['provider'],trajectory))==source_body
        and ref(context['normalization_ref']['path'])==context['normalization_ref'],
        'WINDOW_FINAL_SOURCE_PREFIX')
    report={'status':'COMPLETE'if complete else'RESUMABLE','trajectory_plan':trajectory_ref,
        'execution_plan':execution_ref,'completed_updates':cursor,'restored_prefix_updates':restored,
        'current_window_updates':cursor-restored,'requested_stop_after_cursor':stop_after_cursor,
        'soft_deadline_reached':soft_stop or time.monotonic()>=deadline-reserve,
        'checkpoint_receipt':latest,'checkpoint':latest_body['checkpoint'],'checkpoint_receipts':saved,
        'initial_checkpoint':initial_checkpoint,'final_checkpoint':final_checkpoint,
        'initial_state_sha256':initial_sha,'final_state_sha256':canonical_model_state_sha256(model.state_dict()),
        'state_proof':latest_body['state_proof'],'steps':steps,'actual_counts':counts,
        'restored_prefix_counts':checkpoint_prefix_counts(restored),
        'logical_prefix_counts':checkpoint_prefix_counts(cursor),
        'logical_total_counts':{**checkpoint_prefix_counts(cursor),'model_forwards':cursor+
            (sum(map(len,panels.values()))if complete else 0)},
        'physical_work_after_parent_receipt':'UNKNOWN'if parent_receipt else'NO_PARENT',
        'axis':context['axis'],'normalization':context['normalization_ref'],'prior':context['prior_ref'],
        'normalization_population':context['normalization']['fit_population'],
        'normalization_fits':0,'prior_fits':0,'selected_cohorts':selected_report,
        'initial':initial,'final':scores,'prior_initialization':prior_initialization,
        'initial_baseline_reused':True,'prepared_initial_parent':trajectory['metadata']['prepared']['result'],
        'initial_score_artifact':trajectory['metadata']['prepared']['initial_score_artifact'],
        'captures':{'INITIAL':trajectory['metadata']['prepared']['initial_capture'],'FINAL':capture_ref},
        'checkpoints':[read(binding)['checkpoint']for binding in saved],
        'canonical_optimizer_updates':cursor,'evaluation_batches_per_phase':sum(map(len,panels.values())),
        'training_schedule':trajectory['training_schedule'],
        'actual_source_row_schedule_sha256':digest(np.concatenate(batches[:cursor]))if cursor else digest(np.empty(0,np.int64)),
        'train_batch_rows':recipe['batch_rows'],'score_batch_rows':recipe['score_batch_rows'],
        'declared_passes':1,'qualification_weights_loaded':False,
        'score_lane':'eval_no_grad_same_frozen_TRAIN_stats_qualified_execution_route',
        'score_scope':'five_year_TRAIN_and_Jan_Jun2026_development_VAL_all_rows_including_unknown_tails',
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'test_data_used':False,'economic_edge_proven':False}
    need(report['cuda_peak_reserved_bytes']<=LARGER_VRAM_MAX,'WINDOW_VRAM_LIMIT')
    return report


def run_checkpoint_window(plan_path,plan_sha):
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    began=time.monotonic();guard=require_guarded_cuda_trainer_execution();common.firewall_state()
    plan,plan_ref,policy_ref=inspect_checkpoint_window(plan_path,plan_sha,admitted=True)
    root=Path(plan['output_root']);root.mkdir(exist_ok=False)
    claim=publish(root/'CLAIM.json',{'stage':plan['stage'],'status':'CLAIMED_ONE_USE','plan':plan_ref,
        'policy_at_claim':policy_ref,'process_identity':checkpoint_process_identity(),
        'max_invocations':1,'remaining_invocations':0,'test_data_used':False})
    trajectory_ref=plan['inputs']['trajectory_plan'];trajectory=read(trajectory_ref)
    deadline=began+plan['execution']['internal_seconds'];old=signal.getsignal(signal.SIGALRM)
    result=None
    def expired(*args):raise TimeoutError('NATIVE_M5_WINDOW_INTERNAL_LIMIT')
    try:
        signal.signal(signal.SIGALRM,expired);remaining=deadline-time.monotonic()
        need(remaining>0,'WINDOW_TIME_BEFORE_INPUTS');signal.setitimer(signal.ITIMER_REAL,remaining)
        start_receipt=None
        if 'checkpoint_receipt'in plan['inputs']:
            parent=read(plan['inputs']['checkpoint_receipt'])
            start_receipt=consume_checkpoint_receipt(parent,plan['inputs']['checkpoint_receipt'],
                execution_ref=plan_ref,claim=claim)
        report=checkpoint_window_physical(trajectory,trajectory_ref,root,execution_ref=plan_ref,
            claim=claim,limits=plan['limits'],stop_after_cursor=plan['recipe']['stop_after_cursor'],
            deadline=deadline,checkpoint_receipt=plan['inputs'].get('checkpoint_receipt'),
            execution_start_receipt=start_receipt)
        need(larger_source_clean(plan)and ref(POLICY)==policy_ref and ref(plan_path)==plan_ref,
            'WINDOW_SOURCE_POLICY_BOOKEND')
        result=publish(root/'RESULT.json',{'stage':plan['stage'],'status':report['status'],
            'plan':plan_ref,'claim':claim,'report':report,'guard':guard,'source_unchanged':True,
            'test_data_used':False,'optimizer_steps':report['actual_counts']['optimizer_steps'],
            'remaining_invocations':0,'relaunch_allowed':False,'elapsed_seconds':time.monotonic()-began,
            'clean_process_io_firewall':common.firewall_state()})
        terminal=publish(root/'TERMINAL.json',{'stage':plan['stage'],'status':report['status'],
            'exit_code':0,'plan':plan_ref,'claim':claim,'result':result,
            'checkpoint_receipt':report['checkpoint_receipt'],'remaining_invocations':0,
            'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False})
        return {'result':result,'terminal':terminal}
    except BaseException as error:
        publish(root/'TERMINAL.json',{'stage':plan['stage'],'status':'FAILED_CONSUMED','exit_code':1,
            'plan':plan_ref,'claim':claim,'result':result,'error':type(error).__name__+':'+str(error)[:2000],
            'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


def checkpoint_window_cli(args):
    _,_,root=checkpoint_window_paths(args.plan)
    need(args.device=='cuda'and args.out_dir==str(root),'CHECKPOINT_WINDOW_CLI')
    if args.preflight:
        plan,plan_ref,policy_ref=inspect_checkpoint_window(args.plan,args.plan_sha256,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_CHECKPOINT_WINDOW_METADATA_ONLY','plan':plan_ref,
            'policy':policy_ref,'claim_created':False,'model_constructions':0,'optimizer_steps':0,
            'start_cursor':plan['recipe']['start_cursor'],'stop_after_cursor':plan['recipe']['stop_after_cursor']}))
    else:print(json.dumps(run_checkpoint_window(args.plan,args.plan_sha256),sort_keys=True))


# Exact CPU-only technical prefix; no older fit/qualification scope is reopened.
FUTURE_CPU_STAGE='native_m5_future_state_cpu_mechanics_20261121_001'
FUTURE_CPU_POLICY='pretrain_12_checkpoint_review_20261121_001'
FUTURE_CPU_PLAN=REPO/'configs/research/NATIVE_M5_FUTURE_STATE_CPU_MECHANICS_20261121_001.json'
FUTURE_CPU_ROOT=Path('/home/andre2/GX1_RUNS/NATIVE_M5_FUTURE_STATE_CPU_MECHANICS_20261121_001')
FUTURE_CPU_B3=Path('/home/andre2/GX1_RUNS/GX1_NATIVE_FUTURE_STATE_QUALIFICATION_20261121_001/MODEL_FORWARD_PLAN001.json')
FUTURE_CPU_RECIPE=REPO/'configs/research/NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_003.json'
FUTURE_CPU_TRAIN_START=1609459200000000000
FUTURE_CPU_TRAIN_UPPER=1767225600000000000
FUTURE_CPU_TRAIN_SHA='c0805f39affb9bd80e679e0b41f69a558ea975d0e479ddeee815c7dd370997ea'
FUTURE_CPU_TRUTH_KEYS=('issue_ns','target_start_ns','validity_ns','target_present','mature',
    'nominal_complete','state_window_complete','partial_range_active','activity_known','activity','side_known','side_up')
FUTURE_CPU_SOURCE_PATHS=tuple(dict.fromkeys((*SOURCE_PATHS,
    'gx1/contracts/market_state_ta_integrated_model_v1.py','gx1/contracts/market_state_ta_integrated_objective_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh',
    'tests/test_native_future_state_cpu_mechanics_v1.py')))
FUTURE_CPU_LIMITS={lane:{'model_constructions':1,'optimizer_constructions':1,
    'model_forwards':13 if lane=='continuous'else 9,'backbone_forwards':13 if lane=='continuous'else 9,
    'autograd_calls':3 if lane=='continuous'else 0,'backward_calls':4 if lane=='continuous'else 1,
    'optimizer_steps':4 if lane=='continuous'else 1,'checkpoint_loads':0 if lane=='continuous'else 1,
    'checkpoint_writes':2 if lane=='continuous'else 1,'checkpoint_readbacks':2 if lane=='continuous'else 1,
    'normalization_fits':0,'prior_fits':0}for lane in ('continuous','replay')}


def future_cpu_read(binding):
    need(type(binding)is dict and set(binding)=={'path','sha256'}and type(binding['path'])is str
        and Path(binding['path']).is_absolute()and '..'not in Path(binding['path']).parts
        and Path(binding['path']).suffix=='.json'and type(binding['sha256'])is str
        and len(binding['sha256'])==64 and all(c in '0123456789abcdef'for c in binding['sha256']),
        'FUTURE_CPU_JSON_REFERENCE')
    return read(binding)


def future_cpu_recipe(reference):
    r=reference['recipe']
    need(r['optimizer']==ADAMW_RECIPE and r['scheduler']is None and r['gradient_clip_norm']==1.
        and type(r['seed'])is int,'FUTURE_CPU_QUALIFIED_RECIPE')
    return {key:r[key]for key in ('optimizer','gradient_clip_norm','scheduler','seed')}|{
        'batch_rows':16,'logical_updates':4,'checkpoint_cursor':3}


def future_cpu_metadata(plan):
    from gx1.contracts.market_state_ta_integrated_model_v1 import native_future_state_model_contract
    from gx1.contracts.market_state_ta_integrated_objective_v1 import require_native_future_state_train_prior
    need(plan['inputs']['forward_plan']['path']==str(FUTURE_CPU_B3)
        and plan['inputs']['recipe_plan']['path']==str(FUTURE_CPU_RECIPE),'FUTURE_CPU_REFERENCE_PATHS')
    parent=future_cpu_read(plan['inputs']['forward_plan']);meta=parent['metadata']
    need(plan['metadata']==meta and plan['recipe']==future_cpu_recipe(future_cpu_read(plan['inputs']['recipe_plan']))
        and meta['seed']==plan['recipe']['seed'],'FUTURE_CPU_FROZEN_METADATA_RECIPE')
    norm=future_cpu_read(meta['normalization']);prior=future_cpu_read(meta['prior'])
    data=future_cpu_read(meta['dataset']);snapshot=future_cpu_read(meta['source_snapshot'])
    manifest=future_cpu_read(meta['native_manifest'])
    prior=require_native_future_state_train_prior(prior,target_contract=meta['target_contract'])
    need(native_future_state_model_contract(meta['input_binding'],architecture=meta['architecture'],
        train_prior=prior,target_contract=meta['target_contract'])==meta['model_contract'],'FUTURE_CPU_MODEL_IDENTITY')
    pop=norm['fit_population'];p=prior['fit_population']
    need(pop['issue_rows']==p['issue_rows']==354550 and pop['selected_issue_ns_sha256']==p['issue_ns_sha256']==FUTURE_CPU_TRAIN_SHA
        and pop['cutoff_ns']==p['cutoff_ns']==FUTURE_CPU_TRAIN_UPPER
        and all(pop[k]==p[k]for k in ('first_issue_ns','last_issue_ns')),'FUTURE_CPU_WHOLE_TRAIN_STATISTICS')
    need(data['contract']==meta['target_contract']and data['cutoff_ns']==FUTURE_CPU_TRAIN_UPPER
        and data['TRAIN_issue_ns_sha256']==FUTURE_CPU_TRAIN_SHA and data['model_input']is False
        and snapshot['schema_version']=='gx1_native_m5_bounded_feature_snapshot_v1'
        and len(snapshot['arrays'])==27 and manifest['market_closure_contract']==meta['target_contract']['source_closure_contract'],
        'FUTURE_CPU_DATASET_SOURCE')
    bridge=future_cpu_read(plan['inputs']['baseline_component'])
    need(bridge['status']=='COMPLETE_FRESH_NATIVE_BODY_IDENTITY_COMPONENT'
        and bridge['source_unchanged']is True and bridge['test_data_used']is False
        and bridge['report']['body_identity']['all_persistent_parameters_and_buffers_exact']is True
        and bridge['report']['candidate_state_loads']==0
        and bridge['report']['composite_wholecohort_baseline_qualified']is False,'FUTURE_CPU_BASELINE_COMPONENT_ONLY')
    return norm,prior,data,snapshot,bridge


def future_cpu_replay_admission(plan_ref,policy_ref):
    """Authenticate the completed first lane and its unchanged admission policy."""
    parent=FUTURE_CPU_ROOT/'continuous'
    terminal=future_cpu_read(ref(parent/'TERMINAL.json'))
    need(terminal['status']=='COMPLETE_CONSUMED'and terminal['exit_code']==0
        and terminal['plan']==plan_ref
        and terminal['claim']['path']==str(parent/'CLAIM.json'),'FUTURE_CPU_REPLAY_PARENT_TERMINAL')
    claim=future_cpu_read(terminal['claim'])
    need(claim['plan']==plan_ref and claim['lane']=='continuous'
        and claim['policy_at_claim']==policy_ref,'FUTURE_CPU_INTERLANE_POLICY_FREEZE')


def inspect_future_cpu(plan_sha,lane,*,admitted):
    need(lane in FUTURE_CPU_LIMITS and Path.cwd().resolve()==REPO,'FUTURE_CPU_CWD_LANE')
    binding={'path':str(FUTURE_CPU_PLAN),'sha256':plan_sha};plan=future_cpu_read(binding)
    need(plan['schema_version']=='gx1_native_future_state_cpu_mechanics_plan_v1'
        and plan['stage']==FUTURE_CPU_STAGE and plan['output_root']==str(FUTURE_CPU_ROOT)
        and plan['limits']==FUTURE_CPU_LIMITS
        and plan['execution']=={'class':'producer','device':'cpu','memory':'12G','swap':'512M',
            'numerical_threads':1,'tasks_max':64,'outer_seconds':900,'internal_seconds':870,'kill_after_seconds':10}
        and plan['batch_rule']=='first_64_chronological_whole_TRAIN_input_rows_four_disjoint_B16_no_target_filter'
        and plan['diagnostic_routing']=='fully_attached_only_for_this_technical_prefix_not_production_policy',
        'FUTURE_CPU_EXACT_SCOPE')
    need(set(plan['source_owners'])==set(FUTURE_CPU_SOURCE_PATHS)
        and all(Path(k).suffix in ('.py','.sh')and not Path(k).is_absolute()and '..'not in Path(k).parts for k in plan['source_owners']),
        'FUTURE_CPU_SOURCE_INVENTORY')
    need(all(sha(REPO/k)==v for k,v in plan['source_owners'].items()),'FUTURE_CPU_SOURCE_HASH')
    need(subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current',
        'FUTURE_CPU_BRANCH')
    subprocess.run(['git','merge-base','--is-ancestor',plan['source_base_commit'],'HEAD'],cwd=REPO,check=True)
    future_cpu_metadata(plan)
    policy_ref=ref(POLICY);policy=future_cpu_read(policy_ref);gate=policy[FUTURE_CPU_POLICY]['future_mechanics_001']
    need(gate['plan']==binding and gate['limits']==FUTURE_CPU_LIMITS and gate['relaunch_allowed']is False
        and gate['lanes'][lane]['remaining_invocations']==1,'FUTURE_CPU_GATE_BINDING')
    if admitted:
        need(gate['status']=='ADMITTED_EXACT_TECHNICAL_PREFIX'and gate['execution_authorized']is True
            and gate['lanes'][lane]['execution_authorized']is True
            and policy['training_enabled']is True and policy['native_training_enabled']is True,
            'FUTURE_CPU_EXACT_TRAINING_ADMISSION')
    else:
        need(gate['execution_authorized']is False and policy['training_enabled']is False
            and policy['native_training_enabled']is False,'FUTURE_CPU_PREFLIGHT_CLOSED')
    need(all(policy[k]is False for k in ('native_launch_authorized','full_epoch_training_allowed','full_val_allowed')),
        'FUTURE_CPU_NO_OTHER_AUTHORITY')
    if admitted and lane=='replay':future_cpu_replay_admission(binding,policy_ref)
    need(not(FUTURE_CPU_ROOT/lane).exists(),'FUTURE_CPU_ONE_USE_LANE')
    return plan,binding,policy_ref


def future_cpu_rng():
    import random,torch
    name,keys,position,gauss,cached=np.random.get_state()
    need(name=='MT19937','FUTURE_CPU_NUMPY_RNG')
    return {'python':random.getstate(),'numpy_name':name,'numpy_keys':torch.from_numpy(keys.copy()),
        'numpy_position':int(position),'numpy_has_gauss':int(gauss),'numpy_cached':float(cached),
        'torch_cpu':torch.get_rng_state().clone()}


def future_cpu_restore_rng(value):
    import random,torch
    need(type(value)is dict and set(value)=={'python','numpy_name','numpy_keys','numpy_position',
        'numpy_has_gauss','numpy_cached','torch_cpu'}and value['numpy_name']=='MT19937'
        and isinstance(value['numpy_keys'],torch.Tensor)and value['numpy_keys'].dtype==torch.uint32
        and value['numpy_keys'].ndim==1 and isinstance(value['torch_cpu'],torch.Tensor)
        and value['torch_cpu'].dtype==torch.uint8 and value['torch_cpu'].ndim==1,'FUTURE_CPU_RNG_SCHEMA')
    random.setstate(value['python'])
    np.random.set_state((value['numpy_name'],value['numpy_keys'].cpu().numpy(),value['numpy_position'],
        value['numpy_has_gauss'],value['numpy_cached']))
    torch.set_rng_state(value['torch_cpu'].cpu())


def future_cpu_process_identity():
    import os
    fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return {'pid':os.getpid(),'start_ticks':int(fields[19]),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def future_cpu_parent_stopped(identity):
    need(type(identity)is dict and set(identity)=={'pid','start_ticks','boot_id'}
        and type(identity['pid'])is int and identity['pid']>0
        and type(identity['start_ticks'])is int and identity['start_ticks']>0,'FUTURE_CPU_PARENT_IDENTITY')
    if identity['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():return
    path=Path('/proc')/str(identity['pid'])/'stat'
    try:actual=int(path.read_text().rsplit(')',1)[1].split()[19])
    except FileNotFoundError:return
    need(actual!=identity['start_ticks'],'FUTURE_CPU_PARENT_STILL_RUNNING')


def future_cpu_batches(provider):
    rows=np.flatnonzero((provider.issue_ns>=FUTURE_CPU_TRAIN_START)&(provider.issue_ns<FUTURE_CPU_TRAIN_UPPER)).astype(np.int64)
    need(len(rows)==354550 and provider.input_ready[rows].all()
        and digest(provider.issue_ns[rows])==FUTURE_CPU_TRAIN_SHA,'FUTURE_CPU_FULL_INPUT_AXIS')
    batches=[rows[i:i+16]for i in range(0,64,16)]
    need(len(np.unique(np.concatenate(batches)))==64,'FUTURE_CPU_DISTINCT_PREFIX')
    return rows,batches


def future_cpu_load(plan):
    import torch
    from gx1.audit.build_native_m5_ta_feature_cache_v1 import load_snapshot
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    norm,prior,data,snapshot,bridge=future_cpu_metadata(plan)
    provider=load_snapshot(plan['metadata']['source_snapshot']);rows,batches=future_cpu_batches(provider)
    need(provider.input_binding(norm)==plan['metadata']['input_binding']
        and digest(provider.source_row[rows])==norm['fit_population']['selected_source_rows_sha256'],'FUTURE_CPU_PHYSICAL_BINDING')
    order=[{'source_rows_sha256':digest(provider.source_row[b]),'issue_ns_sha256':digest(provider.issue_ns[b])}for b in batches]
    need(ref(data['arrays']['path'])==data['arrays'],'FUTURE_CPU_TARGET_ARCHIVE')
    truth={}
    with np.load(data['arrays']['path'],allow_pickle=False)as archive:
        for key in FUTURE_CPU_TRUTH_KEYS:
            a=archive[key];header=data['array_headers'][key]
            need(a.dtype.str==header['dtype']and list(a.shape)==header['shape']and digest(a)==header['sha256'],
                'FUTURE_CPU_TRUTH_ARRAY:'+key)
            truth[key]=a
    need(np.array_equal(truth['issue_ns'],provider.issue_ns[rows])
        and objective._native_prior_array_sha(truth)==prior['fit_population']['truth_sha256'],'FUTURE_CPU_PRIOR_TRUTH')
    objective._native_future_state_truth(truth,rows=len(rows),fold_upper_ns=FUTURE_CPU_TRAIN_UPPER,
        target_contract=plan['metadata']['target_contract'])
    packets=[]
    for b in batches:
        raw=provider.gather(provider.source_row[b],max_rows=16)
        positions=np.searchsorted(rows,b)
        packets.append({'features':{k:torch.from_numpy(v)for k,v in raw.items()},
            'truth':{k:v[positions]for k,v in truth.items()}})
    return {'provider':provider,'normalization':norm,'prior':prior,'order':order,'packets':packets,'bridge':bridge}


def future_cpu_score(model,packets,plan,prior,*,initial):
    import torch
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    before=state_sha(model.state_dict());rng=shared_gradient_rng();training=model.training;reports=[]
    model.eval()
    try:
        with torch.no_grad():
            for packet in packets:
                output=model(**packet['features'])
                kwargs={'truth':packet['truth'],'fold_upper_ns':FUTURE_CPU_TRAIN_UPPER,
                    'target_contract':plan['metadata']['target_contract']}
                loss=objective.native_future_state_loss(model_output=output,**kwargs)
                scores=objective.native_future_state_score_sums(model_output=output,**kwargs)
                if initial:
                    reference=objective.native_future_state_train_prior_logits(prior,rows=len(packet['truth']['issue_ns']),
                        target_contract=plan['metadata']['target_contract'])
                    need(all(torch.equal(output[k],v)for k,v in reference.items()),'FUTURE_CPU_INITIAL_PRIOR_EXACT')
                    need(scores==objective.native_future_state_score_sums(model_output=reference,**kwargs),
                        'FUTURE_CPU_INITIAL_PROPER_SCORE_EXACT')
                total=sum(cell['partial_log_loss_sum']for cell in scores['per_tf'].values())
                reports.append({'known_activity_cells':loss['count'],'partial_NLL':float(loss['loss']),
                    'proper_partial_log_loss_sum':total,'loss_score_abs_delta':abs(float(loss['loss'])*loss['count']-total),
                    'scores':scores,'output_sha256':{k:digest(v.detach().cpu().numpy())for k,v in output.items()}})
    finally:
        model.train(training);shared_gradient_restore_rng(rng)
    need(state_sha(model.state_dict())==before,'FUTURE_CPU_SCORE_STATE_MUTATION')
    count=sum(r['known_activity_cells']for r in reports)
    return {'rows':sum(len(p['truth']['issue_ns'])for p in packets),'batches':reports,
        'known_activity_cells':count,'proper_partial_NLL':sum(r['proper_partial_log_loss_sum']for r in reports)/count if count else None,
        'initial_prior_exact':initial,'scope':'same_64_TRAIN_trajectory_prefix_not_fullcohort_or_independent_FIT'}


def future_cpu_family_gradients(names,g5,gaux):
    """Diagnostic parameter-name partition; no additional derivative calculation."""
    import torch
    from gx1.features.entry_specialist_feature_groups_v1 import MODEL_NATIVE_TRAINING_SPECIALISTS
    families=tuple(MODEL_NATIVE_TRAINING_SPECIALISTS)
    need(len(names)==len(g5)==len(gaux),'FUTURE_CPU_FAMILY_GRADIENT_AXIS')
    groups={name:[]for name in (*families,'unassigned_shared_mix')}
    for i,name in enumerate(names):
        matches=[family for family in families if family in name]
        groups[matches[0]if len(matches)==1 else'unassigned_shared_mix'].append(i)
    total=tuple(b if a is None else a if b is None else a+b for a,b in zip(g5,gaux))
    result={}
    for group,indices in groups.items():
        left=tuple(g5[i]for i in indices);right=tuple(gaux[i]for i in indices)
        joint=tuple(total[i]for i in indices);measures=shared_gradient_measures(left,right)
        result[group]={**measures,'parameter_tensor_count':len(indices),
            'gtotal_l2':shared_gradient_measures(joint,(None,)*len(joint))['g5_l2'],
            'gaux_to_gM5_l2_ratio':measures['gaux_l2']/measures['g5_l2']if measures['g5_l2']>0 else None,
            **{label:sum(value is not None and bool(torch.count_nonzero(value))for value in values)
               for label,values in (('gM5_nonzero_parameter_tensors',left),
                   ('gaux_nonzero_parameter_tensors',right),('gtotal_nonzero_parameter_tensors',joint))}}
    return {'per_group':result,'assignment':'one_canonical_family_substring_else_unassigned_shared_mix',
        'scope':'shared_parameter_data_gradient_connectivity_not_individual_feature_or_TF_utility'}


def future_cpu_gradient_probe(model,packet,plan,counts):
    import torch
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_future_state_loss
    before=state_sha(model.state_dict());rng=shared_gradient_rng();training=model.training
    need(all(p.grad is None for p in model.parameters()),'FUTURE_CPU_PROBE_GRAD_NONE')
    model.eval()
    try:
        output=model(**packet['features'])
        loss=native_future_state_loss(model_output=output,truth=packet['truth'],fold_upper_ns=FUTURE_CPU_TRAIN_UPPER,
            target_contract=plan['metadata']['target_contract'])
        n=loss['count'];need(n>0,'FUTURE_CPU_PROBE_SUPPORT')
        sums=loss['loss_sums'];named=tuple((name,p)for name,p in model.backbone.named_parameters()if p.requires_grad)
        parameters=tuple(p for _,p in named)
        terms=(sums['activity'][0]/n,sums['side'][0]/n,(sums['activity'][1:].sum()+sums['side'][1:].sum())/n)
        gradients=[]
        for i,term in enumerate(terms):
            gradients.append(torch.autograd.grad(term,parameters,retain_graph=i<2,allow_unused=True))
            counts['autograd_calls']+=1
        combined=tuple(b if a is None else a if b is None else a+b for a,b in zip(gradients[0],gradients[1]))
        result={'M5_activity_vs_side':shared_gradient_measures(gradients[0],gradients[1]),
            'M5_vs_auxiliary':shared_gradient_measures(combined,gradients[2]),
            'family_shared_gradients':future_cpu_family_gradients(tuple(name for name,_ in named),combined,gradients[2]),
            'denominator_known_activity_cells':n,
            'support':{k:v.detach().cpu().tolist()for k,v in loss['counts'].items()},
            'role':'post_first_update_data_gradients_before_clip_not_AdamW_or_optimal_task_weights'}
    finally:
        model.train(training);shared_gradient_restore_rng(rng)
    need(state_sha(model.state_dict())==before and all(p.grad is None for p in model.parameters()),
        'FUTURE_CPU_PROBE_STATE_GRAD_MUTATION')
    return result


def future_cpu_step(model,optimizer,packet,plan,counts,index):
    import torch
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_future_state_loss
    optimizer.zero_grad(set_to_none=True);model.train()
    output=model(**packet['features'])
    loss=native_future_state_loss(model_output=output,truth=packet['truth'],fold_upper_ns=FUTURE_CPU_TRAIN_UPPER,
        target_contract=plan['metadata']['target_contract'])
    need(not loss['skip_update']and bool(torch.isfinite(loss['loss'])),'FUTURE_CPU_TRAIN_SUPPORT')
    loss['loss'].backward();counts['backward_calls']+=1
    gradients=[p for p in model.parameters()if p.grad is not None]
    need(gradients and finite_all_per_device(p.grad for p in gradients),'FUTURE_CPU_FINITE_DATA_GRADIENT')
    data={}
    for group,parameters in (('backbone',model.backbone.parameters()),('readout',model.readout.parameters())):
        data[group+'_data_gradient_l2']=float(torch.sqrt(sum((p.grad.detach().double().square().sum()
            for p in parameters if p.grad is not None),torch.zeros((),dtype=torch.float64))))
    before_body=[p.detach().clone()for p in model.backbone.parameters()]if index==0 else None
    if index==0:need(data['backbone_data_gradient_l2']==0 and data['readout_data_gradient_l2']>0,
        'FUTURE_CPU_ZERO_HEAD_INITIAL_DATA_GRADIENT')
    norm=torch.nn.utils.clip_grad_norm_(gradients,plan['recipe']['gradient_clip_norm'],error_if_nonfinite=True)
    optimizer.step();counts['optimizer_steps']+=1
    need(finite_all_per_device(model.parameters()),'FUTURE_CPU_FINITE_UPDATED_PARAMETERS')
    if before_body is not None:
        data['backbone_actual_parameter_delta_l2']=float(torch.sqrt(sum(((p.detach().double()-old.double()).square().sum()
            for p,old in zip(model.backbone.parameters(),before_body)),torch.zeros((),dtype=torch.float64))))
        data['weight_decay']=plan['recipe']['optimizer']['weight_decay']
        data['interpretation']='zero_initial_backbone_data_gradient_does_not_disable_AdamW_decay_FP32_delta_reported_separately'
        need(any(bool(torch.count_nonzero(h.weight))for h in model.readout.activity_heads.values()),
            'FUTURE_CPU_HEAD_WEIGHT_DATA_UPDATE')
    optimizer.zero_grad(set_to_none=True)
    return {'logical_step':index+1,'loss':float(loss['loss'].detach()),'gradient_norm_pre_clip':float(norm),
        'support':{k:v.detach().cpu().tolist()for k,v in loss['counts'].items()},
        'output_sha256':{k:digest(v.detach().cpu().numpy())for k,v in output.items()},**data}


def future_cpu_bindings(plan,plan_ref):
    return {'plan':plan_ref,'metadata':plan['metadata'],'recipe':plan['recipe'],
        'scope':'technical_prefix_of_whole_TRAIN_trajectory_whole_TRAIN_prior_and_statistics_not_small_FIT_comparison'}


def future_cpu_checkpoint(model,optimizer,plan,plan_ref,order,cursor,steps):
    return {'schema_version':'gx1_native_future_state_cpu_technical_checkpoint_v1','cursor':cursor,
        'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'scheduler_state':None,
        'rng_state':future_cpu_rng(),'training':model.training,'batch_order':order,
        'optimizer_parameter_names':[n for n,_ in model.named_parameters()],
        'bindings':future_cpu_bindings(plan,plan_ref),'steps':steps}


def future_cpu_proof(checkpoint):
    return {key:state_sha(checkpoint[key])for key in ('model_state','optimizer_state','scheduler_state',
        'rng_state','cursor','training','batch_order','optimizer_parameter_names','bindings','steps')}


def future_cpu_publish_checkpoint(path,checkpoint,counts):
    import torch
    before=state_sha(future_cpu_rng());stream=io.BytesIO();torch.save(checkpoint,stream)
    payload=stream.getvalue();stream.close();binding=publish(path,payload);counts['checkpoint_writes']+=1
    need(ref(path)==binding,'FUTURE_CPU_CHECKPOINT_FILE_HASH')
    loaded=torch.load(path,map_location='cpu',weights_only=True);counts['checkpoint_readbacks']+=1
    need(state_sha(loaded)==state_sha(checkpoint),'FUTURE_CPU_CHECKPOINT_READBACK')
    need(state_sha(future_cpu_rng())==before,'FUTURE_CPU_CHECKPOINT_RNG_MUTATION')
    return binding


def future_cpu_restore(model,optimizer,checkpoint,plan,plan_ref,order):
    import torch
    fields={'schema_version','cursor','model_state','optimizer_state','scheduler_state','rng_state','training',
        'batch_order','optimizer_parameter_names','bindings','steps'}
    need(type(checkpoint)is dict and set(checkpoint)==fields
        and checkpoint['schema_version']=='gx1_native_future_state_cpu_technical_checkpoint_v1'
        and checkpoint['cursor']==3 and checkpoint['scheduler_state']is None and checkpoint['training']is True
        and checkpoint['batch_order']==order and len(checkpoint['steps'])==3
        and checkpoint['optimizer_parameter_names']==[n for n,_ in model.named_parameters()]
        and checkpoint['bindings']==future_cpu_bindings(plan,plan_ref),'FUTURE_CPU_CHECKPOINT_BINDINGS')
    state=checkpoint['optimizer_state'];fresh=optimizer.state_dict()
    need(type(state)is dict and set(state)=={'state','param_groups'}
        and state['param_groups']==fresh['param_groups']and len(state['param_groups'])==1,
        'FUTURE_CPU_OPTIMIZER_GROUPS')
    ids=state['param_groups'][0]['params'];parameters=list(model.parameters())
    need(len(ids)==len(parameters)and set(state['state'])<=set(ids),'FUTURE_CPU_OPTIMIZER_PARAMETER_IDS')
    for index,item in state['state'].items():
        p=parameters[ids.index(index)]
        need(type(item)is dict and set(item)=={'step','exp_avg','exp_avg_sq'}
            and all(isinstance(v,torch.Tensor)and bool(torch.isfinite(v).all())for v in item.values())
            and item['step'].numel()==1 and 1<=float(item['step'])<=3
            and all(item[k].shape==p.shape and item[k].dtype==p.dtype for k in ('exp_avg','exp_avg_sq')),
            'FUTURE_CPU_OPTIMIZER_STATE')
    previous=future_cpu_rng()
    try:
        future_cpu_restore_rng(checkpoint['rng_state'])
        need(state_sha(future_cpu_rng())==state_sha(checkpoint['rng_state']),'FUTURE_CPU_RNG_ROUNDTRIP')
    finally:future_cpu_restore_rng(previous)
    model.load_state_dict(checkpoint['model_state'],strict=True)
    optimizer.load_state_dict(state);model.train(True);future_cpu_restore_rng(checkpoint['rng_state'])
    need(state_sha(model.state_dict())==state_sha(checkpoint['model_state'])
        and state_sha(optimizer.state_dict())==state_sha(state),'FUTURE_CPU_RESTORED_STATE')


def build_future_cpu_plan(*,inputs,source_base_commit=None):
    parent=future_cpu_read(inputs['forward_plan'])
    plan={'schema_version':'gx1_native_future_state_cpu_mechanics_plan_v1','stage':FUTURE_CPU_STAGE,
        'status':'PREPARED_NOT_ADMITTED','output_root':str(FUTURE_CPU_ROOT),
        'source_base_commit':source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'source_owners':{path:sha(REPO/path)for path in FUTURE_CPU_SOURCE_PATHS},'inputs':inputs,
        'metadata':parent['metadata'],'recipe':future_cpu_recipe(future_cpu_read(inputs['recipe_plan'])),
        'execution':{'class':'producer','device':'cpu','memory':'12G','swap':'512M','numerical_threads':1,
            'tasks_max':64,'outer_seconds':900,'internal_seconds':870,'kill_after_seconds':10},
        'limits':FUTURE_CPU_LIMITS,
        'batch_rule':'first_64_chronological_whole_TRAIN_input_rows_four_disjoint_B16_no_target_filter',
        'diagnostic_routing':'fully_attached_only_for_this_technical_prefix_not_production_policy',
        'whole_cohort_CPU_initialbaseline_qualified':False,'test_data_used':False}
    future_cpu_metadata(plan)
    return plan


def future_cpu_physical(plan,plan_ref,root,lane):
    import random,torch
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    context=future_cpu_load(plan);meta=plan['metadata'];counts=dict.fromkeys(FUTURE_CPU_LIMITS[lane],0)
    torch.set_num_threads(1);random.seed(plan['recipe']['seed']);np.random.seed(plan['recipe']['seed'])
    torch.manual_seed(plan['recipe']['seed']);torch.use_deterministic_algorithms(True)
    model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=meta['input_binding'],
        architecture=meta['architecture'],device=torch.device('cpu'),train_prior=context['prior'],target_contract=meta['target_contract'])
    counts['model_constructions']+=1
    initial_state=canonical_model_state_sha256(model.state_dict())
    need(initial_state==context['bridge']['report']['fresh_new_full_model_state_sha256'],'FUTURE_CPU_FRESH_COMPONENT_IDENTITY')
    options={**plan['recipe']['optimizer'],'betas':tuple(plan['recipe']['optimizer']['betas'])}
    optimizer=torch.optim.AdamW(model.parameters(),**options);counts['optimizer_constructions']+=1
    need(not optimizer.state,'FUTURE_CPU_FRESH_ADAMW')
    def count(name):
        def hook(*args):
            counts[name]+=1;need(counts[name]<=FUTURE_CPU_LIMITS[lane][name],'FUTURE_CPU_FORWARD_BUDGET')
        return hook
    hooks=[model.register_forward_pre_hook(count('model_forwards')),
           model.backbone.register_forward_pre_hook(count('backbone_forwards'))]
    packet_before=state_sha(context['packets']);normalization_before=state_sha(context['normalization'])
    prior_before=state_sha(context['prior']);steps=[];gradient=None;checkpoint3=None;reference=None
    try:
        # Every actual update is gated by this exact fresh 64-row CPU baseline.
        initial=future_cpu_score(model,context['packets'],plan,context['prior'],initial=True)
        need(initial['rows']==64 and len(initial['batches'])==4 and initial['initial_prior_exact']is True,
            'FUTURE_CPU_ALL_INITIAL_BEFORE_UPDATES')
        publish(root/'INITIAL_SCORE.json',initial)
        if lane=='replay':
            parent_root=FUTURE_CPU_ROOT/'continuous';terminal=future_cpu_read(ref(parent_root/'TERMINAL.json'))
            need(terminal['status']=='COMPLETE_CONSUMED'and terminal['exit_code']==0
                and terminal['plan']==plan_ref,'FUTURE_CPU_CONTINUOUS_COMPLETE')
            reference=future_cpu_read(terminal['result'])
            future_cpu_parent_stopped(reference['process_identity'])
            need(reference['plan']==plan_ref and reference['report']['initial']==initial
                and reference['source_unchanged']is True,'FUTURE_CPU_REPLAY_INITIAL_BINDING')
            receipt=future_cpu_read(reference['report']['checkpoint3_receipt'])
            need(receipt['plan']==plan_ref and receipt['cursor']==3
                and receipt['checkpoint']['path']==str(parent_root/'CURSOR003.pt'),'FUTURE_CPU_CURSOR3_AUTHORITY')
            need(ref(receipt['checkpoint']['path'])==receipt['checkpoint'],'FUTURE_CPU_CURSOR3_HASH')
            checkpoint=torch.load(receipt['checkpoint']['path'],map_location='cpu',weights_only=True)
            counts['checkpoint_loads']+=1
            need(future_cpu_proof(checkpoint)==receipt['state_proof'],'FUTURE_CPU_CURSOR3_PROOF')
            future_cpu_restore(model,optimizer,checkpoint,plan,plan_ref,context['order'])
            steps=list(checkpoint['steps']);checkpoint3=reference['report']['checkpoint3_receipt']
        for index in range(0 if lane=='continuous'else 3,4):
            step=future_cpu_step(model,optimizer,context['packets'][index],plan,counts,index);steps.append(step)
            if lane=='continuous'and index==0:
                gradient=future_cpu_gradient_probe(model,context['packets'][0],plan,counts)
            if lane=='continuous'and index==2:
                checkpoint=future_cpu_checkpoint(model,optimizer,plan,plan_ref,context['order'],3,steps)
                binding=future_cpu_publish_checkpoint(root/'CURSOR003.pt',checkpoint,counts)
                checkpoint3=publish(root/'CURSOR003.json',{'plan':plan_ref,'cursor':3,'checkpoint':binding,
                    'state_proof':future_cpu_proof(checkpoint)})
        final=future_cpu_score(model,context['packets'],plan,context['prior'],initial=False)
        checkpoint=future_cpu_checkpoint(model,optimizer,plan,plan_ref,context['order'],4,steps)
        final_binding=future_cpu_publish_checkpoint(root/'FINAL_STATE.pt',checkpoint,counts)
        proof=future_cpu_proof(checkpoint)
        if lane=='replay':
            need(proof==reference['report']['final_state_proof']and steps==reference['report']['steps']
                and final==reference['report']['final'],'FUTURE_CPU_COLD_REPLAY_BIT_EXACT')
        need(counts==FUTURE_CPU_LIMITS[lane]and state_sha(context['packets'])==packet_before
            and state_sha(context['normalization'])==normalization_before and state_sha(context['prior'])==prior_before,
            'FUTURE_CPU_EXACT_COUNTS_FROZEN_INPUTS')
        return {'lane':lane,'counts':counts,'logical_updates':4,'physical_updates_this_lane':counts['optimizer_steps'],
            'batch_order':context['order'],'initial_state_sha256':initial_state,'initial':initial,'final':final,'steps':steps,
            'post_first_update_gradient':gradient,'checkpoint3_receipt':checkpoint3,'final_checkpoint':final_binding,
            'final_state_proof':proof,'cold_replay_bit_exact':lane=='replay',
            'prefix_proper_NLL_gain':initial['proper_partial_NLL']-final['proper_partial_NLL'],
            'whole_TRAIN_prior_and_statistics':True,'small_FIT_comparison':False,
            'whole_cohort_CPU_initialbaseline_qualified':False,'production_gradient_policy_admitted':False,
            'CONTROL_rows':0,'TEST_rows':0,'limits':'Four distinct B16 logical TRAIN-prefix updates; replay duplicates only step4. Small matched TRAIN adaptation and raw gradients do not establish utility or optimal routing.'}
    finally:
        for hook in hooks:hook.remove()


def run_future_cpu(plan_sha,lane):
    from gx1.contracts.gx1_capped_execution_v1 import require_native_future_state_cpu_mechanics_execution,require_capped_lock_ancestry
    began=time.monotonic();guard=require_native_future_state_cpu_mechanics_execution(plan_sha256=plan_sha,lane=lane)
    lock=require_capped_lock_ancestry()
    need(guard['memory_max']==guard['memory_high']==12*1024**3 and guard['swap']==512*1024**2
        and guard['numerical_threads']==1 and lock['scope']==guard['cgroup_path'][len('/sys/fs/cgroup'):],
        'FUTURE_CPU_GUARD')
    plan,plan_ref,policy_ref=inspect_future_cpu(plan_sha,lane,admitted=True)
    root=FUTURE_CPU_ROOT/lane;root.mkdir(parents=True,exist_ok=False)
    process=future_cpu_process_identity()
    claim=publish(root/'CLAIM.json',{'stage':FUTURE_CPU_STAGE,'plan':plan_ref,'policy_at_claim':policy_ref,
        'lane':lane,'process_identity':process,'remaining_invocations':0,'relaunch_allowed':False})
    previous=signal.getsignal(signal.SIGALRM);result=None
    def expired(*args):raise TimeoutError('FUTURE_CPU_INTERNAL_870_SECONDS')
    try:
        remaining=870-(time.monotonic()-began);need(remaining>0,'FUTURE_CPU_TIME_BEFORE_INPUTS')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
        report=future_cpu_physical(plan,plan_ref,root,lane)
        need(all(sha(REPO/k)==v for k,v in plan['source_owners'].items())and ref(POLICY)==policy_ref
            and ref(FUTURE_CPU_PLAN)==plan_ref,'FUTURE_CPU_SOURCE_POLICY_BOOKEND')
        for binding in [*plan['inputs'].values(),*(plan['metadata'][k]for k in ('source_snapshot','normalization','prior','dataset','native_manifest'))]:
            future_cpu_read(binding)
        snapshot=future_cpu_read(plan['metadata']['source_snapshot']);data=future_cpu_read(plan['metadata']['dataset'])
        need(ref(snapshot['archive']['path'])==snapshot['archive']and ref(data['arrays']['path'])==data['arrays'],
            'FUTURE_CPU_PAYLOAD_BOOKEND')
        need(common.firewall_state()['denied_attempts']==0,'FUTURE_CPU_CLEAN_FIREWALL')
        result=publish(root/'RESULT.json',{'stage':FUTURE_CPU_STAGE,'status':'COMPLETE_TECHNICAL_PREFIX',
            'plan':plan_ref,'claim':claim,'process_identity':process,'report':report,'guard':guard,'lock':lock,
            'source_unchanged':True,'test_data_used':False,'elapsed_seconds':time.monotonic()-began})
        terminal=publish(root/'TERMINAL.json',{'stage':FUTURE_CPU_STAGE,'status':'COMPLETE_CONSUMED','exit_code':0,
            'plan':plan_ref,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False})
        return {'result':result,'terminal':terminal}
    except BaseException as error:
        publish(root/'TERMINAL.json',{'stage':FUTURE_CPU_STAGE,'status':'FAILED_CONSUMED','exit_code':1,
            'plan':plan_ref,'claim':claim,'result':result,'error':type(error).__name__+':'+str(error)[:2000],
            'remaining_invocations':0,'relaunch_allowed':False});raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def future_cpu_cli():
    parser=argparse.ArgumentParser();parser.add_argument('--native-future-state-cpu-mechanics',action='store_true',required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--lane',choices=('continuous','replay'),required=True)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args();common.firewall_state()
    if args.preflight:
        _,plan_ref,policy_ref=inspect_future_cpu(args.plan_sha256,args.lane,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_FUTURE_STATE_CPU_MECHANICS_METADATA_ONLY','plan':plan_ref,
            'policy':policy_ref,'lane':args.lane,'model_constructions':0,'optimizer_steps':0,'claim_created':False}))
    else:print(json.dumps(run_future_cpu(args.plan_sha256,args.lane),sort_keys=True))


# New, exact CUDA authority; all older CPU/raw qualification paths stay closed.
FUTURE_CUDA_PARENT='native_future_state_train_readiness_20261121_001'
FUTURE_CUDA_SCOPES={kind:{'stage':'native_m5_future_state_cuda_'+kind+'_20261121_001',
    'plan':REPO/('configs/research/NATIVE_M5_FUTURE_STATE_CUDA_'+kind.upper()+'_20261121_001.json'),
    'root':Path('/home/andre2/GX1_RUNS/NATIVE_M5_FUTURE_STATE_CUDA_'+kind.upper()+'_20261121_001'),
    'gate':'cuda_'+kind+'_001'}for kind in ('mechanics','initial')}
FUTURE_CUDA_INPUTS={'corrected_macro_result','corrected_macro_terminal','normalization_result',
    'normalization_terminal','forward_plan','recipe_plan','runtime_proof'}
FUTURE_CUDA_INITIAL_INPUTS={'mechanics_result','mechanics_terminal','replay_result','replay_terminal'}
FUTURE_CUDA_CORRECTED=Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_COT_CLOCK_CORRECTION_20261121_001')
FUTURE_CUDA_NORM=Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_COT_CLOCK_NORMALIZATION_20261121_001')
FUTURE_CUDA_FIXED_REFS={
    'forward_plan':{'path':str(FUTURE_CPU_B3),'sha256':'06f17df72165c695536c19ccb4e588efa8a3fed680875a6ede96ef37feb41ff1'},
    'recipe_plan':{'path':str(FUTURE_CPU_RECIPE),'sha256':'20c964020752318c507957ec0b71645e419e023defd7275637cc85fd48b2b5fd'},
    'corrected_macro_result':{'path':str(FUTURE_CUDA_CORRECTED/'RESULT.json'),'sha256':'6dd7df0fd2f0f2b736ba8034697e1c6a8f5e2e9c20138d76f3154568dffec206'},
    'corrected_macro_terminal':{'path':str(FUTURE_CUDA_CORRECTED/'TERMINAL.json'),'sha256':'8eaef41cb506f40b20e4c1d0cfd6c76d6469347c211452ab900f42ff8765d50c'},
    'normalization_result':{'path':str(FUTURE_CUDA_NORM/'RESULT.json'),'sha256':'08ee51600f9cc327490dc88268f7ea5c024c4b52375dfc989f0277da660975fb'},
    'normalization_terminal':{'path':str(FUTURE_CUDA_NORM/'TERMINAL.json'),'sha256':'b84f4b50d270ebcb96eda3e5953f036ffe94875766bdfdb715238d65c8bc17b0'}}
FUTURE_CUDA_GPU_UUID='GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29'
FUTURE_CUDA_SOURCE_PATHS=tuple(dict.fromkeys((*FUTURE_CPU_SOURCE_PATHS,
    'gx1/audit/build_native_m5_ta_feature_cache_v1.py','tests/test_native_future_state_cuda_qualification_v1.py')))
FUTURE_CUDA_LIMITS={lane:{'model_constructions':1,'optimizer_constructions':0 if lane=='initial'else 1,
    'model_forwards':{'continuous':17,'replay':9,'initial':5540}[lane],
    'backbone_forwards':{'continuous':17,'replay':9,'initial':5540}[lane],
    'autograd_calls':3 if lane=='continuous'else 0,'backward_calls':{'continuous':6,'replay':1,'initial':0}[lane],
    'optimizer_steps':{'continuous':4,'replay':1,'initial':0}[lane],
    'checkpoint_loads':1 if lane=='replay'else 0,'checkpoint_writes':2 if lane=='continuous'else 1,
    'checkpoint_readbacks':2 if lane=='continuous'else 1,'feature_cache_payload_loads':1,
    'target_archive_loads':1,'truth_fields_decoded':12,'normalization_fits':0,'prior_fits':0,
    'CONTROL_rows_scored':0,'TEST_rows':0}for lane in ('continuous','replay','initial')}


def future_cuda_lane_root(lane):
    need(lane in FUTURE_CUDA_LIMITS,'FUTURE_CUDA_LANE_ROOT')
    return FUTURE_CUDA_SCOPES['initial']['root']if lane=='initial'else FUTURE_CUDA_SCOPES['mechanics']['root']/lane


def future_cuda_execution(kind):
    need(kind in FUTURE_CUDA_SCOPES,'FUTURE_CUDA_KIND')
    return {'class':'trainer','device':'cuda','memory':'20G','swap':'512M','tasks_max':128,
        'numerical_threads':8,'cpu_affinity':list(range(19)),
        'outer_seconds':1800 if kind=='mechanics'else 5400,
        'internal_seconds':1770 if kind=='mechanics'else 5370,'kill_after_seconds':10,
        'cuda_peak_reserved_bytes_max':LARGER_VRAM_MAX,
        'power_limit_W':300,'power_draw_stop_W':310,'GPU_core_stop_C':85,'GPU_memory_stop_C':80}


def future_cuda_source_inventory(owners):
    need(type(owners)is dict and set(owners)==set(FUTURE_CUDA_SOURCE_PATHS),'FUTURE_CUDA_SOURCE_INVENTORY')
    for name,value in owners.items():
        path=Path(name)
        need(type(name)is str and not path.is_absolute()and '..'not in path.parts
            and str(path)==name and path.suffix in ('.py','.sh')and type(value)is str
            and len(value)==64 and all(c in '0123456789abcdef'for c in value),'FUTURE_CUDA_SOURCE_SCHEMA')
        actual=REPO/path
        need(actual.is_file()and not actual.is_symlink()and actual.resolve()==actual,'FUTURE_CUDA_SOURCE_FILE')


def future_cuda_gpu_metadata():
    import csv
    rows=list(csv.reader(subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv,noheader,nounits'],
        text=True,timeout=20).splitlines()))
    need(len(rows)==1 and len(rows[0])==5,'FUTURE_CUDA_GPU_QUERY_SCHEMA')
    index,uuid,name,driver,memory=(x.strip()for x in rows[0])
    need(index=='0'and uuid==FUTURE_CUDA_GPU_UUID and memory.isdecimal()and bool(name)and bool(driver),
        'FUTURE_CUDA_GPU_IDENTITY')
    processes=[]
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name',
        '--format=csv,noheader,nounits'],text=True,timeout=20)
    for row in csv.reader(output.splitlines()):
        need(len(row)==2 and row[0].strip().isdecimal()and bool(row[1].strip()),'FUTURE_CUDA_PROCESS_QUERY_SCHEMA')
        processes.append({'pid':int(row[0].strip()),'process_name':row[1].strip()})
    return {'index':0,'uuid':uuid,'name':name,'driver_version':driver,
        'total_memory_bytes':int(memory)*1024**2},processes


def future_cuda_current_project_processes():
    import os
    excluded=set();pid=os.getpid()
    while pid>0 and pid not in excluded:
        excluded.add(pid)
        try:pid=int((Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[1])
        except FileNotFoundError:break
    found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal()or int(entry.name)in excluded:continue
        try:
            if entry.stat().st_uid!=os.getuid():continue
            args=[part.decode()for part in (entry/'cmdline').read_bytes().split(b'\0')if part]
            if not args or not Path(args[0]).name.startswith('python'):continue
            cwd=(entry/'cwd').resolve(strict=True)
        except (FileNotFoundError,ProcessLookupError):continue
        if cwd==REPO or REPO in cwd.parents:found.append({'pid':int(entry.name),'argv':args,'cwd':str(cwd)})
    return sorted(found,key=lambda x:x['pid'])


def collect_future_cuda_runtime_metadata():
    """Fresh metadata collector only: no CUDA context, model, checkpoint or arrays."""
    import datetime,platform,torch
    need(not torch.cuda.is_initialized(),'FUTURE_CUDA_CONTEXT_BEFORE_CLAIM')
    gpu,processes=future_cuda_gpu_metadata();project=future_cuda_current_project_processes()
    need(not processes and not project,'FUTURE_CUDA_LIVE_IDLE')
    value={'schema_version':'gx1_native_future_state_cuda_runtime_proof_v1',
        'status':'VERIFIED_IDLE_RUNTIME_CURRENT_BOOT','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'wsl_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'physical_windows_boot_id':physical_windows_boot_id(),
        'library_versions':{'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__),
            'torch_cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()},'gpu':gpu,
        'idle':{'gpu_compute_processes':processes,'current_project_model_processes':project},
        'cuda_initialized':False,'no_model_or_checkpoint_opened':True}
    need(not torch.cuda.is_initialized(),'FUTURE_CUDA_METADATA_INITIALIZED_CONTEXT')
    future_cuda_require_runtime(value);return value


def future_cuda_require_runtime(value):
    import datetime
    fields={'schema_version','status','observed_utc','source_commit','wsl_boot_id','physical_windows_boot_id',
        'library_versions','gpu','idle','cuda_initialized','no_model_or_checkpoint_opened'}
    need(type(value)is dict and set(value)==fields and value['schema_version']=='gx1_native_future_state_cuda_runtime_proof_v1'
        and value['status']=='VERIFIED_IDLE_RUNTIME_CURRENT_BOOT'and value['cuda_initialized']is False
        and value['no_model_or_checkpoint_opened']is True,'FUTURE_CUDA_RUNTIME_SCHEMA')
    stamp=datetime.datetime.fromisoformat(value['observed_utc']);need(stamp.tzinfo is not None,'FUTURE_CUDA_RUNTIME_TIMEZONE')
    need(type(value['source_commit'])is str and len(value['source_commit'])==40
        and all(c in '0123456789abcdef'for c in value['source_commit'])
        and type(value['physical_windows_boot_id'])is int and value['physical_windows_boot_id']>0
        and type(value['wsl_boot_id'])is str and bool(value['wsl_boot_id']),'FUTURE_CUDA_RUNTIME_BOOT_SOURCE')
    libs=value['library_versions'];gpu=value['gpu']
    need(type(libs)is dict and set(libs)=={'python','numpy','torch','torch_cuda','cudnn'}
        and all(type(libs[k])is str and bool(libs[k])for k in ('python','numpy','torch','torch_cuda'))
        and type(libs['cudnn'])is int and libs['cudnn']>0,'FUTURE_CUDA_RUNTIME_LIBRARIES')
    need(type(gpu)is dict and set(gpu)=={'index','uuid','name','driver_version','total_memory_bytes'}
        and type(gpu['index'])is int and gpu['index']==0 and gpu['uuid']==FUTURE_CUDA_GPU_UUID
        and all(type(gpu[k])is str and bool(gpu[k])for k in ('name','driver_version'))
        and type(gpu['total_memory_bytes'])is int and gpu['total_memory_bytes']>LARGER_VRAM_MAX,
        'FUTURE_CUDA_RUNTIME_GPU')
    need(value['idle']=={'gpu_compute_processes':[],'current_project_model_processes':[]},'FUTURE_CUDA_RUNTIME_IDLE')
    return value


def future_cuda_runtime_matches(expected,actual):
    future_cuda_require_runtime(expected);future_cuda_require_runtime(actual)
    need(all(actual[k]==expected[k]for k in expected if k!='observed_utc'),'FUTURE_CUDA_CURRENT_RUNTIME_PROOF')


def future_cuda_complete(result_ref,terminal_ref,status):
    result,terminal=future_cpu_read(result_ref),future_cpu_read(terminal_ref)
    need(result['status']==status and terminal['status']=='COMPLETE'and terminal['exit_code']==0
        and terminal['result']==result_ref and terminal['plan']==result['plan']and terminal['claim']==result['claim']
        and result['source_unchanged']is terminal['source_unchanged']is True
        and result['test_data_used']is terminal['test_data_used']is False,'FUTURE_CUDA_COMPLETE_PARENT')
    return result,terminal


def future_cuda_metadata(inputs,kind,policy):
    from gx1.contracts.market_state_ta_integrated_model_v1 import native_future_state_model_contract
    from gx1.contracts.market_state_ta_integrated_objective_v1 import require_native_future_state_train_prior
    from gx1.contracts.market_state_native_m5_targets_v1 import require_fully_future_state_contract
    need(type(inputs)is dict and set(inputs)==FUTURE_CUDA_INPUTS|(FUTURE_CUDA_INITIAL_INPUTS if kind=='initial'else set()),
        'FUTURE_CUDA_INPUT_ROLES')
    need(all(inputs[k]==v for k,v in FUTURE_CUDA_FIXED_REFS.items())
        and inputs['forward_plan']['path']==str(FUTURE_CPU_B3)
        and inputs['recipe_plan']['path']==str(FUTURE_CPU_RECIPE),'FUTURE_CUDA_FIXED_INPUTS')
    corrected,_=future_cuda_complete(inputs['corrected_macro_result'],inputs['corrected_macro_terminal'],
        'COMPLETE_NATIVE_M5_TA_COT_CLOCK_CORRECTION')
    completed,_=future_cuda_complete(inputs['normalization_result'],inputs['normalization_terminal'],
        'COMPLETE_NATIVE_M5_TA_COT_CLOCK_NORMALIZATION')
    report=completed['report'];parent=future_cpu_read(inputs['forward_plan'])['metadata']
    norm=future_cpu_read(report['normalization']);binding=future_cpu_read(report['input_binding'])
    snapshot=future_cpu_read(report['snapshot']);data=future_cpu_read(parent['dataset'])
    target=require_fully_future_state_contract(data['contract'])
    prior=require_native_future_state_train_prior(future_cpu_read(parent['prior']),target_contract=target)
    manifest=future_cpu_read(parent['native_manifest']);runtime=future_cuda_require_runtime(future_cpu_read(inputs['runtime_proof']))
    pop=norm['fit_population'];p=prior['fit_population'];axes=snapshot['surfaces_metadata']['axes']
    need(report['snapshot']==corrected['report']['snapshot']and report['archive']==snapshot['archive']==corrected['report']['archive']
        and report['normalization']['path']==str(FUTURE_CUDA_NORM/'NORMALIZATION.json')
        and report['input_binding']['path']==str(FUTURE_CUDA_NORM/'INPUT_BINDING.json')
        and report['prior_refit_performed']is False and report['initialbaseline_qualified']is False,'FUTURE_CUDA_CORRECTED_INPUT_CHAIN')
    need(pop['issue_rows']==p['issue_rows']==354550 and pop['cutoff_ns']==p['cutoff_ns']==FUTURE_CPU_TRAIN_UPPER
        and pop['selected_issue_ns_sha256']==p['issue_ns_sha256']==data['TRAIN_issue_ns_sha256']==FUTURE_CPU_TRAIN_SHA
        and all(pop[k]==p[k]for k in ('first_issue_ns','last_issue_ns')),'FUTURE_CUDA_EXACT_WHOLE_TRAIN')
    need(snapshot['schema_version']=='gx1_native_m5_bounded_feature_snapshot_v1'and len(snapshot['arrays'])==27
        and axes['contract_sha256']==norm['source_axes_sha256']==binding['source_axes_sha256']==report['source_axes_sha256']
        and binding['input_normalization_sha256']==norm['contract_sha256']
        and norm['native_m5_manifest']==data['native_manifest']==parent['native_manifest']
        and data['model_input']is False and data['cutoff_ns']==FUTURE_CPU_TRAIN_UPPER
        and manifest['market_closure_contract']==target['source_closure_contract'],'FUTURE_CUDA_FEATURE_TARGET_BINDINGS')
    recipe=future_cpu_recipe(future_cpu_read(inputs['recipe_plan']))
    need(recipe['seed']==20260911 and recipe['optimizer']==ADAMW_RECIPE,'FUTURE_CUDA_SOURCE_QUALIFIED_RECIPE')
    recipe.update(batch_rows=64,logical_updates=0 if kind=='initial'else 4,checkpoint_cursor=0 if kind=='initial'else 3)
    model=native_future_state_model_contract(binding,architecture=parent['architecture'],train_prior=prior,
        target_contract=target,backbone_gradient_policy=policy)
    meta={'source_snapshot':report['snapshot'],'normalization':report['normalization'],'input_binding_ref':report['input_binding'],
        'input_binding':binding,'prior':parent['prior'],'dataset':parent['dataset'],'native_manifest':parent['native_manifest'],
        'target_contract':target,'architecture':parent['architecture'],'model_contract':model,'runtime':runtime,
        'TRAIN_rows':354550,'TRAIN_issue_ns_sha256':FUTURE_CPU_TRAIN_SHA,'truth_sha256':p['truth_sha256']}
    if kind=='initial':
        previous=[]
        for lane,result_key,terminal_key in (('continuous','mechanics_result','mechanics_terminal'),('replay','replay_result','replay_terminal')):
            root=FUTURE_CUDA_SCOPES['mechanics']['root']/lane
            need(inputs[result_key]['path']==str(root/'RESULT.json')and inputs[terminal_key]['path']==str(root/'TERMINAL.json'),
                'FUTURE_CUDA_INITIAL_MECHANICS_PATH')
            result,_=future_cuda_complete(inputs[result_key],inputs[terminal_key],'COMPLETE_NATIVE_FUTURE_STATE_CUDA_MECHANICS')
            plan=future_cpu_read(result['plan']);r=result['report']
            need(plan['metadata']['model_contract']==model and plan['metadata']['runtime']==runtime
                and plan['backbone_gradient_policy']==policy and r['lane']==lane,'FUTURE_CUDA_INITIAL_SAME_MECHANICS_FUNCTION')
            previous.append(result)
        a,b=(r['report']for r in previous)
        need(previous[0]['plan']==previous[1]['plan']and a['final_state_proof']==b['final_state_proof']
            and b['cold_replay_bit_exact']is True and a['equal_tf_evaluation']['qualified']is True
            and a['equal_tf_evaluation']['candidate_activation']['batch_eval_active']is True,
            'FUTURE_CUDA_INITIAL_GROUPED_EVAL_QUALIFIED')
    return meta,recipe



def build_future_cuda_plan(*,inputs,lane_group,backbone_gradient_policy,source_base_commit=None):
    need(lane_group in FUTURE_CUDA_SCOPES,'FUTURE_CUDA_KIND')
    future_cuda_source_inventory(dict.fromkeys(FUTURE_CUDA_SOURCE_PATHS,'0'*64))
    meta,recipe=future_cuda_metadata(inputs,lane_group,backbone_gradient_policy);scope=FUTURE_CUDA_SCOPES[lane_group]
    base=source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    need(base==meta['runtime']['source_commit'],'FUTURE_CUDA_RUNTIME_SOURCE_COMMIT')
    lanes=('initial',)if lane_group=='initial'else('continuous','replay')
    return {'schema_version':'gx1_native_future_state_cuda_qualification_plan_v1','status':'PREPARED_NOT_ADMITTED',
        'stage':scope['stage'],'lane_group':lane_group,'output_root':str(scope['root']),'source_base_commit':base,
        'source_owners':{name:sha(REPO/name)for name in FUTURE_CUDA_SOURCE_PATHS},'inputs':inputs,
        'metadata':meta,'recipe':recipe,'backbone_gradient_policy':backbone_gradient_policy,
        'execution':future_cuda_execution(lane_group),'limits':{k:FUTURE_CUDA_LIMITS[k]for k in lanes},
        'batch_rule':'all354550chronological_TRAIN_B64_last54_no_target_filter'if lane_group=='initial'
            else'first256chronological_whole_TRAIN_four_distinct_B64_no_target_filter',
        'parity_tolerance':THROUGHPUT_EVAL_TOLERANCE,'training_authorized':False,'test_data_used':False,
        'production_gradient_policy_selected':False,'whole_TRAIN_generalization_or_edge_qualified':False}


def future_cuda_gate_expected(plan_ref,kind,*,admitted):
    lanes=('initial',)if kind=='initial'else('continuous','replay')
    return {'status':'ADMITTED_EXACT_CUDA_QUALIFICATION'if admitted else'PREPARED_NOT_ADMITTED',
        'plan':plan_ref,'execution_authorized':admitted,'relaunch_allowed':False,
        'limits':{k:FUTURE_CUDA_LIMITS[k]for k in lanes},
        'lanes':{k:{'execution_authorized':admitted,'remaining_invocations':1 if admitted else 0}for k in lanes}}


def future_cuda_validate_gate(policy,plan_ref,kind,*,admitted):
    gate=policy[FUTURE_CUDA_PARENT][FUTURE_CUDA_SCOPES[kind]['gate']]
    need(json.dumps(gate,sort_keys=True)==json.dumps(future_cuda_gate_expected(plan_ref,kind,admitted=admitted),sort_keys=True),
        'FUTURE_CUDA_EXACT_ONEUSE_GATE')
    training=admitted and kind=='mechanics'
    need(policy['training_enabled']is training and policy['native_training_enabled']is training
        and all(policy[k]is False for k in ('native_launch_authorized','full_epoch_training_allowed','full_val_allowed')),
        'FUTURE_CUDA_NO_OTHER_TRAINING_AUTHORITY')


def future_cuda_replay_admission(plan_ref,policy_ref):
    root=FUTURE_CUDA_SCOPES['mechanics']['root']/'continuous';terminal=future_cpu_read(ref(root/'TERMINAL.json'))
    need(terminal['status']=='COMPLETE'and terminal['exit_code']==0 and terminal['plan']==plan_ref
        and terminal['claim']['path']==str(root/'CLAIM.json'),'FUTURE_CUDA_CONTINUOUS_TERMINAL')
    claim=future_cpu_read(terminal['claim'])
    need(claim['plan']==plan_ref and claim['lane']=='continuous'and claim['policy_at_claim']==policy_ref,
        'FUTURE_CUDA_INTERLANE_POLICY_FREEZE')
    result=future_cpu_read(terminal['result']);future_cpu_parent_stopped(result['process_identity'])
    return result


def inspect_future_cuda(plan_sha,lane,*,admitted):
    need(lane in FUTURE_CUDA_LIMITS and Path.cwd().resolve()==REPO,'FUTURE_CUDA_CWD_LANE')
    kind='initial'if lane=='initial'else'mechanics';scope=FUTURE_CUDA_SCOPES[kind]
    binding={'path':str(scope['plan']),'sha256':plan_sha};plan=future_cpu_read(binding)
    future_cuda_source_inventory(plan['source_owners'])
    need(plan==build_future_cuda_plan(inputs=plan['inputs'],lane_group=kind,
        backbone_gradient_policy=plan['backbone_gradient_policy'],source_base_commit=plan['source_base_commit']),
        'FUTURE_CUDA_EXACT_PLAN_SOURCE')
    need(subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current',
        'FUTURE_CUDA_BRANCH')
    policy_ref=ref(POLICY);future_cuda_validate_gate(future_cpu_read(policy_ref),binding,kind,admitted=admitted)
    if admitted and lane=='replay':future_cuda_replay_admission(binding,policy_ref)
    need(not future_cuda_lane_root(lane).exists()and not future_cuda_lane_root(lane).is_symlink(),
        'FUTURE_CUDA_ONE_USE_LANE')
    live=collect_future_cuda_runtime_metadata();future_cuda_runtime_matches(plan['metadata']['runtime'],live)
    return plan,binding,policy_ref,live


def future_cuda_batches(provider,*,initial):
    rows=np.flatnonzero((provider.issue_ns>=FUTURE_CPU_TRAIN_START)&(provider.issue_ns<FUTURE_CPU_TRAIN_UPPER)).astype(np.int64)
    need(len(rows)==354550 and provider.input_ready[rows].all()and digest(provider.issue_ns[rows])==FUTURE_CPU_TRAIN_SHA,
        'FUTURE_CUDA_FULL_UNFILTERED_INPUT_TRAIN')
    selected=rows if initial else rows[:256];batches=all_score_batches(selected,64)
    need(len(batches)==(5540 if initial else 4)and len(np.unique(np.concatenate(batches)))==len(selected)
        and all(len(b)==64 for b in batches[:-1])and len(batches[-1])==(54 if initial else 64),
        'FUTURE_CUDA_EXACT_BATCH_GEOMETRY')
    return rows,batches


def future_cuda_load(plan,counts):
    from gx1.audit.build_native_m5_ta_feature_cache_v1 import load_snapshot
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    m=plan['metadata'];norm=future_cpu_read(m['normalization']);prior=future_cpu_read(m['prior']);data=future_cpu_read(m['dataset'])
    provider=load_snapshot(m['source_snapshot']);counts['feature_cache_payload_loads']+=1
    rows,batches=future_cuda_batches(provider,initial=plan['lane_group']=='initial')
    need(provider.input_binding(norm)==m['input_binding']
        and digest(provider.source_row[rows])==norm['fit_population']['selected_source_rows_sha256'],
        'FUTURE_CUDA_PHYSICAL_INPUT_BINDING')
    need(ref(data['arrays']['path'])==data['arrays'],'FUTURE_CUDA_TRAIN_TARGET_ARCHIVE')
    truth={}
    with np.load(data['arrays']['path'],allow_pickle=False)as archive:
        counts['target_archive_loads']+=1
        for key in FUTURE_CPU_TRUTH_KEYS:
            a=archive[key];h=data['array_headers'][key]
            need(a.dtype.str==h['dtype']and list(a.shape)==h['shape']and digest(a)==h['sha256'],
                'FUTURE_CUDA_TRUTH_BYTES:'+key)
            truth[key]=a;counts['truth_fields_decoded']+=1
    need(np.array_equal(truth['issue_ns'],provider.issue_ns[rows])
        and objective._native_prior_array_sha(truth)==prior['fit_population']['truth_sha256']==m['truth_sha256'],
        'FUTURE_CUDA_CURRENT_TRAIN_TRUTH_PRIOR_PROOF')
    objective._native_future_state_truth(truth,rows=len(rows),fold_upper_ns=FUTURE_CPU_TRAIN_UPPER,target_contract=m['target_contract'])
    order=[{'source_rows_sha256':digest(provider.source_row[b]),'issue_ns_sha256':digest(provider.issue_ns[b]),'rows':len(b)}for b in batches]
    return {'provider':provider,'rows':rows,'batches':batches,'truth':truth,'normalization':norm,'prior':prior,'order':order}


def future_cuda_packet(context,batch):
    import torch
    raw=context['provider'].gather(context['provider'].source_row[batch],max_rows=64)
    at=np.searchsorted(context['rows'],batch)
    need(np.array_equal(context['rows'][at],batch),'FUTURE_CUDA_SOURCE_TRUTH_JOIN')
    return {'features':{k:torch.from_numpy(v)for k,v in raw.items()},
        'truth':{k:v[at]for k,v in context['truth'].items()}}


def future_cuda_forward(model,features,*,grouped=True,equal_tf=False,capture=False):
    import torch
    from contextlib import nullcontext
    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as body
    need(type(grouped)is type(equal_tf)is type(capture)is bool,'FUTURE_CUDA_FORWARD_FLAGS')
    device=next(model.parameters()).device;features={k:v.to(device)for k,v in features.items()};seen=[];hidden=[]
    groups={}
    for tf in ('M5','M15','H1','H4','D1'):groups.setdefault(str(features['seq_'+tf.lower()].shape[1]),[]).append(tf)
    def before(module,args):
        active=body._BATCH_EQUAL_LENGTH_MTF_EVAL.get()and not module.training and not torch.is_grad_enabled()
        seen.append({'batch_eval_active':bool(active),'model_training':module.training,
            'grad_enabled':torch.is_grad_enabled(),'scope_enabled':body._BATCH_EQUAL_LENGTH_MTF_EVAL.get(),
            'eligible_equal_length_TF_groups':groups})
        need(not equal_tf or active and any(len(v)>1 for v in groups.values()),'FUTURE_CUDA_EQUAL_TF_ACTUAL_ACTIVATION')
    def after(module,args,output):
        h=output['entry_q_timeframe_hidden']
        need(h.dtype==torch.float32 and h.ndim==3 and h.shape[:2]==(len(features['seq_x']),5)
            and bool(torch.isfinite(h).all()),'FUTURE_CUDA_ALL_HIDDEN_FINITE_FLOAT32')
        if capture:hidden.append(h.detach().cpu().clone())
    hooks=[model.backbone.register_forward_pre_hook(before),model.backbone.register_forward_hook(after)]
    try:
        with torch.autocast(device_type=device.type,enabled=False),(
                body.model_finite_check_scope('grouped')if grouped else nullcontext()),(
                body.batch_equal_length_mtf_eval_scope()if equal_tf else nullcontext()):
            output=model(**features)
    finally:
        for hook in hooks:hook.remove()
    need(len(seen)==1 and all(v.dtype==torch.float32 and bool(torch.isfinite(v).all())for v in output.values()),
        'FUTURE_CUDA_ONE_BACKBONE_FINITE_OUTPUT')
    return output,hidden[0]if capture else None,seen[0]


def future_cuda_score(model,packets,plan,prior,*,initial,equal_tf):
    import torch
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    before=state_sha(model.state_dict());rng=training_rng();training=model.training;outputs={};truths={};batches=[]
    model.eval()
    try:
        with torch.no_grad():
            for packet in packets:
                output,_,activation=future_cuda_forward(model,packet['features'],equal_tf=equal_tf)
                cpu={k:v.detach().cpu()for k,v in output.items()};kwargs={'truth':packet['truth'],
                    'fold_upper_ns':FUTURE_CPU_TRAIN_UPPER,'target_contract':plan['metadata']['target_contract']}
                loss=objective.native_future_state_loss(model_output=output,**kwargs)
                if initial:
                    reference=objective.native_future_state_train_prior_logits(prior,rows=len(packet['truth']['issue_ns']),
                        target_contract=plan['metadata']['target_contract'])
                    need(all(torch.equal(cpu[k],v)for k,v in reference.items()),'FUTURE_CUDA_INITIAL_EXACT_PRIOR')
                for k,v in cpu.items():outputs.setdefault(k,[]).append(v)
                for k,v in packet['truth'].items():truths.setdefault(k,[]).append(v)
                batches.append({'rows':len(packet['truth']['issue_ns']),'known_activity_cells':loss['count'],
                    'partial_NLL':float(loss['loss']),'output_sha256':{k:digest(v.numpy())for k,v in cpu.items()},
                    'activation':activation})
        output={k:torch.cat(v)for k,v in outputs.items()};truth={k:np.concatenate(v)for k,v in truths.items()}
        kwargs={'truth':truth,'fold_upper_ns':FUTURE_CPU_TRAIN_UPPER,'target_contract':plan['metadata']['target_contract']}
        scores=objective.native_future_state_score_sums(model_output=output,**kwargs)
        if initial:
            reference=objective.native_future_state_train_prior_logits(prior,rows=len(truth['issue_ns']),target_contract=plan['metadata']['target_contract'])
            need(scores==objective.native_future_state_score_sums(model_output=reference,**kwargs),'FUTURE_CUDA_INITIAL_FULL_PROPER_SCORE_EXACT')
        count=sum(b['known_activity_cells']for b in batches);total=sum(x['partial_log_loss_sum']for x in scores['per_tf'].values())
        return {'rows':len(truth['issue_ns']),'batch_count':len(batches),'batches':batches,'scores':scores,
            'known_activity_cells':count,'proper_partial_NLL':total/count if count else None,
            'initial_prior_exact':initial,'output_sha256':{k:digest(v.numpy())for k,v in output.items()},
            'score_scope':'observed_activity_and_certified_side_support_not_unconditional_calibration'}
    finally:
        model.train(training);restore_training_rng(rng)
        need(state_sha(model.state_dict())==before and all(p.grad is None for p in model.parameters()),'FUTURE_CUDA_SCORE_STATE_GRAD_MUTATION')



def future_cuda_array_parity(expected,value):
    a=np.asarray(expected);b=np.asarray(value);t=THROUGHPUT_EVAL_TOLERANCE
    need(a.shape==b.shape and a.dtype==b.dtype and np.isfinite(a).all()and np.isfinite(b).all(),'FUTURE_CUDA_PARITY_DOMAIN')
    delta=np.abs(a.astype(np.float64)-b.astype(np.float64));nonzero=a!=0
    return {'qualified':bool(np.allclose(a,b,atol=t['atol'],rtol=t['rtol'])),'bitexact':bool(np.array_equal(a,b)),
        'max_absolute_difference':float(delta.max())if delta.size else 0.,
        'max_relative_difference_nonzero_reference':float((delta[nonzero]/np.abs(a[nonzero])).max())if np.any(nonzero)else 0.,
        'nonzero_difference_at_zero_reference':int(np.count_nonzero((~nonzero)&(delta!=0)))}


def future_cuda_parity_pair(model,packet,plan,counts,*,equal_tf):
    """Two eval no-grad TF forwards OR two full-gradient finite-check forwards."""
    import torch
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_future_state_loss
    before=state_sha(model.state_dict());rng=training_rng();training=model.training;records=[];reference_grads={};gradient_differences={}
    need(all(p.grad is None for p in model.parameters()),'FUTURE_CUDA_PARITY_CLEAN_GRADS')
    model.eval()if equal_tf else model.train()
    try:
        for candidate in (False,True):
            restore_training_rng(rng);model.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(not equal_tf):
                output,hidden,activation=future_cuda_forward(model,packet['features'],
                    grouped=candidate if not equal_tf else True,equal_tf=candidate if equal_tf else False,capture=True)
                loss=native_future_state_loss(model_output=output,truth=packet['truth'],fold_upper_ns=FUTURE_CPU_TRAIN_UPPER,
                    target_contract=plan['metadata']['target_contract'])
                record={'timeframe_hidden':hidden.numpy(),'loss':loss['loss'].detach().cpu().numpy(),
                    **{k:v.detach().cpu().numpy()for k,v in output.items()},'activation':activation,
                    'support':{k:v.detach().cpu().tolist()for k,v in loss['counts'].items()}}
                if not equal_tf:
                    loss['loss'].backward();counts['backward_calls']+=1
                    for name,p in model.named_parameters():
                        value=None if p.grad is None else p.grad.detach().cpu().numpy().copy()
                        if not candidate:reference_grads[name]=value
                        elif value is None or reference_grads[name]is None:
                            need(value is reference_grads[name],'FUTURE_CUDA_PARITY_UNUSED_PARAMETER')
                        else:gradient_differences[name]=future_cuda_array_parity(reference_grads[name],value)
            records.append(record)
        a,b=records;need(a['support']==b['support'],'FUTURE_CUDA_PARITY_SUPPORT')
        differences={k:future_cuda_array_parity(a[k],b[k])for k in ('timeframe_hidden','activity_logits','side_logits','loss')}
        qualified=all(v['qualified']for v in (*differences.values(),*gradient_differences.values()))
        need(qualified,'FUTURE_CUDA_GROUPED_PARITY')
        return {'qualified':qualified,'tolerance':THROUGHPUT_EVAL_TOLERANCE,'values':differences,
            'parameter_gradient_differences':gradient_differences,'parameter_gradients_compared':len(gradient_differences),
            'reference_activation':a['activation'],'candidate_activation':b['activation'],
            'scope':'equal_TF_eval_no_grad_only_no_gradient_claim'if equal_tf else'finite_check_scope_full_parameter_gradients_no_equal_TF_batching'}
    finally:
        model.zero_grad(set_to_none=True);model.train(training);restore_training_rng(rng)
        need(state_sha(model.state_dict())==before,'FUTURE_CUDA_PARITY_MODEL_MUTATION')


def future_cuda_gradient_probe(model,packet,plan,counts):
    device=next(model.parameters()).device
    packet={**packet,'features':{k:v.to(device)for k,v in packet['features'].items()}}
    rng=training_rng()
    try:return future_cpu_gradient_probe(model,packet,plan,counts)
    finally:
        restore_training_rng(rng)
        need(state_sha(training_rng())==state_sha(rng),'FUTURE_CUDA_PROBE_ALL_RNG')


def future_cuda_step(model,optimizer,packet,plan,counts,index):
    import torch
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_future_state_loss
    optimizer.zero_grad(set_to_none=True);model.train();output,_,_=future_cuda_forward(model,packet['features'])
    loss=native_future_state_loss(model_output=output,truth=packet['truth'],fold_upper_ns=FUTURE_CPU_TRAIN_UPPER,
        target_contract=plan['metadata']['target_contract'])
    need(not loss['skip_update']and bool(torch.isfinite(loss['loss'])),'FUTURE_CUDA_TRAIN_SUPPORT')
    loss['loss'].backward();counts['backward_calls']+=1
    gradients=[p for p in model.parameters()if p.grad is not None]
    need(gradients and finite_all_per_device(p.grad for p in gradients),'FUTURE_CUDA_FINITE_DATA_GRADIENT')
    device=next(model.parameters()).device
    def norm(values):return float(torch.sqrt(sum((v.detach().double().square().sum()for v in values),torch.zeros((),device=device,dtype=torch.float64))))
    data={group+'_data_gradient_l2':norm(p.grad for p in parameters if p.grad is not None)
        for group,parameters in (('backbone',model.backbone.parameters()),('readout',model.readout.parameters()))}
    before=[p.detach().cpu().clone()for p in model.backbone.parameters()]if index==0 else None
    if index==0:need(data['backbone_data_gradient_l2']==0 and data['readout_data_gradient_l2']>0,'FUTURE_CUDA_ZERO_HEAD_INITIAL_GRADIENT')
    clipped=torch.nn.utils.clip_grad_norm_(gradients,plan['recipe']['gradient_clip_norm'],error_if_nonfinite=True)
    optimizer.step();counts['optimizer_steps']+=1
    need(finite_all_per_device(model.parameters()),'FUTURE_CUDA_UPDATED_PARAMETERS_FINITE')
    if before is not None:
        data.update(backbone_actual_parameter_delta_l2=norm(p.detach().double()-old.to(p.device).double()for p,old in zip(model.backbone.parameters(),before)),
            weight_decay=plan['recipe']['optimizer']['weight_decay'],
            interpretation='data_gradient_before_clip_separate_from_AdamW_decay_actual_parameter_delta')
        need(any(bool(torch.count_nonzero(h.weight))for h in model.readout.activity_heads.values()),'FUTURE_CUDA_HEAD_DATA_UPDATE')
    optimizer.zero_grad(set_to_none=True)
    return {'logical_step':index+1,'loss':float(loss['loss'].detach()),'gradient_norm_pre_clip':float(clipped),
        'support':{k:v.detach().cpu().tolist()for k,v in loss['counts'].items()},
        'output_sha256':{k:digest(v.detach().cpu().numpy())for k,v in output.items()},**data}


def future_cuda_bindings(plan,plan_ref):
    return {'plan':plan_ref,'metadata':plan['metadata'],'recipe':plan['recipe'],
        'backbone_gradient_policy':plan['backbone_gradient_policy'],'runtime_proof':plan['inputs']['runtime_proof'],
        'scope':'whole_TRAIN_fresh_initial_no_updates'if plan['lane_group']=='initial'else'technical_prefix_of_whole_TRAIN_trajectory_not_small_FIT_or_production_routing_selection'}


def future_cuda_checkpoint(model,optimizer,plan,plan_ref,order,cursor,steps):
    return {'schema_version':'gx1_native_future_state_cuda_qualification_checkpoint_v1','cursor':cursor,
        'model_state':model.state_dict(),'optimizer_state':None if optimizer is None else optimizer.state_dict(),
        'scheduler_state':None,'rng_state':training_rng(),'training':model.training,'batch_order':order,
        'optimizer_parameter_names':[n for n,_ in model.named_parameters()],
        'bindings':future_cuda_bindings(plan,plan_ref),'steps':steps}


def future_cuda_publish_checkpoint(path,checkpoint,counts):
    rng=state_sha(training_rng());binding=future_cpu_publish_checkpoint(path,checkpoint,counts)
    need(state_sha(training_rng())==rng,'FUTURE_CUDA_DISK_CHECKPOINT_ALL_RNG');return binding


def future_cuda_restore(model,optimizer,checkpoint,plan,plan_ref,order):
    import torch
    need(type(checkpoint)is dict and checkpoint.get('schema_version')=='gx1_native_future_state_cuda_qualification_checkpoint_v1'
        and checkpoint.get('bindings')==future_cuda_bindings(plan,plan_ref),'FUTURE_CUDA_CHECKPOINT_BINDINGS')
    value=checkpoint['rng_state'];old=training_rng()
    need(type(value)is dict and set(value)==set(old)and type(value['torch_cuda'])is list
        and len(value['torch_cuda'])==len(old['torch_cuda'])
        and all(isinstance(v,torch.Tensor)and v.dtype==torch.uint8 and v.ndim==1
            and v.device.type=='cpu'for v in value['torch_cuda']),'FUTURE_CUDA_RNG_HEADER_BEFORE_COPY')
    try:
        restore_training_rng(value)
        need(state_sha(training_rng())==state_sha(value),'FUTURE_CUDA_RNG_ROUNDTRIP_BEFORE_COPY')
    finally:restore_training_rng(old)
    # Reuse proven strict model/AdamW/name/cursor/pre-copy validation through a
    # transient CPU-schema view; the persisted CUDA artifact remains unchanged.
    slots=checkpoint['optimizer_state'];ids=slots['param_groups'][0]['params']
    heads={ids[i]for i,(name,p)in enumerate(model.named_parameters())if name.startswith('readout.')}
    need(heads and heads<=set(slots['state'])and all(float(slots['state'][i]['step'])==3 for i in heads),
        'FUTURE_CUDA_CURSOR3_READOUT_OPTIMIZER_SLOTS')
    view={**checkpoint,'schema_version':'gx1_native_future_state_cpu_technical_checkpoint_v1',
        'rng_state':{k:v for k,v in value.items()if k!='torch_cuda'},'bindings':future_cpu_bindings(plan,plan_ref)}
    future_cpu_restore(model,optimizer,view,plan,plan_ref,order)
    for p,item in optimizer.state.items():
        need(item['step'].device.type=='cpu'and item['exp_avg'].device==p.device
            and item['exp_avg_sq'].device==p.device,'FUTURE_CUDA_RESTORED_OPTIMIZER_DEVICES')
    restore_training_rng(value)
    need(state_sha(model.state_dict())==state_sha(checkpoint['model_state'])
        and state_sha(optimizer.state_dict())==state_sha(checkpoint['optimizer_state'])
        and state_sha(training_rng())==state_sha(value),'FUTURE_CUDA_ALL_RESTORED_STATE')


def future_cuda_seed_and_runtime(plan):
    import os,torch
    seed_native_fit(plan['recipe']['seed']);m=plan['metadata']['runtime'];gpu,processes=future_cuda_gpu_metadata()
    need(gpu==m['gpu']and torch.cuda.device_count()==1,'FUTURE_CUDA_ACTIVE_DEVICE_RUNTIME')
    properties=torch.cuda.get_device_properties(0)
    need(properties.name==gpu['name']and LARGER_VRAM_MAX<properties.total_memory<=gpu['total_memory_bytes']
        and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        and torch.are_deterministic_algorithms_enabled()and torch.backends.cudnn.deterministic
        and not torch.backends.cudnn.benchmark and not torch.backends.cuda.matmul.allow_tf32
        and not torch.backends.cudnn.allow_tf32 and not torch.is_autocast_enabled(),
        'FUTURE_CUDA_DETERMINISTIC_FP32_FLAGS')
    torch.cuda.set_per_process_memory_fraction(LARGER_VRAM_MAX/properties.total_memory,0)
    torch.cuda.reset_peak_memory_stats()
    return {'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],'deterministic_algorithms':True,
        'cudnn_deterministic':True,'cudnn_benchmark':False,'matmul_allow_tf32':False,'cudnn_allow_tf32':False,
        'autocast':False,'parameter_dtype':'float32','device':'cuda:0','gpu':gpu,
        'cuda_usable_memory_bytes':properties.total_memory}



def future_cuda_physical(plan,plan_ref,root,lane):
    import torch
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    counts=dict.fromkeys(FUTURE_CUDA_LIMITS[lane],0);context=future_cuda_load(plan,counts)
    runtime=future_cuda_seed_and_runtime(plan);m=plan['metadata']
    model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=m['input_binding'],
        architecture=m['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=m['target_contract'],
        backbone_gradient_policy=plan['backbone_gradient_policy']);counts['model_constructions']+=1
    need(all(p.dtype==torch.float32 for p in model.parameters()),'FUTURE_CUDA_FLOAT32_PARAMETERS')
    initial_state=canonical_model_state_sha256(model.state_dict());optimizer=None
    if lane!='initial':
        options={**plan['recipe']['optimizer'],'betas':tuple(plan['recipe']['optimizer']['betas'])}
        optimizer=torch.optim.AdamW(model.parameters(),**options);counts['optimizer_constructions']+=1
        need(not optimizer.state,'FUTURE_CUDA_FRESH_ADAMW')
    def count(name):
        def hook(*args):
            counts[name]+=1;need(counts[name]<=FUTURE_CUDA_LIMITS[lane][name],'FUTURE_CUDA_FORWARD_BUDGET')
        return hook
    hooks=[model.register_forward_pre_hook(count('model_forwards')),model.backbone.register_forward_pre_hook(count('backbone_forwards'))]
    norm_before=state_sha(context['normalization']);prior_before=state_sha(context['prior']);steps=[];probe=None;finite=None;equal=None;checkpoint3=None
    packets=None if lane=='initial'else[future_cuda_packet(context,b)for b in context['batches']]
    packet_before=None if packets is None else state_sha(packets)
    try:
        initial=future_cuda_score(model,(future_cuda_packet(context,b)for b in context['batches'])if packets is None else packets,
            plan,context['prior'],initial=True,equal_tf=lane=='initial')
        need(initial['rows']==(354550 if lane=='initial'else 256)and initial['batch_count']==(5540 if lane=='initial'else 4)
            and initial['initial_prior_exact']is True and counts['optimizer_steps']==0,'FUTURE_CUDA_ALL_INITIAL_BEFORE_UPDATES')
        publish(root/'INITIAL_SCORE.json',initial)
        reference=None
        if lane=='replay':
            parent=FUTURE_CUDA_SCOPES['mechanics']['root']/'continuous'
            terminal=future_cpu_read(ref(parent/'TERMINAL.json'));reference=future_cpu_read(terminal['result'])
            future_cpu_parent_stopped(reference['process_identity'])
            need(reference['plan']==plan_ref and reference['report']['initial']==initial
                and reference['report']['initial_state_sha256']==initial_state,'FUTURE_CUDA_REPLAY_FRESH_INITIAL')
            receipt=future_cpu_read(reference['report']['checkpoint3_receipt'])
            need(receipt['plan']==plan_ref and receipt['cursor']==3 and receipt['checkpoint']['path']==str(parent/'CURSOR003.pt')
                and ref(receipt['checkpoint']['path'])==receipt['checkpoint'],'FUTURE_CUDA_CURSOR3_AUTHORITY')
            checkpoint=torch.load(receipt['checkpoint']['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']+=1
            need(future_cpu_proof(checkpoint)==receipt['state_proof'],'FUTURE_CUDA_CURSOR3_PROOF')
            future_cuda_restore(model,optimizer,checkpoint,plan,plan_ref,context['order'])
            steps=list(checkpoint['steps']);checkpoint3=reference['report']['checkpoint3_receipt']
        if lane!='initial':
            for index in range(3 if lane=='replay'else 0,4):
                steps.append(future_cuda_step(model,optimizer,packets[index],plan,counts,index))
                if lane=='continuous'and index==0:
                    probe=future_cuda_gradient_probe(model,packets[0],plan,counts)
                    finite=future_cuda_parity_pair(model,packets[0],plan,counts,equal_tf=False)
                    equal=future_cuda_parity_pair(model,packets[0],plan,counts,equal_tf=True)
                if lane=='continuous'and index==2:
                    checkpoint=future_cuda_checkpoint(model,optimizer,plan,plan_ref,context['order'],3,steps)
                    binding=future_cuda_publish_checkpoint(root/'CURSOR003.pt',checkpoint,counts)
                    checkpoint3=publish(root/'CURSOR003.json',{'plan':plan_ref,'cursor':3,'checkpoint':binding,
                        'state_proof':future_cpu_proof(checkpoint)})
            final=future_cuda_score(model,packets,plan,context['prior'],initial=False,equal_tf=False)
        else:final=initial
        checkpoint=future_cuda_checkpoint(model,optimizer,plan,plan_ref,context['order'],0 if lane=='initial'else 4,steps)
        binding=future_cuda_publish_checkpoint(root/('INITIAL_STATE.pt'if lane=='initial'else'FINAL_STATE.pt'),checkpoint,counts)
        proof=future_cpu_proof(checkpoint)
        if lane=='replay':
            need(proof==reference['report']['final_state_proof']and steps==reference['report']['steps']
                and final==reference['report']['final'],'FUTURE_CUDA_COLD_REPLAY_BIT_EXACT')
        need(counts==FUTURE_CUDA_LIMITS[lane]and state_sha(context['normalization'])==norm_before
            and state_sha(context['prior'])==prior_before and(packets is None or state_sha(packets)==packet_before),
            'FUTURE_CUDA_EXACT_COUNTS_FROZEN_INPUTS')
        if lane=='initial':need(canonical_model_state_sha256(model.state_dict())==initial_state,'FUTURE_CUDA_FULL_INITIAL_STATE_UNCHANGED')
        peak=torch.cuda.max_memory_reserved();need(peak<=LARGER_VRAM_MAX,'FUTURE_CUDA_VRAM_LIMIT')
        return {'lane':lane,'counts':counts,'batch_order':context['order'],'initial_state_sha256':initial_state,
            'initial':initial,'final':final,'steps':steps,'post_first_update_gradient':probe,
            'finite_check_gradient_parity':finite,'equal_tf_evaluation':equal,'checkpoint3_receipt':checkpoint3,
            'final_checkpoint':binding,'final_state_proof':proof,'cold_replay_bit_exact':lane=='replay',
            'runtime_flags':runtime,'cuda_peak_reserved_bytes':peak,'logical_updates':0 if lane=='initial'else 4,
            'physical_updates_this_lane':counts['optimizer_steps'],'whole_TRAIN_initialbaseline_qualified':lane=='initial',
            'proper_prefix_NLL_gain':None if lane=='initial'else initial['proper_partial_NLL']-final['proper_partial_NLL'],
            'production_gradient_policy_selected':False,'generalization_or_economic_edge_qualified':False,
            'scope':'full354550TRAIN_fresh_INITIAL_only'if lane=='initial'else'four_distinct_B64_TRAIN_prefix_updates_replay_duplicates_only_step4_not_independent_small_FIT',
            'observability':'weekly_v2_only_partial_positive_ACTIVITY_never_partial_FLAT_or_side; unchanged_truth_masks; no_new_calendar'}
    finally:
        for hook in hooks:hook.remove()


def run_future_cuda(plan_sha,lane):
    from gx1.contracts.gx1_capped_execution_v1 import require_native_future_state_cuda_qualification_execution
    began=time.monotonic();guard=require_native_future_state_cuda_qualification_execution(plan_sha256=plan_sha,lane=lane)
    plan,plan_ref,policy_ref,live=inspect_future_cuda(plan_sha,lane,admitted=True)
    kind=plan['lane_group'];scope=FUTURE_CUDA_SCOPES[kind];execution=plan['execution']
    need(guard['memory_max']==guard['memory_high']==20*1024**3 and guard['swap']==512*1024**2
        and guard['pids']==128 and guard['numerical_threads']==8 and guard['cpu_affinity']==list(range(19))
        and guard['plan']==plan_ref and guard['outer_seconds']==execution['outer_seconds']
        and guard['internal_seconds']==execution['internal_seconds'],'FUTURE_CUDA_EXACT_GUARD')
    root=future_cuda_lane_root(lane);need(guard['output_root']==str(root),'FUTURE_CUDA_GUARD_OUTPUT_ROOT')
    root.mkdir(parents=True,exist_ok=False);process=future_cpu_process_identity()
    claim=publish(root/'CLAIM.json',{'stage':scope['stage'],'status':'CLAIMED_ONE_USE','plan':plan_ref,
        'policy_at_claim':policy_ref,'lane':lane,'process_identity':process,'fresh_live_idle_runtime':live,
        'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
    previous=signal.getsignal(signal.SIGALRM);result=None
    def expired(*args):raise TimeoutError('FUTURE_CUDA_INTERNAL_DEADLINE')
    try:
        remaining=execution['internal_seconds']-(time.monotonic()-began);need(remaining>0,'FUTURE_CUDA_TIME_BEFORE_PAYLOADS')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
        report=future_cuda_physical(plan,plan_ref,root,lane)
        need(all(sha(REPO/k)==v for k,v in plan['source_owners'].items())and ref(POLICY)==policy_ref
            and ref(scope['plan'])==plan_ref,'FUTURE_CUDA_SOURCE_POLICY_PLAN_BOOKEND')
        for binding in [*plan['inputs'].values(),*(plan['metadata'][k]for k in
                ('source_snapshot','normalization','input_binding_ref','prior','dataset','native_manifest'))]:future_cpu_read(binding)
        snapshot=future_cpu_read(plan['metadata']['source_snapshot']);data=future_cpu_read(plan['metadata']['dataset'])
        need(ref(snapshot['archive']['path'])==snapshot['archive']and ref(data['arrays']['path'])==data['arrays']
            and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id']
            and physical_windows_boot_id()==live['physical_windows_boot_id']
            and subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==live['source_commit'],
            'FUTURE_CUDA_PAYLOAD_BOOT_BOOKEND')
        need(common.firewall_state()['denied_attempts']==0,'FUTURE_CUDA_CLEAN_FIREWALL')
        status='COMPLETE_NATIVE_FUTURE_STATE_CUDA_INITIAL'if kind=='initial'else'COMPLETE_NATIVE_FUTURE_STATE_CUDA_MECHANICS'
        result=publish(root/'RESULT.json',{'stage':scope['stage'],'status':status,'plan':plan_ref,'claim':claim,
            'process_identity':process,'report':report,'guard':guard,'source_unchanged':True,'test_data_used':False,
            'elapsed_seconds':time.monotonic()-began})
        terminal=publish(root/'TERMINAL.json',{'stage':scope['stage'],'status':'COMPLETE','exit_code':0,
            'plan':plan_ref,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,
            'source_unchanged':True,'test_data_used':False})
        return {'result':result,'terminal':terminal}
    except BaseException as error:
        publish(root/'TERMINAL.json',{'stage':scope['stage'],'status':'FAILED_CONSUMED','exit_code':1,
            'plan':plan_ref,'claim':claim,'result':result,'error':type(error).__name__+':'+str(error)[:2000],
            'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':False,'test_data_used':False});raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def future_cuda_cli():
    import sys
    parser=argparse.ArgumentParser();parser.add_argument('--native-future-state-cuda-qualification',action='store_true',required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--lane',choices=('continuous','replay','initial'),required=True)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    need(sys.argv[1:]==['--native-future-state-cuda-qualification','--plan-sha256',args.plan_sha256,'--lane',args.lane]
        +(['--preflight']if args.preflight else []),'FUTURE_CUDA_EXACT_CLI')
    if args.preflight:
        _,plan_ref,policy_ref,live=inspect_future_cuda(args.plan_sha256,args.lane,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_FUTURE_STATE_CUDA_METADATA_ONLY','plan':plan_ref,'policy':policy_ref,
            'lane':args.lane,'cuda_initialized':False,'model_constructions':0,'optimizer_steps':0,
            'claim_created':False,'live_idle_observed_utc':live['observed_utc']}))
    else:print(json.dumps(run_future_cuda(args.plan_sha256,args.lane),sort_keys=True))


if __name__=='__main__':
    import sys
    if '--native-future-state-cuda-qualification'in sys.argv:future_cuda_cli()
    elif '--native-future-state-cpu-mechanics'in sys.argv:future_cpu_cli()
    else:main()
