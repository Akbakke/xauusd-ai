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
    parser.add_argument('--out-dir',required=True);parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();common.firewall_state()
    need(args.plan==str(PLAN)and args.out_dir==str(ROOT),'CLI_SCOPE')
    if args.preflight:
        _,plan_ref,policy_ref=inspect(args.plan_sha256,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_COMPLETED_PARENT_METADATA_ONLY','plan':plan_ref,'policy':policy_ref,
            'claim_created':False,'raw_values_decoded':0,'feature_values_decoded':0,'model_forwards':0,'autograd_calls':0}))
    else:print(json.dumps(run(args.plan_sha256),sort_keys=True))


if __name__=='__main__':main()
