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
        Path('/home/andre2/GX1_RUNS/NATIVE_M5_FULL_GC_SHORT_FIT_20261119_002'))}
ADAMW_RECIPE={'lr':1e-4,'betas':[.9,.999],'eps':1e-8,'weight_decay':1e-4,
    'amsgrad':False,'foreach':None,'maximize':False,'capturable':False,
    'differentiable':False,'fused':None}
FIT_SOURCE_PATHS=tuple(dict.fromkeys((*SOURCE_PATHS,
    'gx1/contracts/market_state_ta_fold2_m5_fit_v1.py',
    'tests/test_gx1_capped_run_contract.py','tests/test_market_state_ta_integrated_objective_v1.py','tests/test_entry_model_native_input_normalization.py')))


def fit_recipe(value,mode):
    fields={'optimizer','gradient_clip_norm','scheduler','seed','batch_rows','score_batch_rows','fit_batches',
        'score_batches_per_split','checkpoint_every_updates','outer_seconds','internal_seconds',
        'required_windows_boot_id_greater_than'}
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
    need(mode in FIT_SCOPES and(mode!='mechanics'or value['fit_batches']==2),'FIT_MODE')
    if mode=='main':
        need(value['fit_batches']==96 and value['checkpoint_every_updates']==32
            and value['outer_seconds']==1800 and value['internal_seconds']==1770,'FIT_SMALL_FIXED_BUDGET')
    need((mode=='main'and value['score_batches_per_split']is None)or
        (mode=='mechanics'and type(value['score_batches_per_split'])is int and value['score_batches_per_split']>0),
        'FIT_SCORE_SCOPE')
    return dict(value)


def fit_metadata(inputs,mode):
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
    result={'schema_version':'gx1_native_m5_source_rank_cohorts_v1',
        'selection':'linspace_source_ranks_before_outcomes_no_observedness_or_class_filter',
        'pool_axis_authority':metadata['mechanics']['result'],'cohorts':{}}
    for name,pool,blocks,batch in [('FIT','TRAIN',96,16),('CONTROL','CONTROL',8,64)]:
        axis=metadata['mechanics']['axis'][pool]
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
        fold_upper_ns=FIT_END_NS,target_contract_sha256=plan['metadata']['target_contract']['contract_sha256'])
    context['prior']=prior
    context['prior_ref']=publish(root/'MATCHED_FIT_PRIOR.json',prior)
    return context['prior_ref']


def fit_limits(recipe,mode,metadata=None):
    updates=3 if mode=='mechanics'else recipe['fit_batches']
    return {'model_constructions':2 if mode=='mechanics'else 1,
        'model_forwards':updates+(4*recipe['score_batches_per_split']if mode=='mechanics'else
            2*sum((item['rows']+recipe['score_batch_rows']-1)//recipe['score_batch_rows']
                for item in small_cohort_manifest(metadata)['cohorts'].values())),
        'backward_calls':updates,'optimizer_steps':updates}


def build_fit_plan(*,inputs,recipe,mode,source_base_commit=None):
    need(mode in FIT_SCOPES,'FIT_MODE');recipe=fit_recipe(recipe,mode)
    stage,path,root=FIT_SCOPES[mode]
    metadata=fit_metadata(inputs,mode)
    head=source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    return {'schema_version':'gx1_native_m5_full_gc_fit_plan_v1','stage':stage,'mode':mode,
        'status':'PREPARED_NOT_ADMITTED','output_root':str(root),'source_base_commit':head,
        'source_owners':{p:sha(REPO/p)for p in FIT_SOURCE_PATHS},'inputs':inputs,'metadata':metadata,
        'recipe':recipe,**({'cohort_selection':small_cohort_manifest(metadata)}if mode=='main'else{}),
        'execution':{'class':'trainer','device':'cuda','memory':'20G','swap':'512M',
            'tasks_max':128,'numerical_threads':8,'outer_seconds':recipe['outer_seconds'],
            'internal_seconds':recipe['internal_seconds'],'kill_after_seconds':10},
        'fit_order':'source_only_evenly_spaced_nonoverlapping_contiguous_batches',
        'score_order':'fixed_source_only_evenly_spaced_nonoverlapping_contiguous_batches',
        'normalization_axis':'all_TRAIN_input_ready_issues_with_any_mature_supervised_cell_before_sampling',
        'GC_observedness_row_filter':False,'scheduler_state':None,'EMA_state':None,
        'limits':fit_limits(recipe,mode,metadata),'qualification_weights_loaded':False,'training_authorized':True,'test_data_used':False,
        'economic_edge_proven':False}


def fit_source_clean(plan):
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
        if plan['mode']=='main':
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


def fresh_fit_model(provider,normalization,plan):
    import torch
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_bar_direction_model
    seed_native_fit(plan['recipe']['seed'])
    model=build_native_bar_direction_model(input_normalization=normalization,
        input_binding=provider.input_binding(normalization),architecture=plan['metadata']['architecture'],
        device=torch.device('cuda',0),gc_mtf_history=normalization['gc_mtf_history'])
    options={**plan['recipe']['optimizer'],'betas':tuple(plan['recipe']['optimizer']['betas'])}
    optimizer=torch.optim.AdamW(model.parameters(),**options)
    need(not optimizer.state,'FIT_FRESH_OPTIMIZER')
    model._native_fit_counts={'model_constructions':1,'model_forwards':0,'backward_calls':0,'optimizer_steps':0}
    def count_forward(module,args):
        module._native_fit_counts['model_forwards']+=1
        need(module._native_fit_counts['model_forwards']<=plan['limits']['model_forwards'],'FIT_FORWARD_LIMIT')
    model.register_forward_pre_hook(count_forward)
    return model,optimizer


def prepare_native_fit(plan,root,*,reload_normalization=None,reload_prior=None):
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


def fit_step(model,optimizer,provider,cohort,rows,plan):
    import torch
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_direction_loss
    model.train();optimizer.zero_grad(set_to_none=True)
    raw=provider.gather(rows,max_rows=plan['recipe']['batch_rows'])
    features={key:torch.from_numpy(value).to(next(model.parameters()).device)for key,value in raw.items()}
    truth=slice_truth(cohort,rows,fit=True)
    output=model(features)
    loss=native_bar_direction_loss(model_output=output,truth=truth,fold_upper_ns=FIT_END_NS,
        target_contract_sha256=plan['metadata']['target_contract']['contract_sha256'])
    need(not loss['skip_update']and bool(torch.isfinite(loss['loss'])),'FIT_SUPERVISED_FINITE_LOSS')
    loss['loss'].backward()
    model._native_fit_counts['backward_calls']+=1
    gradients=[p for p in model.parameters()if p.grad is not None]
    need(gradients and all(bool(torch.isfinite(p.grad).all())for p in gradients),'FIT_FINITE_GRADIENTS')
    norm=torch.nn.utils.clip_grad_norm_(gradients,plan['recipe']['gradient_clip_norm'],error_if_nonfinite=True)
    optimizer.step();model._native_fit_counts['optimizer_steps']+=1;optimizer.zero_grad(set_to_none=True)
    need(all(bool(torch.isfinite(p).all())for p in model.parameters()),'FIT_FINITE_PARAMETERS')
    return {'loss':float(loss['loss'].detach()),'gradient_norm':float(norm),'source_rows_sha256':digest(rows),
        'predictive_counts':{key:value.detach().cpu().tolist()for key,value in loss['counts'].items()},
        'output_sha256':{key:digest(value.detach().cpu().numpy())for key,value in output.items()}}


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
                    output=model({key:torch.from_numpy(value).to(next(model.parameters()).device)for key,value in raw.items()})
                    for key,value in output.items():outputs[key].append(value.detach().cpu().numpy().copy())
                outputs={key:np.concatenate(value)for key,value in outputs.items()}
                cohort=context['cohorts'][name];truth=slice_truth(cohort,rows)
                kwargs={'truth':truth,'fold_upper_ns':cohort['upper_ns'],
                    'target_contract_sha256':plan['metadata']['target_contract']['contract_sha256']}
                reports[name]={'rows':len(rows),'source_rows_sha256':digest(rows),
                    'model':native_bar_direction_score_sums(model_output=outputs,**kwargs),
                    'fixed_TRAIN_prior':native_bar_direction_score_sums(
                        model_output=native_bar_train_prior_logits(context['prior'],rows=len(rows)),**kwargs)}
                captures.update({name+'__source_row':rows,**{name+'__'+key:value for key,value in outputs.items()},
                    **{name+'__truth_'+key:value for key,value in truth.items()}})
    finally:model.train(training);restore_training_rng(rng)
    need(state_sha(model.state_dict())==before,'FIT_SCORE_STATE_MUTATION')
    return reports,captures


def fit_checkpoint(model,optimizer,context,plan,plan_ref,cursor):
    return {'schema_version':'gx1_native_m5_full_gc_training_state_v1','plan':plan_ref,
        'cursor':cursor,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
        'scheduler_state':None,'EMA_state':None,'rng_state':training_rng(),'training':model.training,
        'optimizer_parameter_names':[name for name,_ in model.named_parameters()],
        'bindings':{'input':context['provider'].input_binding(context['normalization']),
            'normalization':context['normalization_ref'],'prior':context['prior_ref'],'target':plan['metadata']['target_contract'],
            'architecture':plan['metadata']['architecture'],'recipe':plan['recipe']}}


def publish_checkpoint(path,checkpoint):
    import torch
    stream=io.BytesIO();torch.save(checkpoint,stream);payload=stream.getvalue();stream.close()
    binding=publish(path,payload)
    # Strict structural parity on serialized state; this does not construct a model.
    reloaded=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
    need(state_sha(reloaded)==state_sha(checkpoint),'FIT_CHECKPOINT_ROUNDTRIP')
    return binding


def restore_fit_checkpoint(model,optimizer,context,plan,plan_ref,checkpoint):
    import torch
    fields={'schema_version','plan','cursor','model_state','optimizer_state','scheduler_state','EMA_state',
        'rng_state','training','optimizer_parameter_names','bindings'}
    need(type(checkpoint)is dict and set(checkpoint)==fields
        and checkpoint['schema_version']=='gx1_native_m5_full_gc_training_state_v1'
        and checkpoint['plan']==plan_ref and checkpoint['scheduler_state']is checkpoint['EMA_state']is None
        and checkpoint['optimizer_parameter_names']==[name for name,_ in model.named_parameters()]
        and checkpoint['bindings']==fit_checkpoint(model,optimizer,context,plan,plan_ref,checkpoint['cursor'])['bindings'],
        'FIT_CHECKPOINT_BINDINGS')
    need(type(checkpoint['cursor'])is int and 0<=checkpoint['cursor']<=plan['recipe']['fit_batches']
        and type(checkpoint['training'])is bool,'FIT_CHECKPOINT_CURSOR')
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


def fit_physical(plan,plan_ref,root):
    import torch
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    prepare_started=time.monotonic()
    context=prepare_native_fit(plan,root);prepare_seconds=time.monotonic()-prepare_started
    provider=context['provider'];recipe=plan['recipe']
    normalization_body=state_sha(context['normalization'])
    source_body=state_sha(provider.source_binding)
    selected_report=None;pool_prior_ref=context['prior_ref']
    if plan['mode']=='main':
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
    model,optimizer=fresh_fit_model(provider,context['normalization'],plan)
    initial_sha=canonical_model_state_sha256(model.state_dict())
    statistics_parent=plan['metadata']['mechanics']if plan['mode']=='main'else plan['metadata']['statistics_parent']
    need(initial_sha==statistics_parent['initial_state_sha256'],'FIT_FRESH_MAIN_INITIAL')
    score_started=time.monotonic();initial,capture_initial=fit_score(model,context,panels,plan)
    torch.cuda.synchronize();initial_score_seconds=time.monotonic()-score_started
    initial_score_ref=publish(root/'INITIAL_SCORE.json',{'axis':context['axis'],
        'axis_role':'original_source_and_normalization_pools','selected_cohorts':selected_report,'scores':initial,
        'model_state_sha256':initial_sha,'normalization':context['normalization_ref'],'prior':context['prior_ref']})
    stream=io.BytesIO();np.savez_compressed(stream,**capture_initial)
    capture_initial_ref=publish(root/'INITIAL_CAPTURE.npz',stream.getvalue());stream.close()
    initial_checkpoint=publish_checkpoint(root/'INITIAL_STATE.pt',fit_checkpoint(model,optimizer,context,plan,plan_ref,0))
    steps=[];checkpoint_refs=[];mechanics_checkpoint=None
    for cursor,rows in enumerate(batches):
        started=time.monotonic();step=fit_step(model,optimizer,provider,context['cohorts']['TRAIN'],rows,plan)
        torch.cuda.synchronize();step['elapsed_seconds']=time.monotonic()-started
        steps.append(step)
        if plan['mode']=='mechanics'and cursor==0:
            mechanics_checkpoint=publish_checkpoint(root/'RESUME_STATE.pt',fit_checkpoint(
                model,optimizer,context,plan,plan_ref,1))
        if (cursor+1)%recipe['checkpoint_every_updates']==0 or cursor+1==len(batches):
            checkpoint_refs.append(publish_checkpoint(root/('CHECKPOINT_'+str(cursor+1).zfill(6)+'.pt'),
                fit_checkpoint(model,optimizer,context,plan,plan_ref,cursor+1)))
    score_started=time.monotonic();final,capture_final=fit_score(model,context,panels,plan)
    torch.cuda.synchronize();final_score_seconds=time.monotonic()-score_started
    need(state_sha(context['normalization'])==normalization_body
        and state_sha(provider.source_binding)==source_body
        and ref(context['normalization_ref']['path'])==context['normalization_ref'],'FIT_NORMALIZATION_SOURCE_BOOKEND')
    final_sha=canonical_model_state_sha256(model.state_dict());need(final_sha!=initial_sha,'FIT_WEIGHTS_UNCHANGED')
    final_checkpoint=publish_checkpoint(root/'FINAL_STATE.pt',fit_checkpoint(
        model,optimizer,context,plan,plan_ref,len(batches)))
    stream=io.BytesIO();np.savez_compressed(stream,**capture_final)
    capture_final_ref=publish(root/'FINAL_CAPTURE.npz',stream.getvalue());stream.close()
    report={'axis':context['axis'],'axis_role':'original_source_and_normalization_pools',
        'normalization':context['normalization_ref'],
        'normalization_population':context['normalization']['fit_population'],'prior':context['prior_ref'],
        'normalization_fits':0,'prior_fits':1 if plan['mode']=='main'else 0,
        'original_TRAIN_pool_prior':pool_prior_ref,'prior_scope':'matched_1536_FIT_predictive_cells'if plan['mode']=='main'else'original_TRAIN_pool',
        'selected_cohorts':selected_report,'statistics_parent':statistics_parent['result'],
        'train_batch_rows':recipe['batch_rows'],'score_batch_rows':recipe['score_batch_rows'],
        'initial_state_sha256':initial_sha,'final_state_sha256':final_sha,'initial':initial,'final':final,
        'initial_checkpoint':initial_checkpoint,'final_checkpoint':final_checkpoint,'checkpoints':checkpoint_refs,
        'initial_score_artifact':initial_score_ref,'captures':{'INITIAL':capture_initial_ref,'FINAL':capture_final_ref},
        'prepare_seconds':prepare_seconds,'initial_score_seconds':initial_score_seconds,'final_score_seconds':final_score_seconds,
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'steps':steps,
        'canonical_optimizer_updates':len(batches),'evaluation_batches_per_phase':sum(len(value)for value in panels.values()),
        'actual_counts':dict(model._native_fit_counts),'score_scope':('fixed_source_only_mechanics_panels_not_full_population'if plan['mode']=='mechanics'
            else'all_source_selected_1536_FIT_and_512_CONTROL_rows_including_unknown_last_CONTROL_row'),
        'score_lane':'eval_no_grad_same_batch_geometry_and_frozen_TRAIN_stats',
        'qualification_weights_loaded':False,'test_data_used':False,'economic_edge_proven':False}
    if plan['mode']=='mechanics':
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
    stage,path,root=FIT_SCOPES['mechanics']
    authority=read(ref(root/'COLD_RESUME_AUTHORITY.json'))
    need(plan_ref['path']==str(path)and plan['stage']==stage and authority['plan']==plan_ref
        and authority['parent_pid']==os.getppid()and authority['claim']==ref(root/'CLAIM.json')
        and authority['policy']==ref(POLICY)and fit_source_clean(plan),'FIT_COLD_PARENT')
    checked_fit_inputs(plan);boot=fit_boot_state(plan,admitted=True)
    need(boot['actual_windows_boot_id']==authority['physical_windows_boot_id'],'FIT_COLD_BOOT')
    out=root/'COLD_RESUME';out.mkdir(exist_ok=False)
    context=prepare_native_fit(plan,out,reload_normalization=authority['normalization'],reload_prior=authority['prior'])
    model,optimizer=fresh_fit_model(context['provider'],context['normalization'],plan)
    need(ref(authority['checkpoint']['path'])==authority['checkpoint'],'FIT_COLD_CHECKPOINT_HASH')
    checkpoint=torch.load(authority['checkpoint']['path'],map_location='cpu',weights_only=True)
    cursor=restore_fit_checkpoint(model,optimizer,context,plan,plan_ref,checkpoint)
    batches=selected_batches(context['cohorts']['TRAIN']['fit_rows'],plan['recipe']['fit_batches'])
    need(cursor==1 and len(batches)==2,'FIT_COLD_CURSOR')
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
        'actual_counts':dict(model._native_fit_counts),'clean_process_io_firewall':common.firewall_state(),
        'test_data_used':False})
    return actual


def require_cold_resume_proof(proof,*,parent_pid):
    firewall=proof.get('clean_process_io_firewall')
    need(proof.get('bitexact')is True and proof.get('parent_pid')==parent_pid
        and type(firewall)is dict and firewall.get('denied_attempts')==0
        and firewall.get('denied_operations')=={},'FIT_COLD_PROOF')
    return proof


def run_native_fit(plan_sha,mode):
    import os,sys
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    began=time.monotonic();common.firewall_state();guard=require_guarded_cuda_trainer_execution()
    plan,plan_ref,policy_ref=inspect_fit(plan_sha,mode,admitted=True)
    boot=fit_boot_state(plan,admitted=True)
    stage,path,root=FIT_SCOPES[mode];root.mkdir(exist_ok=False)
    claim=publish(root/'CLAIM.json',{'stage':stage,'status':'CLAIMED_ONE_USE','plan':plan_ref,
        'policy_at_claim':policy_ref,'physical_windows_boot_id':boot['actual_windows_boot_id'],
        'max_invocations':1,'remaining_invocations':0,'test_data_used':False})
    old=signal.getsignal(signal.SIGALRM);result=None
    def expired(*args):raise TimeoutError('NATIVE_M5_FIT_INTERNAL_LIMIT')
    try:
        remaining=plan['recipe']['internal_seconds']-(time.monotonic()-began);need(remaining>0,'FIT_TIME_BEFORE_INPUTS')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
        report=fit_physical(plan,plan_ref,root)
        if mode=='mechanics':
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
        checked_fit_inputs(plan)
        need(report['actual_counts']==plan['limits'],'FIT_ACTUAL_COUNT_LIMITS')
        need(fit_source_clean(plan)and ref(POLICY)==policy_ref and ref(path)==plan_ref,'FIT_SOURCE_POLICY_BOOKEND')
        updates=3 if mode=='mechanics'else plan['recipe']['fit_batches']
        result=publish(root/'RESULT.json',{'stage':stage,
            'status':'COMPLETE_NATIVE_M5_FULL_GC_FIT_MECHANICS'if mode=='mechanics'else'COMPLETE_NATIVE_M5_FULL_GC_SHORT_FIT',
            'plan':plan_ref,'claim':claim,'report':report,'guard':guard,'source_unchanged':True,
            'physical_windows_boot_id':boot['actual_windows_boot_id'],
            'optimizer_steps':updates,'training_authorized':True,'test_data_used':False,
            'economic_edge_proven':False,'clean_process_io_firewall':common.firewall_state(),
            'elapsed_seconds':time.monotonic()-began})
        terminal=publish(root/'TERMINAL.json',{'stage':stage,'status':'COMPLETE','exit_code':0,
            'plan':plan_ref,'claim':claim,'result':result,'source_unchanged':True,'remaining_invocations':0,
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
    if args.out_dir==str(root/'COLD_RESUME')and mode=='mechanics'and not args.preflight:
        cold_resume_fit({'path':str(path),'sha256':args.plan_sha256});return
    need(args.out_dir==str(root),'FIT_CLI_OUTPUT')
    if args.preflight:
        plan,plan_ref,policy_ref=inspect_fit(args.plan_sha256,mode,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_NATIVE_M5_FULL_GC_FIT_METADATA_ONLY','plan':plan_ref,
            'policy':policy_ref,'physical_boot':fit_boot_state(plan,admitted=False),
            'claim_created':False,'raw_values_decoded':0,'feature_values_decoded':0,
            'model_constructions':0,'optimizer_steps':0}))
    else:print(json.dumps(run_native_fit(args.plan_sha256,mode),sort_keys=True))


if __name__=='__main__':main()
