"""Two cached CPU evaluations for exact last-head versus hidden-state attribution."""
from pathlib import Path
import json, os, resource, signal, subprocess, time
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from gx1.contracts.unified_exit_selected_sampler_v1 import file_sha256
from gx1.contracts.unified_exit_random_access_model_v1 import strict_load_random_access_v2_state
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.scripts import run_unified_exit_random_access_val_v1 as native
from scripts.collect_gx1_handover_readonly import _native_processes
repo=Path('/home/andre2/src/GX1_CURRENT')
base=Path('/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912')
out=Path(__file__).parent
paired=base/'NATIVE_FROZEN_TRACE_MEMORY_FCD03E88_REFERENCE/LEARNING95_96_20260916'
trace=base/'NATIVE_REAL_TRAIN_TO_VAL_FDD70E5C/OPERATOR_OBSERVATIONS/FROZEN_POLICY_TRACE_NATIVE_BATCH_20260916_V2'
start=time.monotonic();signal.alarm(600)
def require(x,s):
    if not x: raise RuntimeError(s)
def read(p):return json.loads(p.read_text())
def write(n,x):
    with (out/n).open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def git(*a):return subprocess.check_output(['git','-C',str(repo),*a],text=True).strip()
source=git('rev-parse','HEAD')
require(source=='0dd34ff8352a3b56746a00b4a2564d128fb56ac1' and not git('status','--porcelain'),'Source mismatch')
require(not _native_processes(repo) and read(repo/'NEXT_RUN_POLICY.json')['training_enabled'] is False,'Training must be stopped')
write('ATTRIBUTION_PLAN.json',{'source_commit':source,'bound_scope':'One existing16-entry/64-transition batch, ONLINE95 and96 CPU eval. Exactly2 Entry and2 Exit forwards; no teacher, backward, optimizer, new materialization, VAL or TEST.',
 'observed_blocker':'Cache-only diagnosis cannot distinguish constant output shift caused by final head parameters from hidden representations.',
 'predefined_method':'Capture existing exit_random_access_fuse output. For Q=w*h+b, symmetric exact decomposition dQ=dw*mean(h)+mean(w)*dh+db. Report sides separately; compare reconstruction against cached outputs. No architecture/weight modification or parameter transplant.',
 'criteria':'Both Entry/Exit/forecast output arrays must exactly reproduce existing caches. Store algebra residual including float32 rounding. Do not claim raw gradient causality or generalization; no training authorization.'})
recipe=read(base/'NATIVE_FROZEN_TRACE_MEMORY_FCD03E88_REFERENCE/NATIVE_RECIPE.json')
for k,v in recipe['recipe_env'].items():os.environ[str(k)]=str(v)
os.environ['CUDA_VISIBLE_DEVICES']=''
trainer._set_deterministic(int(recipe['trainer_cli']['seed']),torch.device('cpu'),'deterministic_fp32')
torch.set_num_threads(min(4,len(os.sched_getaffinity(0))))
inputs=trace/'BATCH_INPUTS.pt'
require(file_sha256(inputs)=='5cff4b8ab5406f2cf68875d9e340ef617f1b877e22c56f5e54e67ae0af68e234','Input mismatch')
cache=torch.load(inputs,map_location='cpu',weights_only=True,mmap=True)
local=read(paired/'FIVE_STEP_LOCAL_BATCH.json')['row']
proven=read(paired/'RESULT.json')
model=native._model(cache['meta'],cache['child_contract'],torch.device('cpu'));model.requires_grad_(False).eval()
captures={};active=[];arms={}
def hook(module,args,output):
    require(not active,'Unexpected duplicate fuse forward');active.append(output.detach().cpu().double().numpy().copy())
handle=model.exit_random_access_fuse.register_forward_hook(hook)
for arm in ['BEFORE95','AFTER96']:
    binding=proven['checkpoint_bindings'][arm];path=Path(binding['path'])
    require(file_sha256(path)==binding['sha256'],'Checkpoint mismatch')
    state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    require(strict_load_random_access_v2_state(model,state['model_state'])==proven['model_shas'][arm],'Model state mismatch')
    native.bind_preserved_v7_input_normalization(model,cache['old_norm']);model.eval();active.clear()
    with torch.inference_mode():
        e=trainer._model_forward_fp32(model,**cache['entry_inputs'],**cache['entry_kwargs'])
        b=cache['batch'];kwargs=dict(b['online_model_inputs'])
        kwargs.update(entry_decision_representation=e[native.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY].index_select(0,b['online_entry_batch_index']),action_valid_mask=b['online_action_valid_mask'],liquidation_relative_values=True)
        q=model.forward_exit_random_access_batch(**kwargs)['exit_action_q_bps']
    for name,value in [('entry_q_bps',e['entry_action_q_bps']),('forecast_bps',e['forecast_pred']),('exit_q_bps',q)]:
        require(np.array_equal(value.numpy(),np.asarray(local['arms'][arm][name])),'Cached output mismatch '+arm+' '+name)
    require(len(active)==1 and active[0].shape==(64,2,128),'Unexpected hidden shape')
    weight=model.head_exit_action.weight.detach().double().numpy();bias=model.head_exit_action.bias.detach().double().numpy()
    arms[arm]={'h':active[0],'w':weight[0]-weight[1],'b':float(bias[0]-bias[1]),'q':q.numpy().copy()[:,:,0]}
    require(canonical_model_state_sha256(model.state_dict())==proven['model_shas'][arm],'Model state changed')
    del state
handle.remove()
a=arms['BEFORE95'];b=arms['AFTER96']
head=(b['h']+a['h'])/2 @ (b['w']-a['w'])
representation=(b['h']-a['h']) @ ((b['w']+a['w'])/2)
bias_delta=b['b']-a['b'];delta=b['q'].astype(float)-a['q'].astype(float)
residual=delta-head-representation-bias_delta
def stats(x):return {'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),'max':float(x.max())}
result={'source_commit':source,'cached_all_outputs_exact':True,'entry_forward_count':2,'exit_forward_count':2,'teacher_forward_count':0,
    'algebra':'dQ = dw*mean(h) + mean(w)*dh + db; symmetric between checkpoints, fp64 analysis of captured FP32 tensors.',
    'head_bias_difference_before':a['b'],'head_bias_difference_after':b['b'],'head_bias_delta_shared_both_sides':bias_delta,
    'maximum_float32_output_reconstruction_residual':float(abs(residual).max()),'sides':{},'checkpoint_bindings':proven['checkpoint_bindings'],
    'limits':'Exact attribution of this stored batch at final linear layer; not a causal attribution to individual upstream modules, Adam momentum, all32 gradients or market edge.',
    'test_data_used':False,'gpu_used':False,'optimizer_or_backward':False}
for s,name in enumerate(['LONG','SHORT']):
    result['sides'][name]={'total_change':stats(delta[:,s]),'head_weight_term':stats(head[:,s]),'hidden_representation_term':stats(representation[:,s]),
        'head_including_bias_mean':float(head[:,s].mean()+bias_delta),'hidden_mean_fraction_of_net_change':float(representation[:,s].mean()/delta[:,s].mean()),
        'before_hidden_across_states_centered_energy_fraction':float(np.mean(np.var(a['h'][:,s],axis=0))/np.mean(a['h'][:,s]**2)),
        'after_hidden_across_states_centered_energy_fraction':float(np.mean(np.var(b['h'][:,s],axis=0))/np.mean(b['h'][:,s]**2))}
require(not torch.cuda.is_initialized() and not _native_processes(repo) and not git('status','--porcelain') and git('rev-parse','HEAD')==source,'Unexpected mutation/process')
result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,operator_sha256=file_sha256(Path(__file__)))
write('ATTRIBUTION_RESULT.json',result);signal.alarm(0)
print(json.dumps(result,indent=2))
