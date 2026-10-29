"""TRAIN-only analytic head-gradient diagnostic from a bound saved M5 fit.

No native model forward, autograd, optimizer, training, or market-input access.
The vectors are derivatives with respect to the post-backbone hidden activations,
not parameter gradients and not gradients recorded during the historical fit.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import numpy as np
import torch

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
OUT = ROOT / 'DIRECTION_M5_HEAD_GRADIENT_20261029_002'
STAGE = 'independent_m5_head_gradient_20261029_002'
BATCH = 16
HORIZONS = tuple(range(1, 61))
ANCHORS = (1, 5, 15)


def require(value, why):
    if not bool(value):
        raise RuntimeError('M5_HEAD_GRADIENT_' + why)


def reordered_weights(anchor, aux):
    """Return rows matching dense direction logits, with exact (1,5,15) anchors."""
    anchor = np.asarray(anchor, dtype=np.float64)
    aux = np.asarray(aux, dtype=np.float64)
    require(anchor.ndim == aux.ndim == 2 and anchor.shape[0] == 3
            and aux.shape == (57, anchor.shape[1]) and anchor.shape[1] > 0
            and np.isfinite(anchor).all() and np.isfinite(aux).all(), 'HEADS')
    auxiliary = tuple(h for h in HORIZONS if h not in ANCHORS)
    order = [ANCHORS.index(h) if h in ANCHORS else len(ANCHORS) + auxiliary.index(h)
             for h in HORIZONS]
    return np.concatenate((anchor, aux), axis=0)[order]


def gradients(logits, sign, observed, terminal, decisions, cutoff, weights, *, batch_size=BATCH):
    """Per-batch eligible BCE derivatives at frozen joint hidden representation."""
    z = np.asarray(logits, dtype=np.float64)
    sign = np.asarray(sign); observed = np.asarray(observed)
    terminal = np.asarray(terminal); decisions = np.asarray(decisions)
    weights = np.asarray(weights, dtype=np.float64)
    require(z.ndim == 2 and z.shape[1] == 60 and z.shape[0] > 0
            and sign.shape == observed.shape == terminal.shape == z.shape
            and sign.dtype == np.int8 and observed.dtype == np.bool_
            and terminal.dtype == decisions.dtype == np.int64
            and decisions.shape == (len(z),) and weights.shape[0] == 60
            and weights.ndim == 2 and np.isfinite(z).all() and np.isfinite(weights).all()
            and np.isin(sign, (-1, 0, 1)).all() and (sign[~observed] == 0).all()
            and type(cutoff) is int and type(batch_size) is int and batch_size > 0, 'INPUT')
    expected = decisions[:, None] + np.arange(1, 61, dtype=np.int64)[None, :] * 60_000_000_000
    require(np.array_equal(terminal, expected) and np.all(decisions < cutoff), 'CLOCK')
    eligible = observed & (sign != 0) & (terminal < cutoff)
    # Stable sigmoid, unlike the potentially overflowing 1/(1+exp(-z)).
    prob = np.empty_like(z); positive = z >= 0
    prob[positive] = 1 / (1 + np.exp(-z[positive]))
    expz = np.exp(z[~positive]); prob[~positive] = expz / (1 + expz)
    residual = np.where(eligible, prob - (sign == 1), 0.0)
    total5 = total_other = total_dot = total_combined = 0.0
    result_batches = []
    for start in range(0, len(z), batch_size):
        stop = min(start + batch_size, len(z)); n = int(eligible[start:stop].sum())
        require(n > 0, 'EMPTY_BATCH')
        g5 = residual[start:stop, 4:5] * weights[4:5] / n
        go = residual[start:stop] @ weights / n - g5
        sq5 = float(np.sum(g5*g5)); sqo = float(np.sum(go*go)); dot = float(np.sum(g5*go))
        combined = float(np.sum((g5+go)*(g5+go)))
        total5 += sq5; total_other += sqo; total_dot += dot; total_combined += combined
        cosine = float(dot / np.sqrt(sq5*sqo)) if sq5 > 0 and sqo > 0 else None
        result_batches.append({'parents': stop-start, 'eligible_cells': n,
                               'eligible_five_minute_cells': int(eligible[start:stop,4].sum()),
                               'cosine_five_vs_other': cosine})
    cosine = float(total_dot / np.sqrt(total5*total_other)) if total5 > 0 and total_other > 0 else None
    return {'parents': len(z), 'batch_size': batch_size, 'batch_count': len(result_batches),
            'eligible_cells': int(eligible.sum()), 'eligible_five_minute_cells': int(eligible[:,4].sum()),
            'eligible_five_minute_cell_fraction': float(eligible[:,4].sum()/eligible.sum()),
            'five_norm': float(np.sqrt(total5)), 'other_norm': float(np.sqrt(total_other)),
            'combined_norm': float(np.sqrt(total_combined)),
            'five_norm_div_other_norm': float(np.sqrt(total5/total_other)) if total_other > 0 else None,
            'five_norm_div_combined_norm': float(np.sqrt(total5/total_combined)) if total_combined > 0 else None,
            'five_dot_other': total_dot, 'five_vs_other_cosine': cosine,
            'batches_negative_cosine': int(sum(x['cosine_five_vs_other'] is not None and x['cosine_five_vs_other'] < 0 for x in result_batches)),
            'batches_positive_cosine': int(sum(x['cosine_five_vs_other'] is not None and x['cosine_five_vs_other'] > 0 for x in result_batches)),
            'batches_zero_or_undefined_cosine': int(sum(x['cosine_five_vs_other'] is None or x['cosine_five_vs_other'] == 0 for x in result_batches)),
            'batch_details': result_batches}


def digest(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path, 'REGULAR')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def bound(ref):
    path = Path(ref['path'])
    require(digest(path) == ref['sha256'], 'SHA:' + str(path))
    return path


def publish(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, 'wb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def main():
    os.chdir(REPO); sys.path.insert(0, str(REPO)); sys.dont_write_bytecode = True
    from gx1.contracts import gx1_capped_execution_v1 as cap
    proof = cap.require_capped_cpu_audit_execution(); lock = cap.require_capped_lock_ancestry()
    require(lock['scope'] == proof['cgroup_path'][len('/sys/fs/cgroup'):], 'LOCK')
    require(subprocess.check_output(['git','branch','--show-current'], text=True).strip() == 'work/gx1-current'
            and not subprocess.check_output(['git','status','--porcelain'], text=True), 'SOURCE')
    head = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    policy_path = REPO/'NEXT_RUN_POLICY.json'; policy_sha = digest(policy_path)
    policy = json.loads(policy_path.read_text()); gate = policy[STAGE]
    require(gate['status'] == 'ADMITTED' and gate['execution_authorized'] is True
            and gate['checkpoint_reads_authorized'] is True and gate['train_prediction_reads_authorized'] is True
            and gate['max_invocations'] == 1 and gate['claimed_invocations'] == gate['completed_invocations'] == 0
            and gate['relaunch_allowed'] is False and gate['test_data_used'] is False, 'GATE')
    require(all(policy[k] is False for k in ('training_enabled','native_training_enabled','full_epoch_training_allowed','native_launch_authorized','full_val_allowed'))
            and all(gate[k] is False for k in ('model_forwards_authorized','optimizer_steps_authorized','broker_access_allowed')), 'CLOSED_GATES')
    require(bound(gate['worker']) == Path(__file__).resolve() and Path(gate['output']) == OUT
            and not OUT.exists() and not OUT.is_symlink(), 'WORKER')
    design = json.loads(bound(gate['design']).read_text())
    require(design['schema_version'] == 'gx1_m5_head_gradient_design_v1'
            and design['status'] == 'PREDECLARED' and design['train_parents'] == 256
            and design['batch_size'] == BATCH and design['primary_horizon_minutes'] == 5
            and design['other_horizons_minutes'] == [h for h in HORIZONS if h != 5]
            and design['test_data_used'] is False and design['optimizer_steps'] == 0, 'DESIGN')
    refs = gate['inputs']
    require(design['inputs'] == refs
            and set(refs) == {'checkpoint','train_report','train_predictions','fit_result','fit_recipe','loss_source','head_source'}, 'INPUT_SET')
    for ref in refs.values(): bound(ref)
    fit = json.loads(Path(refs['fit_result']['path']).read_text())
    recipe = json.loads(Path(refs['fit_recipe']['path']).read_text())
    require(fit['recipe'] == refs['fit_recipe']
            and recipe['source_owners']['gx1/contracts/entry_native_direction_dense_v1.py'] == refs['loss_source']
            and recipe['source_owners']['gx1/contracts/entry_native_direction_research_v1.py'] == refs['head_source'], 'FIT_SOURCE')
    train = json.loads(Path(refs['train_report']['path']).read_text())
    require(fit['outcome'] == 'COMPLETE' and fit['source_unchanged'] is True and fit['test_data_used'] is False
            and fit['full_state'] == refs['checkpoint'] and fit['final_train_measurement'] == refs['train_report']
            and train['role'] == 'FINAL_TRAIN' and train['test_data_used'] is False
            and train['predictions'] == refs['train_predictions'] and fit['fold'] == train['fold'], 'LINEAGE')
    require(not torch.cuda.is_initialized(), 'CUDA_INITIALIZED'); torch.set_num_threads(1)
    OUT.mkdir(mode=0o700); publish(OUT/'CLAIM.json', {'inputs':refs,'policy_sha256':policy_sha,'source_head':head,'test_data_used':False,'relaunch_allowed':False})
    def audit(event, args):
        require(event not in ('os.system','socket.connect','socket.bind','socket.sendto','socket.getaddrinfo'), 'FORBIDDEN_EVENT')
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(args[0])
            require('/TEST' not in path.upper() or path.startswith(str(REPO/'.venv')+'/'), 'SEALED_TEST')
            if path.startswith('/home/andre2/GX1_DATA/') or path.startswith('/home/andre2/GX1_RUNS/'):
                require(path in {ref['path'] for ref in refs.values()} or path.startswith(str(OUT)+'/'), 'UNBOUND_READ:'+path)
    sys.addaudithook(audit)
    try:
        state = torch.load(refs['checkpoint']['path'], map_location='cpu', weights_only=True)
        require(state['optimizer_updates'] == 2976 and state['cursor'] == 47602
                and state['contract_sha256'] == fit['durable_checkpoint']['contract_sha256'], 'CHECKPOINT')
        model = state['model_state']
        require('direction_head.weight' in model and 'dense_aux_head.weight' in model
                and 'dense_reorder' in model, 'EXACT_HEAD_KEYS')
        weights = reordered_weights(model['direction_head.weight'].detach().cpu().numpy(), model['dense_aux_head.weight'].detach().cpu().numpy())
        expected_order = [ANCHORS.index(h) if h in ANCHORS else len(ANCHORS) + tuple(x for x in HORIZONS if x not in ANCHORS).index(h) for h in HORIZONS]
        require(model['dense_reorder'].tolist() == expected_order, 'EXACT_DENSE_REORDER')
        with np.load(refs['train_predictions']['path'], allow_pickle=False) as p:
            arrays = {k: p[k] for k in ('logit_up','sign','observed','terminal_time_ns','decision_ns','parent_rows')}
        require(arrays['logit_up'].shape == (256,60) and arrays['parent_rows'].shape == (256,)
                and len(np.unique(arrays['parent_rows'])) == 256, 'EXACT_TRAIN_SAMPLE')
        cutoff = int(fit['fold']['train_cutoff_ns'])
        result = gradients(arrays['logit_up'], arrays['sign'], arrays['observed'], arrays['terminal_time_ns'], arrays['decision_ns'], cutoff, weights)
        for ref in refs.values(): bound(ref)
        require(digest(policy_path) == policy_sha and not subprocess.check_output(['git','status','--porcelain'], text=True), 'FINAL_SOURCE')
        result = {'schema_version':'gx1_m5_head_gradient_diagnostic_v1', 'status':'COMPLETE', 'source_head':head,
                  'inputs':refs, 'scope':'Analytical BCE gradients at final saved TRAIN evaluation logits and frozen fitted direction heads; 256 diagnostic TRAIN parents in their saved 16-row order.',
                  'gradient_target':'joint hidden activations, per saved row and normalized by each corresponding 16-row batch eligible cell count',
                  'metrics':result, 'model_forwards':0, 'optimizer_steps':0, 'test_data_used':False,
                  'limitations':['Not gradients recorded during optimizer steps: saved predictions are evaluation-mode after the final update.',
                                 'Derivatives of hidden activations are not model parameter gradients; upstream Jacobians can change magnitude and alignment.',
                                 'These 256 saved diagnostic parents are not a representative estimate of the full TRAIN history.',
                                 'No evidence of predictive or economic edge, and no learned effect of changing loss weights.']}
        publish(OUT/'RESULT.json',result); publish(OUT/'TERMINAL.json',{'status':'COMPLETE','result_sha256':digest(OUT/'RESULT.json'),'test_data_used':False})
        print('M5_HEAD_GRADIENT_COMPLETE',json.dumps({'result_sha256':digest(OUT/'RESULT.json'),'metrics':result['metrics']}))
    except Exception as exc:
        publish(OUT/'FAILED_TERMINAL.json',{'status':'FAILED','error':type(exc).__name__+': '+str(exc),'relaunch_allowed':False,'test_data_used':False})
        raise


if __name__ == '__main__': main()
