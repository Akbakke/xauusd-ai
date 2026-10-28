"""Replace only current GC context and its last-row aliases with prior values.

This is an input intervention on a frozen model, not a new prediction method.
Caller must bind previous decision clocks and source arrays before use.
"""
from __future__ import annotations

import torch
from gx1.audit.direction_gc_context_intervention_v1 import require


def replace_current_gc_context(features, prior_snap, prior_ctx, layout):
    signal, context = layout['signal_indices'], layout['ctx_cont_indices']
    require(len(signal) == len(context) > 0 and len(set(signal)) == len(set(context)) == len(signal), 'STALE_LAYOUT')
    seq, snap, ctx = (features[key] for key in ('seq_x', 'snap_x', 'ctx_cont'))
    require(seq.ndim == 3 and snap.ndim == ctx.ndim == 2 and seq.shape[0] == snap.shape[0] == ctx.shape[0]
        and seq.shape[2] == snap.shape[1] and prior_snap.shape == snap.shape and prior_ctx.shape == ctx.shape
        and max(signal) < snap.shape[1] and min(signal) >= 0 and max(context) < ctx.shape[1]
        and min(context) >= 0 and seq.device == snap.device == ctx.device == prior_snap.device == prior_ctx.device,
        'STALE_SHAPES')
    sig = torch.tensor(signal, dtype=torch.long, device=seq.device)
    con = torch.tensor(context, dtype=torch.long, device=seq.device)
    require(torch.equal(seq[:, -1, :], snap)
        and torch.equal(snap.index_select(1, sig), ctx.index_select(1, con))
        and torch.equal(prior_snap.index_select(1, sig), prior_ctx.index_select(1, con)), 'STALE_ORIGINAL_ALIAS_PARITY')
    result = dict(features)
    for name in ('seq_x', 'snap_x', 'ctx_cont'):
        result[name] = features[name].clone()
    prior_values = prior_ctx.index_select(1, con)
    result['seq_x'][:, -1, sig] = prior_values
    result['snap_x'][:, sig] = prior_values
    result['ctx_cont'][:, con] = prior_values
    require(torch.equal(result['seq_x'][:, -1, :], result['snap_x'])
        and torch.equal(result['snap_x'].index_select(1, sig), result['ctx_cont'].index_select(1, con)),
        'STALE_RESULT_ALIAS_PARITY')
    return result
