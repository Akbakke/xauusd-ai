"""Source-bound GC context/temporal ablation for a frozen direction model.

The intervention is an encoder-domain missingness counterfactual, not a claim
about a realizable market state. It preserves the original input tensors.
"""
from __future__ import annotations

import hashlib
import json


def require(ok, reason):
    if not bool(ok):
        raise RuntimeError('DIRECTION_GC_CONTEXT_INTERVENTION_' + reason)


def compile_gc_context_layout(profile):
    from gx1.features import gc_native_context_v1 as gc, micro_structure_v1 as micro
    from gx1.features.entry_specialist_feature_groups_v1 import model_native_context_temporal_alias_policy

    names = profile['physical_field_names']
    fields = tuple(profile['gc_context_fields'])
    require(len(fields) == len(set(fields)) == 2410 and all(f.startswith('gc_') for f in fields), 'GC_OWNER_FIELDS')
    owned = set(fields)
    aliases = model_native_context_temporal_alias_policy(names['signal'])['aliases']
    gc_aliases = [x for x in aliases if x['ctx_cont_field'] in owned]
    require(len(gc_aliases) == len(owned)
        and {x['ctx_cont_field'] for x in gc_aliases} == owned, 'ALIAS_COVERAGE')
    signal = tuple(x['signal_index'] for x in gc_aliases)
    context = tuple(x['ctx_cont_index'] for x in gc_aliases)
    require(len(set(signal)) == len(set(context)) == len(fields)
        and set(signal) <= set(profile['signal_indices'])
        and set(context) <= set(profile['ctx_cont_indices']), 'ACTIVE_GC_COORDINATES')
    for surface in ('signal', 'ctx_cont'):
        selected = set(signal if surface == 'signal' else context)
        surface_names = names[surface]
        pairs = tuple(gc.masking_pairs(surface_names)) + tuple(micro.range_observation_masking_pairs(surface_names))
        gc_pairs = [(value, mask) for value, mask in pairs if value in selected or mask in selected]
        require(all(value in selected and mask in selected for value, mask in gc_pairs), 'CROSS_OWNER_MASK:' + surface)
        values = {value for value, _ in gc_pairs}; masks = {mask for _, mask in gc_pairs}
        require(len(gc_pairs) == len(values) == len(masks) == 1205
            and not (values & masks) and values | masks == selected, 'GC_VALUE_MASK_CLOSURE:' + surface)
        require({i for i, name in enumerate(surface_names) if str(name).removeprefix('ctx_cont.').startswith('gc_')} == selected,
            'GC_NAME_COVERAGE:' + surface)
    require(not any('gc_' in str(name).lower() for surface in ('ctx_cat', 'mtf') for name in names[surface]),
        'OTHER_NAMED_GC_SURFACE')
    ordered = [(x['signal_field'], x['ctx_cont_field'], x['signal_index'], x['ctx_cont_index']) for x in gc_aliases]
    identity = hashlib.sha256(json.dumps(ordered, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    return {'signal_indices': signal, 'ctx_cont_indices': context, 'gc_fields': len(fields),
        'value_mask_pairs': 1205, 'aliases_sha256': identity}


def drop_gc_context(features, layout):
    import torch

    signal, context = layout['signal_indices'], layout['ctx_cont_indices']
    require(len(signal) == len(context) > 0 and len(set(signal)) == len(set(context)) == len(signal), 'LAYOUT')
    seq, snap, ctx = (features[key] for key in ('seq_x', 'snap_x', 'ctx_cont'))
    require(seq.ndim == 3 and snap.ndim == ctx.ndim == 2 and seq.shape[0] == snap.shape[0] == ctx.shape[0]
        and seq.shape[2] == snap.shape[1] and max(signal) < snap.shape[1] and min(signal) >= 0
        and max(context) < ctx.shape[1] and min(context) >= 0, 'SHAPES')
    sig = torch.tensor(signal, dtype=torch.long, device=seq.device)
    con = torch.tensor(context, dtype=torch.long, device=ctx.device)
    require(seq.device == snap.device == ctx.device and torch.equal(seq[:, -1, :], snap)
        and torch.equal(snap.index_select(1, sig), ctx.index_select(1, con)), 'ORIGINAL_ALIAS_PARITY')
    result = dict(features)
    for name, indices in (('seq_x', sig), ('snap_x', sig), ('ctx_cont', con)):
        result[name] = features[name].clone().index_fill_(-1, indices, 0)
    require(torch.equal(result['seq_x'][:, -1, :], result['snap_x'])
        and torch.equal(result['snap_x'].index_select(1, sig), result['ctx_cont'].index_select(1, con)),
        'ABLATED_ALIAS_PARITY')
    return result
