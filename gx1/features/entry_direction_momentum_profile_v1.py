"""Opt-in compact MTF price momentum route; no data reads or training authority."""
from __future__ import annotations
from collections.abc import Mapping

from gx1.features import entry_specialist_feature_groups_v1 as owners

PROFILE_NAME = 'compact_mbp_gc_mtf_geometry_price_momentum_direction_research_v1'
LINEAR_INIT_PROFILE_NAME = 'compact_mbp_gc_mtf_geometry_price_momentum_linear_init_direction_research_v1'
PRICE_MOMENTUM_PROFILE_NAMES = (PROFILE_NAME, LINEAR_INIT_PROFILE_NAME)
MOMENTUM_FAMILY = 'momentum_flow_encoder'
PRICE_MOMENTUM_FIELDS = ('rsi14_centered', 'macd_line_atr', 'macd_hist_atr', 'ret_1')


def price_momentum_initialization(profile_name: str) -> str:
    """Resolve the explicit research profile without changing legacy defaults."""
    if type(profile_name) is not str or profile_name not in PRICE_MOMENTUM_PROFILE_NAMES:
        raise RuntimeError('PRICE_MOMENTUM_PROFILE_NAME_INVALID')
    return 'zero_residual' if profile_name == PROFILE_NAME else 'linear_default'


def compile_price_momentum_profile(
    gc_mtf_history: Mapping[str, object], *, initialization: str = 'zero_residual',
    unobserved_units_profile=None,
) -> dict[str, object]:
    """Add the genuine price route with a separately identified initializer."""
    if type(initialization) is not str or initialization not in ('zero_residual', 'linear_default'):
        raise RuntimeError('PRICE_MOMENTUM_INITIALIZATION_INVALID')
    parent = owners.compile_gc_mtf_geometry_entry_execution_profile(gc_mtf_history,
        **({} if unobserved_units_profile is None else {"unobserved_units_profile": unobserved_units_profile}))
    fields = parent['physical_field_names']
    mtf = fields['mtf']
    all_routes = owners.require_multi_tf_specialist_routing_v4(mtf)
    if len(set(PRICE_MOMENTUM_FIELDS)) != len(PRICE_MOMENTUM_FIELDS):
        raise RuntimeError('PRICE_MOMENTUM_PROFILE_DUPLICATE')
    indices = tuple(mtf.index(name) for name in PRICE_MOMENTUM_FIELDS)
    if (len(mtf) != 253 or len(parent['mtf_indices']) != 178
            or len(set(indices)) != 4 or not set(indices).issubset(all_routes[MOMENTUM_FAMILY])
            or set(indices).intersection(parent['mtf_indices'])
            or MOMENTUM_FAMILY in parent['base_mtf_specialists']):
        raise RuntimeError('PRICE_MOMENTUM_PROFILE_SOURCE_DRIFT')
    selected = set(parent['mtf_indices']) | set(indices)
    mtf_indices = tuple(i for i in range(len(mtf)) if i in selected)
    base_families = tuple(f for f in owners.MODEL_NATIVE_TRAINING_SPECIALISTS
        if f in set(parent['base_mtf_specialists']) | {MOMENTUM_FAMILY})
    routes = {f: tuple(i for i in all_routes[f] if i in selected) for f in base_families}
    if (len(mtf_indices) != 182 or any(not routes[f] for f in base_families)
            or any(routes[f] != parent['mtf_indices_by_specialist'][f]
                   for f in parent['base_mtf_specialists'])
            or set(routes[MOMENTUM_FAMILY]) != set(indices)):
        raise RuntimeError('PRICE_MOMENTUM_PROFILE_ROUTE_DRIFT')
    payload = {key:value for key,value in parent.items() if key != 'contract_sha256'}
    payload.update({'schema_version':'gx1_compact_gc_geometry_price_momentum_profile_v1',
        'name': PROFILE_NAME if initialization == 'zero_residual' else LINEAR_INIT_PROFILE_NAME,
        'research_parent_entry_execution_profile': {'name':parent['name'], 'contract_sha256':parent['contract_sha256']},
        'active_field_names': {**parent['active_field_names'], 'mtf':tuple(mtf[i] for i in mtf_indices)},
        'mtf_indices': mtf_indices, 'base_mtf_specialists':base_families,
        'mtf_indices_by_specialist':routes})
    if initialization == 'linear_default':
        payload['price_momentum_initialization'] = 'linear_default_rng_isolated_v1'
    return {**payload,'contract_sha256':owners._canonical_sha256(payload)}


def require_entry_execution_profile_reference(
    value, *, context='ENTRY_EXECUTION_PROFILE', gc_mtf_history=None, unobserved_units_profile=None,
):
    """Delegate old references unchanged; require the separate new hash exactly."""
    units_kwargs = ({} if unobserved_units_profile is None
                    else {"unobserved_units_profile": unobserved_units_profile})
    if isinstance(value, Mapping) and value.get('name') in PRICE_MOMENTUM_PROFILE_NAMES:
        if set(value) != {'name','contract_sha256'} or gc_mtf_history is None:
            raise RuntimeError(f'{context}_PRICE_MOMENTUM_PROFILE_REFERENCE_INVALID')
        profile = compile_price_momentum_profile(gc_mtf_history,
            initialization=price_momentum_initialization(value['name']), **units_kwargs)
        expected = {'name':profile['name'],'contract_sha256':profile['contract_sha256']}
        if any(type(value[k]) is not str or value[k] != expected[k] for k in expected):
            raise RuntimeError(f'{context}_PRICE_MOMENTUM_PROFILE_REFERENCE_MISMATCH')
        return expected
    return owners.require_compact_entry_execution_profile_reference(
        value, context=context, gc_mtf_history=gc_mtf_history, **units_kwargs)
