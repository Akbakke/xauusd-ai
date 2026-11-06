"""Exact checkpoint identities for the locked October control and age views.

This validates identity only. It grants no data, model, fit, score or launch access.
"""
from __future__ import annotations

from collections.abc import Mapping

from gx1.contracts import local_random_access_campaign_v2 as campaign


# The two SHA values bind the exact saved control/candidate feature-view identities.
IDENTITY_SHA256 = {
    'control': '4635f5861108415791f55fd092140c3d87f7126ff7eeebd1ae3be283a3e9942a',
    'candidate': 'e4e825f17c37a49de4b8ddea6a702306aa356c7a668269e1a600cd88f3e5915e',
}
PREFIX_KEYS = {'binding', 'contract_sha256', 'base_sha256', 'gc_layout_sha256',
               'gc_normalization_sha256', 'qualification_sha256',
               'unobserved_units_profile_sha256', 'october_observer_axes_sha256'}


def _require(ok, reason):
    if not ok:
        raise RuntimeError('OCTOBER_CHECKPOINT_IDENTITY_' + reason)


def require_checkpoint_view(identity):
    """Accept only the two previously bound October control/candidate identities."""
    _require(isinstance(identity, Mapping)
             and set(identity) == {'contract_sha256', 'prefix_normalization',
                                   'research_objective_sha256'}, 'KEYS')
    digest = campaign.canonical_sha256(identity)
    arm = next((name for name, expected in IDENTITY_SHA256.items() if digest == expected), None)
    _require(arm is not None, 'UNREGISTERED_VIEW')
    _require(isinstance(identity['prefix_normalization'], Mapping)
             and set(identity['prefix_normalization']) == PREFIX_KEYS, 'PREFIX_KEYS')
    return arm
