"""Qualified arrays may be rebound only across the two decoder-only flags."""
from dataclasses import MISSING, fields

import pytest

from gx1.contracts.entry_anchor_span_feature_inputs_v1 import QualifiedAnchorSpanInputs
from gx1.contracts.market_state_ta_decoder_view_rebind_v1 import rebind_decoder_only


def _fixture():
    old = {'contract_sha256': 'old', 'research_objective': {'mode': 'shared'},
           'full_ta_feature_profile': True, 'dedicated_m5_side': True,
           'dedicated_tf_activity': True, 'physical_widths': {'signal': 317}}
    new = {**old, 'contract_sha256': 'new',
           'research_objective': {'mode': 'separate'},
           'dedicated_non_m5_side': True, 'direct_tf_hidden': True}
    kwargs = {field.name: None for field in fields(QualifiedAnchorSpanInputs)
              if field.default is MISSING}
    kwargs['view_sha256'] = 'old'
    kwargs['later_coverage_sha256'] = 'qualified_later'
    return old, new, QualifiedAnchorSpanInputs(**kwargs)


def test_decoder_only_rebind_preserves_qualified_later_proof():
    old, new, overlay = _fixture()
    rebound = rebind_decoder_only(old, new, overlay)
    assert rebound.view_sha256 == 'new'
    assert rebound.later_coverage_sha256 == 'qualified_later'
    assert overlay.view_sha256 == 'old'


def test_physical_field_change_or_missing_flag_is_rejected():
    old, new, overlay = _fixture()
    new['physical_widths'] = {'signal': 318}
    with pytest.raises(ValueError, match='PHYSICAL_FIELDS'):
        rebind_decoder_only(old, new, overlay)
    new['physical_widths'] = old['physical_widths']
    new.pop('direct_tf_hidden')
    with pytest.raises(ValueError, match='PHYSICAL_FIELDS'):
        rebind_decoder_only(old, new, overlay)
