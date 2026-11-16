"""Reuse qualified TA span arrays when only the research decoder view changes."""
from __future__ import annotations

from dataclasses import replace

from gx1.contracts.entry_anchor_span_feature_inputs_v1 import QualifiedAnchorSpanInputs


_DIFF_KEYS = frozenset({'contract_sha256', 'research_objective',
                        'dedicated_non_m5_side', 'direct_tf_hidden'})


def rebind_decoder_only(old_view: dict, new_view: dict,
                        overlay: QualifiedAnchorSpanInputs) -> QualifiedAnchorSpanInputs:
    if not (type(old_view) is type(new_view) is dict
            and type(overlay) is QualifiedAnchorSpanInputs
            and overlay.view_sha256 == old_view.get('contract_sha256')
            and set(new_view) == set(old_view) | {'dedicated_non_m5_side', 'direct_tf_hidden'}
            and old_view.get('full_ta_feature_profile') is True
            and old_view.get('dedicated_m5_side') is True
            and old_view.get('dedicated_tf_activity') is True
            and 'dedicated_non_m5_side' not in old_view
            and 'direct_tf_hidden' not in old_view
            and new_view['dedicated_non_m5_side'] is True
            and new_view['direct_tf_hidden'] is True
            and all(old_view[key] == new_view[key]
                    for key in old_view if key not in _DIFF_KEYS)
            and new_view.get('contract_sha256') != old_view.get('contract_sha256')):
        raise ValueError('TA_DECODER_VIEW_REBIND_PHYSICAL_FIELDS')
    return replace(overlay, view_sha256=new_view['contract_sha256'])
