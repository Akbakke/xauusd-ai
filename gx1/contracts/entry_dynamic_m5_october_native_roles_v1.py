"""Exact October native data roles; saved audit plans are source bindings only."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from gx1.contracts import entry_dynamic_m5_october_native_identity_v1 as identity
from gx1.contracts import local_random_access_campaign_v2 as campaign

REPO = Path('/home/andre2/src/GX1_CURRENT')
BATCH_PLAN = {'path': str(REPO / 'configs/research/DYNAMIC_M5_OCTOBER_NATIVE_BATCH_BINDING_20261106_002.json'),
              'sha256': '8287e4b16dcd7a1e6e82162b756b32a17c6030f2b9c9e86d865ac8cc5db8d750'}
REFERENCE_PLAN = {'path': str(REPO / 'configs/research/DYNAMIC_M5_OCTOBER_FIT_REFERENCES_20261106_001.json'),
                  'sha256': '7b8f6664bed2bceabde805c09dc4d5a54d7600999847f173fbe72c48beb973d0'}
PRIVATE_SCOPE = 'private_october_dynamic_mtf_state_qualification_v1'
FIT_SCOPE = 'fit_october_dynamic_mtf_state_onepass_v1'
SCORE_SCOPE = 'score_october_dynamic_mtf_state_once_v1'
SCOPE_ROLES = {PRIVATE_SCOPE: 'fit', FIT_SCOPE: 'fit', SCORE_SCOPE: 'score'}


def _require(ok, reason):
    if not ok:
        raise RuntimeError('OCTOBER_NATIVE_ROLE_' + reason)


def _read(ref):
    return campaign.read_bound_json(Path(ref['path']), ref['sha256'])


def require_role_binding(launch, *, read=None):
    """Bind one arm and one target role before any feature or target array is opened.

    This checks metadata identity only; it grants no campaign admission or model run.
    """
    _require(isinstance(launch, Mapping), 'LAUNCH_MAPPING')
    role = SCOPE_ROLES.get(launch.get('run_scope'))
    arm = launch.get('arm')
    _require(role is not None and arm in ('control', 'candidate')
             and launch.get('input_role') == role
             and launch.get('candidate_plan') == BATCH_PLAN
             and launch.get('reference_plan') == REFERENCE_PLAN, 'SCOPE')
    reader = _read if read is None else read
    _require(callable(reader), 'READER')
    batch, references = reader(BATCH_PLAN), reader(REFERENCE_PLAN)
    _require(batch.get('schema_version') == 'gx1_dynamic_m5_october_native_batch_binding_plan_v1'
             and references.get('schema_version') == 'gx1_dynamic_m5_october_fit_references_plan_v1'
             and batch['october_series'] == references['october_series']
             and batch['control_view'] == references['control_view']
             and batch['fit_prefix']['rows'] == 29849
             and batch['fit_prefix']['cutoff_ns'] == references['fit_cutoff_ns']
             and batch['evaluation_axis']['cutoff_ns'] == references['development_cutoff_ns']
             and batch['target_source_refs'] == references['target_source_refs'],
             'PLAN_JOIN')
    # The consumed plans bind inputs, never reauthorize their original executions.
    expected_view = batch[arm + '_view']
    expected_target = references['fit_series'] if role == 'fit' else references['october_series']
    _require(launch.get('feature_view') == expected_view
             and launch.get('target_manifest') == expected_target
             and launch.get('causal_inputs') == batch['causal_inputs']
             and launch.get('observer_axes') == batch['observer_axes'], 'EXACT_ROLE_REFS')
    view, target, axes = reader(expected_view), reader(expected_target), reader(batch['observer_axes'])
    bound_arm = identity.require_checkpoint_view({'contract_sha256': view['contract_sha256'],
        'prefix_normalization': view['prefix_normalization'],
        'research_objective_sha256': view['research_objective']['contract_sha256']})
    _require(bound_arm == arm and view['october_observer_axes'] == axes
             and view.get('confirmed_run_age_input', False) is (arm == 'candidate')
             and view['native_endpoint_loss'] is True
             and target['metadata']['parents'] == (29849 if role == 'fit' else 36172)
             and target['metadata']['cutoff_ns'] == (1727740800000000000 if role == 'fit' else 1730419200000000000)
             and target['metadata']['test_data_used'] is False
             and target['metadata']['source_refs'] == batch['target_source_refs'], 'ROLE_METADATA')
    return {'role': role, 'arm': arm, 'view': view, 'target': target,
            'axes': axes, 'batch': batch, 'references': references}
