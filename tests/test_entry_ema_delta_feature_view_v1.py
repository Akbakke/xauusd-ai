"""Genuine artifact identity and fail-closed metadata routing for 318/256."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import numpy as np

from gx1.contracts import entry_ema_delta_feature_view_v1 as ema
from gx1.contracts import entry_anchor_span_feature_inputs_v1 as overlay
from gx1.contracts import entry_native_direction_feature_view_v1 as views


def _read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture(scope='module')
def bundle():
    fit = Path(ema.NORMALIZATION_ROOT)
    materialized = Path(ema.MATERIALIZER_ROOT)
    old_view = _read('/home/andre2/GX1_RUNS/TA_M5_COLD_FULL_LATER_INPUTS_20261117_001/FEATURE_VIEW.json')
    sources = copy.deepcopy(old_view['source_bindings'])
    for key, ref in sources.items():
        source = Path(ref['path'])
        if str(source).startswith('/home/andre2/src/GX1_CURRENT/') and source.is_file():
            sources[key] = {'path': str(source),
                            'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    sources['ema_delta_inputs_source'] = dict(sources['anchor_span_inputs_source'])
    return ema.build_ema_delta_metadata_bundle(
        parent_packet=_read(ema.PARENT_PACKET_REF['path']),
        normalization_result=_read(fit / 'RESULT.json'),
        normalization_terminal=_read(fit / 'TERMINAL.json'),
        materialized_result=_read(materialized / 'OUTPUT/RESULT.json'),
        materialized_terminal=_read(materialized / 'TERMINAL.json'),
        base_normalization=_read(fit / 'EXTENDED_BASE_NORMALIZATION.json'),
        ta_normalization=_read(fit / 'TA_INPUT_NORMALIZATION.json'),
        unobserved_units_profile=_read(fit / 'NEW_UNITS_PROFILE.json'),
        parent_constructor_metadata=_read(ema.anchor.PARENT_CONSTRUCTOR_REF['path']),
        source_bindings=sources)


def test_genuine_bundle_and_generic_dispatch(bundle):
    packet = bundle['prefix_packet']
    view = bundle['feature_view']
    assert packet['contract_sha256'] == 'a864cb9501019e99fdaa88dc13b068c55048c1cff5ae6d6a3e959ebeb1cc701b'
    assert bundle['artifact_refs']['PREFIX_PACKET.json'] == ema.PACKET_REF
    assert bundle['artifact_refs']['CONSTRUCTOR_METADATA.json'] == ema.CONSTRUCTOR_REF
    assert view['physical_widths']['signal'] == 318
    assert view['physical_widths']['mtf'] == 256
    assert ema.require_ema_delta_prefix_packet(packet) == packet
    rebound = views.bind_native_direction_feature_view(
        prefix_normalization=packet,
        prefix_normalization_ref=ema.PACKET_REF,
        constructor_metadata=bundle['constructor_metadata'],
        constructor_metadata_ref=ema.CONSTRUCTOR_REF,
        source_bindings=view['source_bindings'],
        max_batch_rows=16,
        unobserved_units_profile=packet['unobserved_units_profile'],
        full_feature_profile=True, market_state_time_query_v2=True,
        full_ta_feature_profile=True, dedicated_m5_side=True,
        dedicated_non_m5_side=True, dedicated_tf_activity=True,
        direct_tf_hidden=True)
    assert rebound == view


def test_marker_joins_genuine_overlay_payloads(bundle):
    view = bundle['feature_view']
    result = _read(ema.MATERIALIZED_RESULT_REF['path'])
    manifest_ref = result['manifest']
    manifest = _read(manifest_ref['path'])
    payloads, spans = {}, {}
    for lane, item in manifest['surfaces'].items():
        with np.load(item['artifact']['path'], allow_pickle=False) as archive:
            payloads[lane] = {key: archive[key] for key in archive.files}
        with np.load(item['span_artifact']['path'], allow_pickle=False) as archive:
            spans[lane] = {key: archive[key] for key in archive.files}
    selected = overlay._require_ema_delta_bundle(view, dict(
        result_ref=ema.MATERIALIZED_RESULT_REF, result=result,
        manifest_ref=manifest_ref, manifest=manifest, payloads=payloads),
        spans, None)
    assert set(selected) == set(ema.SIDECAR_REFS)
    assert all(len(selected[lane]) == ema.SIDECAR_ROWS[lane]
               and np.isfinite(selected[lane]).all() for lane in selected)


def test_tampered_physical_ref_fails_after_view_reseal(bundle):
    view = copy.deepcopy(bundle['feature_view'])
    view['ema_delta_input_extension']['sidecars']['M5']['sha256'] = '0' * 64
    view.pop('contract_sha256')
    view['contract_sha256'] = views._sha(view)
    with pytest.raises(RuntimeError, match='EMA_MARKER'):
        views._require_view(view, unobserved_units_profile=bundle['prefix_packet']['unobserved_units_profile'])


def test_tampered_parent_ref_fails_fixed_packet_identity(bundle):
    packet = copy.deepcopy(bundle['prefix_packet'])
    packet['parent_anchor_packet_reference']['sha256'] = '0' * 64
    packet.pop('contract_sha256')
    packet['contract_sha256'] = views._sha(packet)
    with pytest.raises(RuntimeError, match='PACKET_SEAL'):
        ema.require_ema_delta_prefix_packet(packet)
