"""Authenticate and structurally join a separately qualified decimal target.

This data-only owner grants no file, model, optimizer or trading authority.
The caller must bind these exact references in its own admitted recipe, enforce
its data scope and freeze the verified files for the entire execution. The
legacy training scope and its anchor checks remain unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from gx1.contracts import local_random_access_campaign_v2 as binding
from gx1.contracts.entry_native_direction_decimal_data_v1 import (
    DECIMAL_TARGET_IDENTITY, bind_decimal_grid_dense_targets,
)
from gx1.contracts.entry_native_direction_dense_data_v1 import AXES, TARGETS, DENSE_MEMBERS

REF_KEYS = frozenset(('legacy', 'original', 'original_result', 'corrected', 'corrected_result',
    'consumer_plan', 'consumer_result', 'materialization_plan', 'audit_result'))


def require(ok, reason):
    if not bool(ok):
        raise RuntimeError('DECIMAL_DIRECTION_INPUT_' + reason)


def _ref_from_output(value):
    require(isinstance(value, Mapping) and {'path', 'sha256'} <= set(value), 'OUTPUT_REFERENCE')
    return {key: value[key] for key in ('path', 'sha256')}


def _load_exact_archive(ref, names, *, exact):
    with np.load(ref['path'], allow_pickle=False) as archive:
        require(set(names) <= set(archive.files) and (not exact or set(archive.files) == set(names))
            and len(archive.files) == len(set(archive.files)), 'ARCHIVE_MEMBERS')
        return {name: archive[name].copy() for name in names}


def load_bound_decimal_direction_targets(*, refs, axes, legacy):
    """Verify known receipt links and return corrected, immutable target arrays.

    Hash binding establishes byte identity, not authority or truth of arbitrary
    caller-selected receipts. Caller admission must pin independently reviewed
    receipt references and retain input freeze and chronological fold gates.
    """
    require(isinstance(refs, Mapping) and set(refs) == REF_KEYS, 'REFERENCE_SET')
    with binding.fresh_file_verification_scope():
        verified = {key: binding.require_binding(refs[key], label='decimal ' + key, verify_file=True)
            for key in REF_KEYS}
        require(len({value['path'] for value in verified.values()}) == len(verified), 'DISTINCT_REFERENCES')
        metadata = {key: binding.read_bound_json(Path(verified[key]['path']), verified[key]['sha256'])
            for key in ('original_result', 'corrected_result', 'consumer_plan', 'consumer_result',
                'materialization_plan', 'audit_result')}
        old = metadata['original_result']; corrected = metadata['corrected_result']; consumer = metadata['consumer_result']
        consumer_plan = metadata['consumer_plan']; materialization_plan = metadata['materialization_plan']; audit = metadata['audit_result']
        require(old.get('schema_version') == 'gx1_direction_dense_target_result_v1'
            and old.get('status') == 'COMPLETE_REAL_TARGETS_ONLY' and old.get('test_data_used') is False
            and _ref_from_output(old.get('target')) == verified['original']
            and old.get('legacy_anchor_byte_parity') == {'1': True, '5': True, '15': True}
            and old.get('horizons_minutes') == list(range(1, 61)), 'ORIGINAL_EVIDENCE')
        require(corrected.get('schema_version') == 'gx1_direction_decimal_targets_result_v1'
            and corrected.get('status') == 'COMPLETE' and corrected.get('test_data_used') is False
            and corrected.get('new_model_forwards') == corrected.get('optimizer_steps') == 0
            and corrected.get('target_identity') == DECIMAL_TARGET_IDENTITY
            and corrected.get('target') == verified['corrected'] and corrected.get('plan') == verified['materialization_plan']
            and corrected.get('source_clock_bid_ask_and_decimal_grid_parity') is True
            and corrected.get('independent_audit_confusions_reproduced') is True
            and corrected.get('nonzero_sign_flips') == 0 and corrected.get('horizons_minutes') == list(range(1, 61)),
            'CORRECTED_EVIDENCE')
        require(audit.get('schema_version') == 'gx1_direction_decimal_sign_audit_result_v1'
            and audit.get('status') == 'COMPLETE' and audit.get('test_data_used') is False
            and audit.get('new_model_forwards') == audit.get('optimizer_steps') == 0
            and audit.get('target_source_rows_and_masks_parity') is True
            and audit.get('vendor_to_canonical_float64_and_clock_parity') is True, 'AUDIT_EVIDENCE')
        require(materialization_plan.get('schema_version') == 'gx1_direction_decimal_targets_plan_v1'
            and materialization_plan.get('test_data_used') is False
            and materialization_plan.get('inputs', {}).get('dense_targets') == verified['original']
            and materialization_plan.get('inputs', {}).get('legacy_targets') == verified['legacy']
            and materialization_plan.get('inputs', {}).get('audit_result') == verified['audit_result'], 'MATERIALIZATION_LINKS')
        require(consumer_plan.get('schema_version') == 'gx1_direction_decimal_consumer_plan_v1'
            and consumer_plan.get('test_data_used') is False and all(consumer_plan.get('inputs', {}).get(key) == verified[key]
                for key in ('legacy', 'original', 'corrected', 'audit_result', 'materialization_plan'))
            and consumer_plan.get('inputs', {}).get('materialization_result') == verified['corrected_result'], 'CONSUMER_PLAN_LINKS')
        require(consumer.get('schema_version') == 'gx1_direction_decimal_consumer_result_v1'
            and consumer.get('status') == 'COMPLETE' and consumer.get('test_data_used') is False
            and consumer.get('new_model_forwards') == consumer.get('optimizer_steps') == 0
            and consumer.get('plan') == verified['consumer_plan'] and consumer.get('target_identity') == DECIMAL_TARGET_IDENTITY
            and consumer.get('corrected_axes_clocks_rows_masks_and_domain_verified') is True
            and consumer.get('legacy_anchor_bytes_verified_on_original_only') is True
            and consumer.get('independent_60_horizon_confusion_matrices_reproduced') is True
            and consumer.get('nonzero_sign_flips') == corrected.get('nonzero_sign_flips') == 0
            and consumer.get('all_false_moves_removed') == corrected.get('all_false_moves_removed')
            and consumer.get('horizons_minutes') == list(range(1, 61)), 'CONSUMER_EVIDENCE')
        require(isinstance(axes, Mapping) and isinstance(legacy, Mapping) and all(k in legacy for k in AXES + TARGETS), 'CALLER_INPUTS')
        legacy_disk = _load_exact_archive(verified['legacy'], AXES + TARGETS, exact=False)
        for name, disk in legacy_disk.items():
            memory = legacy[name]
            require(isinstance(memory, np.ndarray) and disk.dtype == memory.dtype and disk.shape == memory.shape
                and np.ascontiguousarray(disk).tobytes() == np.ascontiguousarray(memory).tobytes(), 'LEGACY_MEMORY:' + name)
        original = _load_exact_archive(verified['original'], DENSE_MEMBERS, exact=True)
        new = _load_exact_archive(verified['corrected'], DENSE_MEMBERS, exact=True)
        require(old.get('parents') == corrected.get('parent_rows') == consumer.get('parents') == len(axes['parent_rows']), 'PARENT_COUNT')
        result = bind_decimal_grid_dense_targets(axes=axes, legacy=legacy, original_dense=original,
            corrected_dense=new, target_identity=DECIMAL_TARGET_IDENTITY)
        snapshot = binding.capture_verified_file_snapshot(); snapshot.require_unchanged()
        return result
