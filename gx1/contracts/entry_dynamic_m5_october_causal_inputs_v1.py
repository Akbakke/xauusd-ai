"""Read the locked October issue-time inputs without exposing future labels."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as state


_MANIFEST_KEYS = frozenset(('schema_version', 'candidate_design', 'target_source_series',
    'observer_axes', 'fit_parent_rows', 'development_parent_rows', 'cutoff_ns_exclusive',
    'clock_basis', 'future_labels_included', 'arrays', 'parent_rows_sha256', 'issue_ns_sha256'))
_NAMES = ('parent_rows', 'issue_ns', 'current_state', 'current_observed',
          'current_path_observed', 'current_confirmed_run_age_features')
_SHAPES = {'parent_rows': (36172,), 'issue_ns': (36172,),
           'current_state': (36172, 5), 'current_observed': (36172, 5),
           'current_path_observed': (36172, 5),
           'current_confirmed_run_age_features': (36172, 5, 2)}
_DTYPES = {'parent_rows': 'int64', 'issue_ns': 'int64', 'current_state': 'int8',
           'current_observed': 'bool', 'current_path_observed': 'bool',
           'current_confirmed_run_age_features': 'float32'}


def _require(ok, reason):
    if not ok:
        raise RuntimeError('OCTOBER_CAUSAL_INPUT_' + reason)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _verified_json(ref):
    _require(isinstance(ref, Mapping) and set(ref) == {'path', 'sha256'}, 'REF_KEYS')
    path = Path(ref['path'])
    _require(path.is_absolute() and path.is_file() and not any(p.is_symlink()
             for p in (path, *path.parents)) and _sha(path) == ref['sha256'], 'REF_IDENTITY')
    return json.loads(path.read_text())


@dataclass(frozen=True)
class OctoberCausalInputs:
    """Verified full tape; gather returns only selected issue-time feature copies."""
    arrays: Mapping[str, np.ndarray]
    manifest_ref: Mapping[str, str]

    def gather(self, *, parent_rows, issue_ns, role, include_age):
        _require(role in ('fit', 'score') and type(include_age) is bool, 'ROLE')
        _require(isinstance(parent_rows, np.ndarray) and parent_rows.dtype == np.int64
                 and parent_rows.ndim == 1 and 0 < len(parent_rows) <= 16
                 and isinstance(issue_ns, np.ndarray) and issue_ns.dtype == np.int64
                 and issue_ns.shape == parent_rows.shape
                 and len(np.unique(parent_rows)) == len(parent_rows), 'BATCH_AXES')
        source = self.arrays
        at = np.searchsorted(source['parent_rows'], parent_rows)
        _require(np.all(at < 36172) and np.array_equal(source['parent_rows'][at], parent_rows)
                 and np.array_equal(source['issue_ns'][at], issue_ns)
                 and (np.all(at < 29849) if role == 'fit' else np.all(at >= 29849)),
                 'EXACT_PARENT_ISSUE_ROLE')
        keys = _NAMES[2:] if include_age else _NAMES[2:-1]
        return {key: np.ascontiguousarray(source[key][at]) for key in keys}


def load_october_causal_inputs(binding, *, observer_axes_ref, candidate_design_ref,
                               target_series_ref):
    """Verify the complete immutable artifact once before batch selection."""
    manifest = _verified_json(binding)
    scope = _verified_json(observer_axes_ref)
    _require(set(manifest) == _MANIFEST_KEYS
             and manifest['schema_version'] == 'gx1_dynamic_m5_october_causal_input_artifact_v1'
             and manifest['candidate_design'] == candidate_design_ref
             and manifest['target_source_series'] == target_series_ref
             and manifest['observer_axes'] == observer_axes_ref
             and manifest['future_labels_included'] is False
             and manifest['clock_basis'] == 'historical_event_time_only_online_delivery_not_qualified'
             and manifest['fit_parent_rows'] == scope['fit']['rows'] == 29849
             and manifest['development_parent_rows'] == scope['held_out']['rows'] == 6323
             and scope['rows'] == 36172
             and manifest['cutoff_ns_exclusive'] == scope['held_out']['cutoff_ns']
             and manifest['parent_rows_sha256'] == scope['parent_rows_sha256']
             and manifest['issue_ns_sha256'] == scope['decision_ns_sha256']
             and set(manifest['arrays']) == set(_NAMES), 'MANIFEST_SCOPE')
    root = Path(binding['path']).parent
    arrays = {}
    for name in _NAMES:
        desc = manifest['arrays'][name]
        path = root / (name + '.npy')
        _require(set(desc) == {'path', 'sha256', 'size_bytes', 'dtype', 'shape'}
                 and desc['path'] == str(path) and path.is_file() and not path.is_symlink()
                 and path.stat().st_size == desc['size_bytes'] and _sha(path) == desc['sha256']
                 and desc['dtype'] == _DTYPES[name] and desc['shape'] == list(_SHAPES[name]),
                 'ARRAY_IDENTITY:' + name)
        value = np.load(path, mmap_mode='r', allow_pickle=False)
        _require(value.dtype.name == _DTYPES[name] and value.shape == _SHAPES[name]
                 and not value.flags.writeable, 'ARRAY_HEADER:' + name)
        arrays[name] = value
    rows, issues = arrays['parent_rows'], arrays['issue_ns']
    _require(np.all(rows[1:] > rows[:-1]) and np.all(issues[1:] > issues[:-1])
             and issues[29848] < scope['fit']['cutoff_ns'] <= issues[29849]
             and issues[-1] < scope['held_out']['cutoff_ns']
             and hashlib.sha256(rows.astype('<i8', copy=False).tobytes()).hexdigest() == scope['parent_rows_sha256']
             and hashlib.sha256(issues.astype('<i8', copy=False).tobytes()).hexdigest() == scope['decision_ns_sha256'],
             'AXES')
    for block, part in ((scope['fit'], slice(0, 29849)),
                        (scope['held_out'], slice(29849, 36172))):
        _require(hashlib.sha256(rows[part].astype('<i8', copy=False).tobytes()).hexdigest()
                 == block['parent_rows_sha256']
                 and hashlib.sha256(issues[part].astype('<i8', copy=False).tobytes()).hexdigest()
                 == block['decision_ns_sha256'], 'ROLE_AXES')
    observed, path = arrays['current_observed'], arrays['current_path_observed']
    classes = arrays['current_state']
    age = arrays['current_confirmed_run_age_features']
    mask = age[..., 1]
    _require(np.all((classes == state.UNKNOWN) == ~observed)
             and np.all((classes[observed] >= state.DOWN) & (classes[observed] <= state.UP))
             and np.isfinite(age).all()
             and np.all((mask == 0) | (mask == 1))
             and np.all((mask == 0) | (observed & path))
             and np.all(age[..., 0][mask == 0] == 0)
             and np.all(age[..., 0][mask == 1] > 0), 'CURRENT_VALUE_MASK')
    return OctoberCausalInputs(MappingProxyType(arrays), dict(binding))
