"""Process-local file-access guard for native M5 direction entrypoints.

Install before feature/model imports. Legacy Python code may be imported, but
legacy M1 artifacts and the source-bound GC049 registry may never be accessed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

_STATE = None
_M1_TOKEN = re.compile(r'(^|[^a-z0-9])m1([^a-z0-9]|$)', re.IGNORECASE)
_OLD_REGISTRY = 'NATIVE_GC_CONTEXT_FIELDS_20261018_049.json'
_CODE_SUFFIXES = ('.py', '.pyc', '.so', '.pyd', '.dll', '.dylib')


def contract():
    value = {'schema_version': 'native_m5_direction_io_firewall_v1',
        'forbidden_registry_basename': _OLD_REGISTRY,
        'forbidden_artifact_token': 'M1_delimited_by_non_alphanumeric_or_path_edges',
        'code_suffixes_allowed': list(_CODE_SUFFIXES),
        'guards': ['CPython_open_audit_event', 'os.stat', 'os.lstat'],
        'installer': 'before_any_feature_model_or_source_import',
        'disable_or_override': False}
    raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def _forbidden(path):
    if isinstance(path, int):
        return False
    try:
        text = os.fsdecode(os.fspath(path))
    except TypeError:
        return False
    if os.path.basename(text) == _OLD_REGISTRY:
        return True
    return bool(_M1_TOKEN.search(text) and not text.lower().endswith(_CODE_SUFFIXES))


def install_native_m5_direction_io_firewall():
    """Install once; return live denied-attempt counters for terminal reporting."""
    global _STATE
    if _STATE is not None:
        return _STATE
    state = {'contract': contract(), 'denied_attempts': 0, 'denied_operations': {}}

    def check(path, operation):
        if _forbidden(path):
            state['denied_attempts'] += 1
            counts = state['denied_operations']
            counts[operation] = counts.get(operation, 0) + 1
            raise PermissionError('DIRECTION_M1_IO_FORBIDDEN:' + os.fsdecode(os.fspath(path)))

    def audit(event, arguments):
        if event == 'open' and arguments:
            check(arguments[0], 'open')

    sys.addaudithook(audit)
    class GuardedOperation:
        # A callable object retains builtin-style non-descriptor behavior when
        # pathlib binds os.stat into its accessor during a later import.
        def __init__(self, original, operation):
            self.original, self.operation = original, operation

        def __call__(self, path, *args, **kwargs):
            check(path, self.operation)
            return self.original(path, *args, **kwargs)

    for name in ('stat', 'lstat'):
        setattr(os, name, GuardedOperation(getattr(os, name), name))
    _STATE = state
    return state
