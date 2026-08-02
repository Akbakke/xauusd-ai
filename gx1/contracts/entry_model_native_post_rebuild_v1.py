"""Exact post-rebuild admission contract for model-native seq513 Entry data.

The current direct capped rebuild binds ``DATASET_BUILD_PROOF.json`` as its
completion authority.  The historical chain-terminal schema remains readable
for already-authoritative chain events, but no path may pass without one of
those exact completion contracts.
"""

from __future__ import annotations


SCHEMA_VERSION = "entry_model_native_seq513_post_rebuild_readiness_v1"
READY_DECISION = "READY_FOR_MODEL_NATIVE_SEQ513_POST_REBUILD_REVIEW"
EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SEQ513_POST_REBUILD_READINESS"

REQUIRED_PROOF_CHECKS = (
    "rebuild chain terminal is exact green",
    "rebuild preflight is exact ready",
    "full-input liveness is exact pass",
    "pretrain audit is exact pass",
    "split identities are exact and XAU-bound",
)

SIDE_EFFECT_KEYS = (
    "dataset_rebuild",
    "training",
    "replay",
    "iql_distillation",
    "shadow",
    "live",
)
