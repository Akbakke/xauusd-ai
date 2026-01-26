# gx1/utils/feature_context.py
# -*- coding: utf-8 -*-
"""
Context variables for feature state management.
Allows feature-building functions to access persistent state without signature changes.
"""
import contextvars
from typing import Optional

from gx1.features.feature_state import FeatureState

# Context variable for feature state
FEATURE_STATE: contextvars.ContextVar[Optional[FeatureState]] = contextvars.ContextVar(
    "FEATURE_STATE", default=None
)


def get_feature_state() -> Optional[FeatureState]:
    """Get the current feature state from context."""
    return FEATURE_STATE.get()


def set_feature_state(state: FeatureState) -> contextvars.Token:
    """Set the feature state in context. Returns token for reset()."""
    return FEATURE_STATE.set(state)


def reset_feature_state(token: contextvars.Token) -> None:
    """Reset the feature state context to previous value."""
    FEATURE_STATE.reset(token)

