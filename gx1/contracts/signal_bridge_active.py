"""Active signal bridge — env-gated v1/v2/v3 selector.

Default (unset or "3"): v3 contract (2026-Q2 wave 2, canonical_v3 stack —
SEQ/SNAP 37, ctx_tag CTX6CAT5, ctx_cont_dim 142, ctx_cat_dim 5,
seq_len 96).

Set `GX1_SIGNAL_BRIDGE_VERSION=2` for the wave 1 v2 contract
(SEQ/SNAP 37, ctx_cont 43, ctx_cat 6, seq_len 96).

Set `GX1_SIGNAL_BRIDGE_VERSION=1` for the legacy v1 contract
(SEQ/SNAP 7, ctx_cont 6+ prefix, ctx_cat 6, seq_len model-defined).

This is a transparent re-export — every public name from the selected version
is forwarded. Files that don't care which version is active import from here:

    from gx1.contracts.signal_bridge_active import (
        ORDERED_FIELDS, SEQ_SIGNAL_DIM, validate_seq_signal,
        get_canonical_ctx_contract, ...
    )

Files that need a SPECIFIC version (e.g., bundle-metadata writers, contract
audits) should keep importing the explicit `signal_bridge_v1` /
`signal_bridge_v2` / `signal_bridge_v3`.
"""
from __future__ import annotations

import os

_VERSION = os.getenv("GX1_SIGNAL_BRIDGE_VERSION", "3").strip()

if _VERSION == "1":
    from gx1.contracts.signal_bridge_v1 import *  # noqa: F401,F403
    from gx1.contracts.signal_bridge_v1 import (  # noqa: F401
        SIGNAL_BRIDGE_ID,
        ORDERED_FIELDS,
        SEQ_SIGNAL_DIM,
        SNAP_SIGNAL_DIM,
        CONTRACT,
        CONTRACT_SHA256,
        ORDERED_CTX_CONT_NAMES_EXTENDED,
        ORDERED_CTX_CAT_NAMES_EXTENDED,
        N_CTX_CONT_EXTENDED,
        N_CTX_CAT_EXTENDED,
        ORDERED_CTX_CONT_NAMES_BASELINE,
        ORDERED_CTX_CAT_NAMES_BASELINE,
        N_CTX_CONT_BASELINE,
        N_CTX_CAT_BASELINE,
        CTX_CONT_COL_D1_DIST,
        CTX_CONT_COL_H1_COMP,
        CTX_CONT_COL_D1_ATR_PCTL252,
        CTX_CONT_COL_M15_COMP,
        CTX_CAT_COL_H4_TREND_SIGN,
        ALLOWED_CTX_CONT_DIMS,
        ALLOWED_CTX_CAT_DIMS,
        validate_seq_signal,
        validate_snap_signal,
        validate_contract_in_truth,
        validate_bundle_ctx_contract_in_strict,
        get_canonical_ctx_contract,
    )
    ACTIVE_VERSION = 1
elif _VERSION == "2":
    from gx1.contracts.signal_bridge_v2 import *  # noqa: F401,F403
    from gx1.contracts.signal_bridge_v2 import (  # noqa: F401  (explicit for static checkers)
        SIGNAL_BRIDGE_ID,
        ORDERED_FIELDS,
        SEQ_SIGNAL_DIM,
        SNAP_SIGNAL_DIM,
        CONTRACT,
        CONTRACT_SHA256,
        ORDERED_CTX_CONT_NAMES_EXTENDED,
        ORDERED_CTX_CAT_NAMES_EXTENDED,
        N_CTX_CONT_EXTENDED,
        N_CTX_CAT_EXTENDED,
        ORDERED_CTX_CONT_NAMES_BASELINE,
        ORDERED_CTX_CAT_NAMES_BASELINE,
        N_CTX_CONT_BASELINE,
        N_CTX_CAT_BASELINE,
        CTX_CONT_COL_D1_DIST,
        CTX_CONT_COL_H1_COMP,
        CTX_CONT_COL_D1_ATR_PCTL252,
        CTX_CONT_COL_M15_COMP,
        CTX_CAT_COL_H4_TREND_SIGN,
        ALLOWED_CTX_CONT_DIMS,
        ALLOWED_CTX_CAT_DIMS,
        validate_seq_signal,
        validate_snap_signal,
        validate_contract_in_truth,
        validate_bundle_ctx_contract_in_strict,
        get_canonical_ctx_contract,
    )
    ACTIVE_VERSION = 2
else:
    from gx1.contracts.signal_bridge_v3 import *  # noqa: F401,F403
    from gx1.contracts.signal_bridge_v3 import (  # noqa: F401  (explicit for static checkers)
        SIGNAL_BRIDGE_ID,
        ORDERED_FIELDS,
        SEQ_SIGNAL_DIM,
        SNAP_SIGNAL_DIM,
        CONTRACT,
        CONTRACT_SHA256,
        ORDERED_CTX_CONT_NAMES_EXTENDED,
        ORDERED_CTX_CAT_NAMES_EXTENDED,
        N_CTX_CONT_EXTENDED,
        N_CTX_CAT_EXTENDED,
        ORDERED_CTX_CONT_NAMES_BASELINE,
        ORDERED_CTX_CAT_NAMES_BASELINE,
        N_CTX_CONT_BASELINE,
        N_CTX_CAT_BASELINE,
        CTX_CONT_COL_D1_DIST,
        CTX_CONT_COL_H1_COMP,
        CTX_CONT_COL_D1_ATR_PCTL252,
        CTX_CONT_COL_M15_COMP,
        CTX_CAT_COL_H4_TREND_SIGN,
        ALLOWED_CTX_CONT_DIMS,
        ALLOWED_CTX_CAT_DIMS,
        validate_seq_signal,
        validate_snap_signal,
        validate_contract_in_truth,
        validate_bundle_ctx_contract_in_strict,
        get_canonical_ctx_contract,
    )
    ACTIVE_VERSION = 3
