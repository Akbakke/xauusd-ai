"""Single source of truth for IQL reward variant names.

2026-05-24: previously scattered across materialize_build_entry_iql_v2.py and
materialize_build_exit_iql_v2.py. Consolidated here for clarity + avoid
hardcoded-string drift between trainer and live runtime.
"""
ENTRY_REWARD_VARIANTS = (
    "R_NET_REAL", "R_TERMINAL_K24", "R_TERMINAL_K96",
    "R_PATIENT_K96_LAM03", "R_PATIENT_K96_LAM05", "R_PATIENT_K96_LAM10",
    "R_CLEAN_K96_TOL10", "R_CLEAN_K96_TOL20", "R_CLEAN_K96_TOL40",
    "R_QUAD_K96_LAM05", "R_QUAD_K96_LAM10",
    "R_QUALITY_K96",
    "R_SOFT_K96_TOL20", "R_SOFT_K96_TOL40",
    "R_ASYMMETRIC_K96_LAM05", "R_ASYMMETRIC_K96_LAM10",
    "R_WAIT_OPP_K96_LAM05", "R_WAIT_OPP_K96_LAM10", "R_WAIT_OPP_K96_LAM20",
    "R_WAIT_OPP_K96_LAM30", "R_WAIT_OPP_K96_LAM50",
    "R_WAIT_OPP_K48_LAM05", "R_WAIT_OPP_K48_LAM10",
    "R_HYBRID_K96_TOL20", "R_HYBRID_K96_TOL40",
)
EXIT_REWARD_VARIANTS = (
    "R_NET_REAL", "R_GATED", "R_REGRET", "R_NET_V2",
    "R_PEAK_AWARE_MILD", "R_PEAK_AWARE_MED", "R_PEAK_AWARE_HARSH",
    "R_PEAK_QUALITY", "R_PEAK_QUALITY_QUAD",
)
