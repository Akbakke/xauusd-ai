# gx1/scripts/_truth_lane.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import sys

@dataclass(frozen=True)
class TruthXGBLane:
    readme_rel: str = "gx1/scripts/README_TRUTH_XGB.md"
    required_env: str = "GX1_DATA_ROOT"
    required_gx1_data: str = "/home/andre2/GX1_DATA"
    required_manifest_rel: str = "data/data/prebuilt/BASE28_CANONICAL/CURRENT_MANIFEST.json"
    required_xgb_bundle_rel: str = "models/models/xgb_universal_multihead_v2__CANONICAL"

    banned_tokens = (
        "TRIAL160",
        "features_v10_ctx",
        "FULLYEAR_2025_GATED_FUSION",
        "PRUNE14",
        "PRUNE20",
        "v13_refined3",
        "CTX_CONT6_CAT6",
        "REFINED3",
    )

def _workspace_root_from_file(file: str) -> Path:
    return Path(file).resolve().parents[2]

def truth_readme_path(file: str) -> Path:
    root = _workspace_root_from_file(file)
    return root / TruthXGBLane().readme_rel

def require_truth_xgb_lane(file: str, extra_context: str = "") -> Path:
    """
    Hard-fail guard for scripts that claim to be TRUTH XGB lane.
    Returns resolved GX1_DATA_ROOT path if OK.
    """
    lane = TruthXGBLane()
    root = _workspace_root_from_file(file)
    readme = root / lane.readme_rel

    gx1_data = os.environ.get(lane.required_env, "")
    if not gx1_data:
        raise RuntimeError(
            f"TRUTH_LANE_VIOLATION: env {lane.required_env} is not set.\n"
            f"Read: {readme}"
        )

    gx1_data_path = Path(gx1_data).resolve()
    if str(gx1_data_path) != lane.required_gx1_data:
        raise RuntimeError(
            f"TRUTH_LANE_VIOLATION: {lane.required_env} must be {lane.required_gx1_data}, got {gx1_data_path}\n"
            f"Read: {readme}"
        )

    manifest = gx1_data_path / lane.required_manifest_rel
    if not manifest.exists():
        raise RuntimeError(
            f"TRUTH_LANE_VIOLATION: missing CURRENT_MANIFEST.json at:\n  {manifest}\n"
            f"Read: {readme}"
        )

    bundle = gx1_data_path / lane.required_xgb_bundle_rel
    if not bundle.exists():
        raise RuntimeError(
            f"TRUTH_LANE_VIOLATION: missing canonical XGB bundle at:\n  {bundle}\n"
            f"Read: {readme}"
        )

    # Optional: sanity-check the script path itself for banned tokens
    this = str(Path(file).resolve())
    for tok in lane.banned_tokens:
        if tok.lower() in this.lower():
            raise RuntimeError(
                f"TRUTH_LANE_VIOLATION: script path contains banned token '{tok}': {this}\n"
                f"Read: {readme}"
            )

    if extra_context:
        # Extra context may include paths; enforce bans there too.
        low = extra_context.lower()
        for tok in lane.banned_tokens:
            if tok.lower() in low:
                raise RuntimeError(
                    f"TRUTH_LANE_VIOLATION: banned token '{tok}' in context.\n"
                    f"Context: {extra_context}\n"
                    f"Read: {readme}"
                )

    return gx1_data_path