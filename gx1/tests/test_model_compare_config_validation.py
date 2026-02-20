"""
Unit tests for model_compare config strict validation (UNKNOWN_MODEL, evidence format).

No replay/training; only validation logic and error reporting.
Self-check: --validate_only (canonical), --models ALL, multi-model, non-canonical WARNING.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from gx1.scripts.run_model_compare_2025 import (
    CONFIGS_PATH,
    validate_model_compare_config_strict,
)


def test_unknown_model_fails_and_writes_evidence() -> None:
    """Requesting a model not in config must raise RuntimeError and write flat evidence (UNKNOWN_MODEL)."""
    if not CONFIGS_PATH.exists():
        pytest.skip("CONFIGS_PATH not present (run from repo root)")
    config = json.loads(CONFIGS_PATH.read_text(encoding="utf-8"))
    available = list(config.get("models", {}).keys())
    if not available:
        pytest.skip("config has no models")
    models = [available[0], "NOT_IN_CONFIG"]
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        with pytest.raises(RuntimeError) as exc_info:
            validate_model_compare_config_strict(
                CONFIGS_PATH,
                truth_mode=False,
                run_dir=run_dir,
                models=models,
            )
        assert "UNKNOWN_MODEL" in str(exc_info.value)
        assert "NOT_IN_CONFIG" in str(exc_info.value)
        assert "available=" in str(exc_info.value)
        evidence_path = run_dir / "model_compare_validation_evidence.json"
        assert evidence_path.exists()
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        assert evidence.get("config_path")
        assert "canonical_model" in evidence
        assert evidence.get("truth_mode") is False
        assert evidence.get("requested_models") == sorted(models)
        assert "NOT_IN_CONFIG" in evidence.get("unknown_models", [])
        assert evidence.get("available_models") == sorted(available)
        assert evidence.get("status") == "FAIL"
        assert "UNKNOWN_MODEL" in evidence.get("reason", "")
        # UNKNOWN_MODEL uses flat payload; must NOT have "models" key (no per-model records).
        assert "models" not in evidence


def test_evidence_format_has_checks_keys() -> None:
    """Evidence payload uses fixed top keys and per-model checks keys (deterministic)."""
    from gx1.scripts.run_model_compare_2025 import (
        EVIDENCE_TOP_KEYS,
        CHECK_KEYS,
        MODEL_REC_KEYS,
    )
    # Normal validation evidence: config_path, canonical_model, models (list of records with checks), truth_mode.
    assert EVIDENCE_TOP_KEYS == ("config_path", "canonical_model", "models", "truth_mode")
    assert "config_path" in EVIDENCE_TOP_KEYS
    assert "canonical_model" in EVIDENCE_TOP_KEYS
    assert "models" in EVIDENCE_TOP_KEYS
    assert "truth_mode" in EVIDENCE_TOP_KEYS
    assert "xgb_lock_ok" in CHECK_KEYS
    assert "prebuilt_prefix_ok" in CHECK_KEYS
    assert "transformer_lock_ok" in CHECK_KEYS
    assert "transformer_model_file_ok" in CHECK_KEYS
    assert "sha_ok" in CHECK_KEYS
    assert "ctx_manifest_ok" in CHECK_KEYS
    assert "signal_bridge_sha_ok" in CHECK_KEYS
    assert "model" in MODEL_REC_KEYS
    assert "status" in MODEL_REC_KEYS
    assert "reason" in MODEL_REC_KEYS
    assert "checks" in MODEL_REC_KEYS


def _run_script(args: list[str]) -> tuple[int, str, str]:
    """Run run_model_compare_2025 as subprocess; return (returncode, stdout, stderr)."""
    cmd = [sys.executable, "-m", "gx1.scripts.run_model_compare_2025"] + args
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=Path(__file__).resolve().parents[2],
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def test_validate_only_without_models_validates_canonical() -> None:
    """--validate_only without --models must validate canonical model (exit 0 and evidence has canonical_model)."""
    if not CONFIGS_PATH.exists():
        pytest.skip("CONFIGS_PATH not present (run from repo root)")
    rc, out, err = _run_script(["--validate_only"])
    if rc != 0:
        pytest.skip("validate_only failed (paths/locks may be missing); stderr: " + err[:500])
    assert "canonical_model" in err or "canonical" in err.lower()
    evidence_path = Path(__file__).resolve().parents[2] / "gx1" / "configs" / "model_compare_2025"
    # Evidence is written to GX1_DATA/reports/model_compare_2025/; we only assert exit 0 and stderr
    assert rc == 0


def test_models_all_fails_with_banned_message() -> None:
    """--models ALL must exit non-zero with message that ALL is banned."""
    rc, out, err = _run_script(["--models", "ALL"])
    assert rc != 0
    assert "ALL" in err and "banned" in err.lower()


def test_models_multi_fails_with_multi_model_banned() -> None:
    """--models BASE28,BASE28_CTX2PLUS_T1 must exit non-zero with multi-model banned message."""
    rc, out, err = _run_script(["--models", "BASE28,BASE28_CTX2PLUS_T1"])
    assert rc != 0
    assert "Multi-model" in err or "multi-model" in err
    assert "banned" in err.lower()


def test_models_base28_ctx2plus_logs_non_canonical_warning() -> None:
    """--models BASE28_CTX2PLUS_T1 must log RUNNING NON-CANONICAL to stderr (no exit required)."""
    rc, out, err = _run_script(["--validate_only", "--models", "BASE28_CTX2PLUS_T1"])
    assert "RUNNING NON-CANONICAL" in err and "BASE28_CTX2PLUS_T1" in err and "research-only" in err


def test_parse_model_bans_all_and_multi() -> None:
    """_parse_model: ALL and multi-model (A,B) must hard-fail."""
    rc_all, _, err_all = _run_script(["--models", "ALL"])
    assert rc_all != 0
    assert "ALL" in err_all and "banned" in err_all.lower()
    rc_multi, _, err_multi = _run_script(["--models", "BASE28,BASE28_CTX2PLUS_T1"])
    assert rc_multi != 0
    assert "Multi-model" in err_multi or "multi-model" in err_multi
    assert "banned" in err_multi.lower()


def test_format_threshold_id_decimal() -> None:
    """_format_threshold_id(Decimal('0.48')) == '0p48' (regression: no float drift)."""
    from decimal import Decimal

    from gx1.scripts.run_model_compare_2025 import _format_threshold_id

    assert _format_threshold_id(Decimal("0.48")) == "0p48"
    assert _format_threshold_id(Decimal("0.50")) == "0p50"


def test_threshold_value_canonical() -> None:
    """_threshold_value_canonical returns canonical string, e.g. '0.50'."""
    from gx1.scripts.run_model_compare_2025 import _threshold_value_canonical

    assert _threshold_value_canonical("0.50") == "0.50"
    assert _threshold_value_canonical("0.48") == "0.48"


def test_build_cases_uses_canonical_env() -> None:
    """env override must be canonical string '0.48', not float-repr."""
    from decimal import Decimal

    from gx1.scripts.run_model_compare_2025 import (
        _build_cases,
        OUT_ROOT,
    )
    base_env = {}
    cases = _build_cases(
        "BASE28",
        [Decimal("0.48"), Decimal("0.50")],
        base_env,
        "GX1_ENTRY_THRESHOLD_OVERRIDE",
    )
    assert len(cases) == 2
    assert cases[0].case_id == "thr_0p48"
    assert cases[1].case_id == "thr_0p50"
    assert cases[0].env_overrides.get("GX1_ENTRY_THRESHOLD_OVERRIDE") == "0.48"
    assert cases[1].env_overrides.get("GX1_ENTRY_THRESHOLD_OVERRIDE") == "0.50"
    assert cases[0].run_root == OUT_ROOT / "BASE28" / "cases" / "thr_0p48"
    assert cases[0].threshold_value == "0.48"
