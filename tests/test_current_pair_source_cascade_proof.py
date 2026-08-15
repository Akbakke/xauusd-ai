from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.xau_tape_provenance_v1 import (
    SEQ513_SOURCE_CASCADE_PAIR_PROOF_SCHEMA_VERSION,
)
from gx1.scripts import materialize_current_pair_source_cascade_proof_v1 as owner


RUN_ID = "CURRENT_PAIR_SOURCE_PROOF_TEST_V1"


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, monkeypatch):
    times = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
    close = np.arange(4, dtype=np.float64) + 2_000.0
    market = pd.DataFrame(
        {
            "time": times,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.arange(4, dtype=np.float64) + 10.0,
            "bid_close": close - 0.05,
            "ask_close": close + 0.05,
        }
    )
    source = tmp_path / "model_source.parquet"
    canonical = tmp_path / "canonical_v2.parquet"
    cache_source = tmp_path / "cache_source.parquet"
    market.to_parquet(source, index=False)
    market.assign(active_feature=np.arange(4, dtype=np.float64)).to_parquet(
        canonical,
        index=False,
    )
    market.to_parquet(cache_source, index=False)
    cache_dir = tmp_path / "MULTI_TF_V4_CACHE"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    cache = SimpleNamespace(
        cache_identity_sha256="c" * 64,
        m5_prebuilt_source=str(cache_source),
        m5_prebuilt_source_sha256=_sha(cache_source),
    )
    monkeypatch.setattr(owner, "load_multi_tf_v4_cache", lambda _path: cache)
    monkeypatch.setattr(owner, "require_offline_scope", lambda _scope: None)
    pair_manifest = tmp_path / "PAIR_MANIFEST.json"
    pair_manifest.write_text(
        json.dumps({"pair_generation_id": "pair-generation-1"}),
        encoding="utf-8",
    )
    return times, source, canonical, cache_dir, pair_manifest


def test_current_pair_proof_emits_and_revalidates_exact_bindings(
    tmp_path,
    monkeypatch,
) -> None:
    times, source, canonical, cache_dir, pair_manifest = _fixture(
        tmp_path,
        monkeypatch,
    )
    proof_path = tmp_path / "SOURCE_CASCADE_PROOF.json"
    payload = owner.emit(
        run_id=RUN_ID,
        source_parquet=source,
        canonical_v2_parquet=canonical,
        mtf_cache_dir=cache_dir,
        pair_manifest=pair_manifest,
        required_history_start="2026-01-01T00:00:00Z",
        out=proof_path,
    )

    binding = owner.validate_current_pair_source_cascade_proof(
        proof_path,
        expected_run_id=RUN_ID,
        expected_source_parquet=source,
        expected_canonical_v2_parquet=canonical,
        expected_mtf_cache_dir=cache_dir,
        expected_history_start_utc="2026-01-01T00:00:00Z",
        expected_time_max_utc=times[-1],
    )

    assert payload["schema_version"] == (
        SEQ513_SOURCE_CASCADE_PAIR_PROOF_SCHEMA_VERSION
    )
    assert binding["schema_version"] == (
        SEQ513_SOURCE_CASCADE_PAIR_PROOF_SCHEMA_VERSION
    )
    assert binding["source_parquet_sha256"] == _sha(source)
    assert binding["canonical_v2_sha256"] == _sha(canonical)
    assert binding["pair_generation_id"] == "pair-generation-1"


def test_current_pair_validator_rejects_legacy_proof_schema(
    tmp_path,
    monkeypatch,
) -> None:
    times, source, canonical, cache_dir, pair_manifest = _fixture(
        tmp_path,
        monkeypatch,
    )
    proof_path = tmp_path / "SOURCE_CASCADE_PROOF.json"
    owner.emit(
        run_id=RUN_ID,
        source_parquet=source,
        canonical_v2_parquet=canonical,
        mtf_cache_dir=cache_dir,
        pair_manifest=pair_manifest,
        required_history_start="2026-01-01T00:00:00Z",
        out=proof_path,
    )
    payload = json.loads(proof_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "seq513_source_cascade_proof_v8"
    proof_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="PROOF_SCHEMA_MISMATCH"):
        owner.validate_current_pair_source_cascade_proof(
            proof_path,
            expected_run_id=RUN_ID,
            expected_source_parquet=source,
            expected_canonical_v2_parquet=canonical,
            expected_mtf_cache_dir=cache_dir,
            expected_history_start_utc="2026-01-01T00:00:00Z",
            expected_time_max_utc=times[-1],
        )
