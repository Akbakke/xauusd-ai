import gzip
import hashlib
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pytest

from gx1.scripts.gc_order_flow_download_v1 import (
    _decompress_csv_zstd,
    _same_scope,
    HistoricalAPI,
    validate_budget,
)


def test_budget_guard_enforces_exact_120_dollar_ceiling():
    inside = [{"quoted_cost_usd": "119.9999999999"}]
    assert validate_budget(inside) == Decimal("119.9999999999")
    with pytest.raises(ValueError, match="GC_HARD_COST_CAP_EXCEEDED"):
        validate_budget([{ "quoted_cost_usd": "120.0000000001" }])
    with pytest.raises(ValueError, match="GC_HARD_COST_CAP_EXCEEDED"):
        validate_budget(inside, Decimal("120.01"))


def test_existing_batch_job_must_match_full_scope_and_output_contract():
    scope = {
        "dataset": "GLBX.MDP3", "schema": "mbp-1", "symbols": ["GCM4", "GCQ4"],
        "start": "2024-05-01T00:00:00Z", "end": "2024-05-31T00:00:00Z",
        "stype_in": "raw_symbol",
    }
    job = {
        "dataset": "GLBX.MDP3", "schema": "mbp-1", "symbols": "GCM4,GCQ4",
        "start": "2024-05-01T00:00:00.000000000Z", "end": "2024-05-31T00:00:00.000000000Z",
        "stype_in": "raw_symbol", "stype_out": "instrument_id", "encoding": "csv",
        "compression": "zstd", "pretty_px": False, "pretty_ts": False,
        "map_symbols": True, "split_symbols": True, "split_duration": "none", "delivery": "download",
    }
    assert _same_scope(job, scope)
    assert not _same_scope({**job, "start": "2024-05-01T01:00:00.000000000Z"}, scope)
    assert not _same_scope({**job, "compression": "none"}, scope)


def test_streaming_zstd_conversion_produces_hash_bound_gzip_csv(tmp_path: Path):
    header = (
        "ts_recv,ts_event,rtype,publisher_id,instrument_id,action,side,price,size,flags,sequence,"
        "bid_px_00,ask_px_00,bid_sz_00,ask_sz_00,symbol\n"
    )
    row = "1,1,1,1,1,T,B,100,1,0,1,99,101,1,1,GCM4\n"
    original = (header + row) * 500
    source = tmp_path / "sample.csv.zst"
    output = tmp_path / "sample.csv.gz"
    source.write_bytes(pa.compress(original.encode(), codec="zstd"))

    digest, fields, expanded, symbol = _decompress_csv_zstd(source, output, "mbp-1")

    assert set(("ts_recv", "instrument_id", "bid_px_00", "ask_px_00")).issubset(fields)
    assert expanded == len(original.encode())
    assert symbol == "GCM4"
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    with gzip.open(output, "rt", encoding="utf-8") as stream:
        assert stream.readline().startswith("ts_recv,ts_event,")


def test_mbp_download_rejects_wrong_vendor_csv_header(tmp_path: Path):
    source = tmp_path / "bad.csv.zst"
    output = tmp_path / "bad.csv.gz"
    source.write_bytes(pa.compress(b"timestamp,close\n1,100\n", codec="zstd"))
    with pytest.raises(ValueError, match="GC_MBP1_CSV_HEADER_INVALID"):
        _decompress_csv_zstd(source, output, "mbp-1")
    assert not output.exists()



def test_post_request_encodes_form_parameters_in_request_body(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(
        "gx1.scripts.gc_order_flow_download_v1.urllib.request.urlopen",
        fake_urlopen,
    )
    params = {"dataset": "GLBX.MDP3", "symbols": "GCM4,GCQ4", "schema": "mbp-1"}
    assert HistoricalAPI("unused-test-key").request("batch.submit_job", params, post=True) == {}

    request = captured["request"]
    assert request.full_url == "https://hist.databento.com/v0/batch.submit_job"
    assert request.method == "POST"
    assert request.get_header("Content-type") == "application/x-www-form-urlencoded"
    assert request.data.decode("utf-8") == "dataset=GLBX.MDP3&symbols=GCM4%2CGCQ4&schema=mbp-1"
