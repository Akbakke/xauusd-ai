"""Mechanical byte/algebra checks only; no data, model or utility evidence."""

from __future__ import annotations

import gc
import hashlib
import math
import weakref
from fractions import Fraction

import numpy as np
import pytest

from gx1.contracts.entry_exit_feature_usefulness_v1 import (
    STANDARD_ERROR_METHOD,
    StreamingArrayDigest,
    StreamingPairedSummary,
    _require_summary,
    _STREAM_BUFFER_BYTES,
    _stream_array_blocks,
)


DOMAIN = b"mechanical_streaming_usefulness\0"


def _whole_array_digest(values: np.ndarray, domain: bytes = DOMAIN) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _whole_array_summary(values: np.ndarray) -> dict[str, object]:
    """Freeze the pre-streaming reference, independently of the routed script."""

    raw = np.ascontiguousarray(np.asarray(values, dtype="<f8").reshape(-1))
    count = int(raw.size)
    variance = float(raw.var(ddof=1)) if count > 1 else 0.0
    return {
        "count": count,
        "sum": float(raw.sum(dtype=np.float64)),
        "mean": float(raw.mean(dtype=np.float64)),
        "sample_variance": variance,
        "standard_error": math.sqrt(variance / count),
        "standard_error_method": STANDARD_ERROR_METHOD,
        "minimum": float(raw.min()),
        "maximum": float(raw.max()),
        "positive_count": int((raw > 0.0).sum()),
        "zero_count": int((raw == 0.0).sum()),
        "negative_count": int((raw < 0.0).sum()),
        "paired_vector_sha256": _whole_array_digest(raw),
    }


def _summarize(values: np.ndarray, widths: tuple[int, ...]) -> dict[str, object]:
    assert sum(widths) == values.size
    stream = StreamingPairedSummary(values.size, DOMAIN)
    offset = 0
    for width in widths:
        stream.update(values[offset:offset + width])
        offset += width
    result = stream.finalize()
    _require_summary(result, count=values.size, label="MECHANICAL")
    return result


def _gamma(operations: int) -> float:
    unit_roundoff = math.ulp(1.0) / 2
    return operations * unit_roundoff / (1 - operations * unit_roundoff)


def _variance_roundoff(values: np.ndarray, updates: int) -> float:
    """Normal-range fixtures only, not a universal Chan error certificate.

    Bound both paths' rounded reductions/subtractions/squares by ten scalar
    operations per row, and block divisions, compensated sums and Chan merges
    by twenty per update, plus final divisions. Use squared data diameter,
    not a guessed absolute epsilon or the large common input offset.
    """

    diameter = float(max(values) - min(values))
    operations = 10 * values.size + 20 * updates + 4
    return _gamma(operations) * diameter**2 * values.size / (values.size - 1)


@pytest.mark.parametrize("dtype", ["?", "u1", "<i2", ">i8", "<f4", "<f8", ">f8"])
@pytest.mark.parametrize("shape", [(13,), (13, 5), (13, 3, 2)])
@pytest.mark.parametrize("widths", [(13,), (1, 4, 3, 5), (1,) * 13])
def test_digest_matches_complete_c_order_bytes(dtype, shape, widths) -> None:
    values = np.arange(math.prod(shape)).reshape(shape).astype(dtype)
    values = np.asfortranarray(values)[::-1]
    stream = StreamingArrayDigest(values.shape, values.dtype, DOMAIN)
    offset = 0
    for width in widths:
        assert stream.update(values[offset:offset + width]) is None
        offset += width
    assert stream.finalize() == _whole_array_digest(values)
    assert stream.finalize() == _whole_array_digest(values)


def test_digest_binds_domain_shape_dtype_endian_and_signed_zero() -> None:
    values = np.array([0.0, -0.0, 1.0, -1.0], dtype="<f8")
    signatures = set()
    for array, domain in (
        (values, DOMAIN),
        (values, b"other"),
        (values, b""),
        (values.reshape(2, 2), DOMAIN),
        (values.astype("<f4"), DOMAIN),
        (values.astype(">f8"), DOMAIN),
        (np.array([0.0, 0.0, 1.0, -1.0]), DOMAIN),
    ):
        stream = StreamingArrayDigest(array.shape, array.dtype, domain)
        stream.update(array)
        signature = stream.finalize()
        assert signature == _whole_array_digest(array, domain)
        signatures.add(signature)
    assert len(signatures) == 7


def test_digest_streams_rows_wider_than_its_scratch_buffer() -> None:
    width = _STREAM_BUFFER_BYTES // 8 + 1
    values = np.arange(3 * width, dtype="<f8").reshape(3, width)[:, ::-1]
    stream = StreamingArrayDigest(values.shape, values.dtype, DOMAIN)
    stream.update(values[:1])
    stream.update(values[1:])
    assert stream.finalize() == _whole_array_digest(values)


@pytest.mark.parametrize("widths", [(8,), (1, 3, 4), (1,) * 8])
@pytest.mark.parametrize("dtype", ["<f8", ">f8", "<f4", "i8"])
def test_summary_preserves_reference_surface_and_canonical_vector(dtype, widths) -> None:
    values = np.array([0.0, -0.0, 1.0, -2.0, 4.0, -8.0, 16.0, 3.0], dtype=dtype)
    summary = _summarize(values, widths)
    reference = _whole_array_summary(values)
    assert summary.keys() == reference.keys()
    for field in reference.keys() - {"sample_variance", "standard_error"}:
        assert summary[field] == reference[field]
    bound = _variance_roundoff(values.astype("<f8"), len(widths))
    assert abs(summary["sample_variance"] - reference["sample_variance"]) <= bound
    assert abs(summary["standard_error"] - reference["standard_error"]) <= (
        math.sqrt(bound / values.size) + math.ulp(reference["standard_error"])
    )


@pytest.mark.parametrize("values", [
    np.array([False, True, False, True]),
    np.array([2**63, 2**63 + 1, 2**63 + 2, 2**63 + 3], dtype="u8"),
    np.array([1, 2, 3, 4], dtype=">f4"),
])
def test_readonly_numeric_vectors_use_existing_float64_canonicalization(values) -> None:
    values.setflags(write=False)
    stream = StreamingPairedSummary(np.int64(values.size), DOMAIN)
    stream.update(values[:2])
    stream.update(values[2:])
    summary = stream.finalize()
    reference = _whole_array_summary(values)
    assert summary == reference


@pytest.mark.parametrize("value", [-8.0, -0.0, 0.0, 4.0, np.finfo(np.float64).max])
def test_singleton_exact_summary_and_sign_counters(value) -> None:
    values = np.array([value], dtype="<f8")
    assert _summarize(values, (1,)) == _whole_array_summary(values)


@pytest.mark.parametrize("widths", [(7,), (2, 3, 2), (1,) * 7])
def test_sign_counters_include_both_signed_zeros_without_clipping(widths) -> None:
    tiny = np.nextafter(0.0, 1.0)
    values = np.array([-tiny, tiny, -0.0, 0.0, -1.0, 1.0, tiny])
    summary = _summarize(values, widths)
    assert summary["positive_count"] == 3
    assert summary["negative_count"] == 2
    assert summary["zero_count"] == 2
    assert summary["sum"] == tiny
    assert summary["paired_vector_sha256"] == _whole_array_digest(values)


@pytest.mark.parametrize("widths", [(3,), (2, 1), (1, 2), (1, 1, 1)])
def test_cancellation_recovers_block_sum_rounding_residual(widths) -> None:
    values = np.array([2.0**53, 1.0, -(2.0**53)])
    summary = _summarize(values, widths)
    assert summary["sum"] == math.fsum(values) == 1.0
    assert summary["mean"] == 1.0 / 3
    exact = [Fraction(float(value)) for value in values]
    exact_mean = sum(exact) / len(exact)
    exact_variance = sum((value - exact_mean) ** 2 for value in exact) / 2
    assert abs(summary["sample_variance"] - float(exact_variance)) <= (
        _variance_roundoff(values, len(widths))
    )
    assert summary["paired_vector_sha256"] == _whole_array_digest(values)


@pytest.mark.parametrize("widths", [(8,), (3, 2, 3), (1,) * 8])
def test_large_offset_small_variance_against_exact_binary_rationals(widths) -> None:
    origin = 2.0**40
    values = origin + math.ulp(origin) * np.array([-4, -2, -1, 0, 0, 1, 2, 4])
    summary = _summarize(values, widths)
    exact = [Fraction(float(value)) for value in values]
    exact_mean = sum(exact) / len(exact)
    exact_variance = sum((value - exact_mean) ** 2 for value in exact) / (len(exact) - 1)
    reference = _whole_array_summary(values)
    bound = _variance_roundoff(values, len(widths))
    assert summary["mean"] == float(exact_mean) == origin
    assert summary["sum"] == float(sum(exact))
    assert summary["sample_variance"] > 0.0
    assert abs(summary["sample_variance"] - float(exact_variance)) <= bound
    assert abs(summary["sample_variance"] - reference["sample_variance"]) <= bound
    assert summary["paired_vector_sha256"] == reference["paired_vector_sha256"]


@pytest.mark.parametrize("widths", [(2,), (1, 1)])
def test_sub_ulp_mean_rounding_is_not_a_byte_parity_claim(widths) -> None:
    values = np.array([1.0, np.nextafter(1.0, 2.0)])
    summary = _summarize(values, widths)
    separation = Fraction(float(values[1])) - Fraction(float(values[0]))
    exact_variance = float(separation**2 / 2)
    assert summary["sample_variance"] == exact_variance > 0.0
    assert summary["sample_variance"] != _whole_array_summary(values)["sample_variance"]
    assert summary["paired_vector_sha256"] == _whole_array_digest(values)


@pytest.mark.parametrize("shape", [None, (), [], "2", 2, (0,), (-1,), (2, 0), (True,), (np.bool_(True),), (1.0,), ("1",)])
def test_digest_rejects_invalid_declared_shapes(shape) -> None:
    with pytest.raises(RuntimeError, match="STREAM_SHAPE_INVALID"):
        StreamingArrayDigest(shape, "<f8", DOMAIN)


@pytest.mark.parametrize("shape", [(2**63,), (2**62, 4), (1, 2**63)])
def test_digest_rejects_declared_shape_or_byte_size_overflow(shape) -> None:
    with pytest.raises(RuntimeError, match="STREAM_SHAPE_OVERFLOW"):
        StreamingArrayDigest(shape, "<f8", DOMAIN)


@pytest.mark.parametrize("dtype", [None, "invalid", "O", "S8", "U8", "c16", "M8[ns]", "m8[ns]", "V8", [("value", "f8")], ("f8", (2,)), np.dtype("f8", metadata={"unit": "bps"})])
def test_digest_rejects_noncanonical_dtypes(dtype) -> None:
    with pytest.raises(RuntimeError, match="STREAM_DTYPE_INVALID"):
        StreamingArrayDigest((2,), dtype, DOMAIN)


@pytest.mark.parametrize("factory", [StreamingArrayDigest, StreamingPairedSummary])
@pytest.mark.parametrize("domain", [None, "text", 1, bytearray(b"domain")])
def test_streams_require_bytes_domain(factory, domain) -> None:
    arguments = ((2,), "f8", domain) if factory is StreamingArrayDigest else (2, domain)
    with pytest.raises(RuntimeError, match="STREAM_DOMAIN_INVALID"):
        factory(*arguments)


@pytest.mark.parametrize("count", [None, 0, -1, 1.0, "1", True, np.bool_(True), 2**63])
def test_summary_rejects_invalid_or_overflowing_expected_count(count) -> None:
    with pytest.raises(RuntimeError, match="STREAM_SHAPE_(INVALID|OVERFLOW)"):
        StreamingPairedSummary(count, DOMAIN)


@pytest.mark.parametrize("chunk", [[], [1.0], 1.0, np.array(1.0), np.array([]), np.ones((1, 1)), np.array(["1"]), np.array([1j]), np.array([object()]), np.ma.array([1.0], mask=[True])])
def test_summary_rejects_bad_chunk_types_shapes_and_dtypes(chunk) -> None:
    stream = StreamingPairedSummary(1, DOMAIN)
    with pytest.raises(RuntimeError, match="FEATURE_USEFULNESS_STREAM_"):
        stream.update(chunk)
    stream.update(np.array([1.0]))
    assert stream.finalize() == _whole_array_summary(np.array([1.0]))


@pytest.mark.parametrize("chunk", [np.ones((2,), dtype="f4"), np.ones((2,), dtype=">f8"), np.ones((1, 2)), np.ones((1, 1, 2)), np.ones((1, 0)), np.array([]), [1.0]])
def test_digest_rejects_mismatched_or_empty_chunks_without_consumption(chunk) -> None:
    stream = StreamingArrayDigest((2,), "<f8", DOMAIN)
    with pytest.raises(RuntimeError, match="FEATURE_USEFULNESS_STREAM_"):
        stream.update(chunk)
    values = np.array([1.0, 2.0], dtype="<f8")
    stream.update(values)
    assert stream.finalize() == _whole_array_digest(values)


def test_digest_rejects_wrong_trailing_shape_even_with_same_element_count() -> None:
    stream = StreamingArrayDigest((2, 3), "f8", DOMAIN)
    with pytest.raises(RuntimeError, match="STREAM_CHUNK_SHAPE_INVALID"):
        stream.update(np.ones((3, 2)))


@pytest.mark.parametrize("factory", [StreamingArrayDigest, StreamingPairedSummary])
def test_finalization_requires_exact_count_and_seals_updates(factory) -> None:
    values = np.array([1.0, 2.0])
    stream = (
        factory(values.shape, values.dtype, DOMAIN)
        if factory is StreamingArrayDigest else factory(values.size, DOMAIN)
    )
    with pytest.raises(RuntimeError, match="STREAM_COUNT_INCOMPLETE"):
        stream.finalize()
    stream.update(values[:1])
    with pytest.raises(RuntimeError, match="STREAM_COUNT_INCOMPLETE"):
        stream.finalize()
    with pytest.raises(RuntimeError, match="STREAM_COUNT_OVERFLOW"):
        stream.update(values)
    stream.update(values[1:])
    with pytest.raises(RuntimeError, match="STREAM_COUNT_OVERFLOW"):
        stream.update(values[:1])
    result = stream.finalize()
    expected = (
        _whole_array_digest(values)
        if factory is StreamingArrayDigest else _whole_array_summary(values)
    )
    assert result == expected
    assert stream.finalize() == expected
    with pytest.raises(RuntimeError, match="STREAM_ALREADY_FINALIZED"):
        stream.update(values[:1])
    if isinstance(result, dict):
        result["count"] = 99
        assert stream.finalize() == expected


@pytest.mark.parametrize("factory", [StreamingArrayDigest, StreamingPairedSummary])
@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_late_nonfinite_chunk_is_transactional(factory, bad_value) -> None:
    values = np.ones(_STREAM_BUFFER_BYTES // 8 + 1)
    stream = (
        factory(values.shape, values.dtype, DOMAIN)
        if factory is StreamingArrayDigest else factory(values.size, DOMAIN)
    )
    invalid = values.copy()
    invalid[-1] = bad_value
    with pytest.raises(RuntimeError, match="STREAM_NONFINITE"):
        stream.update(invalid)
    with pytest.raises(RuntimeError, match="STREAM_COUNT_INCOMPLETE"):
        stream.finalize()
    stream.update(values)
    expected = (
        _whole_array_digest(values)
        if factory is StreamingArrayDigest else _whole_array_summary(values)
    )
    assert stream.finalize() == expected


@pytest.mark.parametrize("values,widths", [
    (np.array([np.finfo(np.float64).max] * 2), (2,)),
    (np.array([np.finfo(np.float64).max] * 2), (1, 1)),
    (np.array([np.finfo(np.float64).max, -np.finfo(np.float64).max]), (2,)),
    (np.array([1e200, -1e200]), (2,)),
    (np.array([1e200, -1e200]), (1, 1)),
    (np.array([0.0, 1e154, -1e154]), (3,)),
])
def test_summary_rejects_sum_shift_square_or_moment_overflow(values, widths) -> None:
    with pytest.raises(RuntimeError, match="STREAM_REDUCTION_OVERFLOW"):
        _summarize(values, widths)


def test_late_reduction_overflow_does_not_publish_or_consume_earlier_blocks() -> None:
    values = np.zeros(_STREAM_BUFFER_BYTES // 8 + 2)
    stream = StreamingPairedSummary(values.size, DOMAIN)
    invalid = values.copy()
    invalid[-2:] = [1e200, -1e200]
    with pytest.raises(RuntimeError, match="STREAM_REDUCTION_OVERFLOW"):
        stream.update(invalid)
    with pytest.raises(RuntimeError, match="STREAM_COUNT_INCOMPLETE"):
        stream.finalize()
    stream.update(values)
    assert stream.finalize() == _whole_array_summary(values)


def test_noncontiguous_large_chunks_use_bounded_buffers_without_retention() -> None:
    storage = np.arange((_STREAM_BUFFER_BYTES // 8 + 3) * 4, dtype="f8")
    values = storage[::2][::-1]
    reference = _whole_array_summary(values)
    block_count = 0
    for block in _stream_array_blocks(values, dtype=np.dtype("<f8")):
        assert block.flags.c_contiguous
        assert 0 < block.nbytes <= _STREAM_BUFFER_BYTES
        block_count += 1
    assert block_count > 1
    del block
    stream = StreamingPairedSummary(values.size, DOMAIN)
    digest = StreamingArrayDigest(values.shape, values.dtype, DOMAIN)
    stream.update(values)
    digest.update(values)
    result = stream.finalize()
    assert result["paired_vector_sha256"] == digest.finalize()
    assert result["paired_vector_sha256"] == reference["paired_vector_sha256"]
    reference_to_values = weakref.ref(values)
    reference_to_storage = weakref.ref(storage)
    del values, storage
    gc.collect()
    assert reference_to_values() is None
    assert reference_to_storage() is None


def test_summary_finalization_calls_the_existing_validator(monkeypatch) -> None:
    import gx1.contracts.entry_exit_feature_usefulness_v1 as owner

    calls = []
    original = owner._require_summary

    def validate(value, *, count, label):
        calls.append((count, label))
        original(value, count=count, label=label)

    monkeypatch.setattr(owner, "_require_summary", validate)
    summary = _summarize(np.array([1.0, 2.0]), (2,))
    assert calls == [(2, "STREAMING")]
    assert summary["standard_error_method"] == STANDARD_ERROR_METHOD
