#!/usr/bin/env python3
"""Fit the six immutable TRAIN-only volatility-squeeze clock artifacts.

This command performs no dataset or model build.  M1 and M5 come from their
exact authoritative closed-OHLCV parquets; M15/H1/H4/D1 are aggregated once
from native M5 with the production resample owner, then fitted independently.
Every source and lineage artifact is supplied with a predeclared file hash.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_artifact(path: Path, expected_sha256: str, *, label: str) -> Path:
    artifact = path.expanduser()
    if (
        not artifact.is_absolute()
        or artifact.is_symlink()
        or not artifact.is_file()
        or artifact.resolve(strict=True) != artifact
        or len(expected_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in expected_sha256)
        or _sha256_file(artifact) != expected_sha256
    ):
        raise RuntimeError(f"{label}_IDENTITY_INVALID")
    return artifact


def _read_ohlcv(path: Path) -> pd.DataFrame:
    required = ["time", "open", "high", "low", "close", "volume"]
    frame = pd.read_parquet(path, columns=required)
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    # Preserve source order. The owner must reject a duplicate/reversed tape;
    # silently sorting would turn malformed provenance into a different input.
    return frame.set_index("time")



def native_m5_fit_frames_from_plan(
    plan_path: Path, *, expected_sha256: str,
) -> tuple[dict, dict, dict]:
    """Build the five fit frames from physically verified native M5 bytes.

    The caller owns the capped one-use policy lifecycle. This producer checks
    the exact raw source and reconstructs every fitted frame itself.
    """
    import json
    import pyarrow.parquet as pq
    from gx1.features.htf_features import _resample_ohlcv, multi_tf_resample
    from gx1.features.volatility_squeeze_state_v1 import (
        VOLATILITY_SQUEEZE_DIRECTION_CLOCKS as clocks,
        VOLATILITY_SQUEEZE_DIRECTION_CLOCK_CONTRACT as clock_contract,
        _require_native_m5_provenance, _ohlcv_sha256,
        volatility_squeeze_bar_grid, NATIVE_M5_PARAMETER_TRAIN_END_NS,
    )
    checked_plan = _require_artifact(
        Path(plan_path), expected_sha256, label="NATIVE_M5_SQUEEZE_PLAN")
    plan = json.loads(checked_plan.read_text())
    native = plan["native_m5_manifest"]
    cutoff = plan["fit_cutoff_ns"]
    start = plan["fit_start_ns"]
    if (type(start) is not int or type(cutoff) is not int or start >= cutoff
            or cutoff > NATIVE_M5_PARAMETER_TRAIN_END_NS
            or plan["timeframes"] != list(clocks)
            or plan["source_spec"].get("cutoff_ns") != cutoff):
        raise RuntimeError("NATIVE_M5_SQUEEZE_FIT_SCOPE")
    provenance = {
        clock: {
            "source_artifact": native["path"], "source_sha256": native["sha256"],
            "source_schema_version": "xau_canonical_native_source_v3",
            "source_lane": clock, "native_m5_manifest_artifact": native["path"],
            "native_m5_manifest_sha256": native["sha256"],
            "fit_plan_artifact": str(checked_plan),
            "fit_plan_sha256": expected_sha256, "train_split_id": "TRAIN",
            "clock_contract": clock_contract,
            "bar_grid": volatility_squeeze_bar_grid(clock),
        } for clock in clocks
    }
    _require_native_m5_provenance(
        provenance["M5"], timeframe="M5", fit_cutoff_ns=cutoff, fit_start_ns=start)
    pieces = []
    decoded = 0
    for part in plan["source_spec"]["parts"]:
        source = _require_artifact(
            Path(part["path"]), part["sha256"], label="NATIVE_M5_SQUEEZE_PART")
        before = source.stat()
        parquet = pq.ParquetFile(source)
        names = parquet.schema_arrow.names
        column = names.index("time")
        for group in part["row_groups"]:
            stats = parquet.metadata.row_group(group).column(column).statistics
            if (stats is None or not stats.has_min_max
                    or pd.Timestamp(stats.min).value + 300_000_000_000 >= cutoff
                    or pd.Timestamp(stats.max).value >= pd.Timestamp("2026-07-01T00:00:00Z").value):
                raise RuntimeError("NATIVE_M5_SQUEEZE_ROW_GROUP_SCOPE")
            frame = parquet.read_row_group(
                group, columns=["time", "open", "high", "low", "close", "volume"],
                use_threads=False).to_pandas()
            frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
            decoded += len(frame)
            frame = frame.set_index("time")
            pieces.append(frame.loc[
                (frame.index >= pd.Timestamp(start, tz="UTC"))
                & (frame.index + pd.Timedelta(minutes=5) < pd.Timestamp(cutoff, tz="UTC"))])
        if source.stat() != before:
            raise RuntimeError("NATIVE_M5_SQUEEZE_SOURCE_CHANGED")
    if not pieces:
        raise RuntimeError("NATIVE_M5_SQUEEZE_EMPTY")
    m5 = pd.concat(pieces)
    if not m5.index.is_monotonic_increasing or m5.index.has_duplicates:
        raise RuntimeError("NATIVE_M5_SQUEEZE_SOURCE_ORDER")
    frames = {"M5": m5}
    for clock in clocks[1:]:
        frame = _resample_ohlcv(m5, clock)
        # Empty periods retain volume=sum([])=0 in the OHLCV resampler.
        # Remove only bins with no actual source members. An observed invalid
        # price (even all four missing) remains for the strict fitter to reject.
        members = multi_tf_resample(m5, clock).size().reindex(frame.index)
        empty = frame[["open", "high", "low", "close"]].isna().all(axis=1) & members.eq(0)
        frame = frame.loc[~empty]
        duration = pd.Timedelta(volatility_squeeze_bar_grid(clock)["duration_ns"], unit="ns")
        frames[clock] = frame.loc[
            (frame.index >= pd.Timestamp(start, tz="UTC"))
            & (frame.index + duration < pd.Timestamp(cutoff, tz="UTC"))]
    proof = {
        "native_m5_manifest": native,
        "fit_plan": {"path": str(checked_plan), "sha256": expected_sha256},
        "decoded_native_m5_rows": decoded,
        "fit_frames": {
            clock: {"rows": len(frame), "ohlcv_sha256": _ohlcv_sha256(frame),
                    "first_bar_ns": int(frame.index[0].value),
                    "last_available_ns": int(frame.index[-1].value)
                        + volatility_squeeze_bar_grid(clock)["duration_ns"]}
            for clock, frame in frames.items()},
        "timeframes": list(clocks), "source_clock_contract": clock_contract,
    }
    return frames, provenance, proof


def fit_native_m5_artifacts_from_plan(
    plan_path: Path, *, expected_sha256: str, output_dir: Path,
) -> tuple[Path, dict]:
    """Fit fresh five-clock parameters with a raw-frame identity proof."""
    import json
    from gx1.features.volatility_squeeze_state_v1 import (
        VOLATILITY_SQUEEZE_DIRECTION_CLOCKS as clocks,
        fit_volatility_squeeze_artifact_manifest,
    )
    frames, provenance, proof = native_m5_fit_frames_from_plan(
        plan_path, expected_sha256=expected_sha256)
    plan = json.loads(Path(plan_path).read_text())
    start, cutoff = plan["fit_start_ns"], plan["fit_cutoff_ns"]
    manifest = fit_volatility_squeeze_artifact_manifest(
        frames, declared_train_window_start=pd.Timestamp(start, tz="UTC"),
        declared_train_window_end=pd.Timestamp(cutoff, tz="UTC"),
        source_provenance_by_clock=provenance, output_dir=output_dir, clocks=clocks)
    payload = json.loads(manifest.read_text())
    for clock in clocks:
        if (payload["artifacts"][clock]["train_ohlcv_rows"] != proof["fit_frames"][clock]["rows"]
                or payload["artifacts"][clock]["train_ohlcv_sha256"]
                    != proof["fit_frames"][clock]["ohlcv_sha256"]):
            raise RuntimeError("NATIVE_M5_SQUEEZE_PHYSICAL_FIT_IDENTITY")
    return manifest, proof

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for lane in ("m1", "m5"):
        parser.add_argument(f"--{lane}-source", type=Path, required=True)
        parser.add_argument(f"--{lane}-source-sha256", required=True)
    # tape + pair only. Both exist before this fit runs. A --split-manifest was
    # additionally required until 2026-08-15; the rebuild chain PRODUCES its
    # split manifests from the dataset this fit feeds, so the flag demanded an
    # artifact that cannot exist at fit time. The TRAIN-window invariant it was
    # supposed to carry is enforced by exact timestamp equality against the
    # chain's own --train-start/--train-end at contract-validation.
    for name in ("tape", "pair"):
        parser.add_argument(f"--{name}-manifest", type=Path, required=True)
        parser.add_argument(f"--{name}-manifest-sha256", required=True)
    parser.add_argument("--pair-generation-id", required=True)
    parser.add_argument("--train-window-start", required=True)
    parser.add_argument("--train-window-end", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    m1_path = _require_artifact(
        args.m1_source,
        args.m1_source_sha256,
        label="VOLATILITY_SQUEEZE_M1_SOURCE",
    )
    m5_path = _require_artifact(
        args.m5_source,
        args.m5_source_sha256,
        label="VOLATILITY_SQUEEZE_M5_SOURCE",
    )
    lineage_paths = {
        name: _require_artifact(
            getattr(args, f"{name}_manifest"),
            getattr(args, f"{name}_manifest_sha256"),
            label=f"VOLATILITY_SQUEEZE_{name.upper()}_MANIFEST",
        )
        for name in ("tape", "pair")
    }

    from gx1.features.htf_features import (
        _resample_ohlcv,
        build_multi_tf_v4_closed_timestamp_indices,
    )
    from gx1.features.volatility_squeeze_state_v1 import (
        VOLATILITY_SQUEEZE_CLOCKS,
        VOLATILITY_SQUEEZE_CLOCK_CONTRACT,
        fit_volatility_squeeze_artifact_manifest,
        volatility_squeeze_bar_grid,
    )

    m1 = _read_ohlcv(m1_path)
    m5 = _read_ohlcv(m5_path)
    expected_indices = build_multi_tf_v4_closed_timestamp_indices(m5.index)
    frames: dict[str, pd.DataFrame] = {"M1": m1, "M5": m5}
    for clock in VOLATILITY_SQUEEZE_CLOCKS[2:]:
        aggregated = _resample_ohlcv(m5, clock)
        expected = expected_indices[clock]
        if not expected.isin(aggregated.index).all():
            raise RuntimeError(
                f"VOLATILITY_SQUEEZE_{clock}_CLOSED_SOURCE_INCOMPLETE"
            )
        frames[clock] = aggregated.loc[expected]

    common = {
        "tape_manifest_artifact": str(lineage_paths["tape"]),
        "tape_manifest_sha256": args.tape_manifest_sha256,
        "pair_manifest_artifact": str(lineage_paths["pair"]),
        "pair_manifest_sha256": args.pair_manifest_sha256,
        "pair_generation_id": args.pair_generation_id,
        "pair_symbol": "XAUUSD",
        "train_split_id": "TRAIN",
        "clock_contract": VOLATILITY_SQUEEZE_CLOCK_CONTRACT,
    }
    provenance = {}
    for clock in VOLATILITY_SQUEEZE_CLOCKS:
        source_path = m1_path if clock == "M1" else m5_path
        source_sha = args.m1_source_sha256 if clock == "M1" else args.m5_source_sha256
        provenance[clock] = {
            **common,
            "source_artifact": str(source_path),
            "source_sha256": source_sha,
            "source_schema_version": (
                f"authoritative_closed_{clock.lower()}_ohlcv_v1"
            ),
            "source_lane": clock,
            "bar_grid": volatility_squeeze_bar_grid(clock),
        }

    manifest = fit_volatility_squeeze_artifact_manifest(
        frames,
        declared_train_window_start=args.train_window_start,
        declared_train_window_end=args.train_window_end,
        source_provenance_by_clock=provenance,
        output_dir=args.output_dir.expanduser().resolve(strict=True),
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
