#!/home/andre2/venvs/gx1/bin/python
"""
Output expected ctx_cont columns (SSoT) and diff vs existing prebuilt schema.

Reads expected_ctx_cont_dim from the same source as PREBUILT_CTX_CONT_FAIL gate:
canonical truth -> transformer bundle -> bundle_metadata.json.
Expected columns = ORDERED_CTX_CONT_NAMES_EXTENDED[2:expected_ctx_cont_dim]
(slow-core columns that must be present in prebuilt parquet).

Writes:
  expected_ctx_cont_cols.txt  - one column name per line (contract order)
  missing_in_prebuilt.txt      - columns expected but not in prebuilt schema (if prebuilt given)

Usage:
  python -m gx1.scripts.expected_ctx_cont_cols --truth-file gx1/configs/canonical_truth_signal_only.json [--output-dir .]
  python -m gx1.scripts.expected_ctx_cont_cols --truth-file ... --prebuilt /path/to/prebuilt.parquet

No fallback: if bundle metadata missing or expected_ctx_cont_dim invalid, hard-fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from gx1.contracts.signal_bridge_v1 import ORDERED_CTX_CONT_NAMES_EXTENDED


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_expected_ctx_cont_cols_from_bundle(bundle_dir: Path) -> List[str]:
    """
    Return ORDERED_CTX_CONT_NAMES_EXTENDED[2:expected_ctx_cont_dim] from bundle_metadata.json.
    SSoT: same as runner/gate.
    """
    meta_path = bundle_dir / "bundle_metadata.json"
    if not meta_path.exists():
        raise RuntimeError(
            f"[expected_ctx_cont_cols] bundle_metadata.json not found: {meta_path}"
        )
    meta = _load_json(meta_path)
    dim = meta.get("expected_ctx_cont_dim")
    if dim is None:
        raise RuntimeError(
            f"[expected_ctx_cont_cols] expected_ctx_cont_dim missing in {meta_path}"
        )
    dim = int(dim)
    if dim not in (2, 4, 6):
        raise RuntimeError(
            f"[expected_ctx_cont_cols] expected_ctx_cont_dim={dim} not in (2,4,6)"
        )
    # Slow-core columns that must be in prebuilt: indices 2..dim-1
    return list(ORDERED_CTX_CONT_NAMES_EXTENDED[2:dim])


def get_prebuilt_columns(prebuilt_path: Path) -> List[str]:
    """Return column names of parquet (no data load)."""
    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(prebuilt_path)
        return [s.name for s in schema]
    except Exception:
        import pandas as pd
        df = pd.read_parquet(prebuilt_path, columns=[])
        return list(df.columns)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write expected ctx_cont columns (SSoT) and missing_in_prebuilt diff."
    )
    parser.add_argument(
        "--truth-file",
        type=Path,
        required=True,
        help="Path to canonical_truth_signal_only.json",
    )
    parser.add_argument(
        "--prebuilt",
        type=Path,
        default=None,
        help="Prebuilt parquet path (default: canonical_prebuilt_parquet from truth)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for expected_ctx_cont_cols.txt and missing_in_prebuilt.txt",
    )
    args = parser.parse_args()

    if not args.truth_file.exists():
        print(f"[expected_ctx_cont_cols] truth file not found: {args.truth_file}", file=sys.stderr)
        return 1

    truth = _load_json(args.truth_file)
    bundle_dir = Path(truth.get("canonical_transformer_bundle_dir", ""))
    if not bundle_dir or not bundle_dir.exists():
        print(
            f"[expected_ctx_cont_cols] canonical_transformer_bundle_dir missing or not found: {bundle_dir}",
            file=sys.stderr,
        )
        return 1

    expected = get_expected_ctx_cont_cols_from_bundle(bundle_dir)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_file = out_dir / "expected_ctx_cont_cols.txt"
    expected_file.write_text("\n".join(expected) + "\n", encoding="utf-8")
    print(f"Wrote {len(expected)} expected columns to {expected_file}")

    prebuilt_path = args.prebuilt or truth.get("canonical_prebuilt_parquet")
    if prebuilt_path:
        prebuilt_path = Path(prebuilt_path)
        if prebuilt_path.exists():
            prebuilt_cols = get_prebuilt_columns(prebuilt_path)
            prebuilt_set = set(prebuilt_cols)
            missing = [c for c in expected if c not in prebuilt_set]
            missing_file = out_dir / "missing_in_prebuilt.txt"
            missing_file.write_text("\n".join(missing) + "\n" if missing else "", encoding="utf-8")
            print(f"Wrote {len(missing)} missing columns to {missing_file}")
            if missing:
                print(f"  Missing: {missing}", file=sys.stderr)
                return 2
        else:
            print(
                f"[expected_ctx_cont_cols] prebuilt not found (skipping diff): {prebuilt_path}",
                file=sys.stderr,
            )
    else:
        print("[expected_ctx_cont_cols] no prebuilt path (truth or --prebuilt); skipping diff", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
