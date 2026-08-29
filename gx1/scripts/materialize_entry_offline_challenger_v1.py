"""Publish one immutable, review-only offline champion/challenger comparison.

This tool reads two already-materialized rolling-OOS result events.  It never
trains, loads a model, fetches market data, schedules work, or has activation
authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from gx1.contracts.entry_offline_challenger_v1 import (
    OfflineChampionChallengerError,
    publish_offline_challenger_comparison,
)


def _binding(path: str) -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise OfflineChampionChallengerError("candidate result path is unavailable")
    return {
        "json_path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion-result-json", required=True)
    parser.add_argument("--challenger-result-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--created-utc", default=None)
    args = parser.parse_args()
    path, report = publish_offline_challenger_comparison(
        out_dir=Path(args.out_dir),
        champion_result=_binding(args.champion_result_json),
        challenger_result=_binding(args.challenger_result_json),
        created_utc=args.created_utc or _utc_now(),
    )
    print(
        json.dumps(
            {
                "json_path": str(path),
                "decision": report["decision"],
                "activation_authority": report["activation_authority"],
                "promotion_allowed": report["promotion_allowed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
