"""Materialize the immutable benchmark-selected TRAIN sampler admission."""

from __future__ import annotations
import argparse
import json
import os
import tempfile
from pathlib import Path
from gx1.contracts.unified_exit_selected_sampler_v1 import (
    build_selected_sampler_artifact,
    require_selected_sampler_artifact,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-receipt", type=Path, required=True)
    parser.add_argument("--candidate-set", type=Path, required=True)
    parser.add_argument("--random-access-root", type=Path, required=True)
    parser.add_argument("--equivalence-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    artifact = build_selected_sampler_artifact(
        benchmark_receipt_path=args.benchmark_receipt.expanduser().resolve(),
        candidate_set_path=args.candidate_set.expanduser().resolve(),
        random_access_root_path=args.random_access_root.expanduser().resolve(),
        equivalence_receipt_path=args.equivalence_receipt.expanduser().resolve(),
    )
    require_selected_sampler_artifact(artifact, verify_files=True)
    if output.exists() or output.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_OUTPUT_EXISTS")
    if args.publish:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode()
        fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    print(
        json.dumps(
            {
                "decision": "PASS",
                "published": bool(args.publish),
                "output": str(output),
                "artifact_sha256": artifact["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
