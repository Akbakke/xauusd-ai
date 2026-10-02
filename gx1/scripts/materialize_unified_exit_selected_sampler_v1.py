"""Materialize the immutable benchmark-selected TRAIN sampler admission."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from gx1.scripts.benchmark_unified_exit_random_access_train_v1 import _atomic_write_new_json
from gx1.contracts.unified_exit_selected_sampler_v1 import (
    build_selected_sampler_artifact,
    require_selected_sampler_artifact,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-receipt", type=Path, required=True)
    parser.add_argument("--candidate-set", type=Path, required=True)
    parser.add_argument("--random-access-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--equivalence-receipt", type=Path,
                      help="Preserve the historical V3-to-V4 transfer route.")
    mode.add_argument("--direct-benchmark", action="store_true",
                      help="Admit a fresh measured benchmark on this exact full-TRAIN root.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.expanduser().absolute()
    artifact = build_selected_sampler_artifact(
        benchmark_receipt_path=args.benchmark_receipt.expanduser().resolve(),
        candidate_set_path=args.candidate_set.expanduser().resolve(),
        random_access_root_path=args.random_access_root.expanduser().resolve(),
        equivalence_receipt_path=(args.equivalence_receipt.expanduser().resolve()
                                 if args.equivalence_receipt is not None else None),
    )
    require_selected_sampler_artifact(artifact, verify_files=True)
    if output.exists() or output.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_OUTPUT_EXISTS")
    if args.publish:
        _atomic_write_new_json(output, artifact)
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
