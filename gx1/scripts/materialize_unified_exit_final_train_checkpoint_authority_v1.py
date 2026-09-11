"""Validate or atomically publish the final guarded TRAIN checkpoint authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gx1.contracts.unified_exit_final_train_checkpoint_authority_v1 import (
    build_final_train_checkpoint_authority,
    file_sha256,
    publish_final_train_checkpoint_authority,
    require_final_train_checkpoint_authority,
)


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-plan", type=Path, required=True)
    parser.add_argument("--campaign-receipt", type=Path, action="append", required=True)
    parser.add_argument("--gpu-selection", type=Path, required=True)
    parser.add_argument("--train-session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    paths = [args.campaign_plan, *args.campaign_receipt, args.gpu_selection, args.train_session]
    resolved = [path.resolve() for path in paths]
    if any(not path.is_file() or path.is_symlink() for path in resolved):
        raise RuntimeError("UNIFIED_EXIT_FINAL_TRAIN_INPUT_INVALID")
    plan = resolved[0]
    receipts = resolved[1 : 1 + len(args.campaign_receipt)]
    selection, session = resolved[-2:]
    value = require_final_train_checkpoint_authority(
        build_final_train_checkpoint_authority(
            campaign_plan_binding=_binding(plan),
            campaign_receipt_bindings=[_binding(path) for path in receipts],
            gpu_selection_binding=_binding(selection),
            train_session_binding=_binding(session),
        ),
        verify_files=True,
    )
    output = args.output.resolve()
    if args.publish:
        publish_final_train_checkpoint_authority(value, output)
    print(json.dumps({
        "decision": "PASS", "published": args.publish, "output": str(output),
        "authority_sha256": value["authority_sha256"],
        "file_sha256": file_sha256(output) if args.publish else None,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
