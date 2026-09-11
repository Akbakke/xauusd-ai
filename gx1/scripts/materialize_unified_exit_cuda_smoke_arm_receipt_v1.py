"""Finalize one campaign-bound guarded CUDA smoke-arm receipt."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from gx1.contracts.unified_exit_gpu_batch_selection_v1 import (
    build_arm_receipt,
    file_sha256,
    require_arm_receipt,
)


def _binding(path: Path) -> dict[str, str]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_GPU_SMOKE_ARM_INPUT_INVALID")
    return {"path": str(path), "sha256": file_sha256(path)}


def materialize(
    *, plan: Path, invocation: Path, campaign_receipt: Path, measurement: Path,
    output: Path, publish: bool,
) -> dict:
    if output.exists() or output.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_GPU_SMOKE_ARM_OUTPUT_INVALID")
    value = require_arm_receipt(
        build_arm_receipt(
            campaign_plan_binding=_binding(plan),
            campaign_invocation_binding=_binding(invocation),
            campaign_receipt_binding=_binding(campaign_receipt),
            measurement_binding=_binding(measurement),
        ),
        verify_files=True,
    )
    if publish:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
        fd, tmp = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, output)
            directory_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return {
        "decision": "PASS", "published": publish, "output": str(output),
        "receipt_sha256": value["receipt_sha256"],
        "file_sha256": file_sha256(output) if publish else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-plan", type=Path, required=True)
    parser.add_argument("--campaign-invocation", type=Path, required=True)
    parser.add_argument("--campaign-receipt", type=Path, required=True)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(materialize(
        plan=args.campaign_plan.resolve(), invocation=args.campaign_invocation.resolve(),
        campaign_receipt=args.campaign_receipt.resolve(), measurement=args.measurement.resolve(),
        output=args.output.resolve(), publish=args.publish,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
