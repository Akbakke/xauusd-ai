"""Materialize immutable GPU batch selection from three finalized smoke receipts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from gx1.contracts.unified_exit_gpu_batch_selection_v1 import (
    build_selection,
    file_sha256,
    require_arm_receipt,
    require_selection,
)


def materialize(*, receipt_paths: list[Path], output: Path, publish: bool) -> dict:
    if len(receipt_paths) != 3 or output.exists() or output.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_OUTPUT_INVALID")
    receipt_bindings = []
    for path in receipt_paths:
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise RuntimeError("UNIFIED_EXIT_GPU_SELECTION_RECEIPT_PATH_INVALID")
        require_arm_receipt(json.loads(path.read_text()))
        receipt_bindings.append({"path": str(path), "sha256": file_sha256(path)})
    value = require_selection(build_selection(receipt_bindings))
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
            dir_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return {
        "decision": "PASS",
        "published": publish,
        "output": str(output),
        "artifact_sha256": value["artifact_sha256"],
        "file_sha256": file_sha256(output) if publish else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            materialize(
                receipt_paths=[p.resolve() for p in args.arm_receipt],
                output=args.output.resolve(),
                publish=args.publish,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
