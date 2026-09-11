"""Validate or atomically publish fixed-step reference/resume equivalence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gx1.contracts.unified_exit_fixed_step_resume_equivalence_v1 import (
    build_equivalence,
    file_sha256,
    publish_equivalence,
    require_equivalence,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-pointer", type=Path, required=True)
    parser.add_argument("--split-pointer", type=Path, required=True)
    parser.add_argument("--reference-schedule", type=Path, required=True)
    parser.add_argument("--split-schedule", type=Path, required=True)
    parser.add_argument("--launch-manifest-sha256", required=True)
    parser.add_argument("--gpu-selection-sha256", required=True)
    parser.add_argument("--selected-sampler-sha256", required=True)
    parser.add_argument("--batch-size", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--epoch-schedule-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    value = require_equivalence(
        build_equivalence(
            reference_pointer_path=args.reference_pointer.resolve(),
            split_pointer_path=args.split_pointer.resolve(),
            reference_schedule_path=args.reference_schedule.resolve(),
            split_schedule_path=args.split_schedule.resolve(),
            expected_launch_manifest_sha256=args.launch_manifest_sha256,
            expected_gpu_batch_selection_artifact_sha256=args.gpu_selection_sha256,
            expected_selected_sampler_artifact_sha256=args.selected_sampler_sha256,
            expected_batch_size=args.batch_size,
            expected_epoch_schedule_sha256=args.epoch_schedule_sha256,
        )
    )
    output = args.output.resolve()
    if args.publish:
        publish_equivalence(value, output)
    print(
        json.dumps(
            {
                "decision": "PASS",
                "published": args.publish,
                "output": str(output),
                "artifact_sha256": value["artifact_sha256"],
                "file_sha256": file_sha256(output) if args.publish else None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
