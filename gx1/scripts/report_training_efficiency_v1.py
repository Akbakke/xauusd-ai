"""Read explicit pre-TEST smoke evidence and publish a non-authorizing report.

Never imports the trainer, loads tensors, opens datasets, or launches CUDA.
Historical logs provide coarse/cold measurements only; absent measurements
are named as missing instead of being replaced with estimates or zeroes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    publish_bundle_directory_noreplace,
    require_bundle_commit_manifest,
)


class EfficiencyEvidenceError(ValueError):
    """Incomplete or mismatched evidence cannot be used as a baseline."""


def _read_bound(path: Path, expected: str) -> tuple[bytes, dict[str, str]]:
    if (not path.is_absolute() or path.is_symlink()
            or path.resolve(strict=True) != path or not path.is_file()
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None):
        raise EfficiencyEvidenceError("expected canonical file and explicit SHA-256")
    data = path.read_bytes()
    observed = hashlib.sha256(data).hexdigest()
    if observed != expected:
        raise EfficiencyEvidenceError(f"SHA-256 mismatch: {path}")
    return data, {"path": str(path), "sha256": observed}


def _kv(line: str) -> dict[str, str]:
    return dict(re.findall(r"([a-zA-Z_][a-zA-Z_0-9]*)=([^\s]+)", line))


def _one(lines: list[str], marker: str) -> str:
    matches = [line for line in lines if marker in line]
    if len(matches) != 1:
        raise EfficiencyEvidenceError(f"expected exactly one {marker}: got {len(matches)}")
    return matches[0]


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise EfficiencyEvidenceError(f"invalid {name}")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise EfficiencyEvidenceError(f"invalid {name}")
    return result


def _stamp(line: str, *, guard: bool = False) -> datetime:
    if guard:
        return datetime.fromisoformat(line.split()[0].replace("Z", "+00:00"))
    return datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=timezone.utc)


def parse_measurements(trainer_text: str, guard_text: str, *, rows: int,
                       batch_size: int) -> dict[str, Any]:
    """Pure parsing; fixture tests prove arithmetic and rejection, not speed."""
    lines, guard = trainer_text.splitlines(), guard_text.splitlines()
    start = _one(guard, "event=start ")
    end = _one(guard, "event=exit ")
    terminal = _kv(end)
    if terminal.get("child_status") != "0" or any(
        token in guard_text for token in ("event=stop ", "event=fatal ")
    ):
        raise EfficiencyEvidenceError("guard did not finish successfully")
    elapsed = _positive((_stamp(end, guard=True) - _stamp(start, guard=True)).total_seconds(), "guard wall time")
    epoch_start = _one(lines, "[TRAIN_RSS] epoch_start ")
    epoch_end = _one(lines, "[TRAIN_LR_SCHEDULE] epoch=1 ")
    seconds = _positive((_stamp(epoch_end) - _stamp(epoch_start)).total_seconds(), "TRAIN log window")
    steps = math.ceil(rows / batch_size)
    profile = _kv(_one(lines, "[TRAIN_PROFILE] "))
    exit_profile = _kv(_one(lines, "[UNIFIED_EXIT_PROFILE] "))
    total = _positive(profile["total_s"], "first batch compute time")
    if profile.get("batch") != "1":
        raise EfficiencyEvidenceError("cold profile must describe first batch")
    rss = [float(x) for x in re.findall(r"rss_gib=([0-9.]+)", trainer_text)]
    if not rss or any(not math.isfinite(x) or x < 0 for x in rss):
        raise EfficiencyEvidenceError("RSS observations missing or invalid")
    telemetry = [_kv(line) for line in guard if "power_limit_w=" in line
                 and ("event=telemetry " in line or "event=heartbeat " in line)]
    if not telemetry or any(x.get("memory_observed") != "true" for x in telemetry):
        raise EfficiencyEvidenceError("observed memory telemetry missing")
    limits = sorted({_positive(x["power_limit_w"], "power limit") for x in telemetry})
    compute = {k: float(profile[k]) for k in (
        "entry_online_forward_s", "entry_target_forward_s", "exit_train_s",
        "post_exit_backward_s", "total_s", "peak_cuda_mib")}
    if any(not math.isfinite(v) or v < 0 for v in compute.values()):
        raise EfficiencyEvidenceError("invalid cold profile")
    if abs(sum(compute[k] for k in ("entry_online_forward_s", "entry_target_forward_s", "exit_train_s", "post_exit_backward_s")) - total) > 0.000005:
        # Four components and total are printed to six decimals in the owner.
        raise EfficiencyEvidenceError("cold profile phases do not sum to total")
    measured = {
        "guard_run": {
            "wall_seconds": elapsed,
            "telemetry_samples": int(_positive(terminal["telemetry_samples"], "telemetry samples")),
            "core_peak_c": _positive(terminal["peak_core_temp_c"], "core peak"),
            "memory_junction_peak_c": _positive(terminal["peak_memory_temp_c"], "memory peak"),
            "power_draw_peak_w": _positive(terminal["peak_power_draw_w"], "power peak"),
            "gpu_resident_peak_mib": _positive(terminal["peak_memory_used_mib"], "resident peak"),
            "observed_configured_power_limits_w": limits,
        },
        "train_log_window": {
            "evidence_class": "derived_from_real_timestamped_logs_including_warmup",
            "rows": rows, "optimizer_steps": steps, "wall_seconds": seconds,
            "samples_per_second": rows / seconds,
            "optimizer_steps_per_second": steps / seconds,
            "milliseconds_per_batch": 1000 * seconds / steps,
            "includes_warmup": True, "timing_resolution_seconds": 0.001,
            "scope": "epoch_start_to_scheduler_log_includes_final_bookkeeping",
        },
        "cold_first_batch": {
            **compute,
            "exit_share_of_compute": compute["exit_train_s"] / total,
            "exit_materialization_share_of_compute": float(exit_profile["materialize_s"]) / total,
            "exit_breakdown": {k: float(v) for k, v in exit_profile.items()},
            "scope": "one_cold_batch_excludes_initial_loader_and_initial_H2D",
            "post_exit_backward_includes": ["Entry_losses", "main_backward", "gradient_checks", "clipping", "AdamW", "EMA", "bookkeeping"],
        },
        "rss_sampled_max_gib": max(rss),
        "rss_scope": "maximum_of_sparse_process_RSS_logs_not_process_or_cgroup_peak",
    }
    for marker, key in (("TRAIN_EFFICIENCY_WINDOW", "post_warmup_train_window"),
                        ("TRAIN_EFFICIENCY_BATCH", "cold_batch_detail"),
                        ("TRAIN_EFFICIENCY_NUMERICS", "numerics"),
                        ("TRAIN_EFFICIENCY_VAL", "validation"),
                        ("TRAIN_EFFICIENCY_CHECKPOINT", "checkpoint")):
        matching = [line for line in lines if f"[{marker}] " in line]
        if matching:
            if len(matching) != 1:
                raise EfficiencyEvidenceError(f"multiple {marker} records")
            value = json.loads(matching[0].split(f"[{marker}] ", 1)[1])
            if value.get("report_only") is not True:
                raise EfficiencyEvidenceError(f"invalid {marker} report boundary")
            if key == "post_warmup_train_window":
                elapsed = _positive(value["measured_train_seconds"], "measured train time")
                n = _positive(value["measured_train_rows"], "measured rows")
                count = _positive(value["measured_optimizer_steps"], "measured steps")
                if count + int(value["warmup_optimizer_steps"]) != steps or n > rows:
                    raise EfficiencyEvidenceError("measurement geometry differs from recipe")
                value.update(samples_per_second=n / elapsed, optimizer_steps_per_second=count / elapsed)
            measured[key] = value
    return measured


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    recipe_bytes, recipe_binding = _read_bound(Path(args.recipe), args.recipe_sha256)
    trainer_bytes, trainer_binding = _read_bound(Path(args.trainer_log), args.trainer_log_sha256)
    guard_bytes, guard_binding = _read_bound(Path(args.guard_log), args.guard_log_sha256)
    recipe = json.loads(recipe_bytes)
    bundle = Path(args.bundle)
    if recipe.get("profile") != "smoke" or recipe.get("out_bundle_dir") != str(bundle):
        raise EfficiencyEvidenceError("expected exact smoke recipe output")
    cli = recipe["trainer_cli"]
    if cli["epochs"] != 1 or cli["grad_accum_steps"] != 1 or cli["device"] != "cuda":
        raise EfficiencyEvidenceError("requires one-epoch CUDA smoke without accumulation")
    if cli["execution_tier"] != "canonical":
        raise EfficiencyEvidenceError("requires canonical guarded smoke")
    for path, kind in ((Path(args.trainer_log), "trainer"), (Path(args.guard_log), "guard")):
        if path.parent != bundle.parent or not path.name.startswith(f".{bundle.name}.{kind}."):
            raise EfficiencyEvidenceError("logs must be exact bundle-adjacent sidecars")
    manifest = require_bundle_commit_manifest(bundle)
    if manifest["commit_sha256"] != args.bundle_commit_sha256:
        raise EfficiencyEvidenceError("bundle commit differs from explicit expected identity")
    meta = json.loads((bundle / "bundle_metadata.json").read_text())
    lock = json.loads((bundle / "MASTER_TRANSFORMER_LOCK.json").read_text())
    provenance = meta["recipe_source_provenance"]
    if (provenance != lock["recipe_source_provenance"]
            or provenance["recipe_audit_path"] != recipe_binding["path"]
            or provenance["recipe_audit_sha256"] != recipe_binding["sha256"]
            or provenance["source_commit"] != recipe["source_commit"]):
        raise EfficiencyEvidenceError("recipe/source provenance mismatch")
    lineage = meta["run_lineage"]
    if (lineage != lock["run_lineage"] or lineage["training_profile"] != "smoke"
            or lineage["training_run_id"] != recipe["run_id"]
            or lineage["dataset_run_id"] != recipe["dataset_run_id"]):
        raise EfficiencyEvidenceError("run lineage mismatch")
    train = trainer_bytes.decode()
    if f"[DONE] immutable bundle atomically published: {bundle}" not in train:
        raise EfficiencyEvidenceError("trainer log does not prove publication of exact bundle")
    measurements = parse_measurements(train, guard_bytes.decode(),
        rows=int(lineage["effective_train_rows"]), batch_size=int(cli["batch_size"]))
    missing = ["GPU_utilization_over_time", "time_weighted_power_draw_and_J_per_sample",
               "CPU_utilization", "process_and_cgroup_peak_RAM", "steady_loader_wait",
               "complete_H2D", "attention_FFN_input_projection_timings",
               "full_epoch_and_full_VAL_ETA", "variant_VAL_parity", "variant_resume_proof"]
    if "post_warmup_train_window" not in measurements:
        missing.append("synchronized_post_warmup_throughput")
    if "validation" not in measurements:
        missing.append("synchronized_VAL_time")
    if "checkpoint" not in measurements:
        missing.append("fsync_checkpoint_write_time")
    return {
        "schema_version": "gx1_training_efficiency_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "INCOMPLETE_BASELINE_NOT_OPTIMIZATION_QUALIFIED",
        "report_only": True, "activation_authority": False, "test_accessed": False,
        "authority": {key: False for key in ("training", "test", "promotion", "paper", "live", "cloud_purchase")},
        "source_commit_of_measurement": recipe["source_commit"],
        "inputs": {"recipe": recipe_binding, "trainer_log": trainer_binding,
                   "guard_log": guard_binding, "bundle_commit_sha256": manifest["commit_sha256"]},
        "configuration": cli,
        "population": lineage,
        "model_inputs": {key: meta[key] for key in ("seq_len", "seq_input_dim", "ctx_cont_dim", "ctx_cat_dim")},
        "feature_groups": meta["specialist_fusion"]["group_feature_counts"],
        "measurements": measurements,
        "missing_measurements": missing,
        "limits": ["Historical source does not qualify current dirty worktree.",
                   "Sparse guard heartbeats cannot establish average TRAIN power or utilization.",
                   "Cold first-batch timings cannot establish a steady-state bottleneck.",
                   "Smoke sampling changes the epoch-derived EMA horizon; this is not candidate VAL parity.",
                   "No full-epoch ETA is emitted from this short sample.",
                   "Sidecar hashes preserve the supplied logs; logs were not included in the historical bundle commit."],
    }


def render_markdown(report: dict[str, Any]) -> str:
    m = report["measurements"]
    rows = [("TRAIN rows", m["train_log_window"]["rows"]),
            ("Optimizer steps", m["train_log_window"]["optimizer_steps"]),
            ("TRAIN log window seconds (includes warmup)", m["train_log_window"]["wall_seconds"]),
            ("Samples/s (includes warmup)", m["train_log_window"]["samples_per_second"]),
            ("Optimizer steps/s (includes warmup)", m["train_log_window"]["optimizer_steps_per_second"]),
            ("GPU resident peak MiB", m["guard_run"]["gpu_resident_peak_mib"]),
            ("Sparse RSS maximum GiB", m["rss_sampled_max_gib"]),
            ("Power peak W", m["guard_run"]["power_draw_peak_w"]),
            ("Core peak C", m["guard_run"]["core_peak_c"]),
            ("Memory junction peak C", m["guard_run"]["memory_junction_peak_c"])]
    return ("# GX1 training efficiency evidence\n\n"
        + report["decision"] + "\n\nSource: `" + report["source_commit_of_measurement"] + "`.\n\n"
        + "| Measurement | Value |\n| --- | ---: |\n"
        + "".join(f"| {key} | {value:.6g} |\n" for key, value in rows)
        + "\nMissing measurements:\n\n" + "".join(f"- {x}\n" for x in report["missing_measurements"])
        + "\nEvidence limits:\n\n" + "".join(f"- {x}\n" for x in report["limits"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "recipe-sha256", "trainer-log", "trainer-log-sha256",
                 "guard-log", "guard-log-sha256", "bundle", "bundle-commit-sha256", "output-dir"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    report = build_report(args)
    output = Path(args.output_dir)
    if (not output.is_absolute() or output.exists() or output.is_symlink()
            or output.parent.resolve(strict=True) != output.parent):
        raise EfficiencyEvidenceError("output must be a new canonical directory")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    for name, content in (("report.json", json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"),
                          ("report.md", render_markdown(report))):
        with (stage / name).open("x", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    publish_bundle_directory_noreplace(stage, output)
    print(output / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
