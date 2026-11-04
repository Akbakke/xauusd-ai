"""One capped, data-only publication for the 04 November GX1 fit prefix."""
import argparse
import json
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
from gx1.contracts.local_random_access_campaign_v2 import file_sha256, read_bound_json


REPO = Path("/home/andre2/src/GX1_CURRENT")
ROOT = Path("/home/andre2/GX1_RUNS/DYNAMIC_MTF_TRAIN_GATES_20261104_001/DATA_PREFIX_001")
STAGE = "dynamic_mtf_train_prefix_data_20261104_001"


def require(condition, reason):
    if not condition:
        raise RuntimeError("GX1_TRAIN_PREFIX_DATA_" + reason)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def publish(name, value, limit):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    existing = sum(p.stat().st_size for p in ROOT.glob("*.json"))
    require(existing + len(data) <= limit, "JSON_BUDGET")
    path = ROOT / name
    _publish_bytes(path, data)
    return {"path": str(path), "sha256": file_sha256(path)}


def frozen(binding):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()
    require(Path.cwd() == REPO and git("branch", "--show-current") == "work/gx1-current"
            and git("rev-parse", "HEAD") == binding["source_head"] and not git("status", "--porcelain"),
            "SOURCE_IDENTITY")
    require(file_sha256(REPO / "NEXT_RUN_POLICY.json") == binding["policy_sha256"], "POLICY_CHANGED")


def run(binding_sha256):
    require(ROOT.is_dir() and ROOT.resolve() == ROOT, "OUTPUT_ROOT")
    binding = read_bound_json(ROOT / "EXECUTION_BINDING.json", binding_sha256)
    require(set(binding) == {"stage", "source_head", "policy_sha256"} and binding["stage"] == STAGE, "BINDING")
    frozen(binding)
    policy = read_bound_json(REPO / "NEXT_RUN_POLICY.json", binding["policy_sha256"])
    stage = policy[STAGE]
    prereg = policy["current_work"]["dynamic_mtf_train_gates_20261104"]
    require(prereg["specification"] == stage["specification"]
            and file_sha256(REPO / stage["specification"]["path"]) == stage["specification"]["sha256"]
            and stage["target_result"] == policy["dynamic_mtf_state_target_audit_20261102_005"]["result"]
            and prereg["fit_parents"] == stage["prefix_rows"] == 29849
            and stage["fit_cutoff_ns"] == 1727740800000000000 and stage["run_root"] == str(ROOT), "SCOPE")
    require(stage["status"] == "ADMITTED_ONE_DATA_PUBLICATION_NO_FIT"
            and stage["max_invocations"] == 1 and stage["claimed_invocations"] == stage["completed_invocations"] == 0
            and stage["execution_authorized"] is True and stage["fit_target_view_authorized"] is True
            and all(stage[k] is False for k in ("normalization_fit_authorized", "model_calls_authorized",
                    "optimizer_updates_authorized", "training_enabled", "relaunch_allowed", "test_authorized",
                    "broker_authorized", "trading_authorized"))
            and all(policy[k] is False for k in ("training_enabled", "native_training_enabled",
                    "full_epoch_training_allowed", "full_val_allowed")), "ADMISSION")
    require({p.name for p in ROOT.iterdir()} == {"EXECUTION_BINDING.json"}, "CONSUMED")
    guard = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    require(all(guard[k] == stage["limits"][k] for k in ("memory_max", "swap", "pids", "numerical_threads")), "CAPS")
    budget = stage["limits"]["max_json_bytes"]
    claim = publish("CLAIM.json", {"binding_sha256": binding_sha256, "source_head": binding["source_head"],
                    "started_utc": stamp(), "guard": guard, "lock": lock}, budget)

    def deadline(*_args):
        raise TimeoutError("GX1_TRAIN_PREFIX_DATA_DEADLINE")
    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGTERM, deadline)
    signal.alarm(stage["limits"]["max_wall_seconds"])
    try:
        from gx1.audit.dynamic_mtf_train_prefix_inputs import review_train_gc_d1_donor, verify_train_prefix_input_support
        from gx1.audit.dynamic_mtf_train_prefix_targets import materialize_train_prefix_targets
        support = verify_train_prefix_input_support(policy)
        require(support["status"] == "BLOCK" and support["fit_allowed"] is False
                and support["statistics_fits"] == 0 and support["normalization"] is None, "SAVED_SUPPORT_ROLE")
        support["gc_d1_donor"] = review_train_gc_d1_donor(stage["gc_donor_terminal"], stage["fit_cutoff_ns"])
        support_ref = publish("PREFIX_INPUT_SUPPORT.json", support, budget)
        built = materialize_train_prefix_targets(target_result_ref=stage["target_result"],
            prefix_rows=stage["prefix_rows"], cutoff_ns=stage["fit_cutoff_ns"], output_dir=ROOT / "FIT_TARGETS")
        counts_ref = publish("FIT_COUNTS.json", built["counts"], budget)
        prior_ref = publish("FIT_PRIOR.json", built["prior"], budget)
        require(built["clock_support"]["parents"] == stage["prefix_rows"], "CLOCK_POPULATION")
        frozen(binding)
        require(require_capped_lock_ancestry() == lock, "LOCK_CHANGED")
        result = publish("RESULT.json", {"status": "COMPLETE_DATA_PORT_MODEL_BLOCKED",
            "claim": claim, "specification": stage["specification"], "source_head": binding["source_head"],
            "fit_prefix": stage["prefix_rows"], "fit_cutoff_ns": stage["fit_cutoff_ns"],
            "input_support": support_ref, "series": built["series"], "counts": counts_ref, "prior": prior_ref,
            "clock_support": built["clock_support"], "calendar_binding": built["calendar_binding"],
            "normalization_fits": 0, "model_calls": 0,
            "optimizer_updates": 0, "oof_predictions": 0, "test_data_used": False}, budget)
    except Exception as exc:
        publish("TERMINAL.json", {"status": "FAILED", "claim": claim, "finished_utc": stamp(),
                "exception": type(exc).__name__, "detail": str(exc)[:1200]}, budget)
        raise
    finally:
        signal.alarm(0)
    terminal = publish("TERMINAL.json", {"status": "COMPLETE", "claim": claim, "result": result,
                       "finished_utc": stamp()}, budget)
    print(json.dumps(terminal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding-sha256", required=True)
    run(parser.parse_args().binding_sha256)
