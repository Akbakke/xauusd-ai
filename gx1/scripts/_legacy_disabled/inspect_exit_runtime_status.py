# ARCHIVED (NON-EXECUTABLE): kept for reference; wrapped to be syntactically parseable.
# # ARCHIVED (NON-EXECUTABLE): kept for reference; wrapped to be syntactically parseable.
# if False:
#     r"""
# #!/home/andre2/venvs/gx1/bin/python
# raise RuntimeError("LEGACY_DISABLED: Use gx1/scripts/README_TRUTH_XGB.md (TRUTH XGB lane only).")
# """
# Deterministic status dump for exit strategy of a GO run.
# Reads chunk_footer.json, run_header.json, and optionally exits_*.jsonl.
# Output: key=value lines; ends with VERDICT=OK_MASTER_ACTIVE | NOT_MASTER_ACTIVE | NO_EXITS_JSONL_FOUND.
# 
# Example (with LAST_GO.txt):
#   echo /home/andre2/GX1_DATA/reports/truth_e2e_sanity/E2E_SANITY_20260219_084126 \\
#     > /home/andre2/GX1_DATA/reports/truth_e2e_sanity/LAST_GO.txt
# 
#   /home/andre2/venvs/gx1/bin/python -m gx1.scripts.inspect_exit_runtime_status
# 
# Example (no LAST_GO dependency):
#   /home/andre2/venvs/gx1/bin/python -m gx1.scripts.inspect_exit_runtime_status \\
#     --run-dir /home/andre2/GX1_DATA/reports/truth_e2e_sanity/E2E_SANITY_20260219_084126
# """
# 
# from __future__ import annotations
# 
# import argparse
# import json
# import sys
# from collections import Counter
# from pathlib import Path
# 
# LAST_GO_DEFAULT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/LAST_GO.txt")
# MAX_JSONL_LINES = 50_000
# 
# 
# def _resolve_run_dir(args_run_dir: str | None) -> Path:
#     if args_run_dir:
#         return Path(args_run_dir).expanduser().resolve()
#     if not LAST_GO_DEFAULT.exists():
#         print(f"Missing default run-dir file: {LAST_GO_DEFAULT}. Use --run-dir <path>.", file=sys.stderr)
#         sys.exit(1)
#     path = Path(LAST_GO_DEFAULT.read_text().strip()).expanduser().resolve()
#     if not path.exists():
#         print(f"Run dir from LAST_GO.txt does not exist: {path}", file=sys.stderr)
#         sys.exit(1)
#     return path
# 
# 
# def _load_json(path: Path) -> dict:
#     if not path.exists():
#         print(f"Missing: {path}", file=sys.stderr)
#         sys.exit(1)
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)
# 
# 
# def _reason_from_record(rec: dict) -> str:
#     v = rec.get("reason") or rec.get("close_reason") or rec.get("exit_reason") or ""
#     return str(v) if v is not None else ""
# 
# 
# def _source_from_record(rec: dict) -> str:
#     return rec.get("source") or ""
# 
# 
# def _master_exit_seen(reason: str, source: str) -> bool:
#     if "MASTER_" in reason or "MASTER" in reason:
#         return True
#     if "MASTER" in source or "EXIT_MASTER" in source:
#         return True
#     return False
# 
# 
# def main() -> int:
#     ap = argparse.ArgumentParser(description="Inspect exit runtime status for a GO run.")
#     ap.add_argument("--run-dir", type=str, default=None, help="Override run dir (default: read LAST_GO.txt)")
#     args = ap.parse_args()
# 
#     run_dir = _resolve_run_dir(args.run_dir)
#     chunk_dir = run_dir / "replay" / "chunk_0"
#     footer_path = chunk_dir / "chunk_footer.json"
#     header_path = chunk_dir / "run_header.json"
# 
#     footer = _load_json(footer_path)
#     header = _load_json(header_path)
# 
#     run_id = footer.get("run_id", "")
#     exit_type = footer.get("exit_type", "")
#     exit_profile = footer.get("exit_profile", "")
#     router_enabled = footer.get("router_enabled")
#     exit_critic_enabled = footer.get("exit_critic_enabled")
#     policy_path_footer = footer.get("policy_path")
#     policy_path = policy_path_footer
#     if policy_path is None or policy_path == "":
#         policy_path = header.get("policy_path", "")
#     artifacts = header.get("artifacts") or {}
#     policy_artifact = artifacts.get("policy") or {}
#     artifacts_policy_path = policy_artifact.get("path", "")
#     bars_processed = footer.get("bars_processed")
#     bars_evaluated = footer.get("bars_evaluated")
# 
#     print(f"RUN_DIR={run_dir}")
#     print(f"FOOTER_RUN_ID={run_id}")
#     print(f"FOOTER_EXIT_TYPE={exit_type}")
#     print(f"FOOTER_EXIT_PROFILE={exit_profile}")
#     print(f"FOOTER_ROUTER_ENABLED={router_enabled}")
#     print(f"FOOTER_EXIT_CRITIC_ENABLED={exit_critic_enabled}")
#     print(f"POLICY_PATH={policy_path}")
#     print(f"HEADER_POLICY_PATH={header.get('policy_path', '')}")
#     print(f"HEADER_ARTIFACTS_POLICY_PATH={artifacts_policy_path}")
#     if bars_processed is not None:
#         print(f"FOOTER_BARS_PROCESSED={bars_processed}")
#     if bars_evaluated is not None:
#         print(f"FOOTER_BARS_EVALUATED={bars_evaluated}")
#     capsule = footer.get("exit_tuning_capsule")
#     if capsule is not None and isinstance(capsule, dict):
#         print("EXIT_TUNING_CAPSULE=" + json.dumps(capsule, sort_keys=True))
# 
#     allow_ctx_ml = True
#     if capsule is not None and isinstance(capsule, dict):
#         allow_ctx_ml = bool(capsule.get("allow_exit_ml_ctx_per_bar", False))
#     ml_source = (capsule or {}).get("ml_source") if isinstance(capsule, dict) else None
#     if exit_type == "MASTER_EXIT_V1" and allow_ctx_ml is False and ml_source == "ctx":
#         print("VERDICT=CTX_ML_USED_BUT_NOT_ALLOWED")
#         print("DEVIATIONS=ctx_ml_used_but_allow_exit_ml_ctx_per_bar_false", file=sys.stderr)
#         return 1
# 
#     exits_dir = chunk_dir / "logs" / "exits"
#     jsonl_files = sorted(exits_dir.glob("exits_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True) if exits_dir.exists() else []
#     if not jsonl_files:
#         print("NO_EXITS_JSONL_FOUND")
#         print("VERDICT=NO_EXITS_JSONL_FOUND")
#         return 0
# 
#     jsonl_path = jsonl_files[0]
#     print("EXITS_JSONL_PATH=" + str(jsonl_path))
#     source_counts: Counter = Counter()
#     reason_counts: Counter = Counter()
#     master_seen = False
#     line_count = 0
#     with open(jsonl_path, "r", encoding="utf-8") as f:
#         for line in f:
#             if line_count >= MAX_JSONL_LINES:
#                 break
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 rec = json.loads(line)
#             except json.JSONDecodeError:
#                 continue
#             reason = _reason_from_record(rec)
#             source = _source_from_record(rec)
#             if source:
#                 source_counts[source] += 1
#             if reason:
#                 reason_counts[reason] += 1
#             if _master_exit_seen(reason, source):
#                 master_seen = True
#             line_count += 1
# 
#     print("EXITS_JSONL_LINES_READ=" + str(line_count))
#     print("TOP20_SOURCE_COUNTS=" + json.dumps(dict(source_counts.most_common(20))))
#     print("TOP20_REASON_COUNTS=" + json.dumps(dict(reason_counts.most_common(20))))
#     print("MASTER_EXIT_V1_SEEN=" + str(master_seen).lower())
# 
#     ok_master = (
#         exit_type == "MASTER_EXIT_V1"
#         and router_enabled is False
#         and exit_critic_enabled is False
#     )
#     if ok_master:
#         print("VERDICT=OK_MASTER_ACTIVE")
#         return 0
#     deviators = []
#     if exit_type != "MASTER_EXIT_V1":
#         deviators.append("exit_type=" + str(exit_type))
#     if router_enabled is not False:
#         deviators.append("router_enabled=" + str(router_enabled))
#     if exit_critic_enabled is not False:
#         deviators.append("exit_critic_enabled=" + str(exit_critic_enabled))
#     print("VERDICT=NOT_MASTER_ACTIVE")
#     print("DEVIATIONS=" + ";".join(deviators), file=sys.stderr)
#     return 1
# 
# 
# if __name__ == "__main__":
#     sys.exit(main())
# 
#     """
