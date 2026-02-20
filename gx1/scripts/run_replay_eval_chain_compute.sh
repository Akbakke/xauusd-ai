#!/usr/bin/env bash
# Hygiene wrapper: required by ENV gate policy check (pre-commit).
# This script does NOT run the full replay eval chain (no silent pass / false green).
# To run the actual chain, use the TRUTH E2E sanity script or the appropriate pipeline.
set -euo pipefail
echo "[GX1] run_replay_eval_chain_compute.sh: hygiene wrapper only (no-op; not running replay eval chain)"
exit 0
