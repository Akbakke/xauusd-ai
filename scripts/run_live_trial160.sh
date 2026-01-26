#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# Trial 160 — LIVE Trading Runner
# Runs live trading with Trial 160 policy and hard monitoring

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"
test "$(pwd)" = "$ROOT" || { echo "❌ FAIL: Not in repo-root (pwd=$(pwd), root=$ROOT)"; exit 1; }

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== TRIAL 160 — LIVE TRADING ===${NC}"
echo ""

# [RUN_CTX] Header
GIT_HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "[RUN_CTX] root=$ROOT"
echo "[RUN_CTX] head=$GIT_HEAD"
echo "[RUN_CTX] whoami=$(whoami) host=$(hostname)"
echo ""

# Step 1: Doctor check (must pass)
echo -e "${BLUE}[LIVE_TRIAL160] Step 1: Doctor check...${NC}"
if ! ./scripts/doctor_trial160.sh; then
    echo -e "${RED}[LIVE_TRIAL160] ❌ Doctor check FAILED${NC}"
    exit 1
fi
echo -e "${GREEN}[LIVE_TRIAL160] ✅ Doctor check passed${NC}"
echo ""

# Step 2: Load policy and verify
echo -e "${BLUE}[LIVE_TRIAL160] Step 2: Load policy...${NC}"
POLICY_PATH="$ROOT/policies/sniper_trial160_prod.json"
POLICY_ID=$(python3 -c "import json; print(json.load(open('$POLICY_PATH'))['policy_id'])" 2>/dev/null || echo "")
POLICY_SHA=$(python3 -c "import hashlib; print(hashlib.sha256(open('$POLICY_PATH', 'rb').read()).hexdigest())" 2>/dev/null || echo "")

if [ -z "$POLICY_ID" ] || [ "$POLICY_ID" != "trial160_prod_v1" ]; then
    echo -e "${RED}[LIVE_TRIAL160] ❌ Policy ID mismatch${NC}"
    exit 1
fi

if [ -z "$POLICY_SHA" ]; then
    echo -e "${RED}[LIVE_TRIAL160] ❌ Failed to compute policy SHA256${NC}"
    exit 1
fi

echo -e "${GREEN}[LIVE_TRIAL160] ✅ Policy loaded: $POLICY_ID (SHA256: ${POLICY_SHA:0:16}...)${NC}"
echo ""

# Step 3: Hard checks for live trading
echo -e "${BLUE}[LIVE_TRIAL160] Step 3: Hard checks for live trading...${NC}"

if [ "${OANDA_ENV:-}" != "practice" ] && [ "${OANDA_ENV:-}" != "live" ]; then
    echo -e "${RED}[LIVE_TRIAL160] ❌ OANDA_ENV must be 'practice' or 'live'${NC}"
    exit 1
fi

if [ -z "${OANDA_API_TOKEN:-}" ]; then
    echo -e "${RED}[LIVE_TRIAL160] ❌ OANDA_API_TOKEN must be set${NC}"
    exit 1
fi

if [ -z "${OANDA_ACCOUNT_ID:-}" ]; then
    echo -e "${RED}[LIVE_TRIAL160] ❌ OANDA_ACCOUNT_ID must be set${NC}"
    exit 1
fi

if [ "${I_UNDERSTAND_LIVE_TRADING:-}" != "YES" ] && [ "${OANDA_ENV:-}" = "live" ]; then
    echo -e "${RED}[LIVE_TRIAL160] ❌ I_UNDERSTAND_LIVE_TRADING=YES required for live trading${NC}"
    exit 1
fi

echo -e "${GREEN}[LIVE_TRIAL160] ✅ Live trading checks passed${NC}"
echo ""

# Step 4: Set up output directory
echo -e "${BLUE}[LIVE_TRIAL160] Step 4: Set up output directory...${NC}"
RUN_TAG="TRIAL160_LIVE_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="runs/live_demo/${RUN_TAG}"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}[LIVE_TRIAL160] ✅ Output directory ready: $OUTPUT_DIR${NC}"
echo ""

# Step 5: Find bundle directory from policy YAML
echo -e "${BLUE}[LIVE_TRIAL160] Step 5: Find bundle directory...${NC}"
POLICY_YAML="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_LIVE_TRIAL160_V10_CTX.yaml"
BUNDLE_DIR=$(python3 << PY
import yaml
import sys
from pathlib import Path

yaml_path = Path("${POLICY_YAML}")
if not yaml_path.exists():
    print("", file=sys.stderr)
    sys.exit(1)

with open(yaml_path) as f:
    config = yaml.safe_load(f)

# Try to find bundle_dir from entry_models.v10_ctx.bundle_dir
entry_models = config.get("entry_models", {})
v10_ctx = entry_models.get("v10_ctx", {})
bundle_dir = v10_ctx.get("bundle_dir")

if bundle_dir:
    bundle_path = Path(bundle_dir)
    if not bundle_path.is_absolute():
        repo_root = Path("${ROOT}")
        bundle_path = repo_root / bundle_path
    print(str(bundle_path.resolve()))
else:
    print("", file=sys.stderr)
    sys.exit(1)
PY
)

if [ -z "$BUNDLE_DIR" ] || [ ! -d "$BUNDLE_DIR" ]; then
    echo -e "${RED}[LIVE_TRIAL160] ❌ Bundle directory not found${NC}"
    exit 1
fi

echo -e "${GREEN}[LIVE_TRIAL160] ✅ Bundle directory: $BUNDLE_DIR${NC}"
echo ""

# Step 6: Set environment variables
echo -e "${BLUE}[LIVE_TRIAL160] Step 6: Set environment variables...${NC}"

# Live mode: feature building is allowed (not PREBUILT), but schema validation is required
export GX1_FEATURE_BUILD_DISABLED=0  # Live builds features on-the-fly
export GX1_GATED_FUSION_ENABLED=1
export GX1_REQUIRE_XGB_CALIBRATION=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export GX1_XGB_THREADS=1
export GX1_ANALYSIS_MODE=1

# Trial 160 policy parameters (from policies/sniper_trial160_prod.json)
export GX1_ENTRY_THRESHOLD_OVERRIDE="0.102"
export GX1_RISK_GUARD_BLOCK_SPREAD_BPS_GTE_OVERRIDE="2000"
export GX1_RISK_GUARD_BLOCK_ATR_BPS_GTE_OVERRIDE="13.73"
export GX1_RISK_GUARD_COOLDOWN_BARS_AFTER_ENTRY_OVERRIDE="2"
export GX1_MAX_CONCURRENT_POSITIONS_OVERRIDE="2"

# Live monitoring flags
export GX1_LIVE_MONITORING_ENABLED=1
export GX1_LIVE_POLICY_ID="${POLICY_ID}"
export GX1_LIVE_POLICY_SHA256="${POLICY_SHA}"

echo -e "${GREEN}[LIVE_TRIAL160] ✅ Environment variables set${NC}"
echo ""

# Step 7: Create RUN_IDENTITY.json before starting live trading
echo -e "${BLUE}[LIVE_TRIAL160] Step 7: Create RUN_IDENTITY.json...${NC}"
python3 << PY
import sys
import os
from pathlib import Path
sys.path.insert(0, ".")
from gx1.runtime.run_identity import create_run_identity

# Set live mode flag
os.environ["GX1_LIVE_MODE"] = "1"

output_dir = Path("${OUTPUT_DIR}")
identity = create_run_identity(
    output_dir=output_dir,
    policy_id="${POLICY_ID}",
    policy_sha256="${POLICY_SHA}",
    bundle_dir=Path("${BUNDLE_DIR}"),
    prebuilt_path=None,  # Live mode doesn't use prebuilt
    allow_dirty=True,  # Allow dirty for live (but log it)
    is_live=True,  # Explicitly mark as live mode
)
print(f"[LIVE_TRIAL160] ✅ RUN_IDENTITY.json created")
PY

if [ ! -f "$OUTPUT_DIR/RUN_IDENTITY.json" ]; then
    echo -e "${RED}[LIVE_TRIAL160] ❌ RUN_IDENTITY.json not created${NC}"
    exit 1
fi
echo -e "${GREEN}[LIVE_TRIAL160] ✅ RUN_IDENTITY.json created${NC}"
echo ""

# Step 8: Verify policy and bundle match
echo -e "${BLUE}[LIVE_TRIAL160] Step 8: Verify policy and bundle match...${NC}"
python3 << PY
import json
import sys
from pathlib import Path

# Load RUN_IDENTITY
identity_path = Path("${OUTPUT_DIR}/RUN_IDENTITY.json")
identity = json.load(open(identity_path))

# Verify policy match
if identity.get("policy_id") != "${POLICY_ID}":
    print(f"❌ Policy ID mismatch: {identity.get('policy_id')} != ${POLICY_ID}", file=sys.stderr)
    sys.exit(1)

if identity.get("policy_sha256") != "${POLICY_SHA}":
    print(f"❌ Policy SHA256 mismatch", file=sys.stderr)
    sys.exit(1)

# Verify bundle exists
bundle_sha = identity.get("bundle_sha256")
if not bundle_sha:
    print("⚠️  Bundle SHA256 not found (will be validated at runtime)", file=sys.stderr)
else:
    print(f"✅ Bundle SHA256: {bundle_sha[:16]}...")

print("✅ Policy and bundle verification passed")
PY

if [ $? -ne 0 ]; then
    echo -e "${RED}[LIVE_TRIAL160] ❌ Policy/bundle verification failed${NC}"
    exit 1
fi
echo -e "${GREEN}[LIVE_TRIAL160] ✅ Policy and bundle verified${NC}"
echo ""

# Step 9: Start live trading
echo -e "${BLUE}[LIVE_TRIAL160] Step 9: Starting live trading...${NC}"
echo -e "${YELLOW}[LIVE_TRIAL160] ⚠️  LIVE TRADING MODE - Real orders will be sent${NC}"
echo ""

python3 << PYEOF
import sys
import signal
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, ".")

from gx1.execution.oanda_demo_runner import GX1DemoRunner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("${OUTPUT_DIR}/live_trial160.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# Signal handler
shutdown_requested = False
def signal_handler(sig, frame):
    global shutdown_requested
    log.info("[LIVE_TRIAL160] Shutdown requested")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Initialize runner
log.info("=" * 80)
log.info("[LIVE_TRIAL160] Starting LIVE trading (Trial 160)")
log.info("=" * 80)
log.info(f"Policy: ${POLICY_YAML}")
log.info(f"Policy ID: ${POLICY_ID}")
log.info(f"Policy SHA256: ${POLICY_SHA[:16]}...")
log.info(f"Output: ${OUTPUT_DIR}")
log.info(f"Account: ${OANDA_ENV:-practice}")
log.info(f"Instrument: XAUUSD")
log.info(f"Mode: LIVE (real orders)")
log.info("=" * 80)

runner = GX1DemoRunner(
    Path("${POLICY_YAML}"),
    dry_run_override=False,  # REAL ORDERS
    replay_mode=False,  # LIVE mode
    output_dir=Path("${OUTPUT_DIR}"),
)

# Initialize required attributes
if not hasattr(runner, '_shutdown_requested'):
    runner._shutdown_requested = False

# Verify policy match at runtime (hard-fail if mismatch)
actual_policy_id = getattr(runner, 'policy', {}).get('live_config', {}).get('policy_id', '')
if actual_policy_id != "${POLICY_ID}":
    raise RuntimeError(
        f"[LIVE_TRIAL160_FAIL] Policy ID mismatch at runtime: "
        f"expected=${POLICY_ID}, got={actual_policy_id}"
    )

# Verify bundle (will be validated when model loads)
log.info("[LIVE_TRIAL160] Policy and bundle verification passed")

# Run forever
try:
    runner.run_forever()
except KeyboardInterrupt:
    log.info("[LIVE_TRIAL160] Interrupted by user")
except Exception as e:
    log.error(f"[LIVE_TRIAL160] Error: {e}", exc_info=True)
    raise
finally:
    log.info("[LIVE_TRIAL160] Execution stopped")

PYEOF

EXIT_CODE=$?

echo ""
echo -e "${BLUE}=== LIVE TRADING STOPPED ===${NC}"
echo "Exit code: ${EXIT_CODE}"
echo "Output: ${OUTPUT_DIR}"
echo ""
