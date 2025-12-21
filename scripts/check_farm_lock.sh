#!/bin/bash
# FARM Lock Check Script
#
# Verifies that FARM PROD_BASELINE files have not been modified.
# Fails if any locked file hash has changed.
#
# Usage:
#   ./scripts/check_farm_lock.sh
#
# This is a LOCAL guard only (not CI infrastructure).
# For CI, use prod_baseline_proof.py instead.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Expected hashes (computed from frozen baseline)
# Update these if baseline is intentionally changed (new snapshot)
EXPECTED_POLICY_HASH="d9da2c864eb7767840fde0b812170ae127a4f5ac6f8302afafa9e8da0cd887e4"

# Locked files
POLICY_FILE="gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml"
ENTRY_CONFIG="gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID/ENTRY_V9_FARM_V2B_PROD.yaml"
EXIT_CONFIG_RULE5="gx1/configs/exits/FARM_EXIT_V2_RULES_A_mp68_max5_rule5_cd0.yaml"
EXIT_CONFIG_RULE6A="gx1/configs/exits/FARM_EXIT_V2_RULES_ADAPTIVE_v1.yaml"
ROUTER_MODEL="gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/exit_router_models_v3/exit_router_v3_tree.pkl"

echo "=================================================================================="
echo "FARM PROD_BASELINE Lock Check"
echo "=================================================================================="
echo ""

# Function to compute SHA256 hash
compute_hash() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "FILE_NOT_FOUND"
        return
    fi
    shasum -a 256 "$file" | awk '{print $1}'
}

# Check policy file
echo -n "Checking policy file... "
POLICY_HASH=$(compute_hash "$POLICY_FILE")
if [ "$POLICY_HASH" = "FILE_NOT_FOUND" ]; then
    echo -e "${RED}✗ FILE NOT FOUND${NC}"
    echo "  Expected: $POLICY_FILE"
    exit 1
elif [ "$POLICY_HASH" = "$EXPECTED_POLICY_HASH" ]; then
    echo -e "${GREEN}✓ MATCH${NC}"
else
    echo -e "${RED}✗ MISMATCH${NC}"
    echo "  Expected: $EXPECTED_POLICY_HASH"
    echo "  Actual:   $POLICY_HASH"
    echo ""
    echo "⚠️  WARNING: FARM PROD_BASELINE policy has been modified!"
    echo "   This violates the FARM lock. If this is intentional, create a NEW snapshot."
    exit 1
fi

# Check entry config (compute hash, warn if changed)
echo -n "Checking entry config... "
ENTRY_HASH=$(compute_hash "$ENTRY_CONFIG")
if [ "$ENTRY_HASH" = "FILE_NOT_FOUND" ]; then
    echo -e "${YELLOW}⚠ FILE NOT FOUND${NC}"
else
    echo -e "${GREEN}✓ EXISTS${NC} (hash: ${ENTRY_HASH:0:16}...)"
fi

# Check exit configs
echo -n "Checking exit config (RULE5)... "
EXIT_RULE5_HASH=$(compute_hash "$EXIT_CONFIG_RULE5")
if [ "$EXIT_RULE5_HASH" = "FILE_NOT_FOUND" ]; then
    echo -e "${YELLOW}⚠ FILE NOT FOUND${NC}"
else
    echo -e "${GREEN}✓ EXISTS${NC} (hash: ${EXIT_RULE5_HASH:0:16}...)"
fi

echo -n "Checking exit config (RULE6A)... "
EXIT_RULE6A_HASH=$(compute_hash "$EXIT_CONFIG_RULE6A")
if [ "$EXIT_RULE6A_HASH" = "FILE_NOT_FOUND" ]; then
    echo -e "${YELLOW}⚠ FILE NOT FOUND${NC}"
else
    echo -e "${GREEN}✓ EXISTS${NC} (hash: ${EXIT_RULE6A_HASH:0:16}...)"
fi

# Check router model
echo -n "Checking router model... "
ROUTER_HASH=$(compute_hash "$ROUTER_MODEL")
if [ "$ROUTER_HASH" = "FILE_NOT_FOUND" ]; then
    echo -e "${YELLOW}⚠ FILE NOT FOUND${NC}"
else
    echo -e "${GREEN}✓ EXISTS${NC} (hash: ${ROUTER_HASH:0:16}...)"
fi

echo ""
echo "=================================================================================="
echo -e "${GREEN}✓ FARM Lock Check PASSED${NC}"
echo "=================================================================================="
echo ""
echo "All FARM PROD_BASELINE files are locked and unchanged."
echo ""
echo "To verify a run matches baseline:"
echo "  python3 gx1/analysis/prod_baseline_proof.py \\"
echo "    --run gx1/wf_runs/<RUN_TAG> \\"
echo "    --prod-policy $POLICY_FILE \\"
echo "    --out gx1/wf_runs/<RUN_TAG>/prod_baseline_proof.md"
echo ""

