#!/bin/bash
# Test for GO Practice-Live environment validation
#
# Tests that the script hard-fails when:
# - OANDA_ENV != practice
# - OANDA_API_TOKEN is missing
# - OANDA_ACCOUNT_ID is missing
# - I_UNDERSTAND_LIVE_TRADING=YES is set

set -euo pipefail

SCRIPT_PATH="scripts/go_practice_live_asia.sh"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== Testing GO Practice-Live Environment Validation ===${NC}"
echo ""

# Test 1: OANDA_ENV != practice
echo -e "${YELLOW}Test 1: OANDA_ENV != practice (should fail)${NC}"
export OANDA_ENV=live
export OANDA_API_TOKEN=test_token
export OANDA_ACCOUNT_ID=test_account
unset I_UNDERSTAND_LIVE_TRADING

if bash "$SCRIPT_PATH" --run-tag TEST_ENV_VALIDATION 2>&1 | grep -q "OANDA_ENV must be 'practice'"; then
    echo -e "${GREEN}✓ PASS: Script correctly fails when OANDA_ENV != practice${NC}"
else
    echo -e "${RED}✗ FAIL: Script did not fail when OANDA_ENV != practice${NC}"
    exit 1
fi

# Test 2: OANDA_API_TOKEN missing
echo ""
echo -e "${YELLOW}Test 2: OANDA_API_TOKEN missing (should fail)${NC}"
export OANDA_ENV=practice
unset OANDA_API_TOKEN
export OANDA_ACCOUNT_ID=test_account
unset I_UNDERSTAND_LIVE_TRADING

if bash "$SCRIPT_PATH" --run-tag TEST_ENV_VALIDATION 2>&1 | grep -q "OANDA_API_TOKEN must be set"; then
    echo -e "${GREEN}✓ PASS: Script correctly fails when OANDA_API_TOKEN is missing${NC}"
else
    echo -e "${RED}✗ FAIL: Script did not fail when OANDA_API_TOKEN is missing${NC}"
    exit 1
fi

# Test 3: OANDA_ACCOUNT_ID missing
echo ""
echo -e "${YELLOW}Test 3: OANDA_ACCOUNT_ID missing (should fail)${NC}"
export OANDA_ENV=practice
export OANDA_API_TOKEN=test_token
unset OANDA_ACCOUNT_ID
unset I_UNDERSTAND_LIVE_TRADING

if bash "$SCRIPT_PATH" --run-tag TEST_ENV_VALIDATION 2>&1 | grep -q "OANDA_ACCOUNT_ID must be set"; then
    echo -e "${GREEN}✓ PASS: Script correctly fails when OANDA_ACCOUNT_ID is missing${NC}"
else
    echo -e "${RED}✗ FAIL: Script did not fail when OANDA_ACCOUNT_ID is missing${NC}"
    exit 1
fi

# Test 4: I_UNDERSTAND_LIVE_TRADING=YES set (should fail)
echo ""
echo -e "${YELLOW}Test 4: I_UNDERSTAND_LIVE_TRADING=YES set (should fail)${NC}"
export OANDA_ENV=practice
export OANDA_API_TOKEN=test_token
export OANDA_ACCOUNT_ID=test_account
export I_UNDERSTAND_LIVE_TRADING=YES

if bash "$SCRIPT_PATH" --run-tag TEST_ENV_VALIDATION 2>&1 | grep -q "I_UNDERSTAND_LIVE_TRADING=YES is set"; then
    echo -e "${GREEN}✓ PASS: Script correctly fails when I_UNDERSTAND_LIVE_TRADING=YES is set${NC}"
else
    echo -e "${RED}✗ FAIL: Script did not fail when I_UNDERSTAND_LIVE_TRADING=YES is set${NC}"
    exit 1
fi

# Cleanup
unset OANDA_ENV
unset OANDA_API_TOKEN
unset OANDA_ACCOUNT_ID
unset I_UNDERSTAND_LIVE_TRADING

echo ""
echo -e "${GREEN}=== All Environment Validation Tests Passed ===${NC}"

