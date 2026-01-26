#!/bin/bash
#
# ENGINE-ONLY GATE: Hard-fail if data files are detected
#
# Checks:
# - Large files (>20MB) with data extensions
# - Data paths (GX1_DATA, reports/, archive/, etc.)
# - Staging area (git diff --cached)
# - Tracked files (git ls-files)
#

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Data file extensions
DATA_EXTENSIONS=(
    ".parquet" ".csv" ".sqlite" ".db" ".pkl" ".joblib" ".bst"
    ".pt" ".onnx" ".npy" ".npz" ".zip" ".tar" ".7z"
)

# Data path patterns
DATA_PATHS=(
    "GX1_DATA" "reports/" "archive/" "data/" "models/" "quarantine/"
    "replay_eval/" "*/cache/" "*/__pycache__/"
)

# Size threshold (20MB)
SIZE_THRESHOLD=$((20 * 1024 * 1024))

FAILED=0
FAILED_FILES=()

echo "============================================================"
echo "ENGINE-ONLY GATE CHECK"
echo "============================================================"

# Check 1: Large files with data extensions (only tracked files)
echo ""
echo "Checking for large tracked data files (>20MB)..."
while IFS= read -r file; do
    if [ -f "$file" ]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
        if [ "$size" -gt "$SIZE_THRESHOLD" ]; then
            for ext in "${DATA_EXTENSIONS[@]}"; do
                if [[ "$file" == *"$ext" ]]; then
                    echo "  ❌ Large tracked data file: $file ($(numfmt --to=iec-i --suffix=B "$size"))"
                    FAILED=1
                    FAILED_FILES+=("$file")
                    break
                fi
            done
        fi
    fi
done < <(git ls-files || true)

# Check 2: Data paths in repo
echo ""
echo "Checking for data paths..."
for pattern in "${DATA_PATHS[@]}"; do
    # Check tracked files
    while IFS= read -r file; do
        if [[ "$file" == *"$pattern"* ]]; then
            echo "  ❌ Data path detected: $file"
            FAILED=1
            FAILED_FILES+=("$file")
        fi
    done < <(git ls-files | grep "$pattern" || true)
    
    # Check staging (only new/modified files, not deletions)
    while IFS= read -r file; do
        if [[ "$file" == *"$pattern"* ]]; then
            # Check if this is a deletion (D) or addition/modification (A/M)
            status=$(git diff --cached --name-status 2>/dev/null | grep "^[AMD].*$file" | head -1 | cut -c1 || echo "")
            if [[ "$status" != "D" ]]; then
                echo "  ❌ Data path in staging (not deletion): $file"
                FAILED=1
                FAILED_FILES+=("$file")
            fi
        fi
    done < <(git diff --cached --name-only 2>/dev/null | grep "$pattern" || true)
done

# Check 3: Data extensions in tracked files
echo ""
echo "Checking tracked files for data extensions..."
for ext in "${DATA_EXTENSIONS[@]}"; do
    while IFS= read -r file; do
        # Skip if it's a small config/metadata file
        if [ -f "$file" ]; then
            size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
            # Allow small files (<1MB) - might be configs
            if [ "$size" -gt 1048576 ]; then
                echo "  ❌ Large tracked data file: $file ($(numfmt --to=iec-i --suffix=B "$size"))"
                FAILED=1
                FAILED_FILES+=("$file")
            fi
        fi
    done < <(git ls-files | grep -E "\\$ext\$" || true)
    
    # Check staging (only new/modified, not deletions)
    while IFS= read -r file; do
        status=$(git diff --cached --name-status 2>/dev/null | grep -E "[AMD].*$file" | head -1 | cut -c1 || echo "")
        if [[ "$status" == "D" ]]; then
            continue  # Deletions are OK
        fi
        if [ -f "$file" ]; then
            size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
            if [ "$size" -gt 1048576 ]; then
                echo "  ❌ Large data file in staging: $file ($(numfmt --to=iec-i --suffix=B "$size"))"
                FAILED=1
                FAILED_FILES+=("$file")
            fi
        fi
    done < <(git diff --cached --name-only 2>/dev/null | grep -E "\\$ext\$" || true)
done

# Final verdict
echo ""
echo "============================================================"
if [ $FAILED -eq 0 ]; then
    echo "✅ PASS: engine-only"
    exit 0
else
    echo "❌ FAIL: data detected"
    echo ""
    echo "Files/paths that must be removed:"
    for file in "${FAILED_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "HOW TO FIX:"
    echo "1. Remove data files from staging: git reset HEAD <file>"
    echo "2. Remove from tracking: git rm --cached <file>"
    echo "3. Add to .gitignore"
    exit 1
fi
