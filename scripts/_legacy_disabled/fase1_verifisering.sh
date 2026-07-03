#!/bin/bash
# FASE 1 VERIFISERING - Full kartlegging av repo og worktrees
# 100% ikke-destruktivt - kun lesing, ingen endringer

set -euo pipefail

# Timestamp for output-fil
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="reports/ops"
OUTPUT_FILE="${OUTPUT_DIR}/FASE1_VERIFISERING_RUN_${TIMESTAMP}.txt"

# Opprett output-mappe hvis den ikke finnes
mkdir -p "${OUTPUT_DIR}"

# Funksjon for å skrive til både stdout og fil
log() {
    echo "$@" | tee -a "${OUTPUT_FILE}"
}

# Funksjon for å håndtere manglende mapper
safe_cd() {
    local dir="$1"
    if [ -d "$dir" ]; then
        cd "$dir" || return 1
        return 0
    else
        echo "MISSING"
        return 1
    fi
}

log "=================================================================================="
log "FASE 1 VERIFISERING - FULL KARTLEGGING"
log "=================================================================================="
log "Dato: $(date)"
log "Output-fil: ${OUTPUT_FILE}"
log ""

# ============================================================================
# SUMMARY (kort oppsummering øverst)
# ============================================================================
log "=================================================================================="
log "SUMMARY"
log "=================================================================================="
log ""

MAIN_REPO="/Users/andrekildalbakke/Desktop/GX1 XAUUSD"
CURSOR_WT_BASE="/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD"

# Sjekk kanonisk repo
if [ -d "${MAIN_REPO}" ]; then
    cd "${MAIN_REPO}"
    MAIN_HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo "ERROR")
    MAIN_BRANCH=$(git branch --show-current 2>/dev/null || echo "DETACHED")
    MAIN_DIRTY=$(git status --porcelain=v1 2>/dev/null | wc -l | tr -d ' ')
    log "Kanonisk repo: ${MAIN_REPO}"
    log "  HEAD: ${MAIN_HEAD}"
    log "  Branch: ${MAIN_BRANCH}"
    log "  Dirty files: ${MAIN_DIRTY}"
else
    log "❌ Kanonisk repo ikke funnet: ${MAIN_REPO}"
fi

# Sjekk worktrees
log ""
log "Worktrees:"
if [ -d "${CURSOR_WT_BASE}" ]; then
    for wt in aaq cia mkm muo; do
        wt_path="${CURSOR_WT_BASE}/${wt}"
        if [ -d "${wt_path}" ]; then
            cd "${wt_path}" 2>/dev/null || continue
            wt_head=$(git rev-parse --short HEAD 2>/dev/null || echo "ERROR")
            wt_branch=$(git branch --show-current 2>/dev/null || echo "DETACHED")
            wt_dirty=$(git status --porcelain=v1 2>/dev/null | wc -l | tr -d ' ')
            has_replay=$(test -f "gx1/scripts/replay_eval_gated_parallel.py" && echo "YES" || echo "NO")
            log "  ${wt}: HEAD=${wt_head}, Branch=${wt_branch}, Dirty=${wt_dirty}, Has_replay=${has_replay}"
        else
            log "  ${wt}: NOT_FOUND"
        fi
    done
else
    log "  Cursor worktrees ikke funnet: ${CURSOR_WT_BASE}"
fi

log ""
log "=================================================================================="
log ""

# ============================================================================
# 1. KANONISK REPO-INFO (Desktop-repo)
# ============================================================================
log "=================================================================================="
log "1. KANONISK REPO-INFO (Desktop-repo)"
log "=================================================================================="
log ""

if safe_cd "${MAIN_REPO}"; then
    log "Top-level: $(git rev-parse --show-toplevel 2>/dev/null || echo 'ERROR')"
    log ""
    log "HEAD (short): $(git rev-parse --short HEAD 2>/dev/null || echo 'ERROR')"
    log "HEAD (full):  $(git rev-parse HEAD 2>/dev/null || echo 'ERROR')"
    log ""
    log "Branch: $(git branch --show-current 2>/dev/null || echo 'DETACHED')"
    log ""
    log "Remote:"
    git remote -v 2>/dev/null | sed 's/^/  /' || log "  (ingen remote)"
    log ""
    log "Worktrees:"
    git worktree list 2>/dev/null | sed 's/^/  /' || log "  (ingen worktrees)"
else
    log "❌ Kanonisk repo ikke funnet: ${MAIN_REPO}"
fi

log ""
log "=================================================================================="
log ""

# ============================================================================
# 2. PER WORKTREE
# ============================================================================
log "=================================================================================="
log "2. PER WORKTREE"
log "=================================================================================="
log ""

if [ -d "${CURSOR_WT_BASE}" ]; then
    for wt in aaq cia mkm muo; do
        wt_path="${CURSOR_WT_BASE}/${wt}"
        log "----------------------------------------------------------------------"
        log "Worktree: ${wt}"
        log "----------------------------------------------------------------------"
        
        if safe_cd "${wt_path}"; then
            log "Path: ${wt_path}"
            log ""
            log "HEAD (short): $(git rev-parse --short HEAD 2>/dev/null || echo 'ERROR')"
            log "HEAD (full):  $(git rev-parse HEAD 2>/dev/null || echo 'ERROR')"
            log ""
            log "Branch: $(git branch --show-current 2>/dev/null || echo 'DETACHED')"
            log ""
            log "Dirty count: $(git status --porcelain=v1 2>/dev/null | wc -l | tr -d ' ') files"
            log ""
            
            # Sjekk replay_eval_gated_parallel.py
            replay_file="gx1/scripts/replay_eval_gated_parallel.py"
            if [ -f "${replay_file}" ]; then
                log "replay_eval_gated_parallel.py: EXISTS"
                log "  Size: $(stat -f%z "${replay_file}" 2>/dev/null || stat -c%s "${replay_file}" 2>/dev/null || echo 'UNKNOWN') bytes"
                log "  mtime: $(stat -f%Sm "${replay_file}" 2>/dev/null || stat -c%y "${replay_file}" 2>/dev/null || echo 'UNKNOWN')"
                log "  SHA256: $(sha256sum "${replay_file}" 2>/dev/null | cut -d' ' -f1 || echo 'ERROR')"
            else
                log "replay_eval_gated_parallel.py: MISSING"
            fi
        else
            log "Path: ${wt_path}"
            log "Status: NOT_FOUND"
        fi
        log ""
    done
else
    log "❌ Cursor worktrees ikke funnet: ${CURSOR_WT_BASE}"
fi

log "=================================================================================="
log ""

# ============================================================================
# 3. STØYANALYSE
# ============================================================================
log "=================================================================================="
log "3. STØYANALYSE (MAIN REPO)"
log "=================================================================================="
log ""

if safe_cd "${MAIN_REPO}"; then
    log "Topp 20 mapper som dominerer git-status:"
    log ""
    git status --porcelain=v1 2>/dev/null | \
        cut -d' ' -f2- | \
        xargs -I{} dirname {} 2>/dev/null | \
        sed 's|^\.$|(root)|' | \
        sort | \
        uniq -c | \
        sort -rn | \
        head -20 | \
        sed 's/^/  /' || log "  (ingen dirty files)"
    log ""
    
    log "Topp 50 linjer av git status --porcelain (smell test):"
    log ""
    git status --porcelain=v1 2>/dev/null | head -50 | sed 's/^/  /' || log "  (ingen dirty files)"
else
    log "❌ Kanonisk repo ikke funnet"
fi

log ""
log "=================================================================================="
log ""

# ============================================================================
# 4. UNTRACKED STORE FILER
# ============================================================================
log "=================================================================================="
log "4. UNTRACKED STORE FILER (> 1MB)"
log "=================================================================================="
log ""

# Main repo
if safe_cd "${MAIN_REPO}"; then
    log "Main repo:"
    log ""
    found_main=0
    git status --porcelain=v1 2>/dev/null | grep "^??" | sed 's/^?? //' | while read -r file; do
        if [ -f "$file" ]; then
            size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
            if [ "$size" -gt 1048576 ]; then
                size_mb=$(echo "scale=2; $size / 1048576" | bc 2>/dev/null || echo "$size")
                log "  $file: ${size_mb}MB"
                found_main=1
            fi
        fi
    done
    if [ "$found_main" -eq 0 ]; then
        log "  (ingen untracked filer > 1MB)"
    fi
    log ""
fi

# Worktrees
if [ -d "${CURSOR_WT_BASE}" ]; then
    for wt in aaq cia mkm muo; do
        wt_path="${CURSOR_WT_BASE}/${wt}"
        if safe_cd "${wt_path}"; then
            log "${wt} worktree:"
            log ""
            found_wt=0
            git status --porcelain=v1 2>/dev/null | grep "^??" | sed 's/^?? //' | while read -r file; do
                if [ -f "$file" ]; then
                    size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
                    if [ "$size" -gt 1048576 ]; then
                        size_mb=$(echo "scale=2; $size / 1048576" | bc 2>/dev/null || echo "$size")
                        log "  $file: ${size_mb}MB"
                        found_wt=1
                    fi
                fi
            done
            if [ "$found_wt" -eq 0 ]; then
                log "  (ingen untracked filer > 1MB)"
            fi
            log ""
        fi
    done
fi

log "=================================================================================="
log ""

# ============================================================================
# 5. DISKBRUK
# ============================================================================
log "=================================================================================="
log "5. DISKBRUK"
log "=================================================================================="
log ""

if safe_cd "${MAIN_REPO}"; then
    log "Hovedmapper:"
    log ""
    for dir in reports outputs runs logs data; do
        if [ -d "$dir" ]; then
            size=$(du -sh "$dir" 2>/dev/null | cut -f1 || echo "ERROR")
            log "  $dir: ${size}"
        else
            log "  $dir: MISSING"
        fi
    done
    log ""
    
    log "Topp 20 største undermapper i reports/:"
    log ""
    if [ -d "reports" ]; then
        du -sh reports/* 2>/dev/null | sort -h | tail -20 | sed 's/^/  /' || log "  (ingen undermapper)"
    else
        log "  reports/ ikke funnet"
    fi
else
    log "❌ Kanonisk repo ikke funnet"
fi

log ""
log "=================================================================================="
log "FASE 1 VERIFISERING FULLFØRT"
log "Output-fil: ${OUTPUT_FILE}"
log "=================================================================================="
