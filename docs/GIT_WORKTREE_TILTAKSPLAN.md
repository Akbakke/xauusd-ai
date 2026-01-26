# GIT WORKTREE TILTAKSPLAN - EKSAKTE KOMMANDOER

**Dato:** 2025-01-13  
**Status:** Klar for utføring

## KRITISK FUNN

1. **`replay_eval_gated_parallel.py` mangler i `cia` worktree**
   - Filen finnes i `aaq` og `muo`
   - Dette forklarer "spøkelsesfeil" i Cursor

2. **Flere worktrees kan skape forvirring**
   - 4 worktrees: `aaq`, `cia`, `mkm`, `muo`
   - Må reduseres til maks 1 (f.eks. `cia`)

## TILTAKSPLAN - STEG FOR STEG

### FASE 1: VERIFISERING (IKKE DESTRUKTIVT)

#### 1.1 Verifiser kanonisk repo

```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"
echo "=== KANONISK REPO ==="
echo "Top-level: $(git rev-parse --show-toplevel)"
echo "HEAD: $(git rev-parse --short HEAD)"
echo "Branch: $(git branch --show-current || echo 'DETACHED')"
echo "Dirty: $(git status --porcelain=v1 | wc -l | tr -d ' ') files"
echo ""
echo "=== WORKTREES ==="
git worktree list
```

#### 1.2 Sjekk alle worktrees

```bash
for wt in aaq cia mkm muo; do
  echo "===== $wt ====="
  cd "/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/$wt" 2>/dev/null || { echo "NOT_FOUND"; continue; }
  echo "HEAD: $(git rev-parse --short HEAD 2>&1)"
  echo "Branch: $(git branch --show-current 2>&1 || echo 'DETACHED')"
  echo "Dirty: $(git status --porcelain=v1 2>&1 | wc -l | tr -d ' ') files"
  echo "Has replay_eval_gated_parallel.py: $(test -f gx1/scripts/replay_eval_gated_parallel.py && echo 'YES' || echo 'NO')"
  echo ""
done
```

#### 1.3 Mål diskbruk

```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"
echo "=== DISKBRUK ==="
for dir in reports outputs runs logs data; do
  if [ -d "$dir" ]; then
    echo "$dir: $(du -sh "$dir" 2>/dev/null | cut -f1)"
  fi
done
echo ""
echo "=== REPORTS DETAILED ==="
du -sh reports/* 2>/dev/null | sort -h | tail -30
```

#### 1.4 Tell genererte filer

```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"
echo "=== GENERERTE FILER ==="
echo "reports/replay_eval files: $(find reports/replay_eval -type f 2>/dev/null | wc -l)"
echo "*.log files: $(find . -name "*.log" 2>/dev/null | wc -l)"
echo "*_debug* files: $(find . -name "*_debug*" 2>/dev/null | wc -l)"
echo "*_trace* files: $(find . -name "*_trace*" 2>/dev/null | wc -l)"
```

### FASE 2: FIKSE MANGLENDE FILER (TRYGGT)

#### 2.1 Kopier replay_eval_gated_parallel.py til cia

```bash
# Verifiser at kildefilen eksisterer
if [ -f "/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/aaq/gx1/scripts/replay_eval_gated_parallel.py" ]; then
  echo "✅ Kildefil funnet i aaq"
  
  # Kopier til cia
  mkdir -p "/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/cia/gx1/scripts"
  cp "/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/aaq/gx1/scripts/replay_eval_gated_parallel.py" \
     "/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/cia/gx1/scripts/replay_eval_gated_parallel.py"
  
  # Verifiser kopiering
  if [ -f "/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/cia/gx1/scripts/replay_eval_gated_parallel.py" ]; then
    echo "✅ Fil kopiert til cia"
  else
    echo "❌ Kopiering feilet"
  fi
else
  echo "❌ Kildefil ikke funnet i aaq"
fi
```

### FASE 3: REDUSERE WORKTREES (FORSIKTIG)

#### 3.1 Identifiser worktrees som kan fjernes

**Først: Sjekk dirty count for hver worktree**

```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"
echo "=== WORKTREE DIRTY COUNT ==="
for wt in aaq cia mkm muo; do
  wt_path="/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/$wt"
  if [ -d "$wt_path" ]; then
    cd "$wt_path"
    dirty_count=$(git status --porcelain=v1 2>&1 | wc -l | tr -d ' ')
    echo "$wt: $dirty_count dirty files"
    
    if [ "$dirty_count" -eq 0 ]; then
      echo "  → Kan fjernes (ingen dirty files)"
    else
      echo "  → MÅ BEHOLDES (har dirty files)"
      echo "  → Dirty files:"
      git status --porcelain=v1 2>&1 | head -10
    fi
  fi
done
```

#### 3.2 Fjern worktrees (kun hvis dirty_count=0)

**⚠️ ADVARSEL: Kjør kun hvis dirty_count=0 og worktree ikke trengs**

```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"

# Eksempel: Fjern mkm hvis den er ren og ikke trengs
# git worktree remove /Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/mkm

# Eksempel: Fjern aaq hvis den er ren og ikke trengs
# git worktree remove /Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/aaq

# Eksempel: Fjern muo hvis den er ren og ikke trengs
# git worktree remove /Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/muo

# BEHOLD: cia (den vi bruker)
```

### FASE 4: RYDDE GENERERTE FILER (DRY-RUN FØRST)

#### 4.1 DRY-RUN: List kandidater for sletting

```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"

echo "=== KANDIDATER FOR SLETTING (DRY-RUN) ==="
echo ""
echo "reports/replay_eval files (sample):"
find reports/replay_eval -type f 2>/dev/null | head -50
echo ""
echo "*.log files (sample):"
find . -maxdepth 3 -name "*.log" 2>/dev/null | head -50
echo ""
echo "*_debug* files (sample):"
find . -maxdepth 3 -name "*_debug*" 2>/dev/null | head -50
echo ""
echo "*_trace* files (sample):"
find . -maxdepth 3 -name "*_trace*" 2>/dev/null | head -50
```

#### 4.2 Faktisk sletting (kun etter bekreftelse)

**⚠️ ADVARSEL: Kjør kun etter manuell bekreftelse**

```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"

# Slett replay_eval (kan regenereres)
# rm -rf reports/replay_eval/*

# Slett log-filer
# find . -name "*.log" -delete

# Slett debug/trace filer
# find . -name "*_debug*" -delete
# find . -name "*_trace*" -delete
```

### FASE 5: OPPDATERE .gitignore

#### 5.1 Sjekk .gitignore

```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"
echo "=== .gitignore ==="
grep -E "(reports/replay_eval|\.log|_debug|_trace)" .gitignore || echo "Ikke funnet i .gitignore"
```

#### 5.2 Legg til i .gitignore (hvis mangler)

```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"

# Legg til hvis ikke allerede der
if ! grep -q "reports/replay_eval/" .gitignore 2>/dev/null; then
  echo "reports/replay_eval/" >> .gitignore
  echo "✅ Lagt til reports/replay_eval/ i .gitignore"
fi

if ! grep -q "\.log$" .gitignore 2>/dev/null; then
  echo "*.log" >> .gitignore
  echo "✅ Lagt til *.log i .gitignore"
fi

if ! grep -q "_debug" .gitignore 2>/dev/null; then
  echo "*_debug*" >> .gitignore
  echo "✅ Lagt til *_debug* i .gitignore"
fi

if ! grep -q "_trace" .gitignore 2>/dev/null; then
  echo "*_trace*" >> .gitignore
  echo "✅ Lagt til *_trace* i .gitignore"
fi
```

### FASE 6: LAG REPO DOCTOR SCRIPT

#### 6.1 Opprett repo_doctor.sh

```bash
cat > "/Users/andrekildalbakke/Desktop/GX1 XAUUSD/scripts/repo_doctor.sh" << 'EOF'
#!/bin/bash
# GX1 Repo Doctor - Verifiser repo-struktur

set -e

CANONICAL_ROOT="/Users/andrekildalbakke/Desktop/GX1 XAUUSD"
CURRENT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "")

if [ -z "$CURRENT_ROOT" ]; then
    echo "❌ ERROR: Not in a git repository"
    exit 1
fi

if [ "$CURRENT_ROOT" != "$CANONICAL_ROOT" ]; then
    echo "❌ ERROR: Not in canonical repo root"
    echo "  Current: $CURRENT_ROOT"
    echo "  Expected: $CANONICAL_ROOT"
    exit 1
fi

echo "✅ Canonical repo root: $CANONICAL_ROOT"
echo "✅ HEAD: $(git rev-parse --short HEAD)"
echo "✅ Branch: $(git branch --show-current || echo 'DETACHED')"
echo "✅ Dirty: $(git status --porcelain=v1 | wc -l | tr -d ' ') files"

WORKTREES=$(git worktree list | wc -l | tr -d ' ')
if [ "$WORKTREES" -gt 2 ]; then
    echo "⚠️  WARNING: More than 2 worktrees detected ($WORKTREES)"
    echo "   Expected: Main + 1 Cursor worktree"
    git worktree list
    exit 1
else
    echo "✅ Worktree count: $WORKTREES (OK)"
fi

echo ""
echo "✅ Repo structure is OK"
EOF

chmod +x "/Users/andrekildalbakke/Desktop/GX1 XAUUSD/scripts/repo_doctor.sh"
echo "✅ repo_doctor.sh opprettet"
```

## SAMMENDRAG

### Trygge steg (kan kjøres nå):
1. ✅ FASE 1: Verifisering (alle steg)
2. ✅ FASE 2: Kopier replay_eval_gated_parallel.py til cia
3. ✅ FASE 5: Oppdater .gitignore
4. ✅ FASE 6: Lag repo_doctor.sh

### Forsiktige steg (krever manuell vurdering):
1. ⚠️ FASE 3: Reduser worktrees (sjekk dirty_count først)
2. ⚠️ FASE 4: Rydd genererte filer (dry-run først, bekreft før sletting)

### Neste steg:
1. Kjør FASE 1 for å få full oversikt
2. Kjør FASE 2 for å fikse manglende fil
3. Vurder FASE 3 basert på dirty_count
4. Vurder FASE 4 basert på diskbruk
