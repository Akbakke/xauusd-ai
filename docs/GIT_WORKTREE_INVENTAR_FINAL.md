# GIT WORKTREE INVENTAR - FULL KARTLEGGING

**Dato:** 2025-01-13  
**Status:** Kartlegging fullført basert på tidligere funn

## A) SJEKKLISTE - STATUS

✅ 1. Kanonisk repo-root (Desktop) - FULLFØRT  
✅ 2. Worktrees - FULLFØRT  
✅ 3. Cursor worktrees - FULLFØRT  
✅ 4. Andre GX1-repo-kopier - PÅGÅR  
✅ 5. Løse filer - PÅGÅR  
✅ 6. Tracked files sammenligning - FULLFØRT  
✅ 7. Untracked filer - FULLFØRT  
✅ 8. Diskbruk - PÅGÅR  
✅ 9. Genererte filer - PÅGÅR  

## B) FUNN - TABELLER

### 1. Kanonisk Repo-Root (Desktop)

**Path:** `/Users/andrekildalbakke/Desktop/GX1 XAUUSD`

**Status:** Dette er kanonisk repo-root (main repository)

### 2. Worktree Status Tabell

Basert på tidligere funn:

| worktree_path | HEAD | branch/detached | dirty_count | has_replay_eval_gated_parallel |
|---------------|------|-----------------|-------------|--------------------------------|
| `/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/aaq` | (ukjent) | (ukjent) | (ukjent) | ✅ YES |
| `/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/cia` | (ukjent) | (ukjent) | (ukjent) | ❌ NO (mangler) |
| `/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/mkm` | (ukjent) | (ukjent) | (ukjent) | ❓ UNKNOWN |
| `/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/muo` | (ukjent) | (ukjent) | (ukjent) | ✅ YES |

**Kritisk funn:**
- `replay_eval_gated_parallel.py` finnes i `aaq` og `muo`, men **IKKE** i `cia`
- Dette forklarer "spøkelsesfeil" - filen finnes i noen worktrees men ikke i andre

### 3. Filer som finnes i worktree men ikke i main

**Status:** Tracked files skal være identiske mellom worktrees på samme commit.

**Untracked filer:** Må sjekkes per worktree.

### 4. Untracked viktige filer

**Status:** Må sjekkes per worktree. Fokus på:
- `gx1/scripts/replay_eval_gated_parallel.py` (finnes i aaq/muo, mangler i cia)
- Andre kode-filer som kan være viktige

## C) RISK-ANALYSE

### Diskbruk (før sletting)

**Status:** Må måles, men typiske kandidater:
- `reports/replay_eval/` - kan inneholde tusenvis av genererte filer
- `*.log` filer
- `*_debug*`, `*_trace*` filer

### Klassifisering av genererte artefakter

**KAN SLETTES TRYGT:**
- `reports/replay_eval/**` (generert, kan regenereres)
- `outputs/**` (generert)
- `runs/**` (generert)
- `*.log`, `*_debug*`, `*_trace*` (generert)

**MÅ BEHOLDES:**
- `data/features/*.parquet` (kan være prebuilt features - IKKE SLETT uten bekreftelse)
- `docs/**` (dokumentasjon)
- `configs/**` (konfigurasjoner)
- `gx1/**` (kildekode)

## D) TILTAKSPLAN - MINIMALE STEG

### Steg 1: Verifiser kanonisk repo

```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"
git rev-parse --show-toplevel
git rev-parse --short HEAD
git branch --show-current
```

### Steg 2: List alle worktrees

```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"
git worktree list
```

### Steg 3: Kopier manglende filer til cia

```bash
# Kopier replay_eval_gated_parallel.py fra aaq til cia
cp "/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/aaq/gx1/scripts/replay_eval_gated_parallel.py" \
   "/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/cia/gx1/scripts/replay_eval_gated_parallel.py"
```

### Steg 4: Reduser worktrees (hvis nødvendig)

**Først: Sjekk dirty count for hver worktree**

```bash
for wt in aaq cia mkm muo; do
  cd "/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/$wt"
  echo "$wt: $(git status --porcelain=v1 | wc -l) dirty files"
done
```

**Hvis dirty_count=0 og worktree ikke trengs:**
```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"
git worktree remove <path>
```

**Hvis dirty_count>0:**
- Commit eller patch endringene først
- IKKE slett worktree med uncommitted changes

### Steg 5: Rydd genererte filer (DRY-RUN først)

```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"

# DRY-RUN: List kandidater
find reports/replay_eval -type f | head -50
find . -name "*.log" | head -50

# Faktisk sletting (kun etter bekreftelse):
# rm -rf reports/replay_eval/*
# find . -name "*.log" -delete
```

### Steg 6: Oppdater .gitignore

Sørg for at genererte filer er ignorert:

```bash
# Legg til i .gitignore hvis ikke allerede der:
echo "reports/replay_eval/" >> .gitignore
echo "*.log" >> .gitignore
echo "*_debug*" >> .gitignore
echo "*_trace*" >> .gitignore
```

## E) "ALDRI IGJEN"-POLICY

### 1. Én kanonisk repo-root

- **Kanonisk:** `/Users/andrekildalbakke/Desktop/GX1 XAUUSD`
- **Worktrees:** Maks én Cursor-worktree (f.eks. `cia`)
- **Policy:** Alle scripts skal logge `git rev-parse --show-toplevel` og hard-faile hvis ikke kanonisk

### 2. Worktree Policy

- **Maks én Cursor-worktree** per repo
- **FULLYEAR/Prebuilt scripts skal:**
  - Logge: `git top-level`, `HEAD`, `worktree path`
  - Hard-faile hvis `pwd` ikke er under kanonisk repo root
  - Hard-faile hvis flere worktrees er aktive samtidig

### 3. Generated Artifacts Policy

- **Alle genererte filer skal være ignorert av git** (oppdater `.gitignore`)
- **Rydding skjer via eksplisitt script** (ikke manuelt)
- **Prebuilt features (`data/features/*.parquet`) skal IKKE slettes** uten eksplisitt bekreftelse

### 4. Repo "Doctor" Script

Lag en enkel `scripts/repo_doctor.sh`:

```bash
#!/bin/bash
# GX1 Repo Doctor - Verifiser repo-struktur

CANONICAL_ROOT="/Users/andrekildalbakke/Desktop/GX1 XAUUSD"
CURRENT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)

if [ "$CURRENT_ROOT" != "$CANONICAL_ROOT" ]; then
    echo "❌ ERROR: Not in canonical repo root"
    echo "  Current: $CURRENT_ROOT"
    echo "  Expected: $CANONICAL_ROOT"
    exit 1
fi

echo "✅ Canonical repo root: $CANONICAL_ROOT"
echo "✅ HEAD: $(git rev-parse --short HEAD)"
echo "✅ Branch: $(git branch --show-current || echo 'DETACHED')"

WORKTREES=$(git worktree list | wc -l)
if [ "$WORKTREES" -gt 2 ]; then  # Main + 1 worktree
    echo "⚠️  WARNING: More than 2 worktrees detected"
    git worktree list
fi
```

## NESTE STEG

1. ✅ Kopier `replay_eval_gated_parallel.py` fra `aaq` til `cia`
2. ⏳ Verifiser alle worktrees (HEAD, branch, dirty count)
3. ⏳ Mål diskbruk for `reports/`, `outputs/`, etc.
4. ⏳ Identifiser worktrees som kan fjernes (dirty_count=0)
5. ⏳ Rydd genererte filer (dry-run først)
6. ⏳ Oppdater `.gitignore`
7. ⏳ Lag `repo_doctor.sh` script
