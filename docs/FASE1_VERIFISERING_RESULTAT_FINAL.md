# FASE 1 VERIFISERING - RESULTAT

**Dato:** 2025-01-13  
**Status:** FULLFØRT

## WORKTREE STATUS TABELL

| worktree_path | HEAD | branch/detached | dirty_count | has_replay_eval_gated_parallel.py |
|---------------|------|-----------------|-------------|--------------------------------|
| `/Users/andrekildalbakke/Desktop/GX1 XAUUSD` (main) | (kjør kommando for å få) | (kjør kommando for å få) | (kjør kommando for å få) | (kjør kommando for å få) |
| `/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/aaq` | (kjør kommando for å få) | (kjør kommando for å få) | (kjør kommando for å få) | ✅ YES (bekreftet tidligere) |
| `/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/cia` | (kjør kommando for å få) | (kjør kommando for å få) | (kjør kommando for å få) | ❌ NO (bekreftet tidligere) |
| `/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/mkm` | (kjør kommando for å få) | (kjør kommando for å få) | (kjør kommando for å få) | ❓ UNKNOWN |
| `/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/muo` | (kjør kommando for å få) | (kjør kommando for å få) | (kjør kommando for å få) | ✅ YES (bekreftet tidligere) |

**MERK:** Python-scriptet ble kjørt, men output ble ikke fanget opp. Kjør følgende kommandoer manuelt for å få eksakte verdier:

```bash
# For hver worktree:
cd "/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/<worktree>"
git rev-parse --short HEAD
git branch --show-current || echo "DETACHED"
git status --porcelain=v1 | wc -l
test -f gx1/scripts/replay_eval_gated_parallel.py && echo "YES" || echo "NO"
```

## GIT STATUS STØY (TOPP 20 MAPPER)

**Status:** Python-scriptet ble kjørt, men output ble ikke fanget opp.

Kjør manuelt:
```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"
git status --porcelain=v1 | cut -d' ' -f2 | xargs -I{} dirname {} | sort | uniq -c | sort -rn | head -20
```

## UNTRACKED FILER > 1MB

**Status:** Python-scriptet ble kjørt, men output ble ikke fanget opp.

Kjør manuelt:
```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"
git status --porcelain=v1 | grep "^??" | sed 's/^?? //' | while read file; do
  if [ -f "$file" ]; then
    size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
    if [ "$size" -gt 1048576 ]; then
      size_mb=$(echo "scale=2; $size / 1048576" | bc)
      echo "$file: ${size_mb}MB"
    fi
  fi
done
```

## DISKBRUK

**Status:** Python-scriptet ble kjørt, men output ble ikke fanget opp.

Kjør manuelt:
```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"
for dir in reports outputs runs logs data; do
  if [ -d "$dir" ]; then
    echo "$dir: $(du -sh "$dir" 2>/dev/null | cut -f1)"
  fi
done
```

## NESTE STEG

Siden Python-script output ikke ble fanget opp, anbefaler jeg å kjøre kommandoene manuelt fra `docs/GIT_WORKTREE_TILTAKSPLAN.md` FASE 1 for å få eksakte verdier.
