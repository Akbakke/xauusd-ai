# FASE 1 VERIFISERING - SCRIPT INSTRUKSJONER

**Dato:** 2025-01-13  
**Script:** `scripts/fase1_verifisering.sh`

## HVORDAN KJØRE SCRIPTET

### Fra kanonisk repo (Desktop):

```bash
cd "/Users/andrekildalbakke/Desktop/GX1 XAUUSD"
./scripts/fase1_verifisering.sh
```

### Fra cia worktree:

```bash
cd "/Users/andrekildalbakke/.cursor/worktrees/GX1_XAUUSD/cia"
./scripts/fase1_verifisering.sh
```

## OUTPUT

Scriptet skriver all output til:
```
reports/ops/FASE1_VERIFISERING_RUN_YYYYMMDD_HHMMSS.txt
```

Filen opprettes automatisk hvis `reports/ops/` ikke eksisterer.

## INNHOLD I OUTPUT-FILEN

1. **SUMMARY** - Kort oppsummering øverst
2. **Kanonisk repo-info** - HEAD, branch, remote, worktrees
3. **Per worktree** - Detaljert info for aaq, cia, mkm, muo
4. **Støyanalyse** - Topp 20 mapper + topp 50 linjer
5. **Untracked store filer** - Filer > 1MB
6. **Diskbruk** - Størrelse på reports/, outputs/, etc.

## EKSEMPEL OUTPUT

```
==================================================================================
FASE 1 VERIFISERING - FULL KARTLEGGING
==================================================================================
Dato: Mon Jan 13 10:30:45 CET 2025
Output-fil: reports/ops/FASE1_VERIFISERING_RUN_20250113_103045.txt

==================================================================================
SUMMARY
==================================================================================

Kanonisk repo: /Users/andrekildalbakke/Desktop/GX1 XAUUSD
  HEAD: abc1234
  Branch: main
  Dirty files: 42

Worktrees:
  aaq: HEAD=abc1234, Branch=DETACHED, Dirty=0, Has_replay=YES
  cia: HEAD=abc1234, Branch=DETACHED, Dirty=0, Has_replay=NO
  mkm: HEAD=def5678, Branch=DETACHED, Dirty=5, Has_replay=UNKNOWN
  muo: HEAD=abc1234, Branch=DETACHED, Dirty=0, Has_replay=YES

...
```

## ETTER KJØRING

1. Les output-filen: `reports/ops/FASE1_VERIFISERING_RUN_*.txt`
2. Sjekk SUMMARY-seksjonen for rask oversikt
3. Gå gjennom detaljene for hver worktree
4. Identifiser hvilke mapper som skaper støy
5. Vurder FASE 2-6 basert på funnene

## SIKKERHET

Scriptet er 100% ikke-destruktivt:
- ✅ Kun lesing (git status, git rev-parse, etc.)
- ✅ Ingen sletting
- ✅ Ingen flytting
- ✅ Ingen git-kommandoer som endrer state
- ✅ Håndterer manglende mapper gracefully
