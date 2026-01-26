# GIT WORKTREE POLICY - "ALDRI IGJEN"

**Dato:** 2025-01-13  
**Mål:** Forhindre "spøkelsesfeil" og repo-forvirring

## KANONISK REPO-ROOT

**Én kilde til sannhet:**
- **Path:** `/Users/andrekildalbakke/Desktop/GX1 XAUUSD`
- **Dette er main repository**
- **Alle scripts skal verifisere at de kjører fra denne root**

## WORKTREE POLICY

### Regel 1: Maks én Cursor-worktree

- **Tillatt:** Main repo + 1 Cursor-worktree (f.eks. `cia`)
- **Ikke tillatt:** Flere Cursor-worktrees samtidig
- **Håndheving:** `repo_doctor.sh` skal hard-faile hvis > 2 worktrees

### Regel 2: Worktree må være synkronisert

- **Tracked files skal være identiske** mellom worktrees på samme commit
- **Untracked filer skal dokumenteres** (spesielt kode-filer)
- **Hvis fil mangler i én worktree:** Kopier fra main eller annen worktree

### Regel 3: FULLYEAR/Prebuilt scripts skal verifisere repo

**Alle scripts skal logge:**
```python
import subprocess
from pathlib import Path

CANONICAL_ROOT = Path("/Users/andrekildalbakke/Desktop/GX1 XAUUSD")
current_root = Path(subprocess.run(
    ["git", "rev-parse", "--show-toplevel"],
    capture_output=True, text=True
).stdout.strip())

if current_root != CANONICAL_ROOT:
    raise RuntimeError(
        f"[REPO_FAIL] Not in canonical repo root. "
        f"Current: {current_root}, Expected: {CANONICAL_ROOT}"
    )

log.info(f"[REPO] Canonical root: {CANONICAL_ROOT}")
log.info(f"[REPO] HEAD: {subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}")
```

## GENERATED ARTIFACTS POLICY

### Regel 4: Genererte filer skal være ignorert

**Alle følgende skal være i `.gitignore`:**
- `reports/replay_eval/`
- `*.log`
- `*_debug*`
- `*_trace*`
- `outputs/`
- `runs/`

### Regel 5: Rydding skjer via eksplisitt script

**Ikke manuell sletting:**
- Bruk `scripts/cleanup_generated.sh` (eller tilsvarende)
- Alltid dry-run først
- **IKKE slett `data/features/*.parquet`** uten eksplisitt bekreftelse

### Regel 6: Prebuilt features er beskyttet

**`data/features/*.parquet` skal IKKE slettes:**
- Dette er prebuilt features
- Tar lang tid å regenerere
- Slett kun hvis eksplisitt bekreftet

## REPO DOCTOR

### Regel 7: Kjør repo_doctor.sh før kritiske operasjoner

**Før FULLYEAR/Prebuilt replay:**
```bash
./scripts/repo_doctor.sh || exit 1
```

**repo_doctor.sh skal sjekke:**
1. ✅ Er vi i kanonisk repo root?
2. ✅ Er worktree count <= 2?
3. ✅ Er HEAD på forventet commit/branch?
4. ✅ Er dirty count akseptabelt?

## FEILHÅNDTERING

### Regel 8: Hard-fail ved repo-feil

**Hvis repo-struktur er feil:**
- **Ikke "continue on error"**
- **Ikke "fallback to alternative path"**
- **Hard-fail med klar feilmelding**

**Eksempel:**
```python
if current_root != CANONICAL_ROOT:
    raise RuntimeError(
        f"[REPO_FAIL] Not in canonical repo root. "
        f"Current: {current_root}, Expected: {CANONICAL_ROOT}. "
        f"Instructions: cd {CANONICAL_ROOT} and retry."
    )
```

## DOKUMENTASJON

### Regel 9: Oppdater docs ved endringer

**Hvis worktree-struktur endres:**
- Oppdater `docs/GIT_WORKTREE_POLICY.md`
- Oppdater `scripts/repo_doctor.sh` hvis nødvendig
- Dokumenter hvorfor endringen var nødvendig

## SAMMENDRAG

**3 hovedregler:**
1. **Én kanonisk repo-root** - alltid verifiser
2. **Maks én Cursor-worktree** - reduser hvis flere
3. **Genererte filer ignorert** - rydd via script, ikke manuelt

**Håndheving:**
- `repo_doctor.sh` før kritiske operasjoner
- Hard-fail ved repo-feil
- Dokumenter endringer
