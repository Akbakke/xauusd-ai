# GX1

Offline XAUUSD-forskning med én lært Entry/Exit-bundle:
Entry på native M5, Exit på native M1, delte encoders og åtte kausale
featurefamilier. Teknisk konsistens er ikke dokumentert læring eller lønnsomhet.

Eneste arbeidsrepo er /home/andre2/src/GX1_CURRENT, branch work/gx1-current.
Les AGENTS.md og GX1_RULES.md før arbeid. Start med:

```bash
bash scripts/gx1_handover.sh --check
```

CURRENT_HANDOVER.md beskriver nåstatus; NEXT_RUN_POLICY.json er eneste
maskinlesbare arbeids-/kjøreautoritet. Ingen gamle checkpoints eller rapporter
gir launchautorisasjon. Benchmarken er avbrutt etter brukerens ønske; trening,
full epoch, full VAL, TEST og handel er stengt.

Dokumentasjon: DOC_INDEX.md. Arkitektur: SYSTEM_MAP.md.
Datakontrakt: docs/DATA_CONTRACT.md. Gjenværende læring: docs/NATIVE_LEARNING.md.
Repo-gjennomgang og opprydding: docs/REPO_REVIEW.md.

Kilde, kontrakter og fokuserte tester versjoneres. Rådata, modellvekter,
hemmeligheter og kjøringsoutput holdes utenfor repoet. Tunge jobber bruker
scripts/gx1_capped_run.sh og de eksisterende maskinvarevaktene.
