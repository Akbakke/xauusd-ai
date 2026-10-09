# GX1

Offline XAUUSD-forskning med én lært Entry/Exit-bundle:
Entry på native M5, Exit på native M1, delte encoders og alle genuine
featurefamilier. Teknisk konsistens er ikke dokumentert læring eller lønnsomhet.

Eneste arbeidsrepo er /home/andre2/src/GX1_CURRENT, branch work/gx1-current.
Les AGENTS.md og GX1_RULES.md før arbeid. Start med:

```bash
bash scripts/gx1_handover.sh --check
```

CURRENT_HANDOVER.md er den ene siste overleveringen: gjort, ikke gjort,
aktuell hindring og neste steg. NEXT_RUN_POLICY.json eier maskinlesbar status,
eksakte bevisbindinger og kjøregrenser. Historiske dokumenter/checkpoints
gir ingen launchautoritet; Git bevarer tidligere oppdateringer.

Dokumentasjon: DOC_INDEX.md. Eiere: SYSTEM_MAP.md.
Datakrav: docs/DATA_CONTRACT.md. Metode: docs/NATIVE_LEARNING.md.
Læringsport: docs/LEARNING_GATE.md. Siste oppryddingsgrense: CURRENT_HANDOVER.md.

Kilde, kontrakter og fokuserte tester versjoneres. Rådata, modellvekter,
hemmeligheter og kjøringsoutput holdes utenfor repoet. Tunge jobber bruker
scripts/gx1_capped_run.sh og eksisterende maskinvarevakter.
