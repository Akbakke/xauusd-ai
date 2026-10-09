# Dokumentindeks

## Les ved overtakelse

- GX1_RULES.md og AGENTS.md / CLAUDE.md — bindende regler.
- CURRENT_HANDOVER.md — kun siste status, gjort/ikke gjort og neste steg.
- NEXT_RUN_POLICY.json — aktuelle kjøregrenser og eksakte bevisbindinger.

`bash scripts/gx1_handover.sh --check` viser kompakt nåstatus, genuin siste
terminal, CURRENT-prosesser og kildeidentitet. `--verbose` legger til bare
den siste overleveringen. Gamle mål-/veikart-/renselogger finnes i Git.

## Les bare når oppgaven trenger det

- SYSTEM_MAP.md — retained kjede og eiere.
- docs/DATA_CONTRACT.md — kilder, kausalitet, lineage og normalisering.
- docs/LEARNING_GATE.md — læring, generalisering og økonomi.
- docs/NATIVE_LEARNING.md — gjenværende metode, ikke launchordre/daglig logg.
- docs/GC_ORDER_FLOW_RESEARCH.md — pauset GC-kildeaudit/A/B/C-protokoll.

## Beholdte immutable kilde-/designbindinger

Frosne configs/research/NATIVE_V38_* og GC-protokoller redigeres ikke for
dokumenthygiene. docs/RISK_OBJECTIVE_20260914.json er hash-bundet av policy.
V29_EVENT_SURFACE_DESIGN_20260811.md, INDICATOR_FIDELITY_AUDIT_20260813.md og
PROJECT_DEEP_REVIEW_20260919.md er referert av beholdte kontrakteiere.
PREREGISTERED_DIRECTION_TEST_20260820.md og de tre
*BASELINES/MECHANISMS_PREREG_20260927.md-dokumentene binder beholdte offline
diagnoseinstrumenter. Ingen av dem er en aktuell kjøreordre.

Bevis-/retentionregistre, rådata, checkpoints og aktive inputmanifester er
ikke fylldokumenter. Avsluttede statusoppdateringer beholdes ikke som nye
repoarkiver; tidligere versjoner gjenopprettes fra Git.
