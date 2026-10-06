# Dokumentindeks

## Gjeldende arbeidsautoritet

- GX1_RULES.md — bindende regler og lukkede omfang.
- AGENTS.md / CLAUDE.md — samme regler for begge agenter.
- CURRENT_HANDOVER.md / NEXT_RUN_POLICY.json — nåstatus og eksakte bindinger.
- GX1_ARBEIDSMAAL.md / VEIEN_VIDERE.md — mål og neste grense.
- SYSTEM_MAP.md — retained kildekjede og ansvarsgrenser.
- docs/DATA_CONTRACT.md — kilder, kausalitet, lineage og normalisering.
- docs/LEARNING_GATE.md — læring, generalisering og økonomi holdes adskilt.
- docs/NATIVE_LEARNING.md — ufullførte punkter 1–6; ingen aktuell launchordre.
- docs/REPO_REVIEW.md — funn, opprydding, kontroller og ubeviste grenser.
- docs/GC_ORDER_FLOW_RESEARCH.md — avgrenset GC-kildeaudit og A/B/C-protokoll;
  ekte GC-inputs og empirisk effektmåling gjenstår, ingen native launch.

## Bevarte kilde-/designbindinger

configs/research/NATIVE_V38_{INPUT_PREPARATION,LEARNING_DESIGN,M1_REALIGNMENT}_20261001.json
er frosne, fullførte v38-bindinger. Ikke rediger dem for dokumenthygiene.

docs/RISK_OBJECTIVE_20260914.json er fortsatt eksplisitt hash-bundet av policy.
V29_EVENT_SURFACE_DESIGN_20260811.md og INDICATOR_FIDELITY_AUDIT_20260813.md
er fortsatt referert av de aktive feature-eierne. PROJECT_DEEP_REVIEW_20260919.md
forklarer den beholdte kausale målkontrakten. PREREGISTERED_DIRECTION_TEST_20260820.md
og de tre *BASELINES/MECHANISMS_PREREG_20260927.md-dokumentene binder fortsatt
beholdte offline diagnoseinstrumenter. De er ikke nye kjøretillatelser.

Avsluttede engangsrapporter, doble statusfiler og gamle forsøksconfigs lagres
ikke som repo-fyll. Tidligere versjoner finnes i Git. Resultater, logger,
checkpoints og aktive inputmanifester utenfor repoet er ikke slettet.
