# GX1 XAUUSD

Krev målbar læring før mer omfattende trening. Lønnsomhet er ikke dokumentert.
Eneste kodebase: /home/andre2/src/GX1_CURRENT, branch work/gx1-current.

Start på Mac med ./handover.sh --check, eller i Linux med
bash scripts/gx1_handover.sh --check. Les deretter:

1. CURRENT_HANDOVER.md — én gjeldende status og eksakt neste handling.
2. GX1_ARBEIDSMAAL.md og AGENTS.md — mål, rammer og arbeidsregler.
3. NEXT_RUN_POLICY.json — faktisk tillatt kjørescope; ingen full epoch nå.
4. docs/LEARNING_GATE_20260916.md — hva som må måles før utvidelse.

I scriptets JSON gjelder current_work dagens arbeidskopi. De øvrige gamle
kilde-/checkpointfeltene er merket completed_run_history. Operatørnotater og
observerte prosesser skilles fra hverandre; ingen håndover starter trening.
COMPLETED_RUN.json bevarer gammel fullført kjøring. handover_snapshot/ inneholder
bevis og eksplisitt merket historikk, aldri alternative oppstartsplaner.
SYSTEM_MAP.md beskriver læringsbanen. Mac-wrapperen eies av
scripts/macos/gx1_takeover.sh i Linux-repositoriet.
