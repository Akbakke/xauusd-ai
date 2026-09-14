# Gjeldende GX1-regler

AGENTS.md inneholder brukerens arbeidsregler. GX1_ARBEIDSMAAL.md angir målet.
CURRENT_HANDOVER.md, COMPLETED_RUN.json og NEXT_RUN_POLICY.json er de eneste
operative overtakelsesdokumentene. Ingen eldre prosedyre kan overstyre disse.

Treningen er stoppet etter full juni-VAL for første femårs-epoch. Ikke tren
videre før risiko-/holdemålet og de påkrevde bevisene er avklart.
Bevar fullført TRAIN, hele VAL, uforanderlig EMA og lagret checkpoint.
TEST er forseglet. Ingen live-/papirhandel, promotering eller ekstern spending.

Bare eksisterende native campaign gjennom gx1_capped_run.sh kan trene.
Neste profil krever batch 256/åtte CPU-arbeidere/tre timers VAL-vinduer,
FP32 uten TF32, alle features, timeframes/familier og uendrede kostnader.
Native budsjett er 12 000 sekunder; ytre vakt 13 800 sekunder.
Ingen stille tilbakegang til et tregere oppsett. GPU-paritet og samlet fart
må bevises; den tidligere CPU-målingen alene er ikke tilstrekkelig.
GPU-clock-launcheren brukes kun under faktisk last, med reset i tomgang.
Alle eksisterende strøm-, temperatur-, RAM-, VRAM-, cgroup- og telemetrigrenser gjelder.
Én agent/én tung jobb. Menneskelig status maksimalt hver time ved lange kjøringer.
