# GX1 — klar for første større ettårslæring, 2026-09-14

Ingen trainer kjører. Windows-task GX1NativeLearningCalibration er Disabled,
siste controller exit 0; fysisk boot 423. NEXT_RUN_POLICY.json aktiverer nå
ettårsutvalget etter fullførte native målinger. Neste handling er å binde en
ren, pushet kilde til ny native recipe og campaign, deretter starte på fersk
fysisk boot. Ingen ny femårs-epoch nå.

TRAIN er 2025-06-01 inklusiv til 2026-06-01 eksklusiv: 65 295 av 313 399
opprinnelige rader. Hele juni-VAL beholder 5 508 rader. Full parenthistorikk,
normalisering, alle 200 features, åtte familier og tidsrammer beholdes.
Modellen videreføres fra femårsfortrening; juni er utviklings-VAL og TEST er
forseglet. Eksisterende øvre ramme er 30 epocher, VAL hver epoch og patience 5.
Første komplette ettårs-epoch og juni-VAL er neste økonomiske vurderingspunkt.

## Ferdige porter — kilde 128c55f2

- Faktisk checkpointovergang og separat CPU-gjenspilling består. Original
  checkpoint 315 / 19 908 steg er uendret. Bare den avtalte Exit-utgangen og
  tilhørende Adam-tilstander nullstilles ved ny v4-overgang; øvrig tilstand bevares.
- Reference 32 steg på boot 421 og split 16 + 16 på boot 422/423 har guard PASS,
  trainer/observer/controller exit 0. Faktiske typed digests er identiske for
  alle 14 sammenlignede komponenter, inkludert modell, target, Adam, EMA,
  scheduler, RNG og treningsrekkefølge. Bare session-/checkpoint-identitet er ulik.
- GPU-batch 256 gir identiske handlinger, største Q-avvik 1.1641532182693481e-10
  Bps og målt inferensspeedup 3.313226 mot evaluatorens referanseoppsplitting.
- Reference-vinduet behandlet 18 852 969 VAL-tilstander på 10 942.512 sekunder
  inkludert oppsett, 1 722.91 tilstander/s. Native total var 11 707.448 sekunder
  inkludert TRAIN32. Samlet relativ speedup er ikke målt. Hele juni er ikke
  fullført i dette kapasitetssnapshotet; det kan ikke velge checkpoint.
- Første lærerbatch velger FLAT 13 / LONG 1 / SHORT 2. Studenten velger fortsatt
  FLAT 0 / LONG 4 / SHORT 12. Frozen HOLD-bootstrap er 0 og HOLD-target følger
  faktisk relativ reward. Forecast-gradient når de fire undersøkte Entry-rutene,
  mens Entry-Q/Exit-gradientene ikke gjør det. Dette er læringsberedskap,
  ikke bevis på kalibrering, profitt eller nyttig samarbeid mellom alle features.

Bevis og begrensninger:
[Samlet beredskap](handover_snapshot/NATIVE_YEAR_LEARNING_READINESS_20260914.json)
og de tilhørende NATIVE_YEAR_*_128C55F2.json-filene. Originale native artifacts
ligger under LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_YEAR_CALIBRATION_EVIDENCE_128C55F2.
Målmodellen oppdateres først etter komplett epoch og VAL. Første epoch lærer
derfor mot den frosne startlæreren; prediktiv kalibrering må måles etter trening.

## Neste kjøring

Bruk ny ordinær native recipe uten native_calibration, med samme v4-økonomimål,
close_now_baseline_v1 og bundet ettårsrot. Start fra den bevarte v2-origin315;
ikke innfør en ny overgang fra de private kontrollsesjonene. Kontrollvekter
og rapporter beholdes. Numerisk kilde er uendret etter de beståtte målingene.

Bruk bare /home/andre2/src/GX1_CURRENT, branch work/gx1-current, via native
campaign og gx1_capped_run.sh. TRAIN16, VAL256, åtte CPU-arbeidere, tre timers
VAL-vinduer, FP32 og alle eksisterende maskinvarevakter beholdes. Start med
bash scripts/gx1_handover.sh --check. Historiske kildekopier er avhengigheter.

Ingen fast holde-/tapsgrense. Vurder samlet kostnadsjustert cash og åpen verdi,
selektivitet og Exit-atferd. Den gamle junistatistikken for 2 227 lukkede av
5 508 handler er ikke samlet profitt; 3 281 var HOLD ved månedsslutt.
Bruk bare første gamle epochs uforanderlige EMA ved sammenligning med gammel juni.

Én tung jobb samtidig; root eier oppstart og commits. Underagenter er autorisert.
Kontroller stabil drift omtrent hver time. Lokal RUNNING_NATIVE_CALIBRATION.json
og CURRENT_HANDOVER.md får faktisk runtime/plan etter oppstart. Ikke gjenta
beståtte målinger eller endre frosset kilde mens kampanjen kjører.
