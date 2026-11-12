# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json eier arbeidsstatus; prosess og terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_DIRECTION_EDGE_CAUSE_NATIVE_HORIZONS_COMPLETE_TRANSFER_UNRESOLVED.
Målet om å avklare og rette svak retningsverdi er fortsatt aktivt og uoppnådd.
Forrige målturn utførte situasjonsproben; denne har faktisk skåret de samme gruppene
på alle fem eksisterende F-endepunkter med alltid-LONG. Begge planer er konsumert.

Neste konkrete skille: oppstår skadelig ekstra tilpasning også på opprinnelige TRAIN-rader
som ikke deltok i de siste4096-radenes oppdateringer, eller først i senere periode?
Gjenbruk eksisterende full-TRAIN-komplementmetode og inspiser dagens benchmark-evidensstater
for én matchet kvalifisering mot eksakt CURRENT INITIAL. Ingen fitting eller modellendring.
Dette er ikke urørt holdout: INITIAL/forhistorien kan allerede ha brukt full TRAIN-populasjon.
Eldre E3-komplementkontroll finnes i docs/TA_M5_E3_FULL_TRAIN_002_REVIEW.json og viste regresjon,
men gjelder annen modelltilstand5e031bd0 og blandet CPU/GPU-protokoll. Ikke gjenåpne gammel plan.
Ingen ny genuine data-/modellkjøring er admittert nå; source/metadata-forberedelse er neste handling.

## Ferdig avlesning av fem F-horisonter

ROOT /home/andre2/GX1_RUNS/MARKET_STATE_TA_SETUP_NATIVE_HORIZONS_20261112_001.
Kilde8b8ce60c; PLAN8f5a0c5d, RESULT74c64743, TERMINALf5c822c4.
Samlet bevis: [TA_DIRECTION_EDGE_CAUSE_REVIEW.json](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json),
setup_native_horizons_result. Alle37grupper og alle primære intervaller finnes i originalresultatet.

| F-horisont | FULL TRAIN-AUC | FULL senere AUC | Senere side-Brier prior / FULL |
| --- | ---: | ---: | ---: |
| 5min | 0,571561 | 0,513672 | 0,249696 / 0,251796 |
| 15min | 0,623356 | 0,509511 | 0,250709 / 0,307638 |
| 60min | 0,645103 | 0,495398 | 0,251352 / 0,316309 |
| 240min | 0,652020 | 0,499505 | 0,249256 / 0,309307 |
| 1440min | 0,714395 | 0,470691 | 0,252789 / 0,303766 |

M15/H1/H4/D1s senere side-Brier-regresjoner mot prior har positive95prosentintervaller
ved både parret UTC-dag og uke. M5-intervallet inkluderer null. AUC er rangering, ikke winrate.
Ingen senere gevinst er vist ved å velge en annen av disse horisontene.

Samme205samstemte situasjoner, med/uten gyldig F-mål:

| Horisont | I oppsettretning | Motsatt | FLAT_NET | UNKNOWN |
| --- | ---: | ---: | ---: | ---: |
| 5min | 49 | 34 | 122 | 0 |
| 15min | 51 | 42 | 112 | 0 |
| 60min | 29 | 44 | 123 | 9 |
| 240min | 44 | 33 | 116 | 12 |
| 1440min | 54 | 34 | 60 | 57 |

Oppsettenes merverdi mot alltid-LONG har intervaller som inkluderer null på alle fem horisonter.
Ingen beste gruppe/horisont velges. HverTF har egen ATR_t og0,5terskel, så tid, skala og
støttemengde endres sammen. Dette isolerer ikke effekten av å vente lenger.
FLAT_NET er netto sluttpunktsbevegelse, ikke fravær av bevegelse underveis eller en Entry-regel.

Alltid-LONG er eksakt(0,0,1); uendelig logloss ved motsatt/FLAT fasit rapporteres eksplisitt,
ikke klippet. Frosne TRAIN-priors er separat endelig sannsynlighetsreferanse. Klassebalanse er ikke PnL.
På alle4096senere rader er råUNKNOWN0/24/141/363/932 for M5/M15/H1/H4/D1, separat fra score-counts.
Eligible-but-immature er0. Alle rader avstemmes; ukjent fylles aldri som FLAT.

## Gjennomføring og gjenbruk

Native F-query[1,3,12,48,288], eksakt modenhetsregel, uendrede grupper og målakser besto.
Seks eksisterende captures og én lagret gruppetildeling; ingen feature-/targettape-lesing,
modell, checkpoint, forward eller fit. Gamle all-TF-skårer ble gjenskapt innen7,31e-16,
og M5-proper-score eksakt. Capped4GiB/512MiB,2,90s,exit0. COMPLETE/CONSUMED:1/1/0.

Opprinnelig situasjonsprobe: MARKET_STATE_TA_SETUP_CONTEXT_20261112_001, RESULT26cbfd65,
TERMINAL984eeea7. Fire M5-nivåkategorier ved0..1ATR ×tre M5-EMA-retninger ×tre HTF-kontekster,
alle36grupper pluss UNKNOWN. Tre av fire M15/H1/H4/D1 må samsvare; terskler er ikke tunet.
Aktuelle M5-relative HTF-ekstrem-/pivotavstander finnes allerede i modellinputs.
Ingen generell mangel på prisgeometri er bevist; lært bruk er fortsatt uavklart.

Fast FULL/M5-only-læringssammenligning er også konsumert. M5-only lærer sitt TRAIN-utvalg
sterkere, men svekker senere sannsynligheter. Ikke fortsett med vilkårlig tapstuning eller indikatorfjerning.

Global training/native_training, full epoch og fullVAL er false; ingen tung jobb kjører.
Kvalifiserte inputs/normalisering og beståtte kontroller gjenbrukes. Originale artefakter beholdes.
Korte runder er bevisst feilsøking. Full kvalifisert TRAIN-historikk og30–40epoker er senere skalering.
TEST, Entry/Exit-evaluering, broker, handel, spending og promotion er stengt; GC er på pause.
Root eier policy, launch og samlet kontroll.
