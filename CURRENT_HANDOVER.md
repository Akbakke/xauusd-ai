# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json eier arbeidsstatus; prosess og terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_DIRECTION_EDGE_CAUSE_FIXED_BENCHMARK_COMPLETE_RETAINED_REVIEW_ADMITTED.
Målet om å avklare og rette svak retningslæring er fortsatt aktivt og uoppnådd.
Den avtalte FULL_OBJECTIVE mot M5_DIRECTION_ONLY er ferdig, ikke en ny launch.
Bare én skåring av de ti lagrede captures er nå tillatt, med eksisterende skåringseier,
alle task/TF-resultater og parret dag-/ukeusikkerhet for M5. Ingen ny modell eller fit.
Stage: market_state_ta_m5_objective_benchmark_20261112_001, retained_review.
4GiB/512MiB via scripts/gx1_capped_run.sh; én invokasjon, 2000 resamples per gruppering.
Kjør REVIEW_FIXED_COMPARISON.py i benchmark-roten; avslutt admisjonen etter kvittering.

## Faktisk fullført sammenligning

ROOT /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_OBJECTIVE_BENCHMARK_20261112_001.
Kilde071724ae, recipe690ae741, RESULT069d4d7c, TERMINALd137b42b.
Campaign receiptb1947323 er COMPLETE, exit0. Windows boot645.
Kontroller avsluttet; GX1M5ObjectiveBenchmark001 er Disabled med siste resultat0.
Native scope er COMPLETE/CONSUMED: claimed1/completed1/remaining0, ingen relaunch.
Global training/native_training, full epoch og fullVAL er false.

Samme modell og genuine inputs, INITIAL, RNG, rekkefølge og queries i begge armer.
TRAIN4096 over mai–september2024; senere4096 over oktober–desember2024.
Tre passeringer per arm, måling0/1/3; pass3 er primært, pass1 er ingen checkpointseleksjon.
Én modell/optimizer,1536updates totalt,4096forwards,0skipped;768updates per arm.
INITIAL-restore, per-pass RNG-likhet, passive evalueringer og uendret kilde besto.
Alle genuine familier/femTF/D1 beholdt. Bare direkte tapsledd var forskjellen.
Foreløpig viser publiserte Brier/log-tall bedre TRAIN-fit, men svakere senere skårer
for M5-only enn FULL i begge M5-mål. FULL svekker også senere Brier/log fra INITIAL.
AUC og parret usikkerhet gjenstår; ingen samlet retningsgevinst eller økonomi er bevist.

## Brukerens spørsmål: støtte/motstand, ATR, EMA og høyere tidsrammer

Aktiv kilde bekrefter EMA-avstand/helning mot ATR, nivå-/trendlinjegeometri,
åtte familier og femTF i den lærte sammensettingen. Konsum beviser ikke at modellen
skiller nyttige oppsett fra ugunstige situasjoner. M5 S/F-målingene gjelder t+5min;
dette er mål, ikke maksimal holdetid. FLAT-mål er ikke demonstrert selektiv Entry.
To-arms-testen undersøker tapsinteraksjon, ikke lønnsomme oppsett eller økonomi.
Før en påstand om oppsettforståelse kreves kausale, forhåndsdeklarerte situasjonsgrupper
med alle utfall og tellinger på senere utviklingsdata. Ikke velg lønnsomme grupper etterpå.
Kildebevis og uavklarte grenser: docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json,
setup_alignment_question_20261112. Ingen ny input-/arkitekturkampanje følger automatisk.

## Gjenbruk og grenser

Kvalifiserte span-inputs og normalisering, mekaniske tester og faktisk INITIAL gjenbrukes.
Forrige to-spanfelt-kontroll er konsumert og viste ingen konsistent prediktiv gevinst.
Eksisterende batchrekkefølge-kontroll bedret totaltap, men svekket M5 Brier; ikke gjenta.
Originale resultater/checkpoints beholdes; eldre detaljer finnes i samlet review og Git.

Korte runder er bevisst feilsøking, ikke en dom over fullhistorikk-kapasitet.
Kvalifisert full TRAIN-historikk og 30–40 epoker er senere skalering.
Senere utviklingsdata er gjenbrukt, ikke uavhengig bekreftelse. AUC er ikke winrate.
TEST, Entry/Exit-evaluering, broker, handel, spending og promotion er stengt; GC er på pause.
Root eier policy, launch og samlet kontroll. Ingen andre tunge jobber er startet.
