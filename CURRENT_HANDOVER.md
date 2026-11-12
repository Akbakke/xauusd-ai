# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json eier arbeidsstatus; prosess og terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_DIRECTION_EDGE_CAUSE_FIXED_BENCHMARK_COMPLETE_NO_LATER_GAIN.
Målet om å avklare og rette svak retningslæring er fortsatt aktivt og uoppnådd.
Den avtalte FULL_OBJECTIVE mot M5_DIRECTION_ONLY og skåringen er ferdig/konsumert.
Neste spørsmål er direkte brukerens: skiller modellen nyttige kausale støtte/motstand-
oppsett med ATR, EMA og høyere tidsrammer fra ugunstige eller uinformative situasjoner?
Knytt eksisterende situasjonsfelt til lagrede senere prediksjoner; spesifiser grupper
og frosne referanser før gruppeutfall leses. Behold alle grupper, FLAT/uklare utfall og tellinger.
Ingen ny modell, trening, terskelsøk eller arkitekturendring er admittert.

## Målt resultat fra fast sammenligning

ROOT /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_OBJECTIVE_BENCHMARK_20261112_001.
Kilde071724ae, recipe690ae741, RESULT069d4d7c, TERMINALd137b42b.
FIXED_COMPARISONda37fcdd, review-terminalc7a44597. Samlet bevis og alle komponentavveininger:
[TA_DIRECTION_EDGE_CAUSE_REVIEW.json](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json),
fixed_objective_benchmark_result.

Samme modell/genuine inputs, INITIAL, RNG, rekkefølge og queries i begge armer.
TRAIN4096 over mai–september2024; senere4096 over oktober–desember2024.
Tre passeringer per arm, måling0/1/3; pass3 primært, ingen checkpointseleksjon.
Én modell/optimizer,1536updates totalt,4096forwards,0skipped;768updates per arm.
INITIAL-restore, per-pass RNG-likhet, passive evalueringer og uendret kilde besto.

| M5 betinget retning | INITIAL | FULL pass3 | M5-only pass3 |
| --- | ---: | ---: | ---: |
| S TRAIN AUC | 0,549685 | 0,564880 | 0,636526 |
| S senere AUC | 0,543411 | 0,548683 | 0,532370 |
| F TRAIN AUC | 0,537196 | 0,571561 | 0,616656 |
| F senere AUC | 0,511290 | 0,513672 | 0,495511 |
| S senere Brier | 0,249426 | 0,250519 | 0,258705 |
| F senere Brier | 0,249553 | 0,251796 | 0,257268 |

M5-only lærer TRAIN sterkere, men begge senere Brier/log-skårer svekkes mot FULL/INITIAL.
2000 parrede resamples over70UTC-dager og14uker: alle disse Brier/log-regresjonenes
95prosentintervaller ligger over null. AUC-punktestimatene svekkes, men forskjellenes
intervaller inkluderer null. Ingen konsistent senere gevinst; M5-only skal ikke promoteres.
Større TRAIN-fit og senere sannsynlighetsfeil er forenlig med overtilpasning/svak overføring;
den unike hovedårsaken er ikke bevist. Ingen generell dom over multitask eller kapasitet.
S-sidestøtte er1403TRAIN/1414senere; F1635/1689 av4096. Dette er utfallsbetinget støtte,
ikke selekterbar Entry-dekning. AUC er rangering, ikke treffprosent eller handelsresultat.
Alle ti captures og publiserte task/TF-skårer ble kontrollert med gammel skåringseier;
største skåringsavvik8,88e-16. Review brukte ingen modell/checkpoint/forward/fit og tok3,04s.

## Brukerens spørsmål og uavklarte grenser

Aktiv kilde bekrefter EMA-avstand/helning mot ATR, nivå-/trendlinjegeometri,
åtte genuine familier og femTF/D1 i den lærte sammensettingen. Konsum beviser ikke
nyttig situasjonsforståelse. M5 S/F-målingene gjelder t+5min; dette er mål, ikke holderegler.
FLAT som markedsmål beviser ikke at modellen kan avstå fra dårlige Entry-oppsett.
To-arms-testen avkreftet ingen universell hypotese: å fjerne andre direkte tapsledd
løste ikke senere M5-verdi i akkurat denne testen. Ikke fortsett med vilkårlig tapstuning.
Situasjonsanalysen må skille faktisk informasjon i oppsett fra modellens evne til å bruke den.
Ingen lønnsom gruppe velges etterpå. Senere data er gjenbrukt utvikling, ikke uavhengig bekreftelse.

## Drift, gjenbruk og avgrensing

Campaign receiptb1947323 er COMPLETE/exit0. Windows boot645.
Kontroller avsluttet; GX1M5ObjectiveBenchmark001 er Disabled med siste resultat0.
Native og review er COMPLETE/CONSUMED: hver claimed1/completed1/remaining0, ingen relaunch.
Global training/native_training, full epoch og fullVAL er false; ingen tung jobb kjører.
Kvalifiserte inputs/normalisering, INITIAL og beståtte kontroller gjenbrukes.
Spanfelt-kontrollen viste ingen konsistent gevinst; batchrekkefølge-kontrollen er allerede utført.
Originale artefakter/checkpoints beholdes; eldre detaljer finnes i review og Git.

Korte runder er bevisst feilsøking, ikke en dom over fullhistorikk-kapasitet.
Kvalifisert full TRAIN-historikk og 30–40 epoker er senere skalering.
TEST, Entry/Exit-evaluering, broker, handel, spending og promotion er stengt; GC er på pause.
Root eier policy, launch og samlet kontroll.
