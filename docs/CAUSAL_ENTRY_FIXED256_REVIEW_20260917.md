# Kausal Entry256 — vurdert, læringsport ikke bestått

Den parete CPU-analysen er fullført. Den kausale targetrettelsen fjerner den
bekreftede etterpåklokskapen, men denne prøven viser ikke nyttig Entry/Exit-læring.
Ingen ny trening, forward, CONTROL, VAL eller TEST ble brukt i analysen.

Samme TRAIN256/256 Exit-ankere/1024 samplede states, alle ni måneder og begge
retninger er sammenlignet. Gamle Entry-prediksjoner vurderes mot korrekt
avledet fasit; Exit-målene er uendret. Originalfiler/checkpoints er bevart.
Checkpoint-payload, modell, lærer, cursor og terminalkvittering er verifisert.
Gamle baseline-metrikker er uavhengig gjenskapt. Eksisterende CPU-vakt ble brukt.

## Entry

| Side | Initial MSE | Connected256 MSE | Kausal256 MSE | TRAIN-konstant |
|---|---:|---:|---:|---:|
| LONG | 634.3625 | 649.5105 | 612.8130 | 608.8028 |
| SHORT | 684.5889 | 681.4859 | 607.4677 | 603.8143 |

Lavere samlet feil kommer fra felles nivå, (LONG+SHORT)/2. Predikert fellesmiddel
er −5,8500 Bps, mot target −5,8550. LONG−SHORT-MSE blir derimot dårligere:
2439,83 mot initial 2435,27 og connected 2429,25. Sentrert kontrastfeil blir også
verre. Kontrastkorrelasjonen er 0,0515, mot 0,0918/0,1034. Prediksjonens
kontrast-standardavvik er 0,1030 Bps, mot target 49,2394. Dette er observasjon av
nesten konstante utdata, ikke bevis for én bestemt kode-/tapsfeil.

Alle 256 Entries velger FLAT: referanseverdi 0 og regret 12,0865 Bps, identisk
konstantbaseline. Dette unngår forrige kandidats negative referanseverdi,
men dokumenterer ikke evne til å finne gode innganger. Fasitens beste realiserte
handlingsvalg er et etterpåklokt diagnostisk sammenligningstak, aldri en
kausal handlingsfasit eller bevis på at alle disse hendelsene kan forutsies.

## Exit

| Måling | Side | Initial MSE | Connected256 MSE | Kausal256 MSE | TRAIN-konstant |
|---|---|---:|---:|---:|---:|
| exit_anchor | LONG | 629.6484 | 624.2090 | 625.7888 | 625.0369 |
| exit_anchor | SHORT | 627.7748 | 620.3908 | 622.3509 | 622.6115 |
| exit_sampled | LONG | 864.7877 | 861.9765 | 864.7067 | 864.7101 |
| exit_sampled | SHORT | 859.5027 | 855.6962 | 858.7182 | 859.4202 |

Alle LONG-states velger HOLD, alle SHORT-states EXIT, både ved inngang og på
samplede states. Alle fire MSE er dårligere enn forrige kandidat. Samplet
referanseverdi er −0,1634/0 Bps mot konstante TRAIN-valg 0/+0,1071.
Bootstrap-komponenten er liten: gjennomsnittlig absolutt omtrent 0,04–0,05 Bps,
mot observert komponent 17–18 Bps. Det beviser ikke fravær av andre målproblemer.

## Alle TRAIN-måneder

Hver celle er kausal MSE / samme globale TRAIN-konstant. Konstanter er ikke
tilpasset måned for måned. Alle øvrige metrikker/valg finnes i JSON-rapporten.

| Måned | Entry LONG | Entry SHORT | Samplet Exit LONG | Samplet Exit SHORT |
|---|---:|---:|---:|---:|
| 2025-06 | 373.35 / 359.09 | 377.83 / 362.42 | 337.00 / 339.59 | 335.66 / 338.49 |
| 2025-07 | 352.41 / 355.66 | 350.35 / 354.84 | 236.10 / 237.20 | 235.98 / 237.29 |
| 2025-08 | 54.00 / 56.94 | 53.58 / 58.66 | 156.32 / 160.22 | 156.60 / 160.55 |
| 2025-09 | 377.26 / 347.58 | 370.40 / 343.77 | 268.54 / 276.58 | 273.07 / 281.84 |
| 2025-10 | 892.03 / 891.05 | 878.26 / 879.30 | 1149.72 / 1142.04 | 1138.31 / 1133.24 |
| 2025-11 | 455.26 / 448.90 | 459.21 / 452.59 | 537.19 / 539.31 | 535.97 / 538.64 |
| 2025-12 | 1057.38 / 1082.19 | 1041.51 / 1065.39 | 1465.06 / 1446.43 | 1456.27 / 1437.98 |
| 2026-01 | 1097.74 / 1081.73 | 1072.81 / 1052.53 | 2631.70 / 2642.42 | 2596.26 / 2608.14 |
| 2026-02 | 998.84 / 1010.17 | 1024.98 / 1036.91 | 1488.42 / 1488.59 | 1485.88 / 1485.91 |

Entry slår konstantens MSE i 4/9 og 5/9 måneder. Samplet Exit forbedrer MSE mot
connected256 i bare 1/9 LONG-måneder og 0/9 SHORT-måneder. Alle måneders
handlingsvalg følger de samme konstante mønstrene.

## Konklusjon og neste handling

REJECT_EXPANSION_CAUSAL_ENTRY_ALL_FLAT_EXIT_FIXED_BY_SIDE. Ingen automatisk
utvidelse eller ny kjøring. TRAIN med fitted-overlapp er ikke generalisering;
referansepolicyverdi er ikke realisert profitt for modellens egen handelsstrategi.

Neste konkrete undersøkelse er om tilstandsavhengig LONG−SHORT-feil når fram
til den faktiske vektede/klippede optimizeroppdateringen. Start med eksisterende
TRAIN-outputs, checkpoint-/optimizerstate, lagrede gradientbevis og kilde.
Entry bruker allerede MSE; ikke anta Huber-klipping eller innfør en tapsendring
fordi utdata har liten variasjon. Gjenbruk tidligere detach-diagnose. En ny
måling må avgrenses og bindes særskilt før utførelse; ingen blind ekstra trening.

Varige artefakter: NATIVE_CAUSAL_ENTRY_FIXED256_20260917/{PAIRED_TRAIN_REVIEW.json,
VERDICT.json,REVIEW_OPERATOR.py,PAIRED_REVIEW.log} under vanlig BASE.
Offentlige kopier inneholder bare aggregater og bindinger, ingen rådata/vekter.
