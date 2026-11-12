# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json eier arbeidsstatus; prosess og terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_DIRECTION_EDGE_CAUSE_SETUP_CONTEXT_COMPLETE_VALUE_UNRESOLVED.
Målet om å avklare og rette svak retningsverdi er fortsatt aktivt og uoppnådd.
Forrige målturn og denne ga faktisk fremdrift: fast læringssammenligning og deretter
kausal situasjonsanalyse er utført. Ikke gjenta dem eller åpne konsumert admisjon.

Neste avgrensede spørsmål er tidshorisonten for de samme situasjonene.
Kontroller først at lagrede F-queries, tidspunkt og native ATR-enheter faktisk matcher
mellom TRAIN/senere på alle fem TF. Deretter kan én separat prespesifisert avlesning
av alle allerede lagrede endepunkter vurderes. M5 beholdes som primært; velg ikke beste
horisont etterpå. Langhorisont krever stående LONG-referanse. Ingen slik scoring er admittert nå.
Ingen nye grupper, terskelsøk, targets, modellendring eller trening er tillatt av dette steget.

## Målt situasjonsanalyse

ROOT /home/andre2/GX1_RUNS/MARKET_STATE_TA_SETUP_CONTEXT_20261112_001.
Kilde d3699f64; PLAN84c54450, RESULT26cbfd65, TERMINAL984eeea7.
Samlet bevis: [TA_DIRECTION_EDGE_CAUSE_REVIEW.json](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json),
setup_context_probe_result. Alle37grupper og primære parrede intervaller er beholdt.

Gruppene ble låst før utfall: fire M5-nivåsituasjoner ved0..1ATR, tre M5-EMA-retninger,
og tre kontekster fra alle fire høyereTF. Fraværende nullnivåer har presence-vakt.
EMA-retning bruker20/50-orden og5-native-barshelning; tre av fire M15/H1/H4/D1 må samsvare.
Dette er én beskrivende probe, ikke en definisjon av alle manuelle tradingoppsett.

| Forhåndsvalgt situasjon | TRAIN | Senere | Senere bevegelse i retning / motsatt / FLAT |
| --- | ---: | ---: | ---: |
| Støtte + M5/HTF opp | 197 | 153 | 38 / 24 / 91 |
| Motstand + M5/HTF ned | 46 | 52 | 11 / 10 / 31 |
| Samlet | 243 | 205 | 49 / 34 / 122 |

Begge paneler har4096rader. FLAT her er F-bevegelse mindre enn0,5M5-ATR etter fem minutter,
ikke en handelsbeslutning. Samlet senere oppsettdekning er5,00prosent; ingen feature-UNKNOWN.
Forventet retningsbalanse utover frossen prior:0,06876;95prosent dagintervall[-0,01116;0,14565],
ukeintervall[-0,02128;0,13373]. Signal i denne proben er derfor ikke bekreftet.
Senere samlet F-treklasse-Brier: prior0,563346, INITIAL0,564316, FULL0,568738,
M5-only0,572955. FULL/prior og M5-only/FULL-intervallene inkluderer null.
FULLs betingede side-Brier er bedre i punktestimat; dette er ikke det samme som bedre
samlet sannsynlighet eller selektiv Entry. Alle grupper, FLAT og uklar støtte beholdes.

All-row S-treklasseskår viser noe verdi mot prior, men2443av4096senere S-rader har aktiv
ukjent side, og én er ineligible/immature. S-state og fremtidig F-retning er ulike mål.
Ingen gevinst i fremtidig retning eller økonomi er etablert gjennom denne forskjellen.

## Kilde og teknisk gjennomføring

EMA/ATR, registry-nivåer, alle femTF og åtte genuine familier finnes i aktiv modellkonsum.
HTF-registryavstand bruker siste HTF-close. Men aktuell M5-relative avstander til HTF-
ekstremer og prior-session-pivoter finnes også i aktive inputs. Ingen generell mangel
på prisgeometri er bevist; lært bruk er fortsatt uavklart.

Eksisterende253-feltscacher og kvalifiserte span-sidecars ble komponert med gammel eier
før canonical n=1-endepunktsuttrekk. Fulle modellvinduer var allerede kvalifisert.
8192rader ×5TF =40960endepunkter; seks lagrede captures, ti spanarkiver, ingen modell,
checkpoint, forward, fit eller ny featureberegning. Skåringsavvik mot originalene5,51e-16.
Capped4GiB/512MiB, tid14,84s, exit0. Engangsjobben er COMPLETE/CONSUMED:1/1/0.

## Tidligere direkte funn og grenser

Fast FULL mot M5-only: samme4096TRAIN/4096senere, INITIAL/RNG/rekkefølge, tre passeringer.
M5-only økte F TRAIN-AUC0,537→0,617, mens senere F-AUC ble0,496 og Brier/log svekket seg.
Å fjerne andre tapsledd løste ikke senere verdi i testen. Original benchmark og review er konsumert.
Ikke bruk dette som generell dom over multitask, genuine indikatorer eller fullhistorikk-kapasitet.

Global training/native_training, full epoch og fullVAL er false; ingen tung jobb kjører.
Kvalifiserte inputs/normalisering og beståtte kontroller gjenbrukes. Originale artefakter beholdes.
Korte runder er bevisst feilsøking. Full kvalifisert TRAIN-historikk og30–40epoker er senere skalering.
TEST, Entry/Exit-evaluering, broker, handel, spending og promotion er stengt; GC er på pause.
Root eier policy, launch og samlet kontroll.
