# C — negativ observert økonomi, statistisk INKONKLUSIV

Målingen er fullført fra ren kilde dab107494538a3e8afdf90bdd0e220641783836c,
etter commit av [registreringen](TA_C_PREREG_20260929.md). Ingen GO.
Én fryst kombinasjon, ingen søk eller omvalg etter resultatet.

## Populasjon og integritet

Juni2025–juni2026:395 kalenderdager,1 649 valgte muligheter,
1 646 utførbare aktive innganger,3 utløpte uten inngang. LONG1 022 / SHORT627.
45 konfliktrader,83 like signalduplikater og5 143 signalrader under opptatt
posisjonsplass. Ingen terminalt sensurerte posisjoner i den faktiske kjøringen.
Alle utløpte og ikke-fylte muligheter inngår i nevneren.

Hele perioden er gjenbrukt utviklingsevidens. Tidligere61-cellers post-hoc valg
og tidligere VAL-bruk blir ikke gjort uavhengig av denne registreringen.
Ingen TEST-utfall eller native modellfit. Inputenes2026-råfil ble hash-lest,
mens dekodede prisrader og utfall ble avgrenset før juli2026.

TERMINAL er COMPLETE. Alle53 artefakters størrelser/hasher er verifisert.
En separat capped audit av cache og kildequotes bekreftet samtlige1 291
passive berøringer, neste-open-klokken, felles utvalg, kostidentitet og
kontantavstemming for alle24 portefølje-/kostscenarier. Ingen ny fit eller søk.

## Kostnader og samlet økonomi

Rå bps per valgt mulighet, inkludert null ved ingen inngang/fylling.
Historisk finansieringsproxy inngår; slippage er per markedsutførelse.
Aktiv/LONG betaler ved inngang og utgang, passiv bare ved utgang.

| Slippage bps | Aktiv | Samme utvalgs LONG | Passiv modell |
|---:|---:|---:|---:|
| 0,0 | -0,550 | -0,324 | -0,614 |
| 0,5 | -1,548 | -1,322 | -1,005 |
| 1,0 | -2,547 | -2,320 | -1,397 |
| 2,0 | -4,543 | -4,317 | -2,180 |

Aktiv mid-bevegelse er+1,465 bps mot spread1,972, slippage1,997 og
finansiering0,043 ved1-bps-scenarioet: netto−2,547. LONG har større mid-bevegelse,
+1,777 bps, men blir også negativ etter kost.

Risiko bruker samme kausale regel og faste initialbudsjett. Tallene nedenfor er
samlet prosent av initialkapital over hele perioden, ikke årlig avkastning.
Sharpe beregnes på kalenderdager mot null kontantrente; drawdown er på quote-gridet.

| Arm,1 bps og finansiering | Samlet netto | Sharpe | Maks drawdown |
|---|---:|---:|---:|
| Aktiv inngang | -24,81 % | -1,81 | 28,63 % |
| LONG på samme muligheter | -20,28 % | -2,12 | 21,71 % |
| Passiv berøringsmodell | -14,91 % | -1,13 | 19,53 % |

Passiv uten finansiering:−14,62 % samlet. Selv ved null slippage er passiv
netto negativ (−0,614 bps per valgt mulighet med finansiering).
De faktiske daglige risikoene er ulike: målt årlig vol er13,99 % aktiv,
9,64 % LONG og12,51 % passiv. Felles risikoregel er ikke lik realisert volatilitet.

## Fyllingsutvalget forklarer en vesentlig begrensning

1 291 av1 646 plasserbare ordre får berøring:78,43 %. Det er78,29 % av alle
1 649 valgte muligheter.355 plasserbare ordre får ingen berøring;3 muligheter
utløper før inngang. Dette er barberøring, ikke observerte ordreutførelser.

På de1 291 berørte mulighetene er etterfølgende mid-bevegelse i signalets
retning−0,751 bps. De355 utførbare uten berøring har+9,536 bps.
Aktiv inngang ville gitt−4,767 netto bps på berøringsutvalget, mot+5,507 på de
utførbare uten berøring. Den passive modellen gir−1,784 netto bps per antatt
fylling ved1-bps-utgang og finansiering; fordelt på alle muligheter er det−1,397.

Dette er en målt utvalgseffekt i den deklarerte modellen: den gunstigere
inngangsprisen beholdt ikke det opprinnelige signalutvalget. Køplass, faktisk
fill, latency og markedspåvirkning er ikke undersøkt. Finansiering bruker
antatt fylling ved barens slutt; faktisk berøringstid er ukjent innen fem minutter.

## Inferens og beslutning

Alle72 deklarerte endepunkter er testbare i felles max-|t|-familie:
63 INKONKLUSIV,9 NO_GO og0 GO. De9 NO_GO gjelder diagnostiske endepunkter;
alle fire nødvendige primære endepunkter er INKONKLUSIV.
Samtidig er den observerte risikojusterte økonomien negativ.

Primær passiv arm ved1 bps og finansiering, rå bps per valgt mulighet:

| Sammenligning | Estimat | Simultan95 % grense | MDE ved80 % | Styrke ved1 /2 /5 bps |
|---|---:|---:|---:|---:|
| Mot FLAT | -1,397 | [-4,480; 1,686] | 4,150 | 4,1 % / 18,5 % / 94,5 % |
| Mot LONG | 0,924 | [-3,332; 5,179] | 5,650 | 2,3 % / 8,9 % / 68,1 % |

Nullfinansiering gir samme hoveddom: netto−1,365 med grense
[−4,444;1,713], differanse mot LONG+0,827 med grense[−3,417;5,072].
Normaliserte og Sharpe-endepunkter, Monte Carlo-feil og alle månedsresultater
ligger i [maskinrapporten](TA_C_RESULT_20260929.json).

Brede intervaller og svak styrke ved små effekter er grunnen til at negativt
punktestimat ikke automatisk blir statistisk NO_GO. De gir heller ikke
grunnlag for en positiv handlingsbeslutning. Styrke/MDE gjelder en betinget
lokasjonsskiftmodell; gjenbrukt historie begrenser videre generalisering.

## Beslutning og bevis

**INKONKLUSIV; ingen GO til utførelsesforskning eller native trening.**
Bevar den fullførte målingen. Ingen ny limit-, celle-, kost- eller terskeltilpasning
på samme resultater. Eventuell ny arm må ha en særskilt begrunnet hypotese,
forhåndsregistrering og avklart informasjons-/datagrense.

Resultat-SHA256:28f6b10db378e24c6e5da881bda42f966a3c0adb3b47e4207ddb6c77f3bfbe73.
Run:/home/andre2/GX1_RUNS/TA_RESEARCH_20260929/MEASUREMENT_C_001.
Etterkontroll:/home/andre2/GX1_RUNS/TA_RESEARCH_20260929/MEASUREMENT_C_001_CACHED_AUDIT.json.

**Målt:** den deklarerte kombinasjonens kostnader, økonomi, berøringsutvalg og
betingede usikkerhet på oppgitt gjenbrukt historie.
**Bevist konsistent:** signalklokke, kilde-/artefaktbinding, kontantavstemming,
kostside og full ny inferensfamilie.
**Ikke undersøkt:** faktiske fills, urørt framtidig generalisering og native
modellens kapasitet under en ny mål-/horisontkontrakt.
