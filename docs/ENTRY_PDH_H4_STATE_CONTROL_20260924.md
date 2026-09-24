# PDH-brudd i bullish H4: kontrollert TRAIN-funn — 24.09.2026

Senere oppfølging: [native synlighet og kausal læringskontroll](ENTRY_PDH_NATIVE_LEARNING_20260924.md).
Gruppelæringen feilet i gjenbrukt juni; native valg på alle 299 hendelser er
nå målt uten ny trening. Resultatene nedenfor er bevart som tidligere TRAIN-evidens.

## Beslutning

Behold denne ene hypotesen som forskningskandidat. Bruddet tilfører en positiv
historisk forskjell utover den målte trend-/tidskontrollen. Nettoresultatet er
fortsatt usikkert; ingen læringsport, modellendring, promotering eller større
trening følger av resultatet. TEST er uåpnet i dette arbeidet.

Dette er en oppfølging av [målereparasjonen](ENTRY_MEASUREMENT_REPAIR_20260924.md).
Alle analyser gjenbruker TRAIN; hypotesen var allerede valgt fra 144 undersøkte
celler. Resultatene er ikke uavhengig bekreftelse eller korrigert for dette utvalget.

## Fast oppsett og kontroll

- Signal: første M5-close over forrige observerte handelsdags high, mens siste
  lukkede H4-bar har close > EMA20 > EMA50 > EMA200. Koden bruker eksisterende
  `pdh_break_trend_H4`; ingen ny terskel, retning eller parameter er valgt.
- Hovedvindu: h48, altså 48 femminuttersbarer. De tidligere h12/h24/h96 beholdes
  som diagnostikk. Det er målevinduer, ikke nye holderegler i native-boten.
- 299 signaler av 313 399 TRAIN-rader (0,0954 %), fordelt 40/49/51/82/77 på
  fem årsperioder fra juni 2021 til mai 2026.
- Referanse: LONG når H4 er bullish uten PDH-hendelsen. Hovedreferansen vektes
  til hendelsenes kalendermåned og UTC-klokketime ved beslutning (M5-start + 5 min).
  Alle 299 hendelser har kontrollrader; minste kontrollgruppe har 10 rader.
  Kontrollens eligibility bruker ingen fremtidige utfall. Vektingen er en
  beskrivende sammenligning, ikke en ferdig handlingsplan eller randomisering.
- Usikkerhet: bidrag fra både hendelser og estimert kontrollsnitt aggregeres per
  kontinuerlig kalenderuke. Eksisterende Newey-West-eier brukes deretter med
  automatisk båndbredde (fire uker samlet, tre per år). Omtrentlige 95 %-intervaller
  er eksplorative; de gir ingen garanti ved lite utvalg eller regimeskifte.
  Se [Newey-West/HAC-metoden](https://www.statsmodels.org/v0.14.3/generated/statsmodels.stats.sandwich_covariance.cov_hac.html).

## Hva signalet tilfører før øvrige kostnader

Spread er inkludert i alle prisutfall. I det opprinnelige close-fill-vinduet:

| Målevindu i M5-barer | PDH/H4, bps | Matchet H4-kontroll, bps | Forskjell, bps |
|---|---:|---:|---:|
| 12 (1 handelstime) | +4,646 | +0,344 | +4,302 |
| 24 (2 handelstimer) | +7,179 | +1,347 | +5,832 |
| **48 (4 handelstimer), hovedmåling** | **+9,562** | **+3,187** | **+6,375** |
| 96 (8 handelstimer) | +13,279 | +5,012 | +8,267 |

Ved h48 er forskjellens omtrentlige intervall +1,314 til +11,435 bps.
Punktforskjellen er positiv i alle fem år (+5,378/+5,786/+8,359/+0,760/+11,932).
Fire av fem enkeltårsintervaller inkluderer null. Dette støtter et samlet
TRAIN-spor, men ikke en påstand om sikker effekt i hvert år.

H4-bull uten hendelsen, uten tidsmatching, ga bare +0,729 bps. Tidsmatching
løfter referansen til +3,187; denne strengere sammenligningen er hovedreferansen.

## Neste tilgjengelige inngang og arkiverte kostnadsscenarioer

En separat plan ble bundet før denne kontrollen. LONG kjøpes til neste
tilgjengelige M5 ask_open etter signalet, og verdsettes til bid_close ved den
samme h48-markeringen. Alle 299 hendelser er med, også tapene. Ingen hendelse
manglet neste bar ved beslutningstidspunktet.

| Scenario | PDH/H4-snitt, bps | Matchet kontroll, bps |
|---|---:|---:|
| Neste åpning, spread inkludert | +9,554 | +3,189 |
| Arkivert lav kostnad: 1 bps per utførelse + finansiering | +7,266 | +0,920 |
| **Arkivert sentralscenario: 2 bps per utførelse + finansiering** | **+5,266** | **−1,080** |
| Arkivert høy kostnad: 4 bps per utførelse + finansiering | +1,266 | −5,080 |

Policyen fra 12.09 er et arkivert prospektivt scenario, ikke sann historisk
meglerkostnad. Den har 0 provisjon, LONG-finansiering 5,4 % årlig på faktisk
veggklokketid med 31 557 600 sekunder per år, og slippage i tillegg til spread.
Ingen meglerkontakt eller ny kostnadsautorisasjon er brukt.

Sentralscenarioets eget snitt har intervall **−0,508 til +11,041 bps**.
Forskjellen mot matchet kontroll er +6,346 med intervall +1,284 til +11,407.
At referansen taper gjør ikke kandidatens netto lønnsomhet bevist.

| Årsperiode | Hendelser | Sentralscenario, bps |
|---|---:|---:|
| 2021–22 | 40 | +0,191 |
| 2022–23 | 49 | +0,251 |
| 2023–24 | 51 | +7,729 |
| 2024–25 | 82 | +0,837 |
| 2025–26 | 77 | +14,179 |

Tre år har under 1 bps margin i sentralscenarioet og blir negative i høyscenarioet.
152 av 299 utfall er positive etter sentralkostnaden (50,84 %): gjennomsnittlig
positivt utfall +38,535 bps og ikke-positivt utfall −29,135 bps. Trefferate alene
beskriver derfor ikke forventet utfall.

Ett par målevinduer overlapper. Maksimal faktisk tid til h48-markeringen er
53 timer over en markedsstenging; gjennomsnittlig finansieringsfradrag er
0,288 bps. Dette er hendelsesutfall, ikke netto porteføljeavkastning.
Ingen posisjonsstørrelse, samtidig eksponering eller native Exit er evaluert.

## Kontroller og evidens

Kilde ved begge kjøringer: GX1_ENGINE, audit/v9-premiere-20260905,
`5f5089dc71df13833dc3d27e46ec449dda1289e6`, clean.
Ingen kanonisk modell-/feature-/treningskode ble endret og ingen modell ble fittet.
Begge analyser brukte audit-vakten med 4 GiB RAM, 512 MiB swap og én numerisk
arbeider. De fullførte 12:59:25 og 13:03:31 UTC, begge rc0.

- TRAIN-predikater ble brukt før materialisering; ingen nye VAL/TEST-rader.
- Uavhengig vektorisert PDH/første-brudd-beregning matchet cache eksakt.
- H4-stack gjenberegnet med EMA- og lukkeklokkeeier matchet cache eksakt.
- Alle gamle pooled-/årsantall og close-fill-snitt ble gjenfunnet eksakt.
- Ingen hendelse har utfall forbi sin rapporterte årsgrense.
- Maksimalt én PDH-hendelse per handelsdag.
- Kilde, planer, instrumenter, cache, policy og resultater er hashbundet.

Evidensrot:
`/home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923/PDH_H4_STATE_CONTROL_20260924/`.
PLAN.json, START.json, TERMINAL.json og RESULT.json binder tilstandskontrollen;
EXECUTION_PLAN.json, EXECUTION_START.json, EXECUTION_TERMINAL.json og
EXECUTION_RESULT.json binder pris-/kostkontrollen. Begge små analyseprogrammer
og alle hendelsesutfall er bevart utenfor kanonisk kilde. Mac-kopien inneholder
bare programmer, planer, logger og aggregerte bevis.

## Neste beslutning

Signaldefinisjon, h48 som hovedvindu, neste-bar-prising, sentralkostnad og
trend-/tidsreferanse er nå fastlagt. Ikke velg en annen horisont, terskel eller
kostnad etter neste resultat. Senere uavhengig evaluering må:
1. binde en ikke tidligere evaluert dataperiode før utfall leses;
2. ta med alle signaler, rapportere antall/manglende utfall og tidsgrenser;
3. måle både absolutt resultat og forskjell mot den samme referansen;
4. rapportere usikkerhet og alle forhåndsfastsatte kostnadsscenarioer;
5. holde eventuell modellimplementering og full porteføljeøkonomi som egne porter.

Ingen uavhengig dataperiode er valgt eller åpnet av dette arbeidet. TEST-forseglingen
består. De 299 TRAIN-hendelsene og gjenbrukt juni kan ikke brukes som ny
bekreftelse. Det er foreløpig ikke grunnlag for større native trening.
