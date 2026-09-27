# Tidlig kalibrering: ferdig sammenligning, NO-GO — 27.09.2026

Tidlig kalibrering og den korrigerte, forhåndsbundne sammenligningen er fullført.
Begge modeller feiler beslutningsporten. Native trening forblir deaktivert.
Maskinprodusert [resultat og kontrollkvittering](HISTORY2009W_EARLY_DECISION_RESULT_20260927.json).

## Hva som er målt

Eksisterende squeeze- og registerkomponenter er kalibrert på 2009-06-01 til
2013-01-01 22:00 UTC. Faktiske artefakter bekrefter grensen. Første indre
modellvalidering starter 2014-08-07 22:00 UTC, første ytre kontroll 2015-06-01.
Alle 241 signaler, 71 kontinuerlige kontekstfelt, sesjonskoding og fire MTF-baner
inngår: 1 076 kolonner. Ti årlige holdouts slutter 2025-06-01. D1-klokke,
1 440 M5-barers utfall, ridge og HGB, én seed og ett kostscenario.

423 ikke-overlappende beslutningsblokker per modell. År vektes likt; tallene
nedenfor er gjennomsnittlig netto bps per beslutningsmulighet, inklusive FLAT=0.
De er ikke årsavkastning, avkastning per handlet posisjon eller porteføljeavkastning.

| Modell / baseline | Netto bps | Forskjell mot alltid LONG | Positive årsfordeler mot LONG |
|---|---:|---:|---:|
| Ridge | -2,37 | -6,74 | 5/10 |
| HGB | +5,71 | +1,34 | 5/10 |
| Alltid LONG | +4,37 | 0 | – |
| Konstant valg lært fra tidligere fit-data | 0 | -4,37 | – |

Den kausale konstanten valgte FLAT i alle ti perioder. Ridge velger LONG 19,
SHORT 14 og FLAT 390 ganger. HGB velger LONG 49, SHORT 66 og FLAT 308 ganger.

HGBs nedre konfidensgrense er -5,12 bps for netto mot FLAT og -22,39 bps for
fordelen mot LONG. Porten bruker én årsverdi per fold og Bonferroni for de to
modellene, ensidig 97,5 prosent Student-t-grense. Modellen slår LONG i fem av
ti perioder; minst åtte kreves. Den slår den kausale konstanten i bare tre år.

HGBs gjennomsnittlige fordel mot LONG er +20,95 bps i de seks foldene hvor
LONG taper, men -28,06 bps i de fire foldene hvor LONG tjener. Regimekravet
feiler. Gruppene defineres etter utfall for evaluering; de er ikke et kausalt
signal som kan brukes til å velge strategi på forhånd.

## Hva kontrollen fant og rettet

En separat kontroll mot BID/ASK avslørte at den eldre forskningskoden brukte
første valgte D1-rad etter en foldgrense som grense for purge og utfall. Åtte
fit-sett fikk én for sen rad; sju holdouts fikk én for sen utfallsrad, hvorav
fem inngikk i blokkstatistikken. Koden bruker nå den deklarerte grensen på
råtapen. Regresjonstester dekker både fit-purge og holdout-slutt.

Bare den korte modellsammenligningen ble beregnet på nytt. Kalibreringsartefakter,
C0, featureflate, modeller, hyperparametere, kostnader og akseptkrav er uendret.
De første resultatene er bevart som erstattet, ikke blandet med de korrigerte.
Eldre analyser med grov klokke er ikke bevis på nøyaktige periodegrenser uten
kontroll av samme feil; ingen historiske modelljobber er gjenkjørt.

Alle produsentsteg og korrigert sammenligning avsluttet med rc0. 54 målrettede
tester består. Separat kontroll av alle 5 394 lagrede prediksjonsrader (begge
modeller samlet) mot rå BID/ASK-priser bekrefter nettoutfall, handlingsvalg og
blokkstatistikk. Største tallavvik er 2,85e-14 bps. Alle evaluerte utfall slutter
før sin foldgrense; alle fit-utfall før neste holdout. MTF-aliasparitet har null avvik.

## Hva konklusjonen betyr

Denne målingen dokumenterer ikke robust positiv beslutningsverdi fra denne
featureflaten med disse to lærerne. Den beviser ikke at XAUUSD er ulærbart eller
at en sekvensmodell aldri kan lære. Positivt gjennomsnitt i HGB alene gir ikke
støtte til mer native trening; usikkerhet, årsdekning og regimekrav feiler.

Kostene er et bundet prospektivt scenario: BID/ASK-spread, 2 bps ugunstig
slippage per utførelse, null ekstra provisjon, 5,4 prosent årlig finansiering
for LONG over faktisk kalenderholdetid og null SHORT-kreditt. Dette er ikke
nyverifisert brokerhistorikk. Samme M5-bars close er forskningskonvensjonen,
ikke dokumentert quote-to-fill-utførelse. Årene har vært inspisert tidligere;
de er kronologisk holdout for disse fittene, ikke et urørt prosjekt-holdout.
TEST og utfall fra juni 2025 eller senere er ikke brukt i sammenligningen.

Neste forsøk krever en konkret, forhåndsbegrunnet endring i tilgjengelig
informasjon eller utførelsesøkonomi. Det skal ikke være flere epochs, en ny
terskelrunde eller et valutabytte begrunnet bare med dette resultatet. Før en
ny modelljobb må endringen ha en målbar mekanisme, datakilde og separat
forhåndsbundet kontroll. Ingen ny jobb er startet eller planlagt automatisk.

## Autoritative artefakter

Runtime: `/home/andre2/GX1_RUNS/HISTORY2009W_EARLY_DECISION_20260927`.

- `PREREGISTRATION.json`: uendrede parametere og akseptkrav.
- `CORRECTION_BINDING.json`: korrigert kilde, gjenbrukte input og evaluatorsjekksummer.
- `STRICT_TERMINAL.json`: korrigert kjøring ferdig 18:13:02 UTC / 20:13:02 Oslo.
- `walkforward_strict/report.json` og `nonoverlap.json`: endelige markedstall.
- `DECISION_GATE_STRICT.json`: anvendelse av den forhåndsregistrerte porten.
- `VERIFICATION.json`: kontroll mot råpriser, eksakte grenser og feature-lineage.
- `FOLD_BOUNDARY_AUDIT.json`: målt feil før rettelsen.

`walkforward/` og `DECISION_GATE.json` er erstattet historikk og skal ikke
brukes som gjeldende konklusjon. Alle opprinnelige checkpoints er bevart.
