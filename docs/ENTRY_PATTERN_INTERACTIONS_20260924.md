# Mønsterfelt i bred ikke-lineær Entry — 24.09.2026

## Avgrensning

Oppfølging av [bred kombinasjonskontroll](ENTRY_BROAD_COMBINATIONS_20260924.md).
Samme full-refit HGB-referanse med 1076 felt ble sammenlignet med 1311 felt:
235 eksisterende mønsterfelt fra M5/H1/H4/D1 ble lagt til. Ingen ny terskel,
horisont, modellvariant eller native kode ble valgt underveis.

Referansens prediksjoner ble gjenbrukt. Ti nye side/fold-modeller brukte samme
fit-rader, h12/ATR-mål, seed 0 og parametere. Gamle iterasjonsvalg var låst:
ni modeller med ett tre, én med tre. Indre iterasjonsvalg/purge ble ikke
gjentatt. Målingen isolerer inputendringen ved denne kompleksiteten.

## Input- og kausalitetskontroll

Hele TRAIN-historikken ble bygget på nytt med mønstereieren og tape avkuttet
før 31.05.2026 kl.23:55 UTC. Alle 313399 rader × 235 felt var bitlike ved
float32 mot den cachede filen som også inneholder juni.

Dette dokumenterer invarians ved denne grensen. Kildegjennomgang viste
swing-bekreftelse etter lookback og siste lukkede bar ved MTF-sampling.
Det er ikke en uttømmende kontroll av alle mulige prefixgrenser.
Ingen TEST ble lest. Kontrollen tok 28,29 sekunder, rc0.

Før fit ble referansens 1076 inputkolonner og tider SHA-sammenlignet.
Alle øvrige bundne tape-/ATR-/utfallsarrayer var identiske. De 263557 senere
tidspunktene hadde bitlike utfall og kostnadsberegninger. Ingen referansefit
ble gjentatt.

## Senere økonomi

Alle valgte innganger markeres ved h12. Spread er i utfallet; arkivert
sentralscenario legger til 4 bps per rundtur og faktisk finansiering.
Close-fill-research, hypotetisk overlapp og gjenbrukt utviklingsdata:
ingen neste-bar-/portefølje-/native Exit- eller uavhengig lønnsomhetsbekreftelse.

| Senere periode | Valgte, 1311 felt | Netto bps/valgt, 1076 felt | Netto bps/valgt, 1311 felt |
|---|---:|---:|---:|
| juni 2022–mai 2023 |1334|−4,582|−1,754|
| juni 2023–mai 2024 |16196|−5,229|−5,468|
| juni 2024–mai 2025 |491|+0,263|−7,804|
| juni 2025–mai 2026 |33|−0,090|+0,241|
| juni 2026 |0|−9,542|ingen valgte innganger|

De fire TRAIN-holdoutene samlet:
- Referanse: 20251 valgte, −5,016 bps/valgt, −0,394 bps/mulighet.
- Mønsterarm: 18054 valgte, −5,246 bps/valgt, −0,367 bps/mulighet.
- Paret forskjell per mulighet: +0,0266 bps, justert intervall [−0,0582;+0,1219].
- Mønsterarm mot FLAT: −0,3670 bps, justert intervall [−0,7775;−0,0674].

Usikkerhet: 2000 seed-0 resamplinger av 48 månedsblokker, kvantiler 0,00625/0,99375
for fire forhåndsbestemte sammenligninger. Juni rapporteres separat med 27
dagblokker. Korreksjonen omfatter disse fire kontrastene innen hvert uttak,
ikke prosjektets historiske antall forsøk.

Juni hadde 5509 FLAT. Dette er ingen demonstrasjon av lønnsom selektivitet.
32 forskjellige mønsterfelt ble faktisk brukt i splittene, inkludert
FVG-retester, order-block-avstander/-alder, flagg, liquidity sweeps,
range-breakout-alder og tid siden PDH-brudd. Fravær av feltbruk forklarer
derfor ikke det negative resultatet.

## Beslutning

Tre økonomiske krav for videreføring feilet; kravet om faktisk mønsterbruk
bestod. STOP_THIS_FIXED_COMPLEXITY_H12_PATTERN_VARIANT_NO_RETUNING_OR_PROMOTION.
Ingen større native trening, tuning mot juni eller promotering følger av dette.

Dette avviser denne inputendringen ved den låste kompleksiteten og h12.
Det avviser ikke alle samspill, lengre observasjonsvinduer, sekvensmodeller
eller adaptiv Exit. h12 er et målevindu, ikke en regel om maksimal holdetid.

Neste avklaring skal gjenbruke eksisterende resultater for flere observasjons-
vinduer etter Entry. Vi må skille svak retning/kalibrering fra et målevindu som
ikke fanger tilstanden modellen forsøker å handle, før noen ny fit avgrenses.
Ikke åpne et nytt bredt søk eller vende tilbake til manuell trening av ett
håndplukket oppsett. Målet om økonomisk nyttig Entry er fortsatt ikke oppnådd.

## Drift og evidens

Kjørekilde 753b73a584ead1e5515eb75dee572f7056fe95d1,
GX1_ENGINE/audit-v9-grenen. Native kode, vekter og Exit uendret.
Fit rc0 kl.14:30:20 UTC /16:30:20 Oslo, 241,29 sekunder.
Producer-cap 10 GiB, 512 MiB swap, én numerisk tråd; ingen CUDA/native optimizersteg.

Rot:
/home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923/HGB_PATTERN_INTERACTIONS_20260924

CAUSAL_PLAN/START/TERMINAL/RESULT, PLAN/START/TERMINAL/INPUT_BINDING/RESULT/REVIEW,
fold0–4, metrics.csv, skript og logger er bevart. Private radprediksjoner
forblir på Linux. Ingen TEST, live/papir, spending eller ENGINE-push.

RESULT SHA: 45f02cf3be4ddd6f313ea7404ac9d0493606d664bd02704b95719e3be9bd26d6.
CAUSAL_RESULT SHA: be42ce0713c57d3ae313194197476e7aa18c0cd778d64669065a40fffab3239b.
