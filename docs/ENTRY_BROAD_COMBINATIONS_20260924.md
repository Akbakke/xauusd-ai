# Bred kombinasjonslæring og HGB-refit — 24.09.2026

## Spørsmålet og omfanget

Brukeren ba om bredere kombinasjonslæring fremfor langvarig arbeid med ett
PDH/H4-oppsett. PDH/H4 beholdes som et kontrolltilfelle. Denne runden gjenbruker
de brede analysene og isolerer én allerede rettet målefeil. Ingen native
modell-/treningskode, vekter eller Exit er endret.

## Hva er allerede undersøkt?

11 bevarte walk-forward-rapporter er inventert i COMBINATION_COVERAGE.json.

| Modell/input | Dekning og begrensning |
|---|---|
| Hånddefinerte oppsett | 36 oppsett × 4 horisonter = 144 celler. Ikke 144 ulike oppsett eller alle kombinasjoner. |
| HGB på snapshot | 316 snapshot-, kontekst- og sessionfelt; ikke-lineære betingelser. |
| HGB på snapshot + MTF | 1076 felt, inkludert sist lukkede M15/H1/H4/D1. To allerede fullførte HGB-varianter. |
| Ridge med mønsterfelt | Opptil 1311 felt. Kan bruke leverte mønsterfunksjoner, men tilpasningen lærer ikke vilkårlige nye kryssprodukter mellom feltene. |
| Ridge med kryssmarkedsfelt | Opptil 1099 felt. Disse ferdige rapportene inneholder ingen HGB-arm med kryssmarkedsfelt. |
| Frossen native representasjon | Den siste lineære avlesningen kan bruke samspill som allerede finnes i representasjonen. Den trener ikke encoderen på nye samspill. |

HGB med standardparametere beholdt én iterasjon i 65 av 80 side/fold-modeller.
For den låste h12/ATR-kontrollen var dette ni av ti; den siste beholdt tre.
Dette er valgt kompleksitet fra en historisk indre kontroll, ikke bevis for at
markedet mangler alle ikke-lineære signaler. Ett tre kan inneholde mange
betingelsesgrener og er ikke det samme som én teknisk kombinasjon.

Det finnes også en separat ferdig klassifikasjonskontroll med 316 felt og
100 iterasjoner på fire horisonter. Høy TRAIN-tilpasning ble ikke overført til
sikker forbedring mot lineær referanse i senere TRAIN-år. Ikke presenter HGB
som en uprøvd ny metode eller øk antall trær blindt.

## Isolert rettelse

Den gamle HGB-koden valgte iterasjon på siste 20 % av en treningsfold, men
predikerte deretter med modellen tilpasset bare de første 80 %. Den rettede
eieren tilpasser valgt kompleksitet på hele den tillatte treningsfolden.

Denne målingen låste de gamle iterasjonstallene, h12, ATR-skalering, seed 0,
læringsrate 0,1, minste bladstørrelse 20, 1076 felt og ytre purge på 289 barer.
Fire senere TRAIN-år og juni-VAL ble paret mot arkiverte prediksjoner.
Dermed isoleres full-refit-effekten. **Korrigert indre purge og valg av
iterasjon er ikke kjørt på nytt.** Dette er ikke en full gjentakelse av den
rettede HGB-kjeden.

- 10 fulle fold/side-tilpasninger + 4 gamle prefix-reproduksjoner.
- 263557 identiske beslutningstidspunkter og bitlike utfall; ingen gamle rader
  måtte utelates.
- Gamle LONG/SHORT-prediksjoner i første og siste fold reprodusert med
  største absolutt avvik 0 bps.
- MTF-aliaser kontrollert eksakt. Faktisk lastede inputarrayer er SHA-bundet.
- Ett separat måleskript feilet først på manglende eksplisitt sklearn-import,
  før noen tilpasning. Feilplan/logg/terminal er bevart. IMPORT_REPAIR er
  fullført; det er den korrigerte måleskriptkjøringen.

## Resultat

Beslutningsregelen er uendret argmax(LONG, SHORT, FLAT=0) på de gamle
spreadinklusive målene. Netto nedenfor trekker i tillegg fra arkivert
sentralscenario: 4 bps per rundtur og faktisk tidsavhengig finansiering.
Dette er scenarioøkonomi, ikke ny bekreftelse av meglervilkår.

| Senere periode | Gamle valgte | Gamle netto bps/valgt | Full refit valgte | Full refit netto bps/valgt |
|---|---:|---:|---:|---:|
| juni 2022–mai 2023 | 1493 | −5,441 | 3238 | −4,582 |
| juni 2023–mai 2024 | 23397 | −5,307 | 16604 | −5,229 |
| juni 2024–mai 2025 | 2602 | −7,038 | 324 | +0,263 |
| juni 2025–mai 2026 | 95 | +1,713 | 85 | −0,090 |
| juni 2026, gjenbrukt VAL | 2 | −5,755 | 8 | −9,542 |

Fire TRAIN-holdouts samlet, kun beskrivende:
- gammel modell: 27587 valgte, −5,453 bps/valgt, −0,583 bps/mulighet;
- full refit: 20251 valgte, −5,016 bps/valgt, −0,394 bps/mulighet.

I juni valgte full refit 2 LONG / 6 SHORT / 5501 FLAT. Forventet
spreadinklusive verdi på de åtte inngangene var +4,158 bps, observert −5,527
før de ekstra kostnadene. Prediksjonsfeilen er derfor større enn bare en
manglende fratrekking av scenarioets kostnader.

Alle valgte innganger er markert ved h12, inkludert hypotetisk samtidige
eksponeringer. Det er ikke en portefølje, neste-bar-fill eller full native
Exit-replay. h12 er målevindu, ikke ny maksimal holdetid. Periodene er
gjenbrukt utviklingsdata; ingen uavhengig økonomisk PASS hevdes.

## Faktisk bruk av kombinasjoner

De ti refit-modellene brukte samlet 204 forskjellige av de 1076 inputkolonnene
i splittene. MTF-felt fra alle åtte familier og alle fire ekstra tidsrammer
ble brukt. En ett-tre-modell hadde 30 splitknuter, 27–30 brukte felt og dybde
10–15; tre-tre-modellen brukte 73 felt. Dette bekrefter bredere lærte
betingelser enn bare PDH/H4. Det beviser ikke at alle mulige samspill er
undersøkt eller at de lærte samspillene generaliserer.

## Konsekvens for videre arbeid

Full-refit-feilen påvirker valg, men forklarer ikke bort den svake senere
økonomien. Kontrollens beslutning er DIAGNOSTIC_ONLY_NO_NATIVE_PROMOTION_NO_RETUNING.
Ikke relanser den eller øk native trening ut fra teknisk bestått kontroll.

Neste avgrensning gjelder samspill mellom mønsterfelt og MTF: de utvidede
mønsterarmene er hittil bare undersøkt med ridge i disse rapportene.
En eventuell ikke-lineær sammenligning må ha en låst referanse, samme rader,
korrekt purging, kjent kostnadsomfang og forhåndslåst senere måling.
Manglende dekning alene beviser ikke en edge og åpner ikke et bredt søk.
Ingen justering velges for å redde juni-resultatet. TEST forblir forseglet.
Målet om økonomisk nyttig Entry er aktivt og ikke oppnådd.

## Drift og bevis

Kjørekilde: GX1_ENGINE, audit/v9-premiere-20260905,
ffdc3b6201ecb9f0dd825b6a513f2b1842ce35f7. Native kilde urørt.

Rot: /home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923/HGB_FULL_REFIT_ISOLATION_20260924
Fullført under IMPORT_REPAIR: PLAN, INPUT_BINDING, START, TERMINAL, RESULT,
REVIEW, fold0–4.json, metrics.csv, audit_full_refit.py og run.log.
Private paired_predictions.parquet bevares på Linux.

Returkode 0, 24.09.2026 kl. 14:15:46 UTC / 16:15:46 Oslo.
254,82 sekunder med producer-cap 10 GiB, 512 MiB swap, én numerisk tråd,
åtte tillatte CPU-er; ingen CUDA/native optimizersteg.
RESULT SHA: af20ca2f584e0e918ebf3153d6edf4ad00978d7d76e156c11e866d0b928c3513.
Ferdige kildebindinger og resultater er bevart. GX1_CURRENT er urørt.
