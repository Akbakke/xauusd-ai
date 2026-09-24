# Entry: prisforløpet etter de samme inngangene — 24.09.2026

Oppfølging av [mønsterkontrollen](ENTRY_PATTERN_INTERACTIONS_20260924.md).
Ingen nye fits, optimizersteg, sidevalg, horisonter eller Exit-regler ble valgt.

## Låst måling

De samme 20251 referanseinngangene og 18054 mønsterinngangene i fire senere
TRAIN-år ble målt ved alle de eksisterende vinduene h12/24/48/96/288.
Juni ble beholdt separat: referansen hadde åtte innganger, mønsterarmen null.

Alle 263557 beslutningstidspunkter er bevart. En inngang som ikke rakk hele
observasjonsvinduet før fold-slutt, ble beholdt som åpen inventarmarkering ved
siste tilgjengelige bar før grensen. Slike merkinger inngår i totalen og
rapporteres også separat. Ingen neste-fold-priser eller TEST-priser ble brukt.

Samme close-fill BID/ASK-research og arkiverte kostnadsscenario som tidligere:
4 bps per rundtur og faktisk tidsavhengig finansiering. Prospektiv lukkekostnad
er også trukket fra inventarmarkeringene for sammenlignbarhet.
Dette er hypotetisk overlappende eksponeringer, ikke kapitalført portefølje.

Kilde, tidligere prediksjoner og lastede tapearrayer er bundet.
H12 gjenga alle tidligere prisutfall, nettoutfall og valgte markeringer bitlikt.

## Mer tid gjorde ikke disse sidevalgene lønnsomme

Netto bps per valgt inngang, fire TRAIN-holdouts samlet:

| Nominelle handelstimer | Referanse, 1076 felt | Mønsterarm, 1311 felt | Mønsterarm åpne ved periodegrense |
|---|---:|---:|---:|
|1|−5,016|−5,246|0|
|2|−4,670|−5,072|4|
|4|−4,171|−5,204|8|
|8|−3,202|−4,661|8|
|24|−4,518|−7,167|24|

Samme innganger og sidevalg ved alle fem målinger. Ingen positiv horisont
er valgt etterpå. Månedsblokkintervallene for paret forskjell mot referansen
inkluderer null ved alle vinduer. Ingen nedre nettogrense mot FLAT er positiv.

2000 seed-0 resamplinger av 48 månedsblokker; kvantiler 0,0025/0,9975 korrigerer
for to kontraster × fem vinduer. Dette korrigerer ikke alle prosjektets gamle
forsøk. Juni gir fortsatt null valgte mønsterinnganger ved alle vinduer.

24 handelstimer tilsvarte i snitt 35,65 faktiske timer for mønsterinngangene.
Markedspauser og helger inngår i finansieringen. Vinduene er målinger, ikke
regler om maksimal holdetid.

## «Var i pluss en gang» er et svakt Entry-mål

Innen h288 eller periodegrensen nådde 89,82 % av mønsterinngangene minst én
positiv nettomarkering på M5-close. Med samme inngangstidspunkter og motsatt
side var andelen 88,46 %. For referansen var tallene 90,10 % mot 87,85 %.

Blant mønsterinngangene som nådde positiv netto var median tid 65 minutter.
Det betyr ikke at 89,82 % kan klassifiseres som gode innganger ved beslutning:
den senere pristuren er ukjent, og også motsatt side var ofte innom pluss.

Mønsterarm ved h288:
- 48,08 % hadde positiv sluttmarkering etter kostnad.
- Gjennomsnittlig positiv nettomarkering: +44,456 bps.
- Gjennomsnittlig negativ nettomarkering: −54,978 bps.
- Samlet: −7,167 bps, inkludert åpne markeringer.
- Med disse observerte gevinst-/tapsstørrelsene ville 55,29 % netto vinnere
  vært nødvendig for balanse. Dette er en etterberegning, ingen ny terskel.

Mid-prisretningen var riktig i 52,46 % av tilfellene, men det ga ikke positiv
økonomi. Størrelsen på bevegelsene og kostnadene må med i vurderingen.

Maksimale gunstige/ugunstige M5-nettomarkeringer er bevart som diagnostikk.
De inkluderer kostnaden ved inngang. En optimal fremtidig topp er ikke
avkastningen til en kjørbar Exit-policy; ingen slik policy ble konstruert.

## Også gjenbrukt evidens for modeller trent på lengre mål

50 gamle per-config-rapporter ble lest for ATR-ridge med 1076 og 1311 felt,
fem målhorisonter og fem senere perioder. Disse modellene var tilpasset hvert
sitt horisontmål, i motsetning til HGB-kontrollen over som låser h12-sidevalg.

For begge ridge-armene er alle fem samlede TRAIN-resultater negative allerede
når bare 4 bps trekkes fra de spreadinklusive resultatene. Dette er en optimistisk
nettoøvre grense, siden arkivert finansiering er ikke-negativ:
- 1076 felt: −4,397 til −5,305 bps per valgt inngang, avhengig av horisont.
- 1311 felt: −5,633 til −11,459 bps.
- Juni er også negativ for alle disse kombinasjonene.

Dette er historiske rapporter før målereparasjonen, ikke nye reparerte fits
eller en isolert HGB/ridge-sammenligning. Det finnes enkelte positive årsuttak
i referansen; de skal ikke presenteres som samlet, stabil gevinst.

## Videre konsekvens

Hverken mer venting på de samme inngangene eller «innom pluss» gir grunnlag
for ny større trening. Ingen modell, side eller horisont er promotert.

Neste nødvendige avklaring gjelder læringsmålet og kalibreringen av native
Entry-verdi mot observerte nettoutfall. Gjenbruk de allerede ferdige
Exit-lærer-/bootstrap-kontrollene før arbeid avgrenses; unngå enda en variant
av snapshot-/mønster-/horisontsøk uten en konkret ny blokkering.
Ingen ny fit er planlagt. Målet om økonomisk nyttig Entry er ikke oppnådd.

## Drift og bevis

Rot:
/home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923/FIXED_ENTRY_TIME_PATH_20260924

PLAN/START/TERMINAL/RESULT, metrics.csv, measure_fixed_choices.py og run.log.
POST_REVIEW.json og review_cached_readouts.py binder gevinst/tap-dekomponering
og de 50 historiske rapportene. Private fixed_choices_marks.parquet blir på Linux.

Kjørekilde 7f9a7696b7018c926a7bdb8da46bdcf739ecfb9e.
Hovedmåling rc0 kl.14:39:41 UTC /16:39:41 Oslo, 5,83 sekunder.
Etterberegning rc0. Audit-cap 4 GiB/512 MiB swap, én numerisk tråd, ingen CUDA.
Native kode, checkpoints og GX1_CURRENT er uendret. Ingen TEST/live/papir/spending.
