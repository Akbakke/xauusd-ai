# Feature-tilpasning krysset walk-forward-grensene — 24.09.2026

## Observert feil

Forsøkene delte modelltilpasningen kronologisk, men brukte en felles featureflate
med nivå-/trendlinjeparametere og volatilitetsparametere valgt på hele TRAIN
frem til **31.05.2026 kl. 23:55 UTC**. Dette gjelder også årsavlesninger som
starter i juni 2022, 2023, 2024 og 2025.

Kildebindingen er direkte:
- V12 TRAIN-manifestets `extra.multi_tf_cache_binding.v29_registry_constants`.
- MTF-cachemanifestets samme constants-payload, contract SHA
  `a09fd98da91bdfe9ed71ff691c3461fbaeb4dc11d89bd30c44ab08d198237284`.
- Bundet volatility-squeeze-manifest med samme deklarerte TRAIN-slutt.
  Fil-SHA `502cd4e480426b7cc40f75b71e6d8ed743ce0f1dc355f780d0ba954dcff8eee7`.

Trendlinjebånd og nivåers levetid påvirker hvilke linjer/nivåer og hendelser
som når modellen. De er ikke bare beskrivende metadata. Registry-eierne og
`htf_features.py` binder dem til faktisk featureberegning. De åtte tidligere
HGB-delmodellene brukte henholdsvis 10/7/6/15/7/7/5/7 slike registry-felt i
splittene. Ren round-number-geometri er eksplisitt utelatt fra disse eksemplene;
den er ikke tilpasset på TRAIN.

Dermed er dette en feil i forskningsoppsettets kronologi: deler av
forbehandlingen er valgt med senere data enn det historiske kontrolltidspunktet.
Korrekt sist-lukket-join og bitlik prefix-beregning med allerede valgte
konstanter oppdager ikke dette.

## Hva funnet betyr — og ikke betyr

De tidligere PnL-beregningene og parede kontrollene er bevart. Den siste
HGB-kontrollens nullforskjell er fortsatt målt korrekt for de identiske inputs.
De berørte årsavlesningene kan likevel ikke brukes som fullt kronologiske
OOS-bevis for læringsporten.

Dette viser ikke at lekkasjen forklarer den svake retningen, at alle features
er feil, eller at korrigert forbehandling vil gi positiv økonomi.
Parametertilpasningen slutter før juni 2026. Denne ytre grensen passerer
denne konkrete datokontrollen; indre hyperparameterseleksjon trenger fortsatt
egen kontroll. Juni er dessuten gjenbrukt utviklings-VAL.

De faste PDH/H4-reglene og andre analyser som kun bruker uavhengige,
ikke-tilpassede prisformler er ikke automatisk omfattet. Omfang må følge
faktiske inputavhengigheter, ikke hele prosjektnavnet.

## Minste rettelse

`gx1/scripts/research_entry_direction_walkforward_v1.py`:
1. Les feature-fit-proveniens for snapshot og eventuelt separat MTF-cache.
2. Kontroller bundet volatility-manifest mot SHA og valider tidssoner/fit-vinduer.
3. Avvis manglende/ugyldig proveniens eller feature-fit etter første ytre
   kontrollstart før dataset-/prisarrays materialiseres.
4. Kontroller også den faktiske indre kontrollstarten før hver modelltilpasning.
5. Bind proveniensen i eksisterende run/input-binding.

Halvåpent fit-vindu som ender nøyaktig ved kontrollstart tillates. Ingen
bypass, nye handelsterskler eller ny modellarkitektur er innført.
Native trener, samtlige features, datasett, originale checkpoints og TEST er
urørt. Ingen ny modellfit eller stor cachebygging er startet.

## Verifikasjon

- Elleve målrettede kontroller bestod under eksisterende audit-vakt, 4 GiB.
  Dekker registry og volatilitet, halvåpen grense, manglende metadata,
  hashavvik, ugyldig tid, ytre og indre avvisning, samt én eksisterende
  syntetisk ende-til-ende-kontroll med lovlig tidligere feature-fit.
- Ekte V12-metadata avvises ved alle fire ytre grenser.
- Kall til selve instrumentet med ekte metadata stopper ved
  `WALKFORWARD_FEATURE_FIT_AFTER_EVALUATION_START`, før dataset-loaderen.
  Observerte materialiseringskall: 0. Modellfits: 0.
- Avgrenset yttergrensekontroll for juni passerer; dette er ingen full
  data-/modellgodkjenning.

Evidens:
`/home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923/FEATURE_FIT_CHRONOLOGY_20260924`.
RESULT, COMPLETED_REVIEW, verifikasjonsskript og logger er bevart.
COMPLETED_REVIEW snevrer det opprinnelige prefiksbaserte feltinventaret inn
til dokumenterte registry-eksempler, uten ren round-number-geometri.

## Rettet rapporttolkning

ENTRY_DIRECTION_SNR_DIAGNOSIS_20260923.md er korrigert:
Lav fremtidskorrelasjon beviser ikke fravær av lekkasje. Det finnes ingen
generell grense ved korrelasjon 0,1 som gjør en lekkasje synlig. Høy Pearson
for 49 referanseformler er heller ikke et semantisk/kausalt bevis for alle
1072 felt eller en full numerisk paritetskontroll.

Tidligere negative modellforsøk beholdes som avgrenset evidens. De skal ikke
omtales som bevis for at alle prisbaserte eller ikke-lineære signaler er umulige.

## Neste nødvendige arbeid

Før en ny walk-forward-fit må hele den brukte featureflaten ha parametere
kalibrert uten data fra både ytre og indre kontrollperioder. En felles,
tidlig frosset parameterpakke kan være tilstrekkelig dersom den ligger før
den tidligste indre kontrollen og har gyldig støtte. Det skal først avklares
med eksisterende eiere og artefakter; det er ikke autorisasjon til blind
full rebuild eller parameterjakt. Alle featurefamilier skal beholdes.

Denne rettelsen gjør målingen strengere. Økonomisk nyttig Entry er fortsatt
ikke dokumentert, og det overordnede målet er aktivt.
