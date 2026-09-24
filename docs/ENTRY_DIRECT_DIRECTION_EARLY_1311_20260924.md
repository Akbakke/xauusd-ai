# Direkte retning med tidlig kalibrerte 1311 inputs — 24.09.2026

## Hva tidligere kontroller dekket

ENTRY_NONLINEAR_RAW_TRAIN_20260923 brukte 241 snapshot-, 71 kontekst- og fire
sesjonsfelt, totalt 316. Fire faste horisonter, 100 trær med sklearn 1.7.2-
standardparametere, 248100 fit-rader og 65295 rader i siste TRAIN-år.
Ingen horisont bestod den parede retningsporten. For h12 var balansert treff
70,97 prosent på fit og 50,81 prosent senere; justert senere intervall
[49,72; 52,05] prosent. Disse rå inputene har den nå dokumenterte gamle
featurekalibreringen. PLAN- og skripthasher er kontrollert mot RESULT/PLAN.

Native forecast-z-BCE og joint-Entry-BCE er allerede prøvd og feilet.
Se ENTRY_DIRECTION_FIRST_20260923.md. De skal ikke presenteres som uprøvde.

Den fullførte HGB_EARLY_CALIBRATED_20260924 brukte de nye 1311 feltene, men
lærte spread-inklusive avkastningsmål med MSE/ATR. Retningsmakrosnittet var
50,16 prosent, og fem av åtte delmodeller tapte mot konstanten. Modellvalgets
manglende nullalternativ er rettet. Dette avkastningsoppsettet er avsluttet.

## Én avgrenset direkte retningskontroll

Bruk nøyaktig samme korrigerte 313399 × 1311 matrise som den fullførte
sammenligningen, med X SHA
c32c992463951313f2c5e804b54961d5d6129050e9fa6f4d2b72eb02cce4027d.
Gjenbruk regenererte lokale blokker, MTF, faste mønsterfelt, tidsakse,
targets og fire årlige TRAIN-kontroller. Behold de samme ytre og indre
maskene, med purge 289 observerte M5-barer og faktisk feature-fit-datovakt.

Målet er fortegnet på mid[t+12]−mid[t]. Nullbevegelser utelates fra binær
fit/log-loss, men den opprinnelige indre delingsgrensen flyttes ikke.
Nullbevegelser beholdes i utfallsvurderingen av alle muligheter. Dette er
et beregningsmål, ingen maksimal holdetid for en handelsbot.

Klassifikatoren bruker den allerede prøvde faste classifier-oppskriften:
100 iterasjoner, seed 0, learning_rate 0,1, min_samples_leaf 20, 31 bladnoder,
uten automatisk tilfeldig tidssplitt eller klassevekting. Indre kronologisk
log-loss velger mellom konstant TRAIN-prior og de 100 trinnene. Ved likhet
vinner konstanten. Deretter gjøres full-fold-refit eller analytisk beregning
av full TRAIN-prior. Fire indre fits og høyst fire tre-refits.

Dette er en direkte klassifikasjonsoppgave med klassifikatorens eksisterende
100-trinnsramme, sammenlignet med avkastningsregresjonens tidligere ramme på
300. Det er ikke et rent tap-bytte med identisk maksimal kapasitet.
Ingen horisont-, terskel-, tapsvekt- eller parametersøk.

Inner- og sluttmodeller/prediksjoner lagres per fold, slik at videre lesing
ikke krever omtrening. To syntetiske mekanikkontroller bestod: konstanten
vinner korrekt og bruker full-fold-prior, og et bedre trinn refittes på hele
folden. Disse er ingen markedsmåling.

## Forhåndsbundet avlesning

Predikert sannsynlighet over/under 0,5 betyr bullish/bearish; nøyaktig likhet
telles som nøytral. Rapporter klassestøtte, begge retningers treff, balansert
treff, log-loss, Brier og AUC per år. Sannsynlighetene er ikke uavhengig
kalibrert. Referansen for retning er de lagrede regresjonsprediksjonene etter
inkludering av konstantalternativet, ikke den feilaktige gamle velgeren.

Usikkerhet: trekk hele måneder innen hvert av de fire årene separat,
2000 ganger med seed 0. Vekt årene likt. Tre samtidige kontraster:
makro-BA minus 0,5; makro-BA-løft over regresjonen; makro-log-loss-løft over
konstant TRAIN-prior. Kvantiler 0,008333333333333333 / 0,9916666666666667.
Alle nedre grenser må være positive, og begge retninger må predikeres,
for videre utviklingsgjennomgang. Dette korrigerer ikke hele prosjektets
historiske utprøving.

Rapporter også kostnadsjusterte h12-utfall for den ufiltrerte retningsregelen
på alle muligheter, med samme arkiverte 4 bps utførelseskostnad og faktisk
finansiering. Ingen confidence-terskel velges. Disse overlappende markeringene
er ikke porteføljeøkonomi eller native Exit. En bestått retningskontroll er
heller ikke en bestått økonomisk læringsport eller tillatelse til større trening.

## Drift og binding

Rot: /home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923/DIRECT_DIRECTION_EARLY_1311_20260924.

PLAN, START, faktisk prosess og TERMINAL avgjør nåstatus. Ingen oppstart
skal antas fra dokumentet alene. Grense 900 s, 10 GiB RAM, 512 MiB swap,
én numerisk tråd, CPU 0–7 og eksisterende producer-vakt/eksklusiv lås.
Gjenbruk delresultater ved stopp; ingen automatisk restart.

Ingen native modell-/treningskodeendring, juni-/VAL-datasett eller TEST.
Ingen live/papir, spending eller offentlig ENGINE-push. Retningsmålet og
kravet om senere økonomisk nyttig LONG/SHORT/FLAT er fortsatt uoppnådd.
