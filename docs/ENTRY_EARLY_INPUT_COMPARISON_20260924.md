# Tidlige forskningsinputs og avgrenset HGB-sammenligning — 24.09.2026

## Inputrettelsen er fullført

EARLY_CALIBRATED_FEATURE_INPUTS_20260924 avsluttet med rc0 kl. 16:55:43 UTC,
429,81 s, fra clean b1b034bf5b841570c501c12aec9c0d40831cb721.
Eksisterende eiere regenererte de berørte blokkene med den fullførte tidlige
kalibreringen. Ingen modell-fit eller native datasettrebuild.

På alle 313399 TRAIN-rader ble gamle 33 nivå-, 31 trendlinje- og tre
squeeze-felt først reprodusert bitlikt. Ny kalibrering endret henholdsvis
121717, 2753788 og 472311 celler. Alle felt beholdes, også uendrede felt
innenfor de berørte eiergruppene.

For M5/M15/H1/H4/D1 ble alle 123 uendrede MTF-felt per klokke og alle native
skalarer gjengitt bitlikt på felles kildeprefiks. Sammenligningen omfattet
464809 / 154969 / 38762 / 10137 / 1692 barer. Seks eksakte ctx-aliaser var
bitlike på alle 313399 beslutningstidspunkter. Ingen toleranse ble utvidet,
og ingen felt eller beslutningsrader ble fjernet.

Egen research_mtf/manifest.json har forskningsskjema og SHA
a87272afff9a645fa53d74eb8891a23f3dfde59aae0a16a94c5f751c2ea19e80.
Sammenkoblet 313399 × 760-matrise har SHA
1d07449227cb543546f8db51cffb1361fe817b5c6daf2a92586809060762b23f.
Native manifest og datasett beholder sin opprinnelige proveniens.

## Kostnadshinder må inn i seleksjonsavlesningen

En kontroll av eksisterende HGB_CORRECTED_SELECTION-prediksjoner tok under
ett sekund og krevde ingen ny fit. Av 18054 valgte hadde 15503 predikert
råavkastning høyst den arkiverte utførelseskostnaden på 4 bps.

Én forhåndsbestemt regel, prediksjon strengt over denne kostnaden, gir 2551
valgte: 650 LONG og 1901 SHORT. Netto er −5,736 bps per valgt og −0,056705
per mulighet. Færre handler reduserer eksponering/tap; retningen og
lønnsom selektivitet er fortsatt ikke bevist. Gamle prediksjoner har den
dokumenterte kalibreringsfeilen og er bare historisk regnegrunnlag.

4 bps følger det eksisterende arkiverte utførelsesscenarioet. Dette er ikke
en terskel valgt fra PnL eller en oppdatert meglerpris. Faktisk fremtidig
finansiering og holdetid brukes bare i utfallsvurdering, aldri i seleksjonen.

## Neste sammenligning, fastlagt før nye prediksjoner

HGB_EARLY_CALIBRATED_20260924 bruker samme eksisterende lærerfunksjon som
HGB_CORRECTED_SELECTION: 1311 felt, h12 close-fill/ATR, seed 0, opptil 300
trær, læringsrate 0,1, min_samples_leaf 20, indre andel 0,2 og purge 289
observerte M5-barer. Fire opprinnelige TRAIN-årsavlesninger. Åtte indre
fits og åtte fullstendige refits; ingen juni-/VAL-datasett eller native steg.

De gamle prediksjonene gjenbrukes uten refit. Nye faktiske inputblokker,
arrayhasher, feltnavn, radrekkefølge, targets og kostnader bindes. Alle
gamle indre/ytre maskers hasher må samsvare. Ny featurekalibrering må ligge
før hver faktisk indre og ytre kontrollstart. En kopi av snapshots i minnet
erstatter kun de regenererte blokkene; originalfilene bevares.

Samme prediksjoner avleses med eksisterende argmax_flat og med kravet om
prediksjon over kjent utførelseskostnad. Ingen ekstra fit eller terskelsøk.
Hver regel sammenlignes med tilsvarende gammel regel, FLAT, alltid LONG
og alltid SHORT. Månedsblokker, 2000 trekk, seed 0, intervallkvantiler
0,003125/0,996875 korrigerer for disse åtte kontrastene, ikke hele prosjektets
historiske søk. Positiv nedre grense mot både gammel regel og FLAT tillater
bare en ny utviklingsgjennomgang, ingen automatisk videre fit/promotering.

Retning måles separat som fortegnet på mid[t+12]−mid[t], med predikert
retning fra LONG-estimat minus SHORT-estimat. Rapporter balansert treff,
klassevis treff, klassestøtte og dekningsgrad på alle og valgte muligheter.
Nullbevegelser beholdes i økonomien og telles separat; prediksjonsties
telles som feil ved retningstreff på faktiske opp-/nedbevegelser.
Retningsavlesningen er beskrivende, uten ny terskel eller egen PASS-port.

Tidsrammen er 1800 s, 10 GiB RAM, 512 MiB swap, én numerisk tråd, CPU 0–7,
eksisterende producer-vakt og eksklusiv tung-jobb-lås. Delresultater bevares
ved stopp. Ingen automatisk restart, større modell eller terskelsøk.

Dette måler virkningen av reparerte inputs innenfor ett bestemt oppsett.
De fire årsperiodene er gjenbrukt utviklingsdata. Uavhengige, overlappende
h12-markeringer er ikke en kapitalført portefølje eller native Exit-livsløp.
Et bedre eller teknisk korrekt resultat er ikke alene en bestått læringsport.

## Evidens

/home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923/

- EARLY_CALIBRATED_FEATURE_INPUTS_20260924: PLAN, RESULT, TERMINAL, lokale
  blokkrapporter og separat research_mtf-pakke.
- HGB_EXECUTION_COST_FLOOR_20260924: resultat fra lagrede prediksjoner.
- HGB_EARLY_CALIBRATED_20260924: avgrensning og kjøreskript; START/TERMINAL
  og faktisk prosess avgjør nåstatus. Ikke anta oppstart fra dette dokumentet.

TEST forblir forseglet. Ingen native modell-/treningskodeendring, spending,
live/papir eller offentlig ENGINE-push. Målet er aktivt og ikke oppnådd.
