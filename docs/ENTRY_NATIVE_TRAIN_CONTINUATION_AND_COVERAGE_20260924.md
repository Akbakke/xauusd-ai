# Frossen native fortsettelse og treningsdekning — 24.09.2026

## Avgrensning og utførelse

20 fortsatt åpne TRAIN-sider fra forrige fortsettelse ble gjenopptatt fra
lagrede recurrent carries. Samme checkpoint, opprinnelige Entry-valg og frosne
v10-funksjon. Ingen fit, optimizer, nye Entry-forwards eller VAL-beslutninger.
Kilde: 64cbed17cbc591b85a917836d252ebf8c682f526. Originale inputs, carries og
modell ble kontrollert uendret. Felles loader materialiserer også pretest-kontekst
til juni; beslutninger/priser i fortsettelsen er TRAIN, TEST ikke åpnet.

20 beregningssekunder per side, 600 sekunder samlet etter initialisering og
1200 sekunder veggklokkevakt. Dette er beregningsgrenser; fortsatt åpne posisjoner
er bevart, ikke tvangslukket. Hovedjobben avsluttet rc0 15:04:18 UTC /17:04:18 Oslo,
502,12 sekunder inkludert lasting; 316,67 sekunder i fortsettelsen.
26 658 nye tilstander. Etterberegning avsluttet rc0.

Evidens:
`/home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923/FROZEN_V10_TRAIN_RESUME20_20260924`.
PLAN, ANALYSIS_PLAN, RESULT, TERMINAL, ANALYSIS, logger og scripts er bevart.
Vekter og carries forblir private på Linux.

## Resultat — alle åpne markeringer medregnet

| Populasjon | Antall | Lukket | Åpne | Gjennomsnitt netto bps |
|---|---:|---:|---:|---:|
| Alle opprinnelige LONG/SHORT-sider |126|113|13|−7,603|
| Opprinnelige Entry-valg |63|57|6|−14,085|
| Alltid LONG, samme tidspunkter |63|58|5|−5,251|
| Alltid SHORT, samme tidspunkter |63|55|8|−9,954|
| Nye lukkinger i denne fortsettelsen |7|7|0|+16,222|
| Fortsatt åpne sider |13|0|13|−177,228|

To av de sju nye lukkingene er negative etter kostnader. De opprinnelige
63 Entry-valgene var −14,157 bps ved forrige beregningsstopp og −6,209 ved
512 tilstander. Ulike observasjonsendepunkter, TRAIN-gjenbruk, hypotetisk
overlapp og uavklarte åpne utfall hindrer tolkning som ferdig livsløpsresultat,
porteføljeavkastning eller uavhengig VAL-bevis. Ingen læringsport bestått.

Netto bruker tidligere arkivert scenario: spread i priser, 2 bps per utførelse,
0 kommisjon, 5,4 % årlig LONG-finansiering, 0 SHORT, faktisk veggklokketid.
Åpne markeringer inkluderer antatt lukkekostnad. Dette er ikke oppdaterte
meglerpriser. 13 sider er beregningssensurert; ingen stoppet ved datagrense.

## Konkret begrensning i den eksisterende treningsruten

V12 TRAIN-manifest oppgir path_state_count=512,
required_observed_m1_rows_per_episode=512 og state_population_per_episode=512.

`gx1/contracts/unified_exit_lifecycle_v1.py:1765` bygger vinduet fra Entry-fill
til start+UNIFIED_EXIT_MAX_PATH_BARS. Trenerens
`EntryV10CtxDataset.materialize_full_exit_episode` bruker denne ruten.
Dette gir supervisjon på de første 512 observerte M1-tilstandene per inngang;
det sampler ikke senere vinduer fra disse posisjonenes livsløp.

`gx1/contracts/unified_exit_fitted_q_v1.py:253` utelater bare ukjent HOLD-label
i siste tilstand fra tapet. HOLD forblir lovlig; ingen tvungen EXIT ved512.
512 observerte barer er heller ikke 512 veggklokkeminutter over markedspauser.

Fortsettelsen når rundt3700 tilstander på de gjenværende åpne sidene.
Dette viser et misforhold mellom direkte treningsdekning og policyens bruk.
Det beviser ikke alene årsaken til svak Entry eller feil livsløpsverdi.
Gjenbruk eksisterende lærer-, kostnads-, gradient- og reward-anchor-kontroller
før en eventuell avgrenset rettelse. Ingen ny fit eller videre fortsettelse
utløses automatisk av denne diagnosen.

## Kombinasjonssøk og neste beslutning

Tidligere brede HGB-kontroller lot modellen bruke1076/1311 felt, inklusive
flere tidsrammer og mønstre. De trener ikke én håndskrevet indikatorregel.
Siste kontroll låste h12/ATR og gammel valgt kompleksitet (ett/tre trær);
den dekker derfor bare en avgrenset klasse samspill, ikke alle kombinasjoner.
De negative resultatene er bevart i tidligere rapporter.

Neste inngrep må knyttes til en målt læringsblokkering, særlig sammenhengen
mellom læringsmål, direkte treningsdekning og observerte utfall. Ingen ny
terskeljakt, full native trening eller juni-tilpasning på dette grunnlaget.
Målet om nyttig og selektiv Entry er fortsatt aktivt, ikke oppnådd.

## Etterkontroll: kjent EXIT-verdi ved de sene tilstandene

LATE_EXIT_VALUE_REVIEW.json sammenligner de samme20 sidene ved forrige og
nåværende endepunkt, uten forwards eller fit. MAE for Q_EXIT mot kjent
spreadinkludert lukkebelønning steg fra6,548 til14,337bps; største avvik
204,838bps. De13 fortsatt åpne sidene har MAE20,506bps. Dette er målbar
feil i en verdi som er kjent i tilstanden, ikke bevis for størrelsen på
feilen i forventet fremtidig HOLD-verdi.

Tidligere kontroll av kjent reward-ankring er gjenbrukt: et felles skift av
begge handlingsverdier retter Q_EXIT, men bevarer HOLD/EXIT-rangeringen
bortsett fra mulige flyttallsbånd. Ingen slik kalibrering er innført som en
påstått Entry-forbedring. De sju nye lukkede utfallene er et selektert utvalg;
de kan ikke alene kalibrere de gjenstående åpne livsløpene.
