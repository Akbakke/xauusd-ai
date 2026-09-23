# Fast lærer og Entry-feedback — 23.09.2026

## Nåstatus

Begge avgrensede native kjøringer er fullført med returkode 0. Ingen trening er
aktiv. Totalt 128 nye optimizersteg er utført i to separate kontroller fra samme
checkpoint844; dette er ikke én sammenhengende trening eller kandidat-resume.
Målet om bedre Entry er aktivt, men læringsporten er ikke bestått.

- Flyttende lærer: ENTRY_V12_NSTEP512_20260923T082726Z, source 88f0cf46,
  08:29:24–08:40:32 UTC, 64 steg, 64 læreroppdateringer.
- Fast lærer: ENTRY_V12_FIXED512_20260923T085645Z, source eddae3c1,
  08:57:29–09:08:33 UTC, 64 steg, ingen læreroppdatering.
- Samme 512 TRAIN og 512 senere utviklings-VAL, startvekter, data, seed og øvrig
  recipe-CLI. Hele før-statistikken og radobservasjonene er eksakt identiske for
  begge splitter. Lærerens digest ble kontrollert uendret etter den faste fitten.
- TEST er forseglet. GX1_CURRENT er urørt. Ingen PC-omstart eller handel er utført.
  Originalcheckpoint og begge nye råvekt-snapshots er bevart.

## Kontrollert resultat

| Råmodell, samme utvalg | Før | Flyttende lærer etter | Fast lærer etter |
|---|---:|---:|---:|
| TRAIN LONG / SHORT / FLAT | 0 / 512 / 0 | 0 / 512 / 0 | 0 / 512 / 0 |
| VAL LONG / SHORT / FLAT | 0 / 512 / 0 | 0 / 512 / 0 | 0 / 512 / 0 |
| TRAIN brutto markert Bps per mulighet | -3,4206 | -3,4709 | -3,2767 |
| TRAIN med arkivert kostnadsscenario | -7,4206 | -7,4709 | -7,2767 |
| VAL brutto markert Bps per mulighet | +3,1800 | +6,1409 | +2,8141 |
| VAL med arkivert kostnadsscenario | -0,8200 | +2,1409 | -1,1859 |
| VAL lukkede / åpne | 430 / 82 | 389 / 123 | 433 / 79 |
| VAL Q mot observert resultat, Spearman | 0,3148 | 0,1503 | 0,2665 |
| TRAIN Entry-MSE mot fast lærer | 264,04 | 351,28 | 258,35 |
| VAL Entry-MSE mot fast lærer | 433,72 | 517,23 | 435,18 |
| TRAIN Exit-MSE mot fast lærer | 8,78 | 261,00 | 7,21 |
| VAL Exit-MSE mot fast lærer | 10,31 | 492,59 | 15,58 |

Fast lærer fjernet mesteparten av verdiustabiliteten i denne kontrollerte prøven,
men ga ingen endrede Entry-valg eller positiv kostnadsjustert VAL. Lavere
treningsfeil er ikke bedre seleksjon. Ikke relanser noen av de fullførte planene.
Ingen større trening er begrunnet av disse resultatene.

Alle valgte handler og åpne posisjoner inngår. Tallene gjelder observerte
vinduer, ikke full livsløps- eller porteføljeavkastning. Arkivert scenario:
4 Bps rundtur, LONG-finansiering 5,4 prosent per år, SHORT-kostnad 0; dette er
gjenbrukt forskningsgrunnlag, ikke bekreftede nåværende meglervilkår.
Samme M1-priser og faktisk klokketid er verifisert mot alle lagrede sideutfall.
Juni 2026 er gjenbrukt utviklings-VAL, ikke uavhengig aksept.

## Tidsfeedback og åpne handler

Etter fast lærer var median holdetid blant lukkede VAL-handler 43 minutter,
p90 268,8 minutter. Summert observert kapitalbinding var 1632,1 notional-timer.
Kostnadsjustert resultat var -0,3720 Bps per observert notional-time.
Lukkede VAL-posisjoner summerte +7252,97 Bps; åpne mark summerte -7860,14 Bps.
Ingen av de 877 lukkede sideforløpene hadde negativt resultat etter scenarioets
kostnader. Det gjør åpne tap avgjørende for vurderingen; lukkede vinnere alene
beskriver ikke strategiens økonomi. Maksimal holdetid er ikke innført.

## Målene Entry faktisk lærer

Gjenbruk av lagrede diagnostikker ga følgende før-fit sammenligning:

| SHORT, brutto Bps | TRAIN | VAL |
|---|---:|---:|
| Gjennomsnittlig n-step læringsmål | +5,3441 | +11,8254 |
| Gjennomsnittlig faktisk vindusmark | -3,4206 | +3,1800 |
| Åpne posisjoner | 149 | 82 |
| Åpen posisjons gjennomsnittlige bootstrap-mål | -21,6258 | -39,4940 |
| Åpen posisjons gjennomsnittlige faktiske mark | -51,7435 | -93,4748 |

Åpne læringsmål er allerede negative i snitt. De må ikke beskrives som bare
positive mål. Forskjellen kommer fra lærerens forventede videre verdi og er
ikke automatisk en kodefeil eller bevist overestimering av sluttresultatet.
Faktisk sluttresultat mangler for disse åpne forløpene. Målmiddel for åpne
posisjoner er rekonstruert fra vektede desiler og observerte lukkede utfall;
det inneholder avrunding fra float32-diagnostikk, ikke gjenopprettede radmål.

En konstant TRAIN-tilpasset verdi per handling gir Entry-MSE 261,09 på TRAIN
og 441,38 på VAL. Fast fit gir 258,35 og 435,18: et lite utslag i verdiestimat,
uten selektiv handling. Alle tre handlinger inngår i denne tapssammenligningen.

## Rangering: ingen enkel terskel er begrunnet

Før- og etter-score ble inndelt i ti grupper med grenser fra TRAIN, deretter
anvendt uendret på senere VAL. Ingen gruppe ble valgt som handelsregel.

- Etter fast fit var TRAIN-gruppenes netto vindusmark, lav til høy score:
  -25,77 / -15,79 / -8,10 / -13,18 / -7,98 / -1,04 / +3,41 / +0,73 / +6,03 / -10,65 Bps.
- Høyeste TRAIN-scoregruppe hadde -10,65 Bps; samme grenser omfattet 188 VAL-rader
  med -6,66 Bps. Høyere score gir dermed ikke jevnt bedre observert økonomi.
- En ren diagnostisk subtraksjon av 4 Bps fra begge side-Q endret null av
  512 Entry-valg i noen split, før eller etter. Dette er ikke implementert som
  postmodell-regel og er ikke en full finansieringsmodell.
- Vinduer og handler overlapper. Gruppene er beskrivende, ikke uavhengige
  signifikansbevis, og må ikke brukes til å velge en vinnende VAL-gruppe.

## Videre arbeid

Ingen ny fit er bestilt av denne rapporten. Neste konkrete avklaring er
læringsmålets kobling til observerbar Entry-kvalitet: hvilke negative utfall
og faktisk tidsbruk som når Entry, og hvilke som erstattes av videreverdi.
Eksisterende pris-/kostnadsdata og hjelpehoder skal gjenbrukes før nye mekanismer.
Både høyere TRAIN-rangering og svak senere VAL må forklares; ikke løs dette med
et vilkårlig scorefilter, tapsvekt eller antatt fremtidig innhenting.

Et mål basert bare på mark ved vindusslutt ville være et endret læringsmål,
ikke målt verdi over hele handlens livsløp. Ikke innfør dette stilltiende, og
ikke innfør EXIT på 512. Full livsløps-VAL er fremdeles ikke dokumentert.

## Etterprøvbart bevis

Runtime: /home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923.

- ENTRY_FIXED512_TARGET_AND_RANK_AUDIT.json
- NATIVE_FIXED512_TIME_COST_ANALYSIS.json
- NATIVE_NSTEP512_TIME_COST_ANALYSIS.json
- Begge run-katalogers TERMINAL.json
- Begge bundles .learning_comparison/LEARNING_COMPARISON.json
- ENTRY_GOAL_PROGRESS.json og RUNNING_NATIVE_ENTRY_LEARNING.json

Rapport-SHA256: flyttende lærer
fa771d8ea2f4c215213faf54b1765457d21c068a173bc40fcaa500172efe53f1;
fast lærer
911ef133d99dc78ca5ad833a24399c6c4e10e5437b79fe361f4f298acfda21ca.
