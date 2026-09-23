# Første native Entry-læringsmåling og fast lærer — 23.09.2026

## Fullført måling

ENTRY_V12_NSTEP512_20260923T082726Z fullførte med returkode0 på source88f0cf46.
08:29:24–08:40:32 UTC, 667,97 sekunder. 512 TRAIN-rader, 512 senere
utviklings-VAL-rader, batch8 og 64 optimizersteg. Native vakter var aktive.
Originalcheckpoint844 er bevart. Råvekter etter fit og komplett før/etter-rapport
ligger i bundle.learning_comparison. Dette er forskningsbevis, ikke promotering.

| Samme utvalg, råmodell | TRAIN før | TRAIN etter | VAL før | VAL etter |
|---|---:|---:|---:|---:|
| LONG / SHORT / FLAT | 0/512/0 | 0/512/0 | 0/512/0 | 0/512/0 |
| Markert brutto Bps per mulighet | -3,4206 | -3,4709 | +3,1800 | +6,1409 |
| Markert Bps med arkivert kostnadsscenario | -7,4206 | -7,4709 | -0,8200 | +2,1409 |
| Lukkede / åpne | 363/149 | 320/192 | 430/82 | 389/123 |
| Median faktisk holdetid, lukkede | 63 min | 88,5 min | 47,5 min | 73 min |
| Summert observert notional-timer | 2893,57 | 3418,90 | 1742,32 | 2525,40 |

Alle valgte handler og åpne posisjoner inngår som lukket resultat eller mark.
Dette gjelder observasjonsvinduene, ikke full livsløps- eller porteføljeavkastning.
Tidsregningen bruker faktisk M1-klokke etter M5-barens tilgjengelighet, verifisert
mot alle 4096 lagrede sideutfall. Det arkiverte kostnadsscenariet er uendret:
4 Bps rundtur og gammel finansieringspolicy, ikke dagens bekreftede meglervilkår.
Ingen framtidig pris ble modellinput. Juni-VAL er gjenbrukt utviklings-VAL.

VAL-rangeringen mellom valgt Q og observert resultat falt fra0,3148 til0,1503.
TRAIN-økonomien ble ikke bedre. VAL-forbedringen kom med flere åpne posisjoner
og lengre holding; Entry endret ingen valg. Læringsporten for bedre selektivitet
er derfor ikke bestått. Ingen full epoch eller full VAL skal følge dette.

## Observert ustabilitet og én avgrenset kontroll

Treningens lærer ble oppdatert på alle64 optimizersteg, etter den eksisterende
native regelen interval=max(1,steps_per_epoch/512). Fasiten flyttet seg dermed
underveis. Mot den uendrede evalueringslæreren ble Entry-MSE på TRAIN264,04→351,28
og på VAL433,72→517,23. Exit-MSE på TRAIN8,78→261,00 og på VAL10,31→492,59.
Den løpende treningsfeilen153,76 mot flyttende mål er ikke samme måling som
før/etter mot fast lærer. Det er ikke dokumentert at læreroppdateringen alene
forårsaket svekkelsen; dette er hypotesen neste kontroll isolerer.

En eksplisitt recipe-bundet freeze_initial_teacher er nå tillatt bare i den
allerede avgrensede initialiserte FP32-smoke-ruten. Resten av native trening,
features, tap, optimizer, batchstørrelse, data, seed og initvekter beholdes.
Flagget setter refresh-intervallet til antall steg pluss én, altså ingen refresh
innen det ene deklarerte budsjettet. Etter fit må lærerens vektdigest fremdeles
være identisk med baseline. Exit-kontraktv4 deklarerer denne smale unntaksruten.
Gamle init-recipes uten flagget beholder sin tidligere oppførsel.

29 avgrensede kontrakt-/launcher-tester bestod. Dette er argument- og
beregningskontroll, ikke et læringsresultat. Neste ene kontroll bruker nøyaktig
samme512 TRAIN/512 VAL, checkpoint844, seed1337 og64 steg. Før-baseline må
sammenlignes direkte med den fullførte målingen før forskjellen tolkes.

Et lavere TRAIN-tap alene er utilstrekkelig. Rapportér faktiske Entry-valg,
senere VAL-økonomi under samme kostnadsscenario, alle åpne mark og tid. Ingen
ny terskel, tapsvekt, arkitektur, handelsregel eller meglerkontakt er lagt til.
TEST er forseglet. GX1_CURRENT er urørt. PC er ikke restartet.

Runtime: /home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923.
Se NATIVE_NSTEP512_TIME_COST_ANALYSIS.json og den fullførte run-katalogens
TERMINAL.json / LEARNING_COMPARISON.json for maskinlesbart bevis.
