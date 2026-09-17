# Avgrenset Entry-signaldiagnose — 17. september 2026

Siste læringsprøve er fortsatt avvist. Lagrede bevis viser at Entry-routing fikk
gradientnorm 21,6508 på første faktiske batch. Sluttvekting er 0,97526 for Entry
og0,97585 for Exit. Optimizer har 256 steg for 726 tilstander; Entry-mikser,
head og rutere har endret seg. Total frakobling eller tapsvekt nær null støttes
ikke. Dette beviser ikke nyttig retningslæring eller utelukker gradientkonflikt.

Én ny måling er bundet før utførelse: native campaign, samme cachede TRAIN16,
startmodell og sluttmodell, to eval-forwards og null optimizersteg. Korrekt
Entry-fasit gjenbrukes; cached input-eier, initialbaseline, modell, cursor,
kilde og policy kontrolleres. Ingen rematerialisering av fungerende inputcache.

Mål variasjon i lokal M5-, fused-, MTF-, global-context- og Entry-hidden-
representasjon. Del Entry-MSE eksakt i fellesnivå, LONG−SHORT og FLAT, og mål
vektede gradientnormer/retning mot hjelpeoppgavene på Entry-private rutere og
head. Dette endrer ikke den trente modellen, tapsformelen eller optimizer.
Ingen fullstendig klippet Adam-oppdatering rekonstrueres; Exit-forward og
treningsmodus/dropout inngår ikke. Resultatet skal tolkes innen dette omfanget.

19 avgrensede diagnostikktester består, inklusive eksakt taps-/gradientidentitet,
to frosne modeller uten akkumulert gradient og avvist endret inputcache.
Beståtte fullsuiter gjentas ikke. Ingen ny trening, CONTROL, VAL, TEST, søk eller
automatisk forlengelse. Kilde fryses under faktisk kjøring som før.

Plan: BASE/NATIVE_ENTRY_SIGNAL_DIAGNOSTIC_20260917/PLAN.json.
BASE er /home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912.
PREPARATION_RESULT.json, prosesser og kvittering avgjør om målingen faktisk har
startet/sluttet. Dette dokumentet er en bundet plan, ikke påstand om utførelse.

## Forsøket er terminalt med feil

Ingen signalrapport ble skrevet. Prediksjonskontrollen feilet; variant og
avviksstørrelse manglet i loggen. Se ENTRY_FORWARD_PARITY_DIAGNOSTIC_20260917.md
og handover_snapshot/ENTRY_SIGNAL_FAILURE_20260917.json. Ikke relanser denne planen.
