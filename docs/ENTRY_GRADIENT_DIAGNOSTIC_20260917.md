# Entry-gradient: én TRAIN16-kontroll uten trening

Den fullførte fast256-modellen overfører ikke læringen robust. Entry-feilen
er eksplisitt koblet fra feature-/MTF-representasjonen i V4. Denne kontrollen
måler den konkrete begrensningen; den åpner ikke et nytt treningsforsøk.

Bruk fast ONLINE256 og de første16 radene i eksisterende fryst TRAIN-probe.
Gjenbruk lagrede mål, prefix-inputs og eksisterende FP32-forward. To eval-
forwards med modellens eksisterende gradientflagg, henholdsvis detached og
connected. Ingen modell-/taps-/samplerendring eller optimizersteg.

Krev maksimalt0,0001Bps avvik fra lagret sluttmåling og identiske handlinger.
Krev bitlike prediksjoner og Entry-hodegradienter mellom variantene, endelige gradienter og null akkumulert
.grad. Mål LONG/SHORT/FLAT, routinggradient og forhold/vinkel mot eksisterende
vektede hjelpeoppgaver. Eval-modus holder dropout fast. Dette er én gjenbrukt
TRAIN-batch, ikke bevis på bedre læring eller overføring til senere data.

Kjør bare gjennom eksisterende native campaign, kildebinding og maskinvare-
vakter. Én invocation, to Entry-forwards, null Exit/CONTROL/TEST-forwards.
Originalt checkpoint og RNG bevares; batchen caches én gang i GX1_DATA.
Ingen automatisk ny trening, parameterendring eller gjentakelse etter resultatet.

Kun de to eksisterende native-eierne utvides for dette avgrensede omfanget.
14 fokuserte syntetiske CPU-kontroller består: scope, endrede inputbindinger,
nullstegsdispatch, eksakt output-/hodegradientparitet, virkelig åpnet
routinggradient og bevart checkpoint/RNG også ved målefeil. Faktiske
plan-/checkpoint-/TRAIN-bindinger er verifisert. Ingen faktisk forward er gjort.
TRAIN16 og vanlig native VAL-profil256/8arbeidere beholdes; VAL kjøres ikke.

Planen og syntetisk resultat er referert i
handover_snapshot/ENTRY_GRADIENT_DIAGNOSTIC_DISPATCH_20260917.json.
training_enabled=false; bare det navngitte diagnostikkunntaket er åpnet.

Første faktiske preflight stoppet før modellarbeid fordi kildebindingene har
python:-prefiks og utvidet metadata. Bare oppslaget og testfixturen ble rettet;
11 valgte scope/dispatch/mask-kontroller består etterpå. Original feillogg og
policykopier er bevart. Faktisk korrigert binding er nå verifisert.

## Første kjøring og minste målerettelse

Første native forsøk på4401c257 sluttet10:42:15UTC med child_status1 på
ENTRY_GRADIENT_FORWARD_VALUES_CHANGED. Faktisk differanse ble ikke lagret;
avrunding er foreløpig en hypotese. Null optimizersteg, ingen gradientrapport.
Original plan/oppskrift/logg/policy bevares i den opprinnelige artifactmappen.

Én korrigert kontroll er separat bundet i NATIVE_ENTRY_GRADIENT_NUMERIC_RETRY_20260917.
Den gjenbruker ONLINE256 og identiske TRAIN16-mål. Toleransen1e-4,rtol0 er
hentet fra eksisterende native FP32 output-paritet (VAL-eier), før dette avviket
er målt. Handlingene må være identiske. Variantparitet er fortsatt bitlik.
Avvik logges før stopp; inputcache skrives før kontroll. Åtte berørte tester
består, inkludert avvisning av endret handling innen toleransen. Modell, trener,
mål og vakter er uendret. Et nytt misforhold gir stopp, ingen toleransesøk.
Se handover_snapshot/ENTRY_GRADIENT_NUMERIC_REPAIR_20260917.json.

## Målt resultat og beslutning

Korrigert kontroll fullført2026-09-17T11:32:26UTC på4f6a64ec. GuardPASS,
trainer0,observer0. Null optimizersteg, to Entry-forwards, ingen Exit/CONTROL/TEST.
Original modell/checkpoint/RNG og inputcache er kontrollert.

| TRAIN16 | Dagens detach | Åpen Entry-forbindelse |
|---|---:|---:|
| Entry routing-gradientnorm |0|0,229012|
| LONG / SHORT / FLAT |0 /0 /0|0,133861 /0,191809 /0,016059|
| Parametertensorer med Entry-gradient |7|541|
| Entry-hodegradientnorm |23,367628|23,367628|

Hjelpeoppgavenes routingnorm er1,936871, Entry/aux-forhold0,118238 og
cosinus0,549554 på denne ene gjenbrukte batchen. Ingen tapsvekter endres ut fra
dette. Variantoutputs og hodegradienter er bitlike. Avvik fra lagret inferens
er0,0000295639Bps med identiske handlinger; første forsøk var for strengt.

Beslutning: fjern bare Entry-Q-kildens detach. Behold Exit-tokenets detach og
alle inputs, tap, priser, tidshorisonter og kausalitet. Dette korrigerer den
påviste gradientbegrensningen; det er ikke en dokumentert læringsgevinst.
Før mer trening kreves én separat bundet, fast TRAIN-only læringskontrast.
Senere kontroll er allerede brukt til utvikling; ingen ommerking til urørt
holdout eller tuning mot dens utfall. TEST forblir forseglet.
