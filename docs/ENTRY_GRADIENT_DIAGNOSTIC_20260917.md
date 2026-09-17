# Entry-gradient: én TRAIN16-kontroll uten trening

Den fullførte fast256-modellen overfører ikke læringen robust. Entry-feilen
er eksplisitt koblet fra feature-/MTF-representasjonen i V4. Denne kontrollen
måler den konkrete begrensningen; den åpner ikke et nytt treningsforsøk.

Bruk fast ONLINE256 og de første16 radene i eksisterende fryst TRAIN-probe.
Gjenbruk lagrede mål, prefix-inputs og eksisterende FP32-forward. To eval-
forwards med modellens eksisterende gradientflagg, henholdsvis detached og
connected. Ingen modell-/taps-/samplerendring eller optimizersteg.

Krev eksakt samme prediksjoner som den lagrede sluttmålingen, identiske
Entry-hodegradienter mellom variantene, endelige gradienter og null akkumulert
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
