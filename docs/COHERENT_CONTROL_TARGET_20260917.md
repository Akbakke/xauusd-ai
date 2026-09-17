# Samme observerte Entry/Exit-fasit i senere kontroll — 2026-09-17

Den eksisterende kontrollen målte Exit mot observerte referansemål, men Entry
mot lærerens rå state0-Q. For den eksplisitte kronologiske CONTROL256-kohorten
bruker begge nå samme beregnede referanse-Q_mu. Entry legger til første
utførbare likvidasjon én gang og velger maksimal gyldig HOLD/EXIT-verdi;
FLAT og likvidasjonsrelativ EXIT_NOW er0. Eksisterende targeteiere gjenbrukes.

Kontrollen krever samme frosne lærer for begge oppgaver. Policy og tidsgrense
leses fra den hashbundne frosne planen. Hele observerte targetstøtten, inkludert
bootstrap-tilstandens M1-sluttid, må ligge innen kontrollperiodens slutt.
Overskridelse avvises før økonomiprojektering eller Exit-statebygging;
forløpet trunkeres ikke. Bootstrap ved observasjonsgrensen bevares.
Kontrollrader beholder eksakte parent-/child-ID-er og fysisk TRAIN-identitet.

Den samme beregnede Exit-targettensoren brukes direkte til Entry. Det trengs
ingen ekstra lærerforward ved state0. Når target og boundary er samme modell,
gjenbrukes også dens Entry-representasjon. Gamle juni-/fullVAL-kohorter
beholder tidligere semantikk; eldre resultater endres ikke eller nytolkes.

24 unike syntetiske CPU-tester består:17 nye og7 berørte legacy-kontroller.
TRAIN og kontroll har eksakt like Entry-/Exit-targettensorer for2,3,121,122 og
600 tilstander, med markedsstenging og høyresensurering. Det inkluderer
kostnader, positiv boundary-bootstrap, klokkegrense, frosne gradients,
rolle-/koordinatbinding og avvisning av blandede lærere. Tre korte forløp
avdekket bare en feil i testens boundary-statebygging; denne ble rettet til
eksisterende predecessor-successor og kun disse tre ble kjørt igjen.
Produksjonskoden var uendret etter første testkjøring; alle logger bevares.

Dette er fasitlikhet, ikke læringsbevis. Immutable referanser til ferdige
normaliseringer og separat bundet faktisk fersk oppstart/måleklargjøring,
inkludert fast ONLINE-snapshot, gjenstår før forsøket. Ingen markedsmodell,
optimizer, refit, native launch, full epoch/VAL eller TEST er kjørt eller åpnet.

Bevis: handover_snapshot/COHERENT_CONTROL_TARGET_SYNTHETIC_20260917.json.
