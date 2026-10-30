# M5 sign-videreføring: korrigert kvitteringsbinding

Den planlagte videreføringen fra sign-ettpass stoppet i metadataforkontroll med KeyError: 'result'. Den fullførte og tidligere verifiserte signkontrollen har feltet fit_result og egne guard-/kilde-/checkpointfelt, mens videreføringskontrollen forventet en annen kontrollskjema-variant. Dette hindret neste forsøk; det er ikke påvist som årsak til de tidligere retningstallene.

Kontrollen godtar nå begge de to kjente skjemaene, men krever fortsatt den eksakte fullførte planen, signert PASS-kvittering, opprinnelige initial- og terminaltilstander, samme foreldrerader og rekkefølge, uendret treningskilde, 2 976 oppdateringer, 47 602 TRAIN-foreldre og forseglet TEST. Den faktiske eldre kvitteringen, rapporten og checkpointene er bevart.

De 64 kildeeierbanene er de samme, men hash for denne kontrollfilen er endret. De øvrige 63 kildeeierne er byteidentiske. Koden krever at abstrakt syntakstre er identisk utenfor den ene videreføringskontrollfunksjonen; signmodell, datarute, tap, optimizer og treningsløp er ikke endret. Dette er et dokumentert kontrollfil-unntak fra den opprinnelig planlagte identiske-kildehash-sammenligningen, ikke en numerisk modellendring.

Den faktiske signterminalen passerte ny metadataforkontroll. 13 fokuserte tester bestod, inkludert avvisning ved ugyldig guard/kvittering, feil checkpoint, feil initialprediksjoner, annen kildeeier, endret kode utenfor kontrollen og endret kontrollsignatur. Fortsettelsesoppskrift, ny engangsplan, fysisk reboot og faktisk andre pass gjenstår. TEST, Exit, broker, ordre, live/paper, spending og promotion er stengt.

Bevis: /home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_M5_SIGN_CONTINUATION_REPAIR_20261030_001/RESULT.json, SHA-256 a7ff371fd0ac2b5af8f4946b6ed90124b825971d2ac14aa879a64eac5ed0bac9. Dette viser at videreføring kan bindes teknisk; det beviser ikke bedre retning eller edge.
