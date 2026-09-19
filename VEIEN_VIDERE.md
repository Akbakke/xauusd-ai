# Neste: avklar Entry-mål for samme frosne Exit-policy

Hele frosne512-policyen er allerede kjørt og vurdert. Gjenbruk
completed_frozen_train_policy og entry_complete_policy_target_alignment i
NEXT_RUN_POLICY.json. Ingen jobb er aktiv; forrige scope er brukt og fjernet.
Ikke gjenta denne evalueringen, rangeringstesten eller de beståtte tekniske testene.

Netto på TRAIN256 er LONG−4,1080 /SHORT−5,1893 Bps, tross bedring mot
umiddelbar EXIT. Øvre rangert halvdel er−3,3433 Bps; alle130 velger LONG.
Faktisk Entry er FLAT256/256. Alle512 forløp er avsluttet av modellen; ingen
åpne posisjoner eller skjulte tap er utelatt. Ingen læringsport eller profittbevis.

Den konkrete gjenværende designforskjellen er nå målt: Entry-target beskriver
Q_mu; handelsforløpet bruker frossen pi512. Referansetarget mot fullpolicy-utfall
har Pearson0,408 LONG/0,391 SHORT; middelavvik+0,358/−2,771 Bps. Entry512s
LONG−SHORT-korrelasjon er0,208 mot referansetarget og0,158 mot fullpolicyutfall.
Dette er globale, deskriptive tall på brukt TRAIN, ikke innenmånedstall eller
fasit på betinget forventningsverdi. Ingen target skal omskrives bare fordi
et annet target passer de observerte vinnerne bedre.

Neste avgrensede arbeid er å lese eksisterende Entry-/referansetarget-eiere og
avklare én kausal beregning av verdien av samme frosne Exit-policy som utføres.
Gjenbruk lagrede fullpolicy-utfall, opprinnelige prediksjoner og samme identiteter
som en cachet baseline. Bevar lærer, alle200 features/åtte familier, priser,
kostnader, FLAT0 og TRAIN-grensen. Etiketter kan bruke senere TRAIN-utfall;
framtidig beste side/gevinst kan aldri brukes som Entry-input eller utvalgsregel.
Avklar korrekt scope og målekrav før eventuell implementering/fit; ingen slike
nye fits eller modellforwards er nå bundet. Dette er én designavklaring, ikke
bredt modell-/terskel-/tapsvektsøk eller tillatelse til ekstra trening.

Selv korrekte policyverdier skaper ikke en handelsfordel som mangler i kausale
inputs. Hvis ingen konkret begrunnet forbedring finnes, behold koden og stopp
eskalering. Separate quote/ordre/fill-logger kan avklare slippage når de foreligger;
manglende kalibrering er ikke tillatelse til å senke kostnader for å få PASS.
Ingen full epoch/full VAL, CONTROL/TEST, live/paper/spending. Målet er aktivt.
