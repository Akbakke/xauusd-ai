# V12 Entry-repair — 2026-09-23

## Nåstatus

Brukeren ba om å stoppe, rette feil og starte på nytt, og har deretter satt
målbar, selektiv Entry som hovedmål. V12-treningen og supervisoren er stoppet.
Checkpoint 844 er arkivert og hashkontrollert. Ingen ny trening eller PC-omstart
er startet. GX1_CURRENT er urørt; denne gjennomgangen gjelder Claude-kjøringen
i GX1_ENGINE, branch audit/v9-premiere-20260905, opprinnelig commit 4f0fea9b.

Arbeidsmål: bedre LONG/SHORT/FLAT-seleksjon må dokumenteres med faktisk observert
handelsutfall, separat fra lærerens Q-estimat. TEST er forseglet. Ingen større
trening før ny baseline og avgrenset læring viser fremgang. Juni er brukt
utviklings-VAL, ikke urørt OOT.

## Bekreftede funn

- Epoch 1 VAL tok 9m22,175s og behandlet 5509 Entry-rader, 5500 gyldige
  episoder × 512 × 2 sider. Batchet GRU, gjenbruk av forwards og mindre
  evalueringsomfang forklarer forskjellen fra eldre VAL. Ikke samme oppgave.
- Entry-valgt gjennomsnitt var -4,406986 Bps per mulighet, med FLAT=0.
  Dette er spreadinkludert brutto, ikke porteføljenetto etter alle kostnader.
- Entry gate-vekt: session 99,7475 %, structure eksakt 0. Admission FAIL;
  ingen valgt beste modell. Exit ga -1,715960 Bps mot umiddelbar -1,569863.
- Entry valgt-Q rangerte negativt selv mot lærerens mål: Spearman -0,261964.
  Dette er ikke korrelasjon mot observerte handelsutfall; sistnevnte manglet.
- 512 var både pakke-/beregningsstørrelse og feilaktig handelsterminal:
  HOLD ble ugyldig, EXIT tvunget, carry utover 512 avvist.
- Supervisoren gjenstartet ved alle kode 75, selv om 75 også brukes ved
  temperatur-/telemetrifeil. Observerte fullførte segmenter var ordinære
  tidsgrensestopp; ingen omgått termisk stopp er påvist.

## Avgrensede rettelser og målinger

1. Entry: stateless LayerNorm bare foran specialist_token_gate; verdi-grenen
   uendret. Modell-/outputskjema v9 krever ny funksjonsbaseline.
   På 32 reelle VAL-rader med arkiverte E2-vekter var token-logittspenn ca334
   mot ca6,5 i basegrenen. Normalisering fjernet 113 eksakte nullceller av256.
   Session var fortsatt 99,5292 % i denne avgrensede kontrafaktiske målingen.
   Dette er numerisk diagnose, ikke dokumentert bedre seleksjon.
2. Exit: beregningsvinduets slutt tvinger ikke EXIT. Både HOLD og EXIT er
   gyldige. Ukjent siste HOLD-label utelates fra tapet. Åpne posisjoner føres
   separat og tas med til observert BID/ASK-verdi. Kontrakter er versjonert.
   Reell VAL-rad0 ble videreført fra512 til513 med eksportert/gjeninnlest carry:
   HOLD gyldig, terminalfalse, Q endelig. Batchet/påfølgende Q-avvik over de
   første512 var maksimalt 0,000068665 Bps. Dette bruker endret funksjon med gamle
   vekter, og gir ingen kandidatadmission.
3. Entry-rapport: LONG/SHORT/FLAT, coverage, valgte handler mot observerte
   utfall, rangkorrelasjon/deciler blant faktiske handler, LONG/SHORT/FLAT-baseliner,
   og åpne/lukkede posisjoner. Avviste epocher får varig JSON-review.
4. Samme HOLD-semantikk er rettet i eksisterende carry-konsumenter.
   Ingen live-/papirhandel er aktivert.
5. Supervisor-rettelse er klargjort utenfor repo. Den er ikke aktivert og er
   fortsatt bundet til gammel recipe. Den må bindes på nytt før eventuell bruk.

VIKTIG: Standard VAL følger fortsatt bare det observerte 512-vinduet og
rapporterer resterende posisjoner som åpne med markedsverdi. Modellen kan
fortsette, men evaluatoren er ennå ikke en full livsløpsreplay.
Dette skal ikke presenteres som fullført evaluering uten horisontavgrensning.
Samme-close-eksekvering, full kostnadsmodell, finansiering og porteføljebegrensning
er heller ikke produksjonsvalidert. Produksjonsporten forblir BLOCK.

## Neste konkrete Entry-arbeid

- Gjenbruk arkivert checkpoint og bundne TRAIN/VAL-input.
- Mål Entry-Q, lærer-Q og observerte utfall hver for seg på et fast,
  kronologisk avgrenset utvalg. Skill feil i mål fra feil i tilpasning.
- Mål seleksjon mot FLAT og alltid LONG/SHORT under samme Exit-policy.
  Ikke sett et vilkårlig coveragekrav eller legg på terskelsøk for å pynte resultatet.
- Bruk målingen til én liten rettelse og en før/etter-måling.
  Ingen flere features, bred refaktorering eller blind epoch-restart.
- Ny funksjon trenger ny baseline, recipe/kildebinding og gjeldende guard.
  Gammel kandidat må aldri resumeres med endret kilde.

## Arkiv og drift

Bevisrot: /home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923.
Checkpoint: STOP_CHECKPOINT/candidate_training_state_slot_1.pt,
SHA256 5acd2947f584f8ad00c98a64c7f173d5d8d14db0be33732131e99fa719e749b3,
epoch_index1, next_batch_offset13952, optimizersteg53127.
Epoch1-snapshot SHA256:
f988c43cc818c39699ee90a6c67096e708a63d9f16fb6cd7ca24e77ed3d2a808.

Første 513-diagnose hadde feil broadcasting i sammenligningen av Q-tensorer.
Den er bevart som superseded_shape_comparison; corrected-loggen og
REAL_VAL_CONTINUATION_513.json er gjeldende. Det ble ikke brukt som paritetsbevis.

Den gamle Windows-oppgaven GX1NativeLearningCalibration er Disabled, med
XML-backup under C:\\Users\\Andre\\GX1_V12_REBOOT_PREP_20260923.
GPU power-limit og host-telemetri er bevart aktive. PC er ikke restartet.
Bugcheck0xA fra21.09 er observert; årsak og effekt av reboot er ikke fastslått.
