# Gjeldende status - 06.10.2026

Målet om punkt 1–6 i docs/NATIVE_V38_EXECUTION_20261005.md er aktivt, ikke fullført.
CPU_WORKLOAD_PROFILE_003 er fullført med exit 0 og uendret kilde. Ikke relanser
den. CPU-budsjettvalget er fortsatt ubekreftet og kreves før ny full benchmark.
Uavhengig av dette er én manglende VAL-inputkontroll nå forhåndsregistrert:
VAL_SEQUENCE_AUDIT_001, gjennom eksisterende auditeier, capped audit4G/512M.
Den verifiserer bare time/seq/snap for alle70880 fysiske VAL-rader mot den
bundne M5-kilden. TRAIN-beviset gjenbrukes; ingen utfall, modellvurdering,
fits, forwards, optimizersteg eller TEST. Plan/operator er bundet i
NEXT_RUN_POLICY.json; engangskravet hindrer relansering. CPU-grensen står.

Målt på ekte, identiske TRAIN16-batcher for alle tre kandidatene:
uinstrumentert tid er 6,53–7,17s, mot 12,25–13,01s før rettelsen (1,81–1,88x).
Instrumentert tid er 13,89–15,76s. Alle originale og instrumenterte batchhasher
er eksakt like. Alle 254 felt, åtte familier og tidsrammer beholdes. Null fits,
modellforwards, optimizersteg og TEST. Dette er CPU-/batchparitet, ikke læring.

Minste rettelse i eksisterende adapter/økonomileverandør gjenbruker ett
eksplisitt referansevindu per view, lazy etter original state-view sin cutoff-
og klokkekontroll. Parent og hvert stegutsnitt valideres. State-view-kilden er
byteidentisk til inputautoriteten; ingen binding/gate omgås eller data refittes.
51 fokuserte syntetiske tester og obligatoriske commitkontroller består.

Den uendrede samplereieren tillater 1800 målte CPU-sekunder per full kandidat.
Lineær fremskrivning fra bare én TRAIN16-batch per kandidat gir omtrent 2,0/
4,4/9,0 timer instrumentert, samlet15,4 timer uten oppstart. Dette er kun
planleggingsanslag: ingen konfidensgrense, fullkapasitetsmåling eller
eligibilitetskonklusjon. CAPACITY_PLANNING_REVIEW skiller dette eksplisitt.

Forslag til operatørvalg: behold alle kvalitets-/maskinvaregrenser, men bind
én full benchmark med CPU-eligibilitetsgrense3 timer per samplerepoch og total
hard kjøretid høyst18 timer. Dette er IKKE godkjent eller implementert.
Alternativet er å beholde30 minutter og avklare videre CPU/designarbeid.
Ingen automatisk flytting av grenser, ny sampler eller relansering.

Resultat/terminal og planleggingsreview er hash-bundet i NEXT_RUN_POLICY.json
og CURRENT_RESTART_POINT.json. Feil og partiale kjøringer består:
- Benchmark første oppstart: feil filrolle, stoppet før kandidatmåling.
- ATTEMPT_002: første kandidat overskred1800s allerede ved1104/8192 Entries;
  stoppet bevart uten samplervalg. Andre kandidater uundersøkt på den kilden.
- CPU_WORKLOAD_PROFILE_001: målinger fullført, sluttpublisering feilet på
  JSON-nøkkeltyper; original rød terminal og verifiserte stagingbytes beholdes.
- CPU_WORKLOAD_PROFILE_002: inputbundet state-view-kildehash avviste første
  rettelse før måling; bevart. Rettelsen flyttet til adapter/økonomileverandør.

Ingen tung/native GX1-jobb kjører ved sluttkontrollen. Ingen sampler er valgt,
ingen koordinater eller fersk native initialisering/initialmåling er produsert,
og ingen256-stegs trening er kjørt. Konstruktørmetadata er publisert; dette
er ikke en initialmåling. Fullførte inputs, normalisering og indeks gjenbrukes.

Authority er /home/andre2/src/GX1_CURRENT på work/gx1-current.
Les CURRENT_RESTART_POINT.json og docs/RESTART_POINT_20261005.md.
Global training_enabled=false; TEST, broker, live/paper, ordre og spending
er stengt. Læring, generalisering, positiv økonomi og paritet for en faktisk
modell er fortsatt ubevist. Utvidelse av treningsbudsjett er betinget av den
senere Entry/Exit-læringsporten, ikke denne CPU-kontrollen.
