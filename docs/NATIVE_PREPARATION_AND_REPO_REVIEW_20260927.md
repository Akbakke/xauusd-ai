# Native-forberedelse og full repo-gjennomgang — 27.09.2026

Brukeren har etter NO-GO-målingen presisert at det konsoliderte native-oppsettet
ikke er epoch-trent, og bedt om å ferdigstille gjenstående arbeid, deretter rydde,
feilsøke og kartlegge hele repoet før eventuell trening. Dette åpner data- og
kodeforberedelse. `training_enabled=false` består; ingen optimizersteg, full epoch,
VAL-evaluering eller automatisk treningsstart er autorisert.

Ridge/HGB-resultatet og dets forhåndsregistrerte NO-GO bevares. Den avgrensede
snapshot-/ukemålingen er ikke et kapasitetsbevis for native sekvenslæring og
samspillet mellom Entry og Exit. Kravet om ny informasjon/kostendring fra den
forrige arbeidsregelen er ikke et absolutt forbud mot brukerens nye forberedelse.

## Første mål: komplette og konsistente native inputs

Bruk full-mode-kjeden som allerede eier M1/M5-par, signalmanifest, separate
Entry/Exit-flater, sekvenser og forseglet TEST. TRAIN/VAL/TEST-vinduene beholdes. Målt gyldig oppvarmingshistorikk starter
2010-06-14 22:00 UTC (se kontrollen under); TRAIN juni 2011–mai 2025, utviklings-VAL juni 2025–juni
2026, mekanisk forseglet TEST juli–august 2026. Ingen TEST-resultater beregnes
eller brukes til valg. Native modellvekter oppdateres ikke.

Ferdig tidlig M5 C0-cache fra `HISTORY2009W_EARLY_DECISION_20260927` gir de
frosne parameterne til parbyggeren. Kalibreringen beholdes fra 2009-06-01 til
2013-01-01 22:00 UTC, indre registergrense 2012-04-11 22:00 UTC. Nytt par og
dataset-run-id krever nye eksakte lineage-bindinger; eksisterende produsenter
lager disse. Ingen manuell endring av artefaktmanifester eller gjenbruk av
avbrutt C0-output som om det var ferdig.

Konkret kodeblokkering: kjeden bandt squeeze-perioden ubetinget til modellens
TRAIN-periode, og registerfit hadde ingen separat start. Eksplisitte fit-vinduer
føres nå gjennom den samme kjeden til hver eksisterende eier. Utelatte grenser
bruker samme TRAIN-verdi som før; oppgitte grenser kontrolleres for UTC, rekkefølge
og slutt senest ved TRAIN-slutt. Artefaktenes faktiske vinduer og parbinding må
stemme eksakt. Ni fokuserte kjedetester består, med eksekverbar kontroll av tidlig,
framtidig og ugyldig kalibrering; dette er mekanikk, ikke læring.

Scope og kvitteringer:
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_PREPARATION_20260927/PLAN.json`.
Nye byggartefakter:
`/home/andre2/GX1_DATA/data/data/prebuilt/HISTORY2009W_NATIVE_EARLY_20260927`.
Én CPU-produsent om gangen gjennom eksisterende capped-run-vakter. Ferdige
kalibrerings- og historikkbevis bevares. Post-rebuild- og lifecycle-bindinger er
ikke erklært klare før de faktisk er kontrollert.

## Deretter: hele repoet før en treningsbeslutning

Kartlegg alle sporede kode-, konfigurasjons-, test- og dokumentfiler, inngangene,
kontrakteierne og deres avhengigheter. Ta med oversikt over ignorerte runtime-stier
uten å eksponere hemmeligheter. Gjenbruk tidligere revisjoner som utgangspunkt,
men kontroller nåværende kode. Skill aktiv kode, historiske bevis og dokumentert
frakoblede filer. Fjern bare dokumentert overflødig innhold; dataopprydding må
fortsatt gå gjennom retention-eieren og ha logg.

Kontroller særlig de åpne trenerfunnene (M3), input-/mål-/normaliseringsbindinger,
kausalitet, Entry/Exit-læring, beslutningsautoritet, fail-closed-vakter, faktiske
oppstartsruter, gamle feilede tester og stale status-/artefaktreferanser. Én samlet
teststatus kan måles som del av denne bestilte revisjonen; gjenta deretter bare
nødvendige kontroller. Lever funn med evidensklasse, rettelse, test og gjenværende
usikkerhet. Teknisk ferdigstilling betyr ikke dokumentert edge eller treningstillatelse.

## Målt M5-blokkering og kontrollert videreføring

Første kjøring fra `f08497c8` avsluttet rødt 20:31 UTC i `m5-model-source`.
Par, tidlig squeeze-fit og komplett M5-flate var ferdige. Leseren sammenlignet
parprodusentens komplette native-deskriptor mot en eldre håndskrevet delmengde;
`explicit_vedtak_id` var første av de ekstra feltene. Vedtaks-ID, native-manifest-
hash og kilde stemmer. Leseren bruker nå parprodusentens eksakte feltprojeksjon;
endrede eller ekstra bindinger avvises fortsatt.

Kjeden har en eksplisitt, hash-bundet gjenbruksinngang for bare den komplette M5-
flaten i samme event/run. Den unntar M5-parquet, manifest, cache og checkpoint fra
freshness-kravet. Alle senere outputs må fortsatt være nye. Registerperioder,
par/run/kildelinje, parquet-hash og skjema og full cache-kontroll utføres før ny
modellkilde publiseres. Ingen automatisk oppdagelse eller delvis output godtas.
Original rødt terminalbevis og logger bevares; videreføringen får egen kjøringsmappe.

22 fokuserte tester består, inkludert ekte guard-eksekvering og endret/manglende
lineage. Separat kontroll på faktiske bytes består: 1 153 078 M5-rader, komplett
MTF-cache og uendrede inputbindinger (`M5_REUSE_VERIFICATION.json`). Dette er
integritetsbevis, ikke læring. Videreføring:
`HISTORY2009W_NATIVE_PREPARATION_20260927/CONTINUE_M5_BINDING_20260927`.

## Brukerens spørsmål om færre features

Kompleksitet inngår nå uttrykkelig i repo-revisjonen. Aktiv kontrakteier returnerer
241 signal-felt per M5-bar. De 71 kontinuerlige kontekstfeltene er allerede
representert i denne flaten (4 obligatoriske + 67 kandidater); ikke legg 71 til
som ny informasjon. Tidsrammesekvenser kommer i tillegg. Antall inputverdier
er ikke antall uavhengige signaler. Alle kandidatfelt tas med av kontrakten;
rangeringen velger ingen bort.
Den tidligere formuleringen «altfor avansert» var sterkere enn evidensen.
Heller ikke den eldre heuristikken «maks omtrent 15 felt» beviser en optimal grense.

Undersøk eksakte/avledede duplikater, nesten konstante felt på faktiske
beslutningsrader, overlapp mellom tidsrammer, endring mellom historiske perioder,
parameterfordeling og aktive hjelpeoppgaver. Skill informasjonens innhold fra
modellens kapasitet. Avhengighetsbevist død kode kan ryddes; prediktiv nytte og
tap ved featurefjerning krever en avgrenset, forhåndsbundet sammenligning med
samme senere perioder, kostmodell og baselines. Ingen featurefamilie slettes
på mistanke, og ingen slik modelltrening startes automatisk. Byggingen bevarer
komplette inputs slik at en senere begrunnet slanking kan gjenbruke dem.

## Målt historikkgeometri, uten endring av TRAIN/VAL/TEST

Videreføringen fra `b01a2d6a` publiserte gyldig M5-modellkilde, men stoppet i
`model-source-identity` 20:46 UTC: første rad er 2010-06-13 22:00, etter planens
historikkstart 2010-06-01. Fullt definerte kontekstfelt begynner 2010-06-14 22:00.
Denne siste grensen brukes nå eksplisitt som oppvarmingsstart. Den eksisterende
D1-eieren måler nøyaktig 252 lukkede D1-barer fra denne historikken før uendret
TRAIN-start 2011-06-01; 252 er arkitektureierens krav. Både den faktiske
kjedens marked-/tidsidentitetsvakt og preflightens komplette MTF-kontroll består
med korrigert historikkstart. Ingen utfall brukes til datovalget, og ingen
kontroll svekkes (`HISTORY_GEOMETRY.json`, `CORRECTED_HISTORY_MTF_PREFLIGHT.json`).

En ekstra eksplisitt SHA256 binder det ferdige modellkildemanifestet. Bare da
kan kilden gjenbrukes: manifest-/payload-hash, output-hash/størrelse, run/par,
enriched-/cache-bindinger, feltliste og gjeldende arkitektur-/featurekontrakter
må stemme. Den etterfølgende markedskontrollen og fulle source-cascade kjøres
fortsatt; rangering og alle senere outputs må være ferske. Ti kjedetester
består, med endret run, par, input, cache og output avvist i ekte guard-kode.
Ny runtime-mappe: `CONTINUE_HISTORY_GEOMETRY_20260927` under samme forberedelse.

Lagringskontrollen i videreføringen avdekket at arkitektureieren sammenlignet
rekkefølge på JSON-objektnøkler. Produsenten publiserer med `sort_keys=True`,
så gyldig persistert arkitektur ble avvist. Eierens kontroll sammenligner nå
nøkkelmengden; eksakte typer, verdier og rekkefølge i modellens lister bevares.
32 fokuserte tester består, inkludert produsentens faktiske JSON-sortering og
avvisning av snudde tidsrammer. Kjedenes faktiske freshness-/gjenbruksvakt er
også eksekvert på de aktuelle ferdige filene og består (`actual_continuation_guard.log`).
Ingen data ble endret. Neste eksplisitte videreføring ligger i
`CONTINUE_SERIALIZED_ARCHITECTURE_20260927`; forrige røde terminal bevares.

## Historiske låste quotes: én kildekontrakt gjennom Entry/Exit

Kjeden fra `769e9080` bestod marked-/historie- og source-cascade-kontroll, men
rangeringen stoppet før checkpoint i M1-geometrileseren. Måling av 5 648 218
M1-rader viste ingen krysset BID/ASK eller brutt OHLC-geometri. Seks open/high/low-
forekomster har lik BID/ASK 2012-12-12; fire M1-close og én M5-close er også låst
på denne datoen. Alle er i TRAIN, ingen i VAL/TEST. Det er datakvalitetsmetadata,
ikke retnings-/avkastningsmåling (`M1_QUOTE_GEOMETRY.json`, `LOCKED_CLOSES.json`).

Kanonical native-eier tillater positive priser med ASK >= BID. Flere nedstrøms
mål-, token-, Exit- og tilstandslesere krevde ASK > BID og ville derfor avvise
den samme tillatte kilden. Disse sammenligningene følger nå native-kontrakten:
lik pris bevares eksakt; kryssede, ikke-finite eller ikke-positive priser avvises.
Per-rad spread kan være null; målenes tilpassede median-hurdle må fortsatt være
strengt positiv. Persistens krever ikke-negativ, prisavstemt entry-spread.
Ingen rådata, slippage, kommisjon, finansiering eller prisformel er endret.

189 fokuserte tester består, inkludert låst quote, uendret pris gjennom M1-
utfall, token-rundtur og fortsatt avvisning av krysset quote. Faktisk M1-leser
og fill-surface-validering består på alle 5 648 218 M1- og 1 153 078 M5-rader;
én M5-beslutning binder en låst M1-fill (`LOCKED_QUOTE_ADMISSION.json`).
Det beviser datakontraktkonsistens, ikke virkelig meglerfill eller edge.
Videreføring: `CONTINUE_NATIVE_QUOTE_GEOMETRY_20260927`, samme ferdige inputs
og tidsvinduer. Historiske terminaler, output og source-cascade-bevis bevares.
