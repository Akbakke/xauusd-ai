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

## M1-minne og gjenbruk av ferdig signalmanifest — 28.09

Rangering fra `7f100ba2` ble ferdig: 67 kandidater, 994 500 TRAIN-rader, ingen
helt konstante eller eksakt like felt innad i kandidatgruppen. Dette utelukker
ikke informasjonsmessig overlapp med basefelter, obligatoriske felt eller MTF.
Signalmanifestet er også ferdig. M1 stoppet 27.09 22:36 UTC med
`before_group_a_attach rss_gib=9.50 ceiling_gib=9.00`. Hele native M1-roten har
6 019 349 rader; parbundet BASE28 brukt i geometrikontrollen har 5 648 218.

Warmup-validering bygget en full float64-matrise og pandas-mellomkopier av ti
felt. Samme finitthets-/prefikskontroll gjøres nå kolonnevis. Group-A-sluttkontroll
bruker også kolonnevis finitthet; serial parity bruker samme fulle kontekst,
men allokerer bare den ene etterspurte outputraden. Uendrede featureverdier,
kausalitetsregler og minnegrenser. 35 fokuserte tester består. Syntetisk
6 019 349 × 10-kontroll ga identisk trim og topp-RSS 1 501,8 → 595,0 MiB;
dette er ikke en måling av hele produksjonsløpet.

`--reuse-signal-manifest` med eksakt SHA256 krever ferdig M5-gjenbruk. Den
allerede eksisterende lineage-eieren revaliderer manifest/rangering/kilde/cache,
par, gjeldende kontrakter og eksakte tidsvinduer. Fit-slutt kommer fra samme
`causal_m1_policy_fit_train_end` som ranker og preflight. Rangering, signal og
source-cascade gjenkjøres ikke. Delvis M1 og tomt gammelt checkpoint bevares;
ny M1 bruker ferskt navn. M1-registry-fit ble ikke publisert før stopp og kan
ikke gjenbrukes. Videreføringen ligger i `CONTINUE_M1_MEMORY_20260928` under
forberedelsesroten. Trening er fortsatt deaktivert; full repo-revisjon gjenstår.

Den faktiske gjenbruksvakten består på de ferdige filene
(`actual_signal_reuse_guard.log`); feil SHA og run-identitet avvises av samme
kjørte guard (`signal_reuse_negative_guard.log`). 19 eksisterende kjede-/signal-
tester samt ny test av fit-grense og avvisning før gjenbruk består. Samlet 55
målrettede tester for denne rettelsen. Ingen native optimizer eller epoch er kjørt.

## Bekreftet Arrow-bufferårsak — minimal videre rettelse

Fullkjøringen fra `73672892` stoppet på samme sted ved 9,48 GiB; det første
valideringsminnetiltaket løste ikke hele blokkeringen. Ingen M1-output eller
Group-A-chunk ble publisert. Rangering og signal ble gjenbrukt korrekt.

Avgrensede native kontekstkontroller brukte samme kilde og funksjoner, men
utelot M1-registry-fit og selve lange Group-A-radløkka. Først ble mulig tidlig
frigjøring av 45 mellomkolonner undersøkt; den er **ikke** innført. Måling av
Arrow-poolen viste den større årsaken: omtrent 2,5 GiB frigjorte bygge-/lesebuffere
ble fortsatt holdt fysisk i prosessen. `default_memory_pool().release_unused()`
returnerer bare ubrukte buffere til operativsystemet.

Endelig kontroll brukte den nye produksjonshjelperen og beholdt alle 131 kolonner:
RSS 8,88 → 6,40 GiB, med identisk hash av alle featureverdier før/etter.
Group-A-kontekst og checkpoint-digest ble deretter bygget ved 6,58 GiB.
`M1_ARROW_RELEASE_VERIFICATION.json` og `m1_arrow_release_verification.log`
inneholder bevis; de er segmentkontroll, ikke full kjøring eller læring.
Ingen glibc/ctypes-operasjon brukes i produksjon, ingen kolonner fjernes, og
RSS-/cgroup-grensene er uendret. Gjenbruksvakten som revaliderer MTF-data er
også satt under eksisterende eksklusive audit-cap 4 GiB/512 MiB.

Ny eksplisitt videreføring: `CONTINUE_M1_ARROW_RELEASE_20260928` under samme
forberedelsesrot. Forrige røde terminal og alle tidligere bevis bevares.

27 målrettede producer-/kjedetester består, inkludert bevaring av aktiv Arrow-
buffer, byteidentiske frameverdier og riktig rekkefølge før Group-A.

## Historisk stopp 28.09 før SMC-rettelsen: minnet rettet

Fullkjøringen fra `10c78d70` fullførte alle 1 382 Group-A-chunks. Arrow-frigjøringen
senket faktisk RSS 9,52 → 6,85 GiB; etter Group-A lå RSS på 5,75 GiB.
Terminalen 02:59:28 UTC er rød med `M1_ENRICHED_OUTPUT_NONFINITE:
smc_pivot_envelope_position`. Det finnes ikke ferdig M1-parquet eller manifest.

Diagnosen finner sju TRAIN-rader etter oppvarmingsgrensen der alle fire bekreftede
pivotpriser er like: to rader 2012-04-06 og fem rader 2019-03-07. Feature-eieren
returnerer tilsiktet NaN ved null bredde; modellinput krever endelige verdier.
Dette er en kontraktkonflikt, ikke begrunnelse for å fylle null, slette radene
eller fjerne en feature uten å definere den nye representasjonen.

Ingen ny rebuild før kontrakten er avklart. Bevar ferdige Group-A-chunks og alle
kvitteringer. Eventuelt gjenbruk må passere kilde-/input-/kontrakthasher; fullført
checkpoint alene gir ikke kompatibilitet etter en featureendring. Repo-revisjonen
fortsetter. Native trening og TEST-resultater forblir stengt.
Se `M1_SMC_ENVELOPE_DIAGNOSIS.json` under native-forberedelsens kjøringsmappe.

## M3: rettet diagnostikk, uendret modellflate

På den aktive trenerstien er følgende verifiserte feil rettet:

- Eksakt `xlogy`-entropi gir null for deterministiske rader; nullsannsynligheter
  får ikke kunstig masse. Den separate kontrollen av brukte ruter består.
- Manglende lagrede masker er manglende bevis, ikke implisitt full supervisjon.
- Liveness skiller eksakt konstant fra liten variasjon uten terskelen 1e-8.
  FLAT=0 er fortsatt et strukturelt konstant target etter gjeldende kontrakt.
- Sesjonsrapporten bruker den eksisterende ASIA/EU/OVERLAP/US-eieren.
- MAE-diagnostikk bruker samme ikke-negativitetskontroll som tapet. Også Inf
  avvises. Eventtap og diagnostikk deler samme binære target-/maskeoverflate;
  ugyldige observerte labels klippes ikke, og maskerte udefinerte cells kommer
  ikke inn i BCE eller gradienten. Gyldige labels og handelsvalg er uendret.

140 fokuserte tester besto. Samlet suite og rettet testgjeld er dokumentert i
[repo-rapporten](REPO_REVIEW_20260928.md): 5 628 bestått / 18 feil / 3 skips;
alle feilede tilfeller består etter triage (133 tester). Metadata-kontroll: 150
bestått. Ingen ny fullsuite. Dette er kilde-/integritetsarbeid, ikke læring.
EMA-warmup og arkitektur-/featureendringer er ikke tatt inn som blind portering.

## Videreføring etter brukerens rettings-/oppryddingsvedtak 28.09

SMC-posisjon er nå implementert som par med observert bredde. Se den etterfølgende
rettelsen i [repo-gjennomgangen](REPO_REVIEW_20260928.md). Kilden går til v37 / 242
signalfelt; de tidligere 241-feltsartefaktene ovenfor er v36-bevis. Gamle outputs
bevares, og gjenbruk må valideres av eksisterende eiere uten manifestomskriving.
Etter første låsblokkering består 281 feature-/integrasjonstester og faktisk
M1-kontroll på 6 019 349 rader, inkludert de sju nullbredderadene. Prosjektvis
låsing er innført etter brukerens vedtak og kontrollert med 294 tester samt
faktisk parallell låsadgang. V37-inputbygging er fortsatt ikke gjennomført.
Dette endrer ikke fit-/TRAIN-/VAL-/TEST-grenser eller autoriserer trening.

## Ny v37-inputbygging — vedtak 28.09 «Fortsett å bygge»

Ny run: `/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V37_20260928`;
ny outputrot: `/home/andre2/GX1_DATA/data/data/prebuilt/HISTORY2009W_NATIVE_V37_20260928`.
Gamle v36-outputs og 1 382 Group-A-chunks bevares. De brukes ikke som ferdige
v37-featurematriser. Den eksisterende par → squeeze(par) → seq513-kjeden
brukes med uendrede fit- og splitgrenser og korrigert historikkstart.

`CALIBRATION_REUSE_VERIFICATION.json` dokumenterer faktisk PASS fra begge
registry-/squeeze-eiere under audit-cap 4 GiB/512 MiB. Bare tidlig frosne
konstanter/parametre gjenbrukes; nye felt beregnes av dagens eiere.
`BINDING.json` opprettes etter ren dokumentasjonscommit. Kilden er frosset
under kjøringen; START/prosess/TERMINAL avgjør faktisk status. Ressursgrenser
og vakter beholdes. Etterpå gjenstår post-build/readiness/lifecycle og resten
av kompleksitetsvurderingen. Ingen trening eller TEST-utfall er autorisert.

## Preflight-publiseringskvittering og gjenbruk — 28.09 kveld

V37 fullførte M1/M5 og 29/29 preflight-kontroller. Kjeden stoppet kl. 13:27 UTC
fordi katalogkontrollen telte den påkrevde `.json.order`-kvitteringen som en
uventet ekstra fil. Kontrollkoden bruker nå den eksisterende publiseringseieren
og avviser endret kvittering/hendelse, flere hendelser og fremmede filer.

14 kjedetester og den virkelige preflight-identiteten består. Videreføringen
`CONTINUE_PREFLIGHT_ORDER_20260928` binder preflight og nøyaktig denne røde
forelderen. Bare kjededriveren kan ha endret kjørende kode; gx1 og øvrige scripts
må være identiske. Gjenbruk rehasher featureflater, manifester og enriched-kilder,
bevarer tidligere M5/signal/cache-vakter og kontrollerer eksakte kommando-/datogrenser.
Datasett/lifecycle/audit-output må fortsatt være ferskt. Ingen metadata omskrives.

Ferdige inputs gjenbrukes; gammelt rødt løp og stoppet POST_BUILD bevares.
Automatisk oppfølging hver 30. minutt er nå faktisk opprettet i samme oppgave.
Tidligere var bare sluttrinn etter vellykket bygging automatisk, uten feiloppfølging.
Trening og TEST-utfall er fortsatt stengt.

## Ressursventing etter kontrollert videreføring

Første videreføring besto fysisk gjenbruk, men stoppet 19:48 UTC på hostens
uendrede minnekrav før model-source-identity. Ingen nye datasettoutputs ble
bygget. `CONTINUE_RESOURCE_WAIT_20260928` venter i runtime på den eksisterende
capped-run-grensen hvert 15. minutt og starter samme kommando én gang.
Egen duplikatlås og eksakt kilde-/filbinding består; post-build får ny forelder.
Dette er kun drift av autorisert inputbygging, ingen modell-/produsentendring.

## Datasettskriving: målt OOM og rettet summary-lagring 28.09

Forrige videreføring ble OOM-drept i sin cgroup 20:51 UTC etter siste flush på
496 640 TRAIN-rader. En voksende liste av dictionaries beholdt alle 97
summary-felter per rad. Kolonnebaserte batcher erstatter denne representasjonen;
selve parquet-emisjonen og targets er uendret. 22 fokuserte tester består.
På 100 000 syntetiske rader med faktisk feltsett er beholdt minne 712 800 984
mot 77 625 088 bytes, med eksakt verdi-/dtype-/rekkefølgeparitet. Dette er ikke
full-run-toppminne eller kvalitetsbevis. Runtime-bevis: `DATASET_SUMMARY_OOM_REPAIR.json`
og `SUMMARY_MEMORY_MEASUREMENT.json` i samme v37-runrot.

`CONTINUE_SUMMARY_MEMORY_20260928` bruker ferdige, revaliderte upstream-inputs og
ny `DATASET_SUMMARY_RECOVERY_20260928`-outputrot. Kjedens eksisterende eiere
utfører ny preflight, rebuild og senere readiness. Oppstrøms kode må være
uendret; kun downstream-datasettfunksjon og summary-helper kan avvike, mens
delte helpers/imports forblir AST-identiske. 10 GiB/512 MiB cap og alle guards
består. Delvis gammel output slettes/overskrives ikke. Trening er stengt.

## 29.09: konkret lifecycle-stopp etter fullført TRAIN-skriving

Minne-reparasjonen passerte TRAIN-skrivingen: 652 552 rader ble lukket uten
ny OOM. Deretter avviste lifecycle tre M1-open-rader fra desember 2012 med
BID lik ASK. Før TEST-grensen hadde kilden ingen kryssede OHLC-priser;
låste priser finnes i open/high/low/close med henholdsvis 3/1/2/4 forekomster.
Kanonisk tape, Entry-utfall og optimal stopping tillater allerede slike priser.
Bygger, lifecycle-leser og offline replay er rettet til samme regel, uten
endring av inputpriser eller mål. Kryssede priser avvises fortsatt.

44 fokuserte tester består. Et faktisk TRAIN-vindu med 2 732 rader og alle de
observerte låste prisene besto kanonisk validering, bygging av fire episoder
og lifecycle-innlasting med eksakt prisbevaring. Dette er teknisk evidens;
TEST og handelsverdi er ikke undersøkt. Rapportene ligger under
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V37_20260928/`:
`M1_SPREAD_FAILURE_AUDIT.json`, `LIFECYCLE_LOCKED_QUOTE_VERIFICATION.json`
og `LIFECYCLE_LOCKED_QUOTE_REPAIR.json`.

Ny engangsvidereføring er `CONTINUE_LOCKED_QUOTES_20260929`; ferske outputs
bygges i `DATASET_LOCKED_QUOTES_RECOVERY_20260929`. Nyeste røde forelder og
dens preflight bindes eksakt; ferdige inputs forblir i opprinnelig CHAIN.
Samme kjedeeier tillater nå en tidligere recovery som forelder og kontrollerer
at kun navngitte downstream-funksjoner avviker, aldri shared helpers/imports
eller upstream-produsenter. Alle fysiske inputhasher kontrolleres og ny
preflight kjøres. Ingen gamle delvise outputs gjenbrukes som ferdige datasett.
Post-build venter på grønn kjede; ingen trening er åpnet.
