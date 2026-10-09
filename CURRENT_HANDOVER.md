# GX1 — siste overlevering, 09.10.2026

Den nye native nullmålingen for Exit-uavhengig Entry er fullført og revidert.
Ingen produksjonsoptimizersteg eller smoke er startet. NEXT_RUN_POLICY.json
åpner nå bare det eksakte, avgrensede Entry-only smoke-forsøket nedenfor.
Stor Entry-trening krever bestått lærings- og driftsport. Senere Exit-trening
som bevarer godkjent Entry er fortsatt uimplementert og ikke admittert.
App-målet i thread 01a1203d-4a95-7d03-bbd9-84017db6b29c er aktivt.

Kun /home/andre2/src/GX1_CURRENT, branch work/gx1-current, én agent og
én tung CURRENT-jobb. Mac er overleveringskopi. Les GX1_RULES.md og kjør
scripts/gx1_handover.sh --check. NEXT_RUN_POLICY.json eier eksakte bindinger.
Fullførte planer og receipts er bevis, aldri relaunch-autoritet.

## Verifisert gjeldende baseline

NATIVE_ENTRY_OBSERVED_INITIAL_20261009_001 fullførte på kilde30708263,
med native terminal 19:08:51 UTC /21:08:51 Oslo, guardPASS og trainer/observer0.
Windows-tasken avsluttet også med0. Dens XML/terminal er bevart, og den brukte
tasken er deaktivert. Hele Windows/native-avslutningen med den korrigerte
post-record-fristen er dermed faktisk verifisert.

INITIAL_MEASUREMENT_AUDIT.json og COMPLETION_REVIEW.json i samme run-root
binder eksakt resultat, recipe, receipt, operatør og original starttilstand.
TRAIN256 og separat CONTROL256 er målt på de opprinnelig frosne koordinatene:
256 Entry-ankre og256 Exit-ankre samt1024 samplede Exit-tilstander per gruppe.
Begge gruppers Entry-target er uavhengig rekonstruert fra originale fysiske
utfallskolonner, klokker og deklarerte kostnader; alle512 rader er eksakt like.
Entry bruker ingen Exit-modell i fasiten. Exit-målingene er egne diagnoser.

Fersk CPU-starttilstand fra NATIVE_COMPONENTS_20261009_003 ble gjenbrukt etter
de kanoniske konstruktør-/source-/state-/normkontrollene.9637663 parametere;
ONLINE/TARGET/EMA startet med hash52cd7442. Native checkpoint og audit
bevarer modell, TARGET, optimizer, EMA, scheduler og CPU/Python/NumPy RNG
bit-likt; CUDA RNG er separat seedet og lagret. Optimizersteg0.
Dette beviser ekte nullmåling og lagring, ikke læring eller lært resume.

Målt native komponentoppbygging1025,994s, initialmålingens samlede native
elapsed1451,936s og checkpointlagring0,675139s. Samme-run guard målte
maks49C core,56C memory-junction,138,04W og2764MiB VRAM.
Første faktiske læringsforsøk må fortsatt måle fremdrift, last/lagring og
resume. Ingen full epoch/full VAL eller generaliserings-/profittpåstand.

## Uavhengig Entry og neste smoke

Operatøren vedtok09.10 at Entry finner retningsmuligheter fra markedet,
uavhengig av et Exit-estimat, før Exit lærer posisjonshåndtering.
configs/research/NATIVE_ENTRY_OBSERVED_MARKET_20261009.json binder målet.
Entry bruker observerte eksekverbare BID/ASK-markouts etter deklarerte
kostnader, LONG/SHORT og FLAT0. Den opprinnelige TRAIN-eide horisonten
er19 M5-barer/95min; dette er ingen maksimal holdetid. Negative utfall
bevares, og framtidsutfall inngår aldri som beslutningsinputs.

Entry-only-treningen kaller ikke TARGET-forward eller Exit-tap. Eksisterende
47 genuine hjelpeutfall, alle features og delte encodere består. Exit-eide
parametere må ha grad=None før optimizersteg, også under AdamW. Delte
encodere lærer fra Entry og kan dermed endre Exit-output; Exit-funksjonen
er ikke frosset i denne fasen. Senere Exit-trening må bevare godkjent Entry.
Samme targetidentitet bindes i recipe, datasett, checkpoint og målinger.
Gamle teacher-baserte baselines eller resume med endret target avvises.

NATIVE_ENTRY_OBSERVED_SMOKE_20261009_001 har eksakt POLICY_SCOPE.json,
PREPARE.py og forhåndsbundet REVIEW_PREREGISTRATION.json /
PAIRED_REVIEW_OPERATOR.py. Recipe og campaign må fortsatt materialiseres
og verifiseres, deretter installeres mot en ny, trygg fysisk Windows-boot.
Én invocation, høyst256 optimizersteg/4096 TRAIN-rader; ingen refit,
teacher-refresh, ny kohort, full epoch/full VAL eller automatisk utvidelse.

Review krever LONG, SHORT og LONG-minus-SHORT mot fersk initialmodell og
TRAIN-konstanter, separat på TRAIN256 og CONTROL256. Rapporter MSE,
sentrert feil, korrelasjon, handlinger, regret, alle deklarerte måneder og
observerte uker. CONTROL krever samme parvise kalenderuke-bootstrap
(5000 trekk, seed20260911) som tidligere bundet review. Manglende
usikkerhet, bare biasflytting, tvetydige handlinger eller konstant
handlingsvalg gir ikke PASS. Sju syntetiske kontroller av reviewlogikken
bestod; de beviser bare måleinstrumentet.

Exit-diagnoser skal rapporteres, men kreves ikke forbedret uten Exit-steg.
Samlet Entry/Exit-port forblir stengt. Numerisk Entry-PASS alene åpner heller
ingen utvidelse: periodestabilitet og faktisk drifts-/resume-evidens må
vurderes. Observert netto markout er ikke realisert porteføljeprofitt.
Konfidens må måles; raw bps er ingen kalibrert sannsynlighet.
CONTROL/Juni2026 er gjenbrukt utvikling. TEST forblir forseglet.

## Omstart og oppfølging

Siste verifiserte fysiske Windows-boot er483,18:38:18.500 UTC /
20:38:18.500 Oslo. WSL-boot41f6a48a. Alle166 kode-/recipe-/
starttilstandsbindinger bestod før/etter omstart; kilde forble uendret.
Native-vindu12000s, ytre guard13800s, faktisk Task Scheduler-grense14400s.
Tidsbudsjett sjekkes etter hvert optimizersteg; checkpoint normalt hvert64.
steg samt ved tids-/sluttpause. Ingen aktiv jobb avbrytes for omstart.

Før hver neste fysiske reboot kreves maskinfelles idle-/writer-bevis,
ledige prosjektjobber/låser/GPU, varige tilstander og bevarte terminaler.
Andre prosjekter kan eksistere; de må også være ledige. Etterpå bekreftes
ny fysisk boot, WSL, kode- og artefaktintegritet før native start.

Verifisert Windows-staging er C:\Users\Andre\GX1_CURRENT_NATIVE_30708263.
Gjenbruk ved byte-lik kode; ikke stol på katalognavnet. Ny task må bruke
dagens recipe/campaign, UseNativeClockProfile og brukerAndre.
GPU-effektvakt og signert telemetribro består. SSH gx1-3090-lan virker.
WSL-shutdown erstatter aldri fysisk reboot.
Thread-heartbeat gx1-f-lg-native-trening følger samme mål hvert30.minutt.
Ingen minuttvis modellpolling; hardwarevaktene eier hyppig telemetri.
Avslutt oppfølgingen når tilhørende mål og kjøring faktisk er ferdige.

## Helrepo-revisjon og bevart historikk

De opprinnelige638 sporede filene ble inventert. Python-AST, JSON,
shell/PowerShell, lokale importer og Markdown-lenker bestod. Hele testutvalget
ga6290 PASS,3 SKIP og6 kartlagte feil. Alle seks ble rettet og kontrollert
fokusert. REPO_AUDIT_20261009_001 binder originalt bevis og senere rettelser.
Entry-endringen ga deduplisert244 PASS,3 SKIP og ingen gjenstående feil
i den fokuserte bølgen. Ekte datakobling for652552 TRAIN og256 CONTROL
bevarte alle47 hjelpekolonner. Syntetisk AdamW-/gradient-/resume-bevis er
separat fra faktisk native læring, som ennå ikke er utført.

Bevarte minimale rettelser: skrivevaktens os-import; eksakt fysisk
foreldremanifest; kanonisk JSON-rekkefølge for åtte familier; checkpointpause
mellom steg;4t Windows-task og riktig klokke-launcher; fersk fysisk campaign
uten gammel full-VAL-modellautoritet. Dobbelt scope i sampler-adgangen ble
fjernet etter måling. Post-record inspect fikk180s etter målt99,547s mot
tidligere90s; oppstart/record90s og vindusgrenser består. Fokuserte tester,
Windows-harness og nå en komplett ekte native-syklus bestod.

NATIVE_INITIAL_20261009_003 med gammel Exit-basert Entry-fasit er historikk.
NATIVE_SMOKE_20261009_001 ble forberedt, men aldri kjørt og er erstattet.
Ingen av planene relanseres. Gamle feil/terminaler, originale checkpoints og
metadataforsøk bevares. CURRENT_HANDOVER er eneste gjeldende fortelling.

## Datasett og fortsatt avgrensning

HISTORY2009W_NATIVE_V38_20261007:5523147 M1-rader,652552/70880 TRAIN/VAL,
254 features, åtte familier, alle tidsrammer og47 kontrollerte targetfelt.
Whole-TRAIN-normalisering,71 eksakte aliaspar og fysiske koordinater består.
Valgt CPU-sampler målte8192 Entry-par på7275,518s; dette er ingen NN-epoch.
Ingen full datarekonstruksjon eller ny feature-/modell-/terskelsøk nå.

Originaldata, caches, checkpoints og ferdige/forkastede analyser bevares.
GC/order-flow og nye footprint/order-block-utvidelser er på pause; de fire
ufullførte GC-trinnene og separat full makro-B består. Ingen broker,
live/paper, handel, spending eller diskopprydding er åpnet.
