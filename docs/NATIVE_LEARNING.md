# Gjenværende avgrenset v38-læring

Brukeren autoriserte punktene 1–6 05.10.2026, men prioriterte 06.10 deretter
kontrollert benchmarkstopp og repo-opprydding. Den 07.10 ble GC/order-flow og
nye order-block-/footprint-utvidelser satt på pause og ny trening med dagens
tekniske indikatoroppsett prioritert. Dette dokumentet beskriver det gjenværende
læringsløpet; faktisk kjøring krever fortsatt en ny eksakt plan i NEXT_RUN_POLICY.
Senere samme dag bestilte brukeren ny repo-revisjon, ferskt datasett,
retention-opprydding og liten smoke før større trening. Det eksplisitte nye
byggevedtaket erstatter tidligere reuse-only for genererte inputs, ikke
forbudet mot relaunch av forbrukte planer.

## Ny kjøreplan — HISTORY2009W_NATIVE_V38_20261007

1. Revider hele tracked inventaret og rett konkrete mismatches. Kjør fokuserte
   regresjoner, ikke gjentatte fullsuiter. Bind ren/committet/pushet kilde før build.
2. Bygg fersk M5/M1 enrichment, MTF, ranking/signalmanifest og Entry/Exit-datasett
   gjennom `scripts/run_seq513_rebuild_chain_v1.sh`. Ny output-identitet;
   gjenbruk kun verifiserte rå-/kalibreringsforeldre. Etter fysisk krasj 07.10
   tillater ny recovery-bestilling kun ferdige samme-generation Oct7-features,
   full byte-/upstreamkontroll og fersk preflight/downstream i CHAIN_RECOVERY_001.
   INPUT_BUILD_001 er avbrutt uten terminal, ikke relaunchet. Ingen gammel
   dataset/normalisering/modell gjenbrukes. Samme historiske perioder,
   alle ekte v38-felt/åtte familier og uendrede maskinvarevakter.
   Planen binder én CPU-build med 64800s endelig ytre sikkerhetsbudsjett;
   recovery deler opprinnelig deadline23:44:34 UTC inkl. downtime, ikke et nytt
   18-timersbudsjett eller gjenåpning av gammel benchmarkgodkjenning.
   INPUT_BUILD_002 stopper ved ekte core-terminal for trygg omstart mellom
   ferdige steg, aldri under aktiv jobb. Dette er ikke komplett input-green.
   Core er nå genuint terminal og forbruker aldri en ny launch. Det nye
   før-smoke-målet gjenbruker disse bytene; forberedelsesfremdrift eies av
   native_v38_rebuild_20261007/pre_smoke_progress i samme policy.
   Nytt app-mål ble genuint aktivert 07.10 kl.20:45:57 UTC uten å fullføre
   det pausede GC-arbeidet. COMPLETE_M1_001 er separat CPU-/output-bundet;
   ekte fasekvitteringer eier kjøringen, og fellesbudsjettet fornyes ikke.
   M1-eieren er nå genuint exit0 07.10 kl.22:27:57 UTC med5523147 rader og
   uendret kilde; tillatelsen er konsumert. INPUT_VALIDATION_001 feilet genuint
   22:49:53 UTC i diagnosekodens kalender/emission-grense; original kilde og
   claim-kvitteringer bevares. Ny INPUT_VALIDATION_002 retter kun datogrense-
   predikatet og binder de samme tre serialiserte capped-kontrollene. Seks
   syntetiske grensetester er PASS, ingen genuine inputaksept. Normal GREEN-
   core post-readiness gjelder fortsatt. Core-liveness/pretrain/overlap gjenbrukes
   bare ved eksakt byte-/eieraksept. Binding er ikke faktisk start eller grønt
   input. CURRENT-slot/GPU0% er målt ved terminal, men beskyttede Windows-
   prosesser er uklassifiserte, derfor ingen maskinvid trygg omstart nå.
3. Før normalisering eller modellsmoke: bygg den komplette pre-TEST M1-flaten
   gjennom eksisterende feature-eier og komplett rå M1-alignmentskilde.
   Den gamle chainens pair-alignerte lifecycle-flate er en annen rolle og er
   ikke full M1-state-dekning. Gjør komplett M1 til obligatorisk byggefase,
   ikke en etterfølgende uregistrert retting. Verifiser hele klokkesuffikset
   etter målt warmup og alle fysiske TRAIN/VAL-visninger. Avklar også den
   målte eksklusjonen fra eksakt fill/komplett native-minutt-labelhorisont
   på hele den fryste TRAIN/VAL-kandidatpopulasjonen: årsak, år/sesjon og
   seleksjon. Den komplette quote-klokken ga samme gyldige rader i ekte
   tidsdiagnose; komplett featurebygg alene er ingen reparasjon. Ingen
   gjettede stengninger, imputering eller periodeendring. Selvstendig
   input-oracle og genuine post-rebuild-readiness kreves. Tilpass ny immutable
   normalisering én gang på hele fysisk TRAIN; ingen gammel normalisering.
4. Aksepter replacement med genuine completion-/sekvens-/økonomi-/readiness-
   kvitteringer. Opplist så eksakte utdaterte DATA/RUNS-leaves med størrelser.
   Bevar transitive råkilder/kalibrering/kostbevis, aktive stier og pausert GC.
   Slett bare gjennom retention plan → godkjenn → utfør og writer-sjekk.
   Dagens policy er hash-bundet fra eksisterende retention-launchrot.
   Ufullstendig closure, ukjent metadata eller forseglet TEST stopper sletting;
   en slik nektelse rapporteres og omgås aldri.
5. Genuine teknisk input-smoke og deretter den eksisterende avgrensede
   læringsprøven nedenfor. Ny sampler/koordinat-/recipe- og hostadgang må først
   bestås; ATTEMPT_003s delresultat gir ingen sampler. Syntetisk hardware-smoke
   demonstrerer kun mekanikk. Fersk Windows-boot/vakter svekkes ikke.
6. Større trening bare etter samlet læringsport og én forhåndsbundet finite
   utvidelse. Ingen automatisk full epoch/full VAL eller TEST-åpning.

Eksakt ny plan: configs/research/NATIVE_V38_REBUILD_AND_SMOKE_20261007.json.
Fremdrift og faktisk autorisasjon står kun i NEXT_RUN_POLICY/current_work og
native_v38_rebuild_20261007, ikke i daterte resultater eller gamle claims.

ATTEMPT_003 er terminal med exit 1 etter operatøravbrudd. Første kandidat
rapporterte 3200/8192 Entry-par. Ingen full benchmark eller sampler er publisert.
Engangsclaim og godkjenning er konsumert; må aldri relanseres.
Eksakte terminal-, plan-, input- og operatorbindinger står i NEXT_RUN_POLICY.json.
Ingen ny benchmark/trening er autorisert av oppryddingen. Den nye prioriteringen
er heller ikke en fornyelse av ATTEMPT_003s konsumerte engangsbudsjett.
Det nye målet avsluttes før smoke med genuine inputaksept/normalisering/
paritet og eksakt finite sampler-/smoke-binding, ikke på en plan alene.
Økonomieierens gross/research-only-grense må oppgis; dette er ikke en
nettoøkonomisk eller produksjonsklar modell. Fysisk PC-omstart legges kun
til ekte terminale faser etter dokumentert maskinvid jobb-/writer-/lås-/
GPU-ledighet, med ny boot-/WSL-/kilde-/artefaktkontroll etterpå.
GC-kvalifisering er ingen forutsetning for det eksisterende v38-oppsettet.

## Beholdt rekkefølge

1. Full workload-matchet TRAIN-only CPU-samplerbenchmark gjennom capped producer,
   faktisk full-TRAIN-indeks og alle 254 felt/åtte familier. Behold eksisterende
   kandidater 32768/65536/131072 overganger, batch16, én full repetisjon og
   utfallsblind rangering. Godkjent tidligere eligibility var 10800s per epoch,
   total hard wall 64800s; dette er ikke en ny kjøringstillatelse.
   Original tracemalloc, 2GiB allokasjons-/1GiB padded-inputgrenser og øvrige
   eiergrenser består. Partial/småbatchbevis er ikke full kapasitetsmåling.
2. Publiser valgt sampler og fryste epoch0/first4096/TRAIN256-koordinater
   gjennom eksisterende eiere. CONTROL256 forblir fryst på separat fysisk VAL.
3. Bind eksisterende native recipe/campaign til genuine sampler/koordinater
   og mål fersk nullstegsinitialisering. ONLINE/TARGET bruker samme nåværende
   funksjon. Historiske vekter eller constructor metadata er ikke initialbevis.
4. Én eksplisitt bundet native prøve: batch16, 256 optimizersteg, høyst4096
   Entries, seed20260911, eksisterende AdamW/scheduler, FP32 og TF32 av.
   Bevar initial/sluttstate, mål, masker og terminale kvitteringer.
   Ingen full epoch eller full VAL.
5. Mål Entry/Exit LONG/SHORT mot fersk initialisering og TRAIN-konstanter,
   identiske fryste mål og ukeblokk-usikkerhet. Rapporter TRAIN-fit og
   CONTROL256-generaliseringsmåling separat. Biasflytting/FLAT/HOLD alene
   er ikke tilstandsavhengig læring. Bruk docs/LEARNING_GATE.md.
6. Bare ved bestått eksisterende samlet læringsport: forhåndsregistrer én
   kontrollert utvidelse med endelig, kapasitetsbegrunnet budsjett og eksakte
   checkpoint-/stopp-/evidenskrav. Ingen automatisk uavgrenset trening.
   Feilet port krever diagnose, ikke en ekstra treningsrunde.

## Forutsetninger for eventuell gjenopptakelse

Det nye byggevedtaket bruker nye genererte data/normaliseringer; gammel
inputkjøring relanseres aldri og dens artefakter er ikke nye outputs.
Gjenbruk bare uendrede genuine rå-/kalibrerings-/kostforeldre med full binding.
Gjeldende design bruker Seq96 og
M5=16/M15=64/H1=96/H4=96/D1=252 fra den faktiske bundne geometrien.
Reference-policy og cutoff kommer fra frosset DESIGN.json.

Ny eksplisitt plan må binde aktuell ren/committet/pushet kilde, inputs,
operator, output-identitet og finite budsjett før launch. Engangsclaim og
supervisor skal avslutte/reape ved hard wall og kreve genuine eiergodkjente
receipts før grønn terminal. Ingen håndlaget seleksjon eller gammelsti-fallback.
Én tung jobb, etablerte cgroup-/GPU-/host-vakter, fryst kilde under kjøring.

Alle genuine features/tidsrammer/targets bevares. Ingen fast tapsgrense eller
maksimal holdetid. TEST, broker, handel, spending og promotion er stengt.
Dette dokumentet demonstrerer ikke læring, full kapasitet eller lønnsomhet.
