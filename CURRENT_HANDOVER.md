# GX1 — siste overlevering, 09.10.2026

Den ekte native nullmålingen er fullført og kontrollert. Smoke er ikke
startet. Neste tillatte arbeid er ett fast256-forsøk gjennom dagens native
campaign, med ny fysisk boot før vinduet. Stor trening er fortsatt portbundet.
App-målet i thread01a1203d-4a95-7d03-bbd9-84017db6b29c er aktivt.

Kun /home/andre2/src/GX1_CURRENT, branch work/gx1-current, én agent og
én tung CURRENT-jobb. Mac er overleveringskopi. NEXT_RUN_POLICY.json eier
eksakte bindinger og autoritet; kjør scripts/gx1_handover.sh --check.
Fullførte planer/receipts er bevis og skal aldri relanseres.

## Nåstatus og ekte modellbevis

NATIVE_COMPONENTS_20261009_003 fullførte fersk CPU-konstruksjon gjennom
eksisterende eiere med9637663 parametere,652552 TRAIN og70880 VAL Entries.
Initialstate116539510 bytes er varig lagret. ONLINE/TARGET/EMA startet
med samme hash52cd7442 og null optimizer-/EMA-steg. Ingen refit/TEST.

NATIVE_INITIAL_20261009_003 fullførte ekte guarded native-vindu med
terminal16:57:08 UTC, guardPASS og både trainer/observer exit0.
TRAIN256 og separat CONTROL256 er målt:256 Entry- og256 Exit-anchor-
observasjoner pluss1024 samplede Exit-tilstander i hver gruppe.
INITIAL_MEASUREMENT_AUDIT.json bekrefter bit-lik modell, TARGET,
optimizer, EMA, scheduler og CPU/Python/NumPy RNG etter native lagring.
CUDA RNG er separat seedet og lagret. Checkpointet står på optimizersteg0.
Dette beviser starttilstand og lagring, ikke læring eller lært resume.

Samme-run guard målte maks49C core,56C memory-junction,134,76W og2764MiB
VRAM. Workload-only klokkeprofil1395–1695/9751MHz er faktisk kvittert
på boot482. Hele campaign-vinduet inklusive kontroller tok omtrent26min.
Les faktisk state/profilering før konklusjoner om NN-treningsytelse.

Nullmålingens opprinnelige prediksjoner og targets er frosset. Initial
Exit valgte HOLD på begge sider overalt; dette er ikke lært tålmodighet.
Ingen profitable beslutninger, OOS-generalisering eller stortrening er bevist.

## Neste forsøk

NATIVE_SMOKE_20261009_001 er bundet i chronological_learning_run:
maks256 optimizersteg/4096 TRAIN Entries, batch16, samme seed20260911,
FP32/TF32 av, fryst TARGET, sluttmåling av ONLINE. Maks én invokasjon,
12000s; full epoch, full VAL, automatisk forlengelse og TEST er stengt.
Recipe-funksjonen krever nøyaktig samme komplette kildeclosure som
nullmålingen. Initialisering, inputs og frosne observasjoner gjenbrukes.

1. Publiser/verifiser recipe og campaign gjennom de kanoniske eierne.
2. Installer den bundne workload-only klokke-launcheren. Kontroller hele
   vertens jobber, filskrivere, låser og GPU; prepare/confirm fysisk omstart.
3. Verifiser ny boot, uendret kilde/checkpoint og signert telemetri. Kjør kun
   det bundne vinduet; bevar checkpoint hvis tidsgrensen nås.
4. Bruk PAIRED_REVIEW_OPERATOR.py på de lagrede målingene. Verifiser også
   lært tilstand ved lasting og mål reell oppstart/lagring/treningskostnad.
5. Større finite trening krever både lærings- og driftsport. Ingen budsjettreset,
   ingen tuning av samme CONTROL etter å ha sett utfallet.

Læringsporten krever forbedring mot initial og TRAIN-lærte konstanter,
tilstandsavhengige/centrerte resultater og LONG–SHORT-kontrast. Alle
forhåndsdeklarerte CONTROL-kontraster må ha negativ øvre95prosent-grense
i parvis uke-bootstrap; rapporter også måneder og handlingskollaps.
Alltid FLAT/HOLD og ren biasflytting er ikke PASS. Se dagens design og
docs/LEARNING_GATE.md. CONTROL/Juni2026 er gjenbrukt utvikling, ikke urørt OOS.

## Omstart og varig fremdrift

Fysisk Windows-omstart via administrativ SSH er verifisert:
boot482 kl.16:27:20.500 UTC /18:27:20.500 Oslo, WSL69a27500.
Alle164 kode-/recipe-/starttilstandsbindinger bestod etter omstart.
Native-vindu12000s, ytre guard13800s, faktisk Task Scheduler-grense14400s.
Tidsbudsjett sjekkes etter hvert fullført optimizersteg; vanlig checkpoint
hvert64.steg samt ved tids-/sluttpause. Ingen blind Windows-omstart:
kontrolleren beholder terminalen og stopper for maskinfelles idle-review.
Andre prosjekter kan eksistere; omstart må ikke avbryte dem.

Dagens Windows-staging er C:\Users\Andre\GX1_CURRENT_NATIVE_6D726F7A.
Filhashene er verifisert mot den aktuelle planens controller/observer.
Installer med UseNativeClockProfile og korrekt brukerAndre.
Den eksisterende GPU-effektvakten og signerte telemetribroen skal bestå.
SSH gx1-3090-lan fungerer. Ikke gjør omstart om til wsl --shutdown.

## Helrepo-revisjon og bevarte rettelser

De opprinnelige638 sporede filene er inventert. Python-AST, JSON,
shell/PowerShell, lokale importer og Markdown-lenker bestod.
Hele testutvalget ble kjørt i to deler:6290 PASS,3 SKIP og6 kartlagte feil.
Alle seks ble rettet og kontrollert fokusert uten å gjenta fullsuiten.
Fokuserte kampanje-/initialkontroller etter siste rettelse:49 PASS.
Se REPO_AUDIT_20261009_001 og current_work.repository_review_20261009.

Minste observerte rettelser: skrivevaktens manglende os-import; foreldremanifest
mot eksakte verifiserte pre-TEST-kilder; JSON-rutingers kanoniske rekkefølge
for alle åtte familier; varig pause mellom steg;4t Windows-task med riktig
klokke-launcher; fersk fysisk campaign uten gammel full-VAL-modellautoritet.
Dobbelt full indeks-/CONTROL-adgangskall ble fjernet: byggingen falt fra
204 til107s; faktisk inspect57s, innen uendret90s Windows-frist.

CPU-forsøk001/002 og metadataforberedelser INITIAL001/002 bevares med
ekte feil/terminaler; ingen relaunch. Completion-review binder originale
kvitteringer som manglet noen rapporteringsfelt, uten å omskrive dem.

## Datasett og fortsatt avgrensning

HISTORY2009W_NATIVE_V38_20261007:5523147 M1-rader,652552/70880 TRAIN/VAL,
254 features, åtte familier, alle tidsrammer og47 kontrollerte targetfelt.
Whole-TRAIN-normalisering,71 eksakte aliaspar og fysiske koordinater består.
Valgt CPU-sampler målte8192 Entry-par på7275,518s. Dette er ikke NN-
treningsytelse eller en hel TRAIN-epoch. Ingen full datarekonstruksjon nå.

Originale datasett, caches, checkpoints, failed receipts og ferdige analyser
bevares. GC/order-flow og nye footprint/order-block-utvidelser er på pause.
Alle fire ufullførte GC-trinn og separat full makro-B består. TEST er
forseglet; ingen broker/live/paper, handel, spending eller diskopprydding.
