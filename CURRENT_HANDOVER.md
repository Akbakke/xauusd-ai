# GX1 — siste overlevering, 09.10.2026

Inputs og CPU-kapasitet er ferdige. Native smoke er ikke startet.
Fysisk omstart er verifisert; native recipe/vindu og fersk initialmåling gjenstår.
Nytt app-mål 09.10 er **aktivt**: helrepo-revisjon, klargjøring, smoke og
portbundet større trening. Det eldre blokkerte målet beholdes som historikk.
Kun /home/andre2/src/GX1_CURRENT, branch work/gx1-current, én agent.

NEXT_RUN_POLICY.json eier nåstatus, eksakte bindinger og kjøregrenser.
`bash scripts/gx1_handover.sh --check` leser siste terminal og faktiske
CURRENT-prosesser. Denne siden erstatter tidligere mål-/veikart-/renselogger;
Git bevarer historikken. Ikke gjennomgå eller kjør alle gamle steg på nytt.

## Pågående repo-revisjon

Alle 638 sporede filer er inventert. Python-AST, JSON, shell/PowerShell-
syntaks, lokale absolutte Python-importer og Markdown-lenker bestod.
Hele testutvalget er gjennomført i to deler uten gjentakelse av rapporterte
tester: 6290 PASS, 3 SKIP, 6 kartlagte feil. Fem var utdaterte testforventninger
eller fixtures; én var en utdatert policyhash i retention-roten. Alle er rettet.
Fokuserte kontroller bestod etter én korrigering av checkpoint-hookens
bakoverkompatibilitet. Sluttkontrollen av koordinator/lagring/prefix hadde
90 PASS; alle 13 PowerShell-filer parse-bestod og Windows-runtime-testen
bestod også den nye omstartsgrensen. Closure-lintvarslene hadde beståtte runtime-
tester og krevde ingen kodeendring.

Skrivevakten er rettet og Windows-oppgavens ytre grense er tilpasset 4t
rundt uendret native 3t20/guard 3t50. Tidsgrensen sjekkes nå etter hvert
optimizersteg, med ekstra varig lagring ved pause; ordinær kadens er 64 steg.
Kontrolleren bevarer segmentkvitteringen og stopper ved omstartsgrensen;
agenten må kontrollere hele verten før eksisterende prepare/confirm-reboot.
Oppstart, checkpoint og resume har separate tidslogger. Genuine v38-tider
og læring er ikke målt ennå. Bevis: REPO_AUDIT_20261009_001 i current_work.

Fersk constructor-metadata og én CPU-komponentplan er nå bundet i
current_work.native_component_preparation_20261009: høyst 1800s, producer
20G/512M, ingen model forwards, optimizersteg eller normaliseringsrefit.
Planen gjenbruker aktuelle inputs/koordinater og lager fersk ONLINE/TARGET.
Dette åpner ikke native GPU-launch; faktisk initialstate og nullstegsmåling
gjenstår.

## Gjort — målt på ekte data

- Nytt HISTORY2009W_NATIVE_V38_20261007 har komplett core/M1, inputkontroll,
  deknings-/gapreview og post-rebuild-readiness. M1-flaten har 5523147 rader;
  fysisk TRAIN/VAL har 652552/70880 Entries. Ukjente gap forblir ukjente og
  right-censored; ingen imputerte priser eller flyttede perioder.
- Whole-TRAIN base-/summary-normalisering og fysiske M1/M5/MTF-visninger er
  produsert. Hele input-/Entry-Exit-kontrollen målte NumPy/Torch maxabs0 og
  71 bit-identiske aliaspar. Ingen fit på VAL/TEST eller lært bundle-paritet.
- Alle tre komplette CPU-samplerkandidater er målt på dagens TRAIN-indeks.
  Eksisterende eier valgte 32768 transitions/batch16: 8192 Entry-par på
  7275,518s. Dette er ikke en målt hel TRAIN-populasjon eller NN-treningsytelse.
- Originaldesignets manglende scope/status er rettet strengt metadata-only;
  originaldesign/benchmark er bevart. Fysisk kontroll av alle47 aktive
  targetfelt og normaliseringskoblinger bestod. Epoch0/first4096/TRAIN256
  og separat CONTROL256 er publisert gjennom eksisterende koordinat-eier.

Siste genuine terminal er NATIVE_PREPROCESSING_001, exit0,
08.10 kl.21:31:56 UTC. ORIGINAL M1-target-support gjenbruker uendret
emitter/manifester som kilde-/algebrabevis, ikke en ny raw quote-scan.
Koordinater er ikke modellmålinger. Alle genuine features og tidsrammer består.

## Ikke gjort / aktuell hindring

- Ingen aktuell native recipe/vindusbinding, fersk ONLINE/TARGET-initialisering,
  modellforward, optimizersteg, smoke, større trening eller full VAL.
- Omstartsblokkeringen er løst via eksisterende administrativ Windows-SSH.
  PID2648/2704 var GPU-effektvakt og telemetribro, bekreftet fra kommandolinjer,
  installert kilde og Task Scheduler-instanser. Ingen ACL-endring.
  Prosjektjobber/låser og GPU-beregninger var ledige før omstart.
- Ny Windows-boot 09.10 kl.10:47:48.500 UTC (12:47:48.500 Europe/Oslo);
  WSL-boot 67c99398. Ren bae53257 og 37 bundne artefakthasher er bevart.
  SSH via gx1-3090-lan, WSL og signert GPU-telemetri er verifisert etterpå.
  HOST_RESTART_20261009_001/POST_RESTART.json er bundet i current_work.
  Dette er kontroll av bundne kvitteringer/koordinater, ikke ny full datascan.
- Krasjårsaken, v38-læring, generalisering, nettoøkonomi og lært train/serve-
  paritet er ikke dokumentert. Mekanikktester/input-PASS beviser ikke dette.
- GC/order-flow og nye footprint/order-block-utvidelser er på pause.
  Alle fire ufullførte GC-trinn og separat full makro-B består.
  TEST, broker, live/paper, handel og spending er stengt.

## Neste steg — i denne rekkefølgen

1. Gjenbruk ferdige inputs, kapasitet og koordinater. Omstart er fullført;
   neste native-invokasjon må fortsatt bestå fersk-boot-/host-/GPU-gatene.
2. Bind aktuell kanonisk native recipe/campaign og separat finite vindu.
   Mål fersk nullstegs ONLINE/TARGET med samme nåværende funksjon.
3. Én portbundet smoke: høyst256 optimizersteg/4096 TRAIN-Entries, batch16,
   FP32/TF32 av, seed20260911. Separate parvise initial-/sluttmålinger på
   fryste TRAIN256 og CONTROL256.
4. Ærlig læringsreview etter docs/LEARNING_GATE.md. Mer trening krever egen
   finite autoritet og bestått læringsport, aldri automatisk utvidelse.

Operatørkrav 09.10: hyppige trygge omstarter mellom målte segmenter, med
varig mellomlagring og målt effektivitet. Nåværende kode har 12000s-vindu og
FP32-checkpoint hver 64 optimizersteg samt ved tidsgrensen. Operatøren har opplevd heng etter
12 timer og valgte 09.10 å beholde eksisterende 3t20-vindu. Ingen forkorting
kreves. V38 resume-ekvivalens og lagre-/lastetid skal fortsatt måles; se
docs/NATIVE_LEARNING.md. Ingen kalenderstyrt omstart avbryter aktivt arbeid.

## Vedlikehold og grenser

Tre overlappende status-/mål-/renselogger er fjernet; Git bevarer dem.
Handover viser kun aktuell status. Etter omstart bestod 63 fokuserte
handoverkontroller gjennom capped audit; ingen fullsuite eller modelltrening.
15 verifiserte cachemapper er fortsatt bevart:
vertens slettingsvern nektet fjerningen før launch. Ingen omgåelse.

Oppstarten skal bare lese denne siste overleveringen og aktuell policy.
docs/NATIVE_LEARNING.md beskriver gjenværende metode, ikke daglig historikk.
Regler: GX1_RULES.md. Eiere: SYSTEM_MAP.md. Dokumenter: DOC_INDEX.md.

Fullførte/feilede claims og deres genuine kvitteringer bevares uten relaunch.
CPU-deadline08.10 kl.22:55:12 UTC er utløpt; ingen budsjettreset.
Timeautomatiseringen er slettet og gjenskapes ikke. DATA/RUNS er ikke slettet:
retention-eierens siste nektelse gjelder uregistrert authority-directory
manifest på dagens run-root. Ingen håndlaget unntak eller sletting av
nødvendige foreldre, rådata, checkpoints, .env, .venv eller .git.
