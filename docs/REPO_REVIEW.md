# Gjennomgang og opprydding av GX1

Siste fasegrense08.10: genuin full fysisk TRAIN/VAL-indeks652552/70880 og
ferske no-cap-inputidentiteter er akseptert, claim konsumert. Ny separat
PRE_SMOKE_READINESS_001 er eksakt bundet/not started under samme18:25:44
CPU-frist: genuine inputrevalidering, fersk utfallsblind CONTROL256/design,
finite fremtidig sampler-/smoke-plan og faktisk retention/nektelse. To
fokuserte kilde-/scalar-kontroller PASS, ingen ekte final-aksept ennå.
Modell, benchmark, TEST, sletting og omstart er ikke åpnet.

Dato: 06.10.2026. Brukeren ba om grundig repo-gjennomgang og minst mulig død
kode/fyll, og prioriterte kontrollert benchmarkstopp. Kildegrunnlag før bølgen:
8f91ecaf006a9d4fa3e53c199a7b1b91ddce7f91, work/gx1-current.
Dette er gjeldende oppryddingsrapport, ikke en trenings- eller sletteautoritet for DATA/RUNS.
08.10: INPUT_VIEWS_001 har genuine separate fysiske TRAIN/VAL/M1-inputs.
BASE_NORMALIZATION_001 er genuine exit0 på uendret1af9a63c, alle652552
TRAIN-Entries/fysiske M1/M5/fem MTF-flater, VAL/TEST-fit0. SUMMARY_NORMALIZATION_001
er genuint exit0,01:19:05 UTC på uendret1a8a0411, claim konsumert: whole-
TRAIN-summary7613258 side-fit-rader, separate VAL-counts uten fit og composite/
split/clock/quote-bindinger. Base ikke refittet. INPUT_TENSOR_AUDIT_003 er
genuint exit0,01:58:31 UTC på uendretb2ad509d og claim konsumert. Hele
M1/M5/Entry/fem kausale MTF-ruter: NumPy/Torch maxabs0,71 aliasbits eksakte,
første-state/480-observed-row mappings source-eksakte. Ingen modell.
NATIVE_INPUT_INDEX_001 er genuint exit0,02:14:19 UTC,555,002s, uendret7c19bb08; for full fysisk
TRAIN/VAL-indeks og ferske no-cap-økonomiidentiteter gjennom eksisterende
eiere/producer20G/512M/samme CPU-frist. To fokuserte metodekontroller PASS;
ingen refit, ny holdetidsregel, samplerbenchmark, modell eller TEST.
Ingen nye feature-/fit-/modellkodeendringer i denne status-/bindingsbølgen.
Staging-feilrettelsene og originale 67/101-case-bevis bevares. Selve nye
indekspublisering, retention og modell-/kvalitetsevidens er ennå ubevist.
Ny status-/bindingsbølge:12 fokuserte handover-regresjoner PASS under4G/512M,
eksterne stagekontroller består syntaks/JSON, diff/stale/source kontrolleres
før ren commit/push. JUnit: BASE_NORMALIZATION_001/HANDOVER_BINDING_TESTS.xml.
Summary-bindingsbølgen:26 fokuserte final-binding/prefix-boundary-regresjoner
PASS under4G/512M; source/fixture-mekanikk, ikke ekte ny summary-fit/paritet.
JUnit: SUMMARY_NORMALIZATION_001/FOCUSED_BINDING_TESTS.xml. Ingen feature-,
fit- eller modellsemantikk endret; eksterne engangskontroller består syntaks.

INPUT_TENSOR_AUDIT_001 feilet før transformene01:38:17 UTC: håndkopiert
VAL-proof-digest var65 tegn, én ekstra bokstav. Kanonisk policy/faktiske
bytes er uendret og matcher. Original exit1/kildebd1b9abc/plan/claim bevares;
ny002 korrigerer kontrollbindingen og avviser malformed digest før hash.
Ingen feature-/normaliserings-/modellkodefeil er påvist av dette stoppet.

INPUT_TENSOR_AUDIT_002 feilet01:47:14 UTC,exit1/314,451s, uendret2e428cfa.
JSON-sorterte nøkler kom i feil deklarasjonsrekkefølge til strict MTF-eier.
Geometri/data uendret; original002 bevares konsumert, uten full inputaksept.
Ny003 bygger bare utført EXPECTED_TFS-rekkefølge med fryste lengder og
prechecker eksisterende eier før transformene. Fire fokuserte kontroller
PASS under4G/512M; fryst scalar-geometri og kilde/syntetisk mekanikk, ikke
full inputparitet. Read-only NumPy-buffere kopieres byte-identisk før Torch-
view; ingen modellkonstruering eller endring av feature-/fit-/eiersemantikk.
JUnit: INPUT_TENSOR_AUDIT_003/FOCUSED_GEOMETRY_TESTS.xml.

Input-audit-bindingen bestod den fokuserte NumPy/Torch-asinh-regresjonen
under audit4G/512M,1 PASS. Kilde-/syntetisk mekanikk, ikke ekte inputparitet.
JUnit: INPUT_TENSOR_AUDIT_001/FOCUSED_TRANSFORM_TEST.xml. Ren kilde og eksakte
eksterne kontroller bindes før claim; ingen feature-/fit-/modellkode endret.

## Ny bølge 07.10.2026 — ferskt datasett og smoke

Nyere brukerbestilling: ny kjøreplan, gjennomgang av hele repoet for feil/
mismatches, ferskt datasett og retirement av utdaterte genererte outputs,
før en liten smoke og betinget større trening. GC forblir på pause.
Den nye kjøreplanen ligger i docs/NATIVE_LEARNING.md og er eksakt bundet i
NEXT_RUN_POLICY/native_v38_rebuild_20261007. Historikken nedenfor beskriver
den tidligere repo-slettebølgen, ikke utført DATA/RUNS-sletting 07.10.

Recovery-bølge etter fysisk BSOD 0xA samme dag: eksisterende chain støttet
RED-terminal recovery, men ikke hard boot-tap uten terminal. Minste retting
tillater eksakt orphan-status på annen boot ved dataset-rebuild, uten gammel
terminal, med uendrede upstreambytes og fersk downstream/preflight. 16 fokuserte
kontrakttester PASS, inkludert åtte negative orphan-identitetsmutasjoner.
Ingen gx1 feature-/modellkilde er endret. INPUT_BUILD_002 er én ny finite
core-kjøring og stopper for trygg omstart før obligatorisk komplett M1.
Gamle receipts beholdes; ingen påstand om gammel exitkode, samlet input-green
eller at hyppig reboot er en dokumentert retting av ukjent driver/hardwareårsak.

Hele inventaret før endring: 640 tracked filer / 581 Python-filer,
14413210 bytes. Mekanisk AST-/lokalimport-/JSON-/shell-/Markdown-/dependency-
kontroll gir ingen syntaksfeil, importhull, doble toppnivådefinisjoner,
brutte lenker, JSON-duplikater eller avhengighetsversjonsmismatches.
69 Ruff-varsler er scope-/fixturefenomener: tre F821 på tidligere kontrollerte
closures og 66 F811 på fixtureimports/parametre. Ingen blind lintopprydding.
Kildereview er hash-bundet i policyen. Manuelt er de risikobærende build-/M1-/
normaliserings-/lifecycle-/status-/retention-grensene og faktiske funn fulgt;
ingen påstand om manuell gjennomgang av hver linje.

Konkrete funn og minste endringer:

- Hardware-testen hadde 242 fra v37 mens utført v38-eier har 254. Signal-/
  context-/MTF-mål kommer nå fra de faktiske kontrakteierne; M5 kopieres ikke
  inn som ekstra Entry-MTF. Ingen arkitekturreduksjon.
- Den syntetiske bounded-parity-fixturen brukte bare price-warmup 219.
  Tosidig sweep-AVWAP var først fullstendig på rad 262 av den deklarerte
  syntetiske kilden. Fixture måler egen SMC-prefix og bevarer ønsket sample-
  antall; produksjons-NaNs og warmup-vakter er uendret.
- Retention-rootene var reelle bevarte sikkerhetsregistre, ikke manglende
  filer. Den tidlige mistanken om slettede registre er trukket tilbake.
  De fulgte imidlertid ikke dagens NEXT_RUN_POLICY-inputbindinger. Nå binder
  samme launchrot eksakt policy/hash. Eksakte {path,sha256}-manifestbindings-
  former følges transitivt og feil hash/ukjent shape feiler lukket. Nested
  binding beholder TEST-rollen før metadataåpning (fire negative cases).
  Den kanoniske kildedirens egen reelle policy følges; tilfeldige DATA-dirs
  med samme filnavn avvises. Faktisk closure-diagnose stopper deretter på
  den nye run-rooten som ennå ikke har genuine registrert completion.
  Den samme exact-target-eieren tillater nå også GX1_RUNS etter brukerens
  uttrykkelige retirement-vedtak; rootdelete, exclusions, aktive writers,
  manglende closure, symlinks og TEST-nektelser er fortsatt forbudt.
- Chainens pair-alignerte M1-lifecycle-flate er ikke komplett M1-state-
  dekning. Nytt build har derfor komplett pre-TEST M1 via samme feature-eier
  som en obligatorisk separat fase før normalisering/smoke. Legacy surface-
  identitet/semantikk endres ikke og komplette raw-minutter fylles aldri inn
  med syntetiske verdier. Selvstendig helt klokkesuffiks må bestås.

Verifikasjon: én fullsuite forsøkt, stoppet fail-fast etter 19 PASS/1 gammel
shape-feil. Ingen gjentatt fullsuite. Siste endrede-gruppe har 368 PASS;
Siste retentiongruppe har 155 PASS. Case-sensitiv JUnit-dedup gir 544 unike
testcases med siste status PASS, ikke full
testsuite-PASS. Alle tunge audits bruker capped 4G/512M, én jobb/tråd.
8 endrede/eksterne Python-filer og tre JSON-autoriteter består syntax/parse.
Råpair-/squeeze-/direkte M1-avhengigheter er genuint revalidert: seks squeeze-
klokker, kontrakteid 254/åtte og 5959045 komplette pre-TEST M1-rader.
Ingen modell-forward, optimizersteg, normaliseringsfit eller TEST-evaluering.

Storage-inventar: cirka 313G DATA og 11G RUNS; 306G ligger i prebuilt.
Det er cirka 546G ledig. Hele v37/v38/BOOTSTRAP-roots kan ikke uten videre
slettes: de inneholder fortsatt genuine rå-/squeeze-/M1-provenanceforeldre.
Ingen DATA/RUNS er slettet i denne bølgen. Etter ny replacement-aksept skal
eksakte foreldede leaves/størrelser tas gjennom retention; run-rooten må
først ha genuine registrert completion og full closure. Ingen håndlagde
unntak dersom current-policy-/directory-/TEST-closure ikke kan bevises.
Bare en genuine completion/clock/readiness-receipt aksepterer nytt input;
ingen modelleffekt, lønnsomhet, ny normalisering eller større trening er bevist.

## Gjennomgått omfang

Hele tracked inventaret: 842 filer, 17219007 bytes, 590 Python-filer /
323500 Python-linjer, 136 Markdown-filer.
Alle tracked tekster inngikk i AST-/import-/literal-/sti-/dokumentreferansescan.
Python ble kontrollert for syntaks og doble top-level-definisjoner: ingen funn.
Manuell gjennomgang prioriterte faktisk arbeidsstatus, source closure, input-/
modell-/økonomieiere, sikkerhetsvakter og konkrete slettekandidaters eierskap.
Dette er ikke en påstand om manuell linje-for-linje-kvalitetsrevisjon av 323500 linjer.

## Konkrete rettelser

- Statusleseren lette etter native modellprosesser, men overså den faktisk
  aktive eksterne CPU-benchmarkoperatoren. Den observerer nå alle Python-jobber
  med CURRENT-interpreter og verifisert cwd, uten egen prosess. Både absolutt
  og relativ .venv/bin/python-invokasjon fra eksisterende capped-runner er dekket.
  Prosess-exit mellom ps og /proc behandles som normal race.
- Doble statusfiler kunne vise gamle «aktive» sesjoner/checkpoints og gamle
  inputbredder. NEXT_RUN_POLICY/current_work er nå eneste arbeidsstatuseier.
  Explicit terminal/hash kontrolleres; manglende binding feiler lukket.
  Ingen historisk checkpointfallback eller implicit dataset-admission.
- Statussti- og repovalidering avviser symlinkforeldre, ikke bare leafsymlinks,
  slik at en tilsynelatende tillatt sti ikke kan peke inn i forseglet TEST.
- M1-vakten matchet gamle Exit-filnavn, men ikke dagens unified_exit-/M1-eiere.
  Den blokkerer nå coarsening på aktuelle stier; seks regresjonscases bevarer
  legitim M5-MTF-bruk. Ingen effekt-/temperatur-/cgroupgrense er svekket.
- Repo-/installerte Claude-vakter hadde gammel worktree/interpretersti.
  Konkrete CURRENT-stier og hook-kommandoer er synkronisert etter særskilt
  brukerautorisasjon. Andre globale innstillinger er bevart.
- Økonomiske enhetstester lastet en kostnadsautoritet fra et gammelt worktree
  og genuine historiske filer. De bygger nå små hash-bundne syntetiske fixtures
  gjennom samme kost-/broker-/step-eiere. Prodsemantikk og nødvendige negative
  kost-/readiness-/hash-/reward-/childclocktester beholdes.
- Dokumentenes prelaunch-/gammelpopulasjonsstatus er erstattet med faktisk
  avbrutt benchmark og gjeldende v38-grenser. Lukket forskningsomfang og full-B-
  mål er konsolidert i de eksisterende reglene, ikke fjernet ved opprydding.

Lintvarsler alene er ikke slettebevis. Tidligere mistanke om udefinerte closure-
variabler og duplikate fixturetester ble trukket tilbake etter faktisk
scope-/AST-kontroll: de var ikke defekter. Dynamisk exec og pytest-fixtureimports
må vurderes med kjørende bruk, ikke bare F401/F811/F821.

## Slettingsgrunnlag

Doble rootstatusfiler, alle resterende handover-snapshots, avsluttede
engangsrapporter/-configs og frakoblede gamle pilot-/benchmark-/oracleverktøy
fjernes bare etter import-/kall-/test-/dokument- og aktuelle inputbindinger.
Syntetiske tester tilhørende en slettet frakoblet implementasjon slettes med
den; tester av beholdte grenser beholdes.
Frosne v38-inputconfigs og kode-refererte design-/preregistreringsdokumenter
beholdes. En CLI uten Python-import er ikke automatisk død kode.

Eksakte slettinger og endringer finnes i denne Git-bølgens diff.
Alle tracked slettinger er gjenopprettbare fra kildegrunnlaget over.
Ingen ny arkiv-/historikkmappe opprettes. Ignorerte, regenererbare cacher
kan fjernes etter eksakt repoavgrensning; hemmeligheter og miljøet bevares.

## Sluttinventar

- Filer: 842 → 635, netto 207 færre (24,58 %).
- Markdown: 136 → 20, 85,29 % færre.
- Python: 590 → 579; linjer 323500 → 320764.
- Eksakt slettebølge: 207 foreldede filer / 2434850 bytes; tre eksisterende
  dokumentautoriteter er omdøpt/konsolidert, ikke et nytt historikkarkiv.
- I tillegg er store status-/designdokumenter forkortet til gjeldende roller.
  Repoets tekstinventar er cirka 14,32 MB mot 17,22 MB før bølgen.
- 17 regenererbare cachemapper / 12445545 bytes er fjernet etter canonical-path,
  ikke-symlink, Git-tracking, scope- og ledig CURRENT-prosjektlås-kontroll.
  Cacher kan gjenoppbygges; de er ikke modeller eller kjøringsbevis.

Beholdt: 257 gx1-filer, alle byteuendret fra grunnlaget. Features, modeller,
execution og alle gjenværende kontrakteiere er ikke refaktorert.
Slettet kode var frakoblet gammel pilotcampaign/telemetri, separat hindsight-
oracle, to gamle microbenchmark-CLI-er og en avsluttet one-shot kildeprobe,
med kun tilhørende frakoblede testsaker. Den aktive samplerbenchmarkeieren består.
Hele snapshot-mappen, tre doble rootstatusfiler, 28 ferdige forsøksconfigs og
gamle evidence-/rapportkopier er fjernet. Aktuelle v38-bindinger består.

Retained dated filnavn er eksplisitte kode-/frozen-inputbindinger, ikke historiske
restartfiler. De gjeldende designdokumentenes gamle framdrifts-/resultatfortellinger
er fjernet. Retention-/incidentregistrene beholdes som aktive slettevakter,
ikke som «fyllmasse». Beholdte offline serving-/persistens- og diagnose-CLI-er
er nødvendige grensesnitt/evidenseiere; manglende import alene er ikke død kode.

## Verifikasjon

Alle testjobber ble kjørt én om gangen gjennom capped audit 4G/512M,
CPU 0–7, én numerisk tråd og TasksMax 64. Dette er syntetisk/kildemekanisk
kontraktbevis, ikke genuine modell-/økonomiske målinger.

- 254 bestått: status, launchhold/datasetautoritet, recipe/handover og vaktkopier.
- 70 bestått: broker/kostpolicy/step-provider og liquidation-relative learning/VAL.
- 184 bestått: beholdt research, incremental carry, outcome-targets og Windows-controller.
- 304 bestått: capped execution/run og M1-/Claude-vakter.
- 327 bestått: signal/feature-layers, modellhandling/sizing, sampler/benchmark,
  state-view/semantiske ruter og retention.
- 54 bestått etter siste vakt-/statusrettelse: inkluderer seks nye M1-path-cases
  og eksplisitt avvisning av gammel native calibration under dagens stop-policy.
  Alle tre prosess-invokasjonsformer er også kontrollert mot fremmed cwd og exit-race.
  Grupper overlapper; disse tallene skal ikke summeres som unike tester.

Hele testsamlingen på da 6170 cases ble samlet uten importfeil; ni senere
regresjonscases er også samlet og bestått i den siste fokuserte jobben.
Ingen full suite er kjørt. Ingen manglende statiske lokale gx1-importer,
Python-syntaksfeil, doble top-level-definisjoner, JSON-parsefeil eller brutte
Markdown-lenker er funnet i det gjenværende inventaret.
10 shellskript (inkludert pre-commit) og 13 PowerShell-filer består syntax/parser.
Referanse-/sti-scannen gir ingen levende kall til de slettede filene;
de eneste retained rootstatus-navnene er negative fraværsasserts i tester.
git diff --check består. Handover skal vise ren kilde etter normal commit,
stoppet benchmark og blokkert next_run, ikke treningsautoritet.

Benchmarken er kontrollert stoppet med bevart exit-1-terminal og source_unchanged.
Ingen ny ekte-data benchmark, native forwards, fits, optimizersteg eller TEST-tilgang er gjort.
Nye kopier av data eller modeller er ikke laget.

## Ikke bevist av denne bølgen

Ingen full samplerbenchmark/valg, fersk initialmåling, 256-stegs prøve,
v38-læring, generalisering, strategi-PnL eller ny train/serve-paritet.
Ingen full testsuite eller manuell revisjon av hver linje.
Ingen Windows-task-/maskinvareinstallasjon, brokerkall eller handel.
Full makro-B er ufullført og erstattes ikke av MACRO_CORE.

Ingen DATA/RUNS, råkilder, modeller/checkpoints, .git, .venv eller .env slettes.
Retention-eierens plan → godkjenning → utføring er eneste DATA/RUNS-rute.
