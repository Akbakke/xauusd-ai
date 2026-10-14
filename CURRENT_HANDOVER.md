# CURRENT HANDOVER — 12.10.2026

Autoritet: /home/andre2/src/GX1_CURRENT, work/gx1-current. NEXT_RUN_POLICY.json er
eneste arbeidsstatus; prosess og terminal overstyrer prosa. Historikk per forsøk
ligger i Git og i policyens bindinger, ikke her.

Nå: den native D1-bølgen er bygget og målt til og med steg 5. Generasjonen
HISTORY2009W_NATIVE_D1_20261011_003 (315 signalfelt med den niende D1-makro-/
posisjoneringsfamilien, 1/5/20-handelsdagers observerte mål i TRAIN/VAL, TEST
forseglet) har grønn kjernekjede, åtte grønne nedstrøms inputsteg og en fersk
v10-nullstegsbaseline. Swing-programmet har funnet et long-timing-signal på 20
dagers horisont som er støttet, men ikke bekreftet; flaten er frosset og
forward-journalen er eneste bekreftelsesrute. Steg 6 (trent native modell mot
D1-ridge-journalen) krever eget vedtak. Ingen dokumentert Entry-edge, ingen
prediktor innrømmet, training_enabled = false.

## Rettelser i HEAD 11.10.2026 (operatørbestilt)

Kode og tester i 56d87f49; filhasher og begrunnelse i
NEXT_RUN_POLICY.current_work.head_deviation_fixes_20261011.

1. Objective-kontrakt v11 bærer `entry_training_phase` (entry_only uten Exit-
   gradienter, joint med frossen Exit-lærer); treneren sender fasen den kjører.
2. Alle encodere avsluttes med parameterfri LayerNorm (F12).
3. Vindus-readout er middel pluss siste lukkede bar på alle fire steder (F27);
   arkitekturversjon v10, samme parametre og state_dict-nøkler, annen funksjon.
4. Vekt-EMA bias-korrigert med min(decay, t/(t+1)) (F18).
5. Checkpoint-policy v6–v8: konstant Entry-handling på VAL kan aldri velges (F15).
6. Forskningseieren merker degenerert referanse eksplisitt.

Ikke endret: hjelpehodene i Entry-only, slippage 2,0 bps, patience/min_delta.
Gamle v9-checkpoints avvises ved bundle-last. Samplerbenchmark: exit 0 på 20ab5fe5,
valgt batch 16 med målt CPU-epoke 7679 s, modelluavhengig.

## v10 nullstegsbaseline 11.10.2026: målt, ingen læring

FRESH_INITIAL_BASELINE_003 (83adab26, eget 6-timers CPU-budsjett PREPARATION_CPU_002,
lukket) målte den ferske v10-funksjonen på samme korrigerte inputs, normalisering og
TRAIN4096/CONTROL4096-kohorter som 002: 9 637 663 parametre, 513 forwards, én
TARGET-paritetsforward med alle hoder bitidentiske, null optimizersteg, 339 s i
producer20G. TRAIN argmax LONG/SHORT/FLAT 1878/733/1485 (002: 2532/717/847), 1046
av 4096 endret; CONTROL 1779/830/1487 (002: 2317/884/895), 1139 endret. Side-MSE
uendret (≈917/878 TRAIN, ≈1877/1879 CONTROL). Utrent initialtilstand; ingen
treningsautoritet.

## Swing-program 11.10.2026: målt status

Kilder (manifestbundne, hashbundne): COT deacot 2009–2026; OANDA H1 for XAU og 10
panelinstrumenter 2009-06..2026-06-30 (USD_CNH feilet lukket); Treasury par-/real-
kurver 2009–2026; Cboe VIX; GLD via Internet Archive (til 2026-02-13) og SPDR-
arbeidsbok (til 2026-10-08); FOMC-sider; Fed H.10 daglig bred dollarindeks. FRED
svarte ikke fra verten; CME og BLS er bak bot-sperrer.

Resultater, alle målt på gjenbrukt utviklingshistorikk 2011–2025 (15 analyser på
samme data) pluss ett ubrukt forward-halvår; eksakte tall og bindinger i
policyblokkene swing_d1_stage_a/b/c, swing_pooled og swing_d1_stage_c_followups:

- Porteføljefamilien (full dekning mot alltid-LONG) er NO_GO eller INKONKLUSIV i
  alle forsøk. Bulk rank-IC ≈ 0.
- Selektivt ved h20: HGB-effekten fra stage A/B overlevde ikke familiejustering
  (minste familie-p 0,23) og replikerte ikke på tvers av instrumenter.
- XAU swing_ridge h20 er treff-familiesignifikant: 008 (73 felt, kanonisk M5-rute)
  0,781 ved 5 % dekning, familie-p 0,043; 009 (to-års folds) 0,872/0,831/0,784/0,698
  ved 2/5/10/25 %, familie-p 0,022/0,0085/0,007/0,026. Fanget mål er aldri
  familiesignifikant under sirkulær-shift-nullen.
- Diagnose: valgte rader er 87–90 % LONG, og alltid-LONG på samme rader treffer
  likt. Effekten er NÅR en 20-dagers long betaler, ikke retning. Blokkmatchet
  long-null: fanget mål p 0,018/0,030 (008) og 0,003/0,004 (009) ved 5/10 %.
- Ablasjon (5 %, treff/familie-p): uten kryss-panel 0,647/0,83 (borte); uten
  drivere 0,721/0,27; uten COT 0,746/0,14; uten kalender 0,776/0,050; uten
  dollarindeks 0,831/0,011. Flaten fryses derfor til 68 felt: A, COT, panel,
  drivere.
- Forward 010 på 2026H1 (108 rader, ubrukt): rank-IC 0,44, treff 0,91 ved 10 %
  (11 rader, 5 episoder) og 0,70 ved 25 % i et halvår med 36 % positive
  20-dagersbevegelser. Forhåndsregelen ble ikke oppfylt (blokk-null-p 0,065 ved
  25 %); INKONKLUSIV med deklarert lav styrke, støttende på alle metrikker.
- Episodeøkonomi (beskrivende, 10 %/20 d, ex finansiering ≈ 30 bps per long-
  episode): 95 episoder +119 bps (008), 90 episoder +169 bps (009), 5 episoder
  +803 bps (010).

Verdikt: ubekreftet long-timing-spor. Ingen prediktor innrømmet.

## Forward-journal (vedtak 11.10 kveld): Q3 2026 scoret, ingen flagg

Eier `run-swing-forward-journal` (ada3b553) på den frosne 68-feltsflaten, XAU via
H1-ruten, swing_ridge h20/h5. Policyblokk forward_journal_20261011 autoriserer
lesing av forseglet periode bare for de fem navngitte kildene til scored_end_utc
2026-10-11; forlengelse er en eksplisitt policyendring. Kildene ble hentet på nytt
under egne manifester (COT 2026 og Treasury 2009–2025 gjenbrukt under hash;
OANDA H1 for 11 instrumenter til 2026-10-09; VIX til 2026-10-09; SPDR-arbeidsbok
til 2026-10-08). Panel 4 718 rader til 2026-10-08.

Første kjøring (012db459): juli, august og september 2026 scoret nøyaktig én gang,
hver med ridge fit på 4 374–4 418 rader hvis 20-dagersutfall lukket før måneden
(alpha 3162 i alle tre). Alle prognoser var LONG i alle tre måneder, i tråd med
long-timing-diagnosen. Ingen rad nådde de forhåndsregistrerte |prognose|-
tersklene (5/10/25 % av modellens egne størrelser siste år), så journalen flagget
null posisjoner. Rank-IC innen måneden ved h20: juli 0,50 (22 rader), august 0,39
(22 rader), september −0,10 (8 kjente rader). Alltid-LONG-treff 0,82 / 0,32 / 0,00.
Juli og august er løst; september løses etter 20 handelsdager. Årsregelen krever
12 løste måneder. Beskrivende tall, ingen inferens, ingen prediktor innrømmet.
Fit skjer bare på historikk før hver måned; scorede rader brukes aldri til fit.

## Native D1-bølge: steg 3 kodet og testet, steg 4 admittert (11.10 natt)

Steg 3 (a33077be): den niende kontekstfamilien er inne i native kontraktene.
Feltnavn og as-of-join eies av gx1/features/d1_macro_context_v1 (forskningseieren
importerer dem); kildespesifikasjonen configs/research/D1_MACRO_CONTEXT_SOURCES_
20261011_001.json er pinnet med hash i koden. Kontekstskjema v13 (132 felt).
Signalkontrakten eksponerer hvert kontinuerlig kontekstfelt til sekvensstien, så
signalflaten vokser fra 254 til 315 felt (signalskjema v39). Alle 61 felt eies
av session_regime_encoder; en kontekst-bare encoder ville vært ny arkitektur,
som vedtaket ikke åpner. Den berikede M1-/M5-lane-produsenten legger familien på
hver rad i avgrensede biter (float32) før felles warmup-trim; builderen regner
den på nytt og krever bitparitet, og legger den selv på bare når en splits kilde
mangler den (TEST-konstruksjon under policyens eksplisitte leseslutt
2026-09-01). Builderen emitterer også y_long/short_gross_at_{1,5,20}d_bps og
medgåtte sekunder (NaN ved splittgrense eller manglende sesjonsoffset). Fokuserte
sweeps: kontrakt, ruting, fixtures, laner, preflight, readiness, modellformer,
builder og forskningseier grønne (den kjente tråd-følsomme exit-token-testen
er grønn ved én tråd).

Steg 4, forsøk 001 (3fbb95c7) stoppet lukket etter 2 013 s i M5-lanen:
D1_MACRO_CONTEXT_BEYOND_READ_END. Lanen kuttet den niende familiens kilder ved
TEST-forseglingen, mens de forseglede pair-rotene går til 2026-08-31 23:55 fordi
kjeden konstruerer og forsegler TEST-splitten. Rettet i lanen (a631e69c): samme
policybundne test-konstruksjonsslutt 2026-09-01 som builderen; eieren nekter
fortsatt enhver slutt policyen ikke autoriserer. Alt fra 001 er bevart.

Steg 4, forsøk 002 (be116811) kom gjennom begge lanene med den niende familien
(M5 1 166 786 rader, M1 5 660 070 rader, alle komplette), signalmanifest, begge
featureflatene (M1 5 570 522 rader med 132 kontekstfelt) og preflight, men
stoppet lukket ved builderens start etter 18 105 s: builderen ba om test_end
+ 1 dag som leseslutt for TEST-kontekst, mens policyen autoriserer 2026-09-01
00:00 (siste TEST-beslutning). Rettet (e183f6bc): builderen binder policyverdien
nøyaktig som lanene og feiler lukket hvis siste TEST-beslutning ligger utenfor.
Horisontjoin herdet (6624014d). Alt fra 002 er bevart.

Steg 4, forsøk 003 (8f5d39fd) er GRØNN: kjernekjeden terminerte 08:42 UTC 12.10
etter 23982 s med exit 0, uendret kilde og forseglet TEST
(CHAIN_TERMINAL_20261012T084236056682Z_GREEN.json). Begge lanene bærer den niende
familien (M5 1 166 786 og M1 5 660 070 komplette rader; M1-featureflate 5 570 522
rader med 132 kontekstfelt). Datasettet: TRAIN 652552 rader med horisontmål
651300/650059/646543 ved 1/5/20 d, VAL 70880 rader med
70573/69535/65772, TEST skrevet og forseglet, ulest. Builderen la den
niende familien på selv (kildeframen bar ikke kolonnene), samme float32-cast som
lanen, så sekvensaliasene og kontekstfeltene er bitlike; aliaslikheten sjekkes
igjen ved modellens forward i baselinen. Radpopulasjonen er identisk med forrige
generasjon (652 552 / 70 880), så koordinatene fra FRESH_INITIAL_BASELINE_003
kan gjenbrukes der tidspunktene mapper.

Trygg omstartsgrense: kjernen er terminert, ingen prosjektlås holdes, ingen GPU-
jobb kjører. Brukeren ba om omstart her. Etter omstart: ny boottid, WSL og uendret
kilde bekreftes før noe bindes. Nedstrøms stegene klones fra forrige generasjon med
operators/clone_stage.py i run-roten (budget → prepare → admit + commit → bind →
launch, ett steg om gangen): COMPLETE_M1, INPUT_VERIFICATION, INPUT_VIEWS,
POSTBUILD_SPECIALIST, BASE_NORMALIZATION, SUMMARY_NORMALIZATION,
INPUT_TENSOR_AUDIT, NATIVE_INPUT_INDEX, så steg 5 FRESH_INITIAL_BASELINE.

Den opprinnelige planen for forsøk 003: run-root /home/andre2/GX1_RUNS/NATIVE_D1_REBUILD_20261011_003 med
PLAN.json lik configs/research/NATIVE_D1_REBUILD_20261011_003.json (run-id
HISTORY2009W_NATIVE_D1_20261011_003, samme vinduer, pair-/squeeze-/M1-
justeringsinputs og 64 800 s kjernetak som den aksepterte generasjonen),
launch_core.py bundet i policyblokken native_d1_rebuild_20261011_003. Lanseringen
fryser HEAD og NEXT_RUN_POLICY.json for hele kjernekjeden; ingen commit før
terminal. Forrige kjernekjede tok 22 130 s. Etter kjernen følger COMPLETE_M1/
inputverifisering, fysiske views og normalisering (132 kontekstfelt), og fersk
nullstegsbaseline (steg 5), hver under egen binding.

## Nedstrøms steg etter omstart (12.10.2026)

Omstarten ved den grønne kjernegrensen er verifisert (HOST_BOUNDARY_001/
POST_REBOOT.json): fysisk reboot, WSL-boot 168c05bf…, kilde 1810bc35 uendret
og rent arbeidstre, alle policybundne kjerneartefakter og de tre splittene
intakte; ingen trening. Det delte CPU-budsjettet for nedstrøms stegene er
PREPARATION_CPU_001/BUDGET.json (64 800 s fra 09:05 UTC 12.10 til 03:05 UTC
13.10; forberedelse og nedetid forbruker det, ingen substage-reset).

COMPLETE_M1_001 feilet lukket i CLOCK_PREFLIGHT etter 17 s: det klonede
preflight-skriptet pinnet forrige generasjons hasher (plan-identitet). Lukket
som complete_m1_stage_failed_001, alt bevart. Kloneren fikk hash-literal-remap,
og COMPLETE_M1_002 (251ce180) terminerte GRØNT 10:39 UTC etter 5 579 s med exit 0 og uendret kilde:
M1-featureflaten COMPLETE_M1/m1_feature_base.parquet har 5 523 147 rader, samme
populasjon som den aksepterte generasjonen; eierens vedtak er «M1 owner complete,
independent input oracle still required». Registrert som complete_m1_stage
COMPLETED_ONCE.

Kloneren operators/clone_stage.py er revidert (revisjon 2, sha b207ebfd…) før
neste steg: stage-aliaser for kryssreferanser (COMPLETE_M1_001 → 002,
COVERAGE_REVIEW_001 → 002); en TERMINAL.json-binding regnes som manglende hvis
exit_code ≠ 0 eller kilden endret seg; argv-par «--x[-json] STI --x-sha256 HASH»
rehashes fra den nye stien (test-seal, train-/val-manifest og -parquet
verifisert); validerings-launcherens fristsjekk er bundet til budsjett-
kvitteringen i stedet for «kjernestart + 64 800 s» (samme konvensjon som alle
senere stage-launchere), og planen binder prerequisites.cpu_budget; stale
hasher fra forrige generasjon feiler lukket; ops dryrun og complete. Den første
COVERAGE_REVIEW_001 i run-roten var forberedt mot den feilede
COMPLETE_M1_001-terminalen og er merket SUPERSEDED.json (ikke admittert, ikke
slettet); COVERAGE_REVIEW_002 erstatter den. INPUT_VERIFICATION_001 (LABEL_COVERAGE →
COMPLETE_M1_ORACLE → POST_REBUILD_READINESS) og COVERAGE_REVIEW_002 er forberedt
med bindinger verifisert mot den nye generasjonen (M1-flate, manifest, terminal,
test-seal, train-/val-manifest og -parquet) og admittert i policyen; bind og
lansering følger denne committen, én tung CPU-jobb. Forrige generasjon brukte
275 s på valideringen.

INPUT_VERIFICATION_001 (ba830e88) terminerte GRØNT 10:47 UTC etter 337 s med
uendret kilde: LABEL_COVERAGE er diagnostikk, ikke inputaksept
(COVERAGE_REVIEW_002/COVERAGE_SELECTION.json); COMPLETE_M1_ORACLE besto
delporten for den komplette M1-flaten uten å innrømme helhetlig input eller
modell; POST_REBUILD_READINESS gir «klar for post-rebuild-review»
(CHAIN/post_rebuild_readiness_001). Registrert som input_validation_stage
COMPLETED_ONCE; label_coverage_stage og complete_input_oracle_stage er
konsumert som faser av samme steg. INPUT_VIEWS_001 (fysiske inputviews,
producer 10G; pilotroten LEARNING_PREPARATION_001 opprettes av produsenten)
er forberedt med alle forutsetninger bundet til den nye generasjonen og
admittert som physical_input_views_stage; forrige generasjon brukte 260 s.

INPUT_VIEWS_001 (911c8ccb) terminerte GRØNT 10:53 UTC etter 332 s med uendret
kilde: fysiske M1-views og ENTRY_WINDOW publisert gjennom eieren under
LEARNING_PREPARATION_001 (652 552 TRAIN- og 70 880 VAL-Entry-rader; 4 884 638
og 382 744 M1-rader), vedtak «views komplette, fersk normalisering gjenstår».
Registrert som physical_input_views_stage COMPLETED_ONCE.

Post-build-auditene planlegges slik den aksepterte generasjonens bevis faktisk
ble til, ikke slik den feilet: der kjørte POSTBUILD_REVIEW_001 sekvensauditene
for TRAIN/VAL og deretter spesialistauditen i én 4G-prosess, ble OOM-drept i
spesialistdelen etter beståtte sekvensaudit, og POSTBUILD_SPECIALIST_001 kjørte
spesialistauditen alene i producer10G. Her er POSTBUILD_REVIEW_001 klonet som
bare sekvensauditene (samme eier, audit 4G; spesialistkallet og dets
outputmappe fjernet fra produsent og produsentplan), og POSTBUILD_SPECIALIST_001
klones etterpå med den kanoniske spesialisteieren i producer10G uten
feilhistorikk-forutsetninger. Ingen algoritme, eier, klasse eller input er
endret. Kloneren er revidert (revisjon 3, sha bundet i policyblokken): gamle
generasjonshasher høstes fra alle bindinger i de aksepterte stage-planene, så
argv-hasher uten stipar (spesialistauditens --train-parquet-sha256) remappes
eksakt; avgrensede skriptpatcher og plan-slettinger; admit kopierer aldri
feilhistorikk. POSTBUILD_REVIEW_001 er forberedt og admittert som
postbuild_source_audit_stage; forrige generasjon brukte 488 s inkludert den
feilede spesialistdelen.

POSTBUILD_REVIEW_001 (23244b15) terminerte GRØNT 11:04 UTC etter 563 s med
uendret kilde: sekvenskilde-rekonstruksjon PASS for TRAIN (652 552 rader) og
VAL (70 880 rader) mot de bundne parquet-/manifesthashene. Registrert som
postbuild_source_audit_stage COMPLETED_ONCE. POSTBUILD_SPECIALIST_001
(kanonisk spesialisteier, producer10G, uten feilhistorikk-forutsetninger) er
forberedt og admittert som specialist_audit_stage. Forberedelsen skjedde i to
trinn: kloneren skrev plan og launcher, men dens andre pass avviste den
allerede omskrevne nye hashen (klonerfeil, rettet); de skrevne filene ble
deretter bevist like en fersk transformasjon av den aksepterte planen uten
manglende forutsetninger før admisjon, ingenting slettet eller omskrevet.
To klonerrettelser er bundet i policyblokken: identitetsmappinger regnes ikke
som stale (de fysiske M1-views er bitidentiske med forrige generasjon fordi de
bygges fra den uendrede kvotekilden), og andre pass godtar nye hasher.

POSTBUILD_SPECIALIST_001 (f2633631) terminerte GRØNT 11:08 UTC etter 72 s med
uendret kilde: åttefamilie-spesialistauditen PASS uten feil på 315 signalfelt,
med sekvensbevisene fra POSTBUILD_REVIEW_001. Registrert som
specialist_audit_stage COMPLETED_ONCE; policyblokken postbuild_review binder
begge sekvensauditene og spesialistauditen. BASE_NORMALIZATION_001 (fersk
basenormalisering tilpasset én gang på hele fysiske TRAIN, producer 20G,
VAL-/TEST-fit null) er forberedt med alle forutsetninger og inputs bundet til
den nye generasjonen og admittert som base_normalization_stage; forrige
generasjon brukte 444 s.

BASE_NORMALIZATION_001 (0f650cef) terminerte GRØNT 11:18 UTC etter 585 s med
uendret kilde: basenormaliseringen er tilpasset én gang på hele fysiske TRAIN
før sampling, 652 552 Entry-rader, 955 670 unike M5-inputrader og 3 978 505
unike M1-inputrader, samme populasjon som den aksepterte generasjonen; VAL- og
TEST-fit er null (LEARNING_PREPARATION_001/BASE_NORMALIZATION/normalization.json
og populasjonsvitnet). Registrert som base_normalization_stage COMPLETED_ONCE.
SUMMARY_NORMALIZATION_001 (fersk TRAIN-summary-fit og splitbindinger uten ny
basefit, producer 20G) er forberedt med alle forutsetninger bundet til den nye
generasjonen og admittert som summary_normalization_stage; forrige generasjon
brukte 313 s.

SUMMARY_NORMALIZATION_001 (01bfc4c1) terminerte GRØNT 11:25 UTC etter 368 s med
uendret kilde: fersk TRAIN-summary-fit og splitbindinger publisert under
LEARNING_PREPARATION_001/FINAL_BINDINGS (komposittnormalisering, sekvens- og
førstetilstandsbindinger for TRAIN/VAL, sampler-kandidater) uten ny basefit;
VAL-/TEST-fit null, 652 552/70 880 Entry-rader. Registrert som
summary_normalization_stage COMPLETED_ONCE. INPUT_TENSOR_AUDIT_001 (faktiske
normaliserte NumPy-/PyTorch-input og aliasparitet, audit 4G, ingen
modelladgang) er forberedt og admittert som input_tensor_audit_stage; forrige
generasjon brukte 333 s. Kloneren fikk en plan-hook for NATIVE_INPUT_INDEX
(økonomi-identiteter beregnes fra denne generasjonens artefakter, den
aksepterte generasjonens kildeendringsgjennomgang beholdes ordrett) og skriver
ingenting når en forutsetning mangler.

INPUT_TENSOR_AUDIT_001 (e78ff5fc) terminerte GRØNT 11:34 UTC etter 470 s med
uendret kilde: faktiske normaliserte NumPy-/PyTorch-input består på hele
TRAIN/VAL med 315 signalfelt og 132 kontekstfelt, rå-/NumPy-/Torch-aliasbit
identiske på alle rader, ingen modelladgang (INPUT_PARITY.json). Registrert som
input_tensor_audit_stage COMPLETED_ONCE. NATIVE_INPUT_INDEX_001 (hele fysiske
TRAIN-/VAL-indeksen og økonomimetoden, producer 20G) er forberedt med
økonomi-identitetene beregnet fra denne generasjonens admisjon, design,
sampler-kandidater og den uendrede kostautoriteten, kildebindingene til de
sju uendrede økonomieierne rekomputert, og den aksepterte generasjonens
kildeendringsgjennomgang beholdt ordrett (klonerens andre pass gjenbruker nå
stage-hookene, så historikkfelt ikke rehashes); admittert som
physical_input_index_stage. Forrige generasjon brukte 544 s.

NATIVE_INPUT_INDEX_001 (d5614cd7) terminerte GRØNT 11:47 UTC etter 671 s med
uendret kilde: hele fysiske TRAIN- (652 552) og VAL-indeksen (70 880) og
økonomimetoden er publisert under LEARNING_PREPARATION_001 og
ECONOMIC_METHOD, uten modelladgang, sampler-valg eller læringsbevis.
Registrert som physical_input_index_stage COMPLETED_ONCE. Dermed er alle
åtte nedstrøms inputsteg grønne for den nye generasjonen.

Steg 5, FRESH_INITIAL_BASELINE_001, er forberedt som klon av den aksepterte
FRESH_INITIAL_BASELINE_003 (samme v10-funksjon, seed 20260911, batch 16, 513
forwards, ingen optimizer, producer 20G) på denne generasjonens inputs:
315 signalfelt med den niende D1-kontekstfamilien og fersk normalisering.
Måles på de samme faste 4096 TRAIN- og 4096 CONTROL-tidsstemplene fra
læringsstudien (uendrede forfedre); sammenligningsinputs peker bevisst på
den aksepterte generasjonens baseline-observasjoner (gammel run-root, aldri
substituert), slik at endringen som måles er input-flaten, ikke funksjonen.
Budsjettet er dagens delte vindu (PREPARATION_CPU_001). Admittert som
fresh_initial_baseline_stage_v10; forrige generasjon brukte 342 s.
Nullstegsprediksjoner er ikke læring eller edge.

FRESH_INITIAL_BASELINE_001 (86a0551c) terminerte GRØNT 11:54 UTC etter 385 s med
uendret kilde: 9 661 087 parametre (seq-/snap-input 315, kontekst 132 + 1
kategorisk; forrige generasjon 9 637 663 på 254 felt), 512 ONLINE-forwards og
én TARGET-paritetsforward med alle hoder bitidentiske, null optimizersteg,
producer20G. Koordinatene er de samme 4096/4096 radene som læringsstudien og
den aksepterte baselinen (ingen manglende tidsstempler); målene er bitidentiske
med forrige kohort. TRAIN argmax LONG/SHORT/FLAT 1055/123/2918 (akseptert
generasjon 1878/733/1485), 2123 av 4096 rader endret argmax mot den aksepterte
baselinen, største prediksjonsendring 3,2 bps; CONTROL 779/173/3144 (1779/830/
1487), 2184 endret, 3,5 bps. Side-MSE ≈908/872 TRAIN og ≈1869/1873 CONTROL,
kontrast-MSE 3 398 og 7 363, praktisk lik forrige generasjon: utrent
initialtilstand med annen input-dimensjon, ikke læring eller edge. Registrert
som fresh_initial_baseline_stage_v10 COMPLETED_ONCE; budsjettet
PREPARATION_CPU_001 er lukket etter målet (ca. 2,9 av 18 timer brukt).
Dermed er steg 3, 4 og 5 i den native D1-bølgen ferdige. Steg 6 er ikke
autorisert: en trent native modell mot den frosne D1-ridge-journalen krever
egen kontrakt, eksakt recipe-binding, GPU-campaign-admisjon under eksisterende
vakter, budsjett og operatørvedtak.

## Native D1-bølge: steg 1 og 2 målt 11.10.2026 (kveld)

Design bundet i docs/NATIVE_D1_CONTEXT_HORIZON_DECISION_20261011.md. Brukeren
bestilte steg 1 og 2; begge er kodet, testet og målt på ekte hashbundne bytes.

Steg 1, D1-kontekst-eier (gx1/features/d1_macro_context_v1.py, 83c8d4f5): 61 felt
(4 COT, 44 panel, 13 drivere) joines as-of til hver M5-beslutningstid med
forskningseierens tilgjengelighetsregler; kildeparsing importeres fra
forskningseieren, ingenting dupliseres, ingenting er innrømmet i native
signalmanifest. Audit D1_MACRO_CONTEXT_AUDIT_20261011_001 (producer 8G): bitidentisk
med journalpanelet ved 4646 D1-beslutningstider før forseglingen;
hele M5-klokken 1,216,924 rader 2009-06..2026-07 med 1,211,069 komplette fra
2009-06-29; brute-force-tilgjengelighet på 2 000 tilfeldige rader uten avvik; det
nyhentede OANDA-panelet er lik forrige fangst før forseglingen for alle 11 instrumenter.

Steg 2, mål på handelsdager (gx1/contracts/entry_observed_market_horizons_v1.py,
49facde6): eksplisitt etterfølger til entry_observed_market_v1 med 1/5/20
observerte sesjoner på UTC-sesjonsklokken (sesjoner uten barer hoppes over, exit
ved samme sesjonsoffset ellers sensurert), 20 d er beslutningsautoritet, kost
gjennom v1-økonomien over faktisk medgått tid, eksakte batchnøkler per horisont.
Audit ENTRY_HORIZON_CLOCK_AUDIT_20261011_001 på native M1-tape, alle 70,922 M5-
beslutninger i 2025: gyldige 70,385/70,324/70,198 ved 1/5/20 d, sensurert
262/323/449; medgått tid median 1/7/28 dager, maks 4/10/32 (helger og
helligdager hoppes over, fredag gir 3 dager ved 1 d); alle exits før forseglingen.
Brutto long 2025 ved 20 d: snitt +451 bps, sd 474 (beskrivende, 2025 var et
sterkt gullår). Ikke koblet til trener eller builder.

Steg 3–6 er ikke startet: signalmanifest med niende familie og spesialisteier,
normalisering og konstruktormetadata (3), datasett-rebuild (4), fersk
nullstegsbaseline (5), recipe/budsjett/trening (6). Hvert steg krever egen
binding og budsjett. training_enabled = false.

## Gjort og akseptert

Featureauditen på 569d0054 reviderte alle 254 lokale signalfelt, 71 eksakte
kontekstaliaser, ett sesjonsfelt og 190 MTF-felt. Fire funksjonsfeil og misvisende
metadata er rettet. 141 fokuserte tester besto; metode, hvert felt og avgrensninger
står i [docs/FEATURE_AUDIT.md](docs/FEATURE_AUDIT.md) og feltregisteret ved siden av.

Ny generasjon: HISTORY2009W_FEATURE_REPAIR_20261010. Korrigerte MTF-/M5-/M1-felt,
berørte labels og immutable kildebindinger er publisert gjennom eksisterende eiere.
Alle 5 523 147 komplette pre-TEST M1-rader og 392 143 437 aliasverdipar er kontrollert.
Alle 652 552 TRAIN-/70 880 VAL-sekvenser og snapshots er rekonstruert mot korrigert M5.
Kanonisk readiness, spesialistruting og hele fysiske TRAIN-/VAL-indeksen består.

Base-normalisering er tilpasset én gang på hele fysiske TRAIN før sampling:
652 552 Entry-rader, 955 670 unike M5-inputrader og 3 978 505 unike M1-inputrader.
Fersk TRAIN-summary-fit og splitbindinger er publisert uten ny basefit.
VAL-/TEST-fit er null. Faktiske normaliserte NumPy-/PyTorch-input, bitlike aliaser,
kausale MTF-ruter og Entry/Exit-førstetilstand består.

FRESH_INITIAL_BASELINE_002 (25cde83c, pre-v10-funksjon) og FRESH_INITIAL_BASELINE_003
(83adab26, v10-funksjon) er begge målt på de samme 4096 TRAIN-/4096 CONTROL-
koordinatene med bitidentiske targets; 003 er gjeldende funksjonsbaseline (tall i
avsnittet om v10-baselinen ovenfor). Initialprediksjoner er ikke læring eller edge.

Samlet aksept og uavhengig kontroll av metrics, koordinater, tilstandsbytes og ti
vellykkede terminalkvitteringer:
/home/andre2/GX1_RUNS/FEATURE_REPAIR_REBUILD_20261010_001/ACCEPTANCE_REVIEW_001/RESULT.json
Alle kjørebevis og originale avvik ligger i samme run-root.

Brukerbestilt omstart etter ferdig baseline er verifisert: Windows boot
10.10.2026 21:51:28 UTC (23:51 norsk tid), ny WSL-boot og signert GPU-telemetri.
Maskinens prosjektjobber, GPU-beregninger og prosjektlåser var ledige før omstart.
Alle 160 native kildebindinger og 30 sentrale filer, inkludert initialtilstandens
bytes, er uendret. Fsync-publiserte kontrollkvitteringer er intakte. Native
treningsoppgave er deaktivert; omstarten startet eller autoriserte ingen trening.
Bevis: samme run-root, HOST_BOUNDARY_002/POST_REBOOT.json.

## Bevart avvik og grenser

Trygg Windows-/WSL-omstart ble gjennomført mellom core og komplett M1. Tre ferske
kontrollfiler uten fsync var tomme etter omstart; tomme originaler er bevart,
observasjoner gjenopprettet fra Mac og atomisk publisering brukt videre.
POSTBUILD_REVIEW_001 fikk OOM ved feil 4 GB-klasse etter beståtte komplette
sekvenskontroller. Kun uferdig spesialistaudit ble kjørt i eksisterende producer10G.
Første baselineoppstart manglet eksplisitt CPU-synlighet og stoppet før konstruksjon.
Etterfølgeren skjulte GPU før PyTorch-import; øvrige metoder og vakter var uendret.

Swing-forsøk 001 (instrumenthelligdager) og 007 (eieren krevde D1-lane) feilet ved
start og er bevart uten resultat; 013 feilet lukket i porteføljefamilien (ridge-
porteføljen gikk til null egenkapital fra 2017-04), og dets selektive tall er regnet
post hoc fra lagrede prognoser med eierens funksjon. GLD-beholdningen fra Internet
Archive slutter 2026-02-13; carry-forward holdt den flat til 2026-06-30 i forsøk
008–015 og forward-sjekken 010 (gld_log_change21D = 0 der). Arbeidsboken fra SPDR
(data til 2026-10-08) brukes av journalen. Kommandomeldingen 18856469 oppgir feil
ubetinget referanse (+29 bps); riktig er +63 bps.

Retention av de feilede NATIVE_D1-kjerneforsøkenes event-røtter (001: 73 MB,
002: 9,6 GB; kvitteringskatalogene beholdes) ble nektet av retention-eieren
12.10: planen stoppet lukket på «unresolved authority directory manifest» for
run-root-stagekatalogen current_work.runtime_root pekte på; stagekataloger
har PLAN.json/TERMINAL.json, ingen av retention-kontraktens
katalogmanifestnavn. Eiernektelsen overstyres ikke. runtime_root peker nå på
den gjeldende generasjonens NATIVE_INPUT_INDEX_001; løsningen krever en egen
avgrenset endring med tester (kontraktens katalogmanifestregel eller
policyens katalogpekere) før plan → godkjenn → utfør. Ingenting er slettet.

Ukjente rådatagap er fortsatt ukjente; ingen imputering eller antatt markedslukking.
Deklarerte kostvilkår er ikke bevis for historisk kostsannhet. CONTROL er gjenbrukt
utviklingsdata. Ingen Exit-rollout, all-trades-/åpen-posisjonsøkonomi eller profitt
ble målt. Nye baselinekoordinater gir ingen native trenings-/sampler-/GPU-autoritet.
Tidligere benchmark- og treningsbevis tilhører sin gamle generasjon.

## Læring og neste beslutning

Forrige læringsstudie ga REJECT_ENTRY_QUALIFICATION: CONTROL4096 valgte 4095 FLAT
og én SHORT. Retnings-MSE 7357,64 mot konstantens 7360,93 ga ingen statistisk påvist
forbedring. v10-baselinen endrer ikke denne konklusjonen. Flere epoker på
95-minuttersmålet er ikke neste steg. D1-kontekst- og horisontbølgen (vedtak
11.10 kveld) er bygget til og med steg 5; neste beslutning er steg 6: trening
på 1/5/20-handelsdagers-målene med eksakt recipe, 3 seeds og forhåndsregistrert
stoppregel, og sammenligning mot D1-ridge-journalen på samme beslutningsdager.
Brukeren vedtok 12.10 å kjøre dette som minimum (GX1_RULES, vedtak 12.10):
Entry-only på 20-dagersmålet, full modell, ett seed, smoke 256 → fit 1 024 →
kurve 4 096 steg. Planen står i docs/NATIVE_D1_MINIMUM_LEARNING_PLAN_20261012.md
og policyblokken native_d1_minimum_learning_20261012: P0 kode/tester (neste),
P1 CPU-forberedelse (design, sampler-kapasitet, preprocessing, koordinater,
metadata) i samme run-root, P2 GPU-campaign med brukerens fysiske omstarter,
P3 gjennomgang. Feature-kutting er ikke åpnet.

P0 er kodet og testet 12.10: horisont-læringsdesignet (recipe-nøkkel
entry_observed_market_horizons, samme prefiks-/kilde-/policy-/kostkontroll som v1
gjennom en delt kjerne), datasettbinding som emitterer de tre eksakte
batchnøklene og admitterer rader komplette ved alle horisonter til og med den
målte cutoffen, trenerdispatch (20-dagersnøkkelen i tapet, fase
entry_only_observed_horizon_20d i objektivkontrakt v12, eligible rader må være
admittert), observatør som rapporterer MSE og rank-IC per horisont, og
læringsstudie v2 (315/132 felt, fit 1 024, kurve 4 096 i ett vindu, kontroll
trukket fra komplette VAL-rader). Målt på den nye generasjonen: 644 255 av
652 552 TRAIN-rader og 65 575 av 70 880 VAL-rader er komplette ved alle tre
horisonter; siste komplette beslutning 2025-05-02 19:20 UTC (TRAIN) og
2026-06-02 23:55 UTC (VAL); sensurering er spredt (3 284 / 308 rader før siste
komplette), så eligibilitet er kompletthet, ikke en tidsgrense alene; 20 dagers
elapsed er 28–32 kalenderdager. Fokuserte og brede regresjoner er grønne.
P1a er kodet og testet: koordinat-kontrakten godtar versjon 2 der de
eligible TRAIN-radene er den frosne delmengden med komplette utfall ved alle
horisonter (eligibility-blokk bundet til horisont-designet og den målte
cutoffen), og koordinat-produsenten publiserer delmengden og den native
samplerens fulle rekkefølge filtrert til den, nøyaktig slik treneren
reproduserer rekkefølgen. P1 er i gang under et nytt CPU-vindu (PREPARATION_CPU_002, 64 800 s):
NATIVE_PREPARATION_001 er klonet fra V38-generasjonens
PRE_SMOKE_READINESS_001 med run-root-generatoren
operators/make_minimum_learning_stages.py (V38-substitusjoner, patcher og
hooks på toppen av clone_stage.py): designkompletteringen fra den nye
prosedyredeklarasjonen configs/research/NATIVE_D1_HORIZON_LEARNING_DESIGN_20261012.json
(samme initialisering, budsjett og referansepolicy som V38; mål, utvalg og
tidskontrakt omskrevet for 20-dagersmålet, 315/132 felt), CONTROL256 trukket
fra de komplette VAL-radene til og med den målte cutoffen, smoke-planen
fjernet (planen er operatørdokumentet), retention-review lesebare. Stage-
lokale kontroller (2 + handover-grensen) er grønne og bundet. Første klon
(NATIVE_PREPARATION_001) stoppet i launcherens preflight før START: V38-
prosjektplanen kalte nådeperioden build_termination_grace_seconds, den nye
heter termination_grace_seconds; ingen fase kjørte, steget er merket
SUPERSEDED og lukket som native_preparation_stage_failed_001.
NATIVE_PREPARATION_002 (launcher patchet) stoppet i produsenten etter 7 s:
malens «reused_completed_owner_hashes» bar V38-tidens hasher av 28 repo-eiere;
lukket som native_preparation_stage_failed_002 med kvitteringer.
NATIVE_PREPARATION_003 binder de 28 eierne ved gjeldende HEAD (git diff fra
kjernekjedens commit er tom for alle) og godtar kjerneterminalens
training_started-flagg; kontroller grønne, admittert som
native_preparation_stage, og terminerte GRØNT 13:33 UTC etter 7 s: det frosne
designet er komplettert uten modell; CONTROL256 er trukket fra de 65 575
VAL-radene med komplette utfall ved alle horisonter til og med 2026-06-02
23:55 UTC (308 sensurerte ekskludert, 4 997 rader etter cutoffen), 2025-06 til
2026-06 fordelt per måned; retention-review er lesebar og rapporterer samme
eiernektelse som før. Registrert COMPLETED_ONCE med design, CONTROL256 og
readiness bundet.
SAMPLER_BENCHMARK_001 er klonet fra V38 SAMPLER_BENCHMARK_002 (samme
benchmark-/launcher-skript, policynøkkel train_capacity_stage i
minimumsblokken, V38-ens «original budget»-input og stavetest fjernet,
nådeperiode-nøkkelen rettet): det bundne forslaget er bygget på nytt fra
denne generasjonens artefakter (SAMPLER_BENCHMARK_PROPOSAL_001/PROPOSAL.json:
geometri 96×315/132 med M5 16/M15 64/H1 96/H4 96/D1 252, budsjetter
32 768/65 536/131 072 ved batch 16, seleksjonspolicy fra benchmark-eieren på
hele fysiske TRAIN-populasjonen 652 552, referansearbeid fra
NATIVE_PREPARATION_003/DESIGN.json). Læringsepoken bruker den
horisont-eligible delmengden (644 255), så kapasiteten er en øvre grense.
Stage-lokalt kapasitetsbudsjett (64 800 s, ett forsøk, ingen relansering),
kontroller 15 + 63 grønne (kvitteringer før launcher-rettelsen; testene
øver bare benchmark.require_scope, BINDING hasher den rettede launcheren),
admittert som train_capacity_stage og lansert 13:35 UTC. Etter 320 målte
entry-par lå raten på 0,955 s/par mot V38-referansens 0,819 s/par tidlig i
samme kandidat (V38 slutt/tidlig-forhold 1,09/1,16/1,17 per kandidat); de tre
fulle kandidatepokene (57 344 par) projiserte til 64 287 s mot vinduets 64 505 s.
Steget ble derfor stoppet av agenten etter 921 s (FAILURE/TERMINAL exit 1,
source_unchanged, delresultater bevart, SUPERSEDED.json), lukket som
train_capacity_stage_failed_001 og aldri relansert. SAMPLER_BENCHMARK_002 er et
nytt engangskrav under eget 86 400 s-vindu (34 % margin over projeksjonen,
basis bundet i SAMPLER_BENCHMARK_PROPOSAL_002/PROPOSAL.json med n=320), samme
skript, kontroller 15 + 63 grønne, admittert som train_capacity_stage. Deretter
NATIVE_METADATA_HANDOFF_001 (SELECTED_SAMPLER + design-metadata) og
NATIVE_PREPROCESSING_001 (labels/normalisering + koordinater v2 over eligible
delmengde), begge under kapasitetsbudsjettets opprinnelige frist.
SAMPLER_BENCHMARK_002 terminerte GRØNT 13.10 05:56 UTC etter 57 822 s (exit 0,
source_unchanged): alle tre kandidater målt komplett; valgt sampler batch 16,
budsjett 32 768 transisjoner per epoke, målt CPU-epoke 7 907 s (V38: 7 276 s på
254 felt), 80 populasjonssykluser. Registrert COMPLETED_ONCE med
CAPACITY_RESULT/BENCHMARK bundet. Forsøk 1 hadde, målt i ettertid, trolig
klart sitt vindu (rateforfallet ble mindre enn V38-mønsteret); stoppet står som
konservativ projeksjon. NATIVE_METADATA_HANDOFF_001 er klonet fra V38
(policynøkkel post_capacity_metadata_stage, produsentens resultatskjema-navn
beholdt som eierens), kontroller 22 + 63 grønne, admittert; den publiserer
SELECTED_SAMPLER og det komplette designet (scope/status fra
prosedyredeklarasjonen) under kapasitetsbudsjettets frist 13:52 UTC.
NATIVE_METADATA_HANDOFF_001 terminerte GRØNT 05:58 UTC etter 5 s: SELECTED_SAMPLER
(budsjett 32 768, batch 16, direct_same_root_benchmark_v1) og det komplette
designet (status DESIGN_AND_CONTROL_IDS_FROZEN_NOT_EXECUTABLE, scope med
entry_only_observed_horizon_20d) publisert; COMPLETED_ONCE.
NATIVE_PREPROCESSING_001 er klonet fra V38 (policynøkkel
native_preprocessing_stage): precheck-produsenten er patchet til å kalle
koordinat-eieren med horizon_eligibility (prosedyredeklarasjonen som bundet
horisontdesign, målt TRAIN-cutoff 2025-05-02T19:20Z) og krever at planens
cutoffs er de målte; signalmanifest, M5-målkilde, byggecommit 8f5d39fd og de
fem måleierne er bundet fra NATIVE_D1-manifestet (uendret HEAD mot
byggecommit). Kontroller 37 + 63 grønne, admittert; publiserer LABELS_RESULT,
NORMALIZATION_RESULT og NATIVE_COORDINATES_001 (v2 over 644 255 eligible rader).
NATIVE_PREPROCESSING_001 terminerte GRØNT 06:04 UTC etter 252 s: LABELS_RESULT
(frosne TRAIN-policyer, komplett pre-TEST målklokkestøtte, CONTROL256 ok),
NORMALIZATION_RESULT (composite + first-state-bindinger, ingen refit) og
NATIVE_COORDINATES_001 (skjema v2: 644 255 eligible TRAIN-rader med komplette
utfall ved alle horisonter til og med 2025-05-02T19:20Z, 3 284 sensurerte
ekskludert; epoke-0-rekkefølge = den native samplerens rekkefølge filtrert til
delmengden; TRAIN4096 og TRAIN256-probe fra den). P1 er DONE; begge
CPU-budsjettene er lukket som konsumert. Ingen modell er kjørt.

**Eierrettelse før P2 (13.10):** tre lesere (prefix-kontrakten i
unified_exit_native_candidate_campaign_v1, treneren og full-train-launcheren)
hardkodet design.budget.planned_optimizer_steps == 256 (V38-smokens budsjett).
Det frosne NATIVE_D1-designet planlegger 1 024 steg (fit-fasen). Én opprinnelse:
`MINIMUM_PLAN_FIT_OPTIMIZER_STEPS = 1024` i gx1/contracts/entry_observed_market_horizons_v1.py,
valgt av `planned_prefix_optimizer_steps(design)` når designets scope
deklarerer entry_only_observed_horizon_20d, ellers V38-konvensjonen 256;
studie v2s faste fit-budsjett leser samme konstant. Tester: planned-steps-valget
og prefix-kontrakten (horisont/1024 godtas, horisont/256 og uten horisont/1024
avvises). «Smoke 256» i planen er fit-fasens observasjon ved steg 256; ingen
egen smoke-recipe finnes for horisontmålet.

**P2 forberedt (13.10):** læringsdesignet
configs/research/NATIVE_ENTRY_OBSERVED_HORIZONS_20261012.json (20-dagersmålet
som eneste tapsmål, 1/5 d observert, entry-only-fase, prefix bundet til det
komplette designet, preprocessing-resultatene og koordinater v2, målte
cutoffs) og studien configs/research/ENTRY_D1_HORIZON_LEARNING_STUDY_20261012.json
(fit 1 024 steg på TRAIN256, kurve 4 096 steg på de første 65 536 eligible
radene, CONTROL4096 trukket deterministisk fra de 65 575 komplette VAL-radene,
ett seed, forhåndsregistrert gjennomgangsregel) er committet. Studieroten er
/home/andre2/GX1_RUNS/ENTRY_D1_HORIZON_LEARNING_20261012_001 med
TRAIN_PROBE4096/CONTROL_POPULATION/CONTROL4096. Fit-fasen er admittert i
policy (entry_learning_study_run, run_id ENTRY_D1_HORIZON_FIT_20261012_001,
én invokasjon, 1 024 steg); recipe bygges gjennom de eksisterende gatene
(observert-scope, studie v2, full-train-recipe-validator). GPU-invokasjonen
krever fersk fysisk Windows-boot (brukeren utfører omstarten) og går gjennom
materialize_local_random_access_campaign_v2 --phase native-candidate med
eksisterende vakter. training_enabled er fortsatt false; ingen trening er
startet.

**Fit forsøk 1 feilet i trener-preflight (14.10, 0 optimizersteg).** Verten
frøs 05:17 CEST (Kernel-Power 41, BugcheckCode 0, ingen dump; ti korrigerte
WHEA-feil i september peker mot maskinvare). Windows-interop er av i
/etc/wsl.conf, så boot-fangst, oppgaveinstallasjon og omstart ble kjørt av
brukeren i Windows PowerShell (fangst transportert via %TEMP%). Kjeden var
grønn: PREPARE/PRE_REBOOT-kontroll, campaign-plan (fil 6672c1c1), oppgave
GX1NativeLearningCalibration Ready, prepare/confirm-reboot, fersk boot 494,
invocation-0001 12:29:41 UTC under capped trainer. Etter 20 min innlasting
stoppet treneren med [CANDIDATE_PREFIX_PHYSICAL_LABEL_MISMATCH]:
`_prefix_candidate_training_binding` krevde at label-bindingens rader
(TRAIN_CALENDAR, 652 552) var lik TRAIN_ELIGIBLE_PARENT_ROWS, som under
koordinater v2 er den eligible delmengden (644 255). V38 hadde de like; P1a
manglet trener-test for v2. Rettet i 528b754d: label-binding og treningsmengde
er separate bindinger, eligible ⊂ label-rader, antall bundet til
eligibility-blokken, sesjonsbindingen bærer begge; tester ende-til-ende på
ekte koordinatbytes, og sammenligningen er replikert felt for felt på
NATIVE_D1-artefaktene (regel 2h). Campaignen for forsøk 1 står i
BLOCKED_RECOVERY_RECEIPT_REQUIRED (controller skrev ingen kvittering) og
beholdes som evidens. Forsøk 2: studiespec-en peker til root
ENTRY_D1_HORIZON_LEARNING_20261012_002 med run_ids _002 og forsøk 1 bokført
under `attempts`; samme rader (byteidentiske), samme design, ny plan,
oppgaveinstallasjon og fersk boot. Vaktene som skulle melde avslutningen
hang fordi pgrep matchet sin egen kommandolinje; fremover sentinelfiler og
guard-loggens `event=exit`.

**Fit forsøk 2 fullført og gjennomgått (14.10 15:01 UTC).** Boot 495,
invocation-0001 RESUMABLE/PASS/exit 0 på 33 min (innlasting 1 130 s, 1 024
steg, 17 checkpoints à 2 s). PHASE_REVIEW (v2): kvittering, cursor, checkpoint
ved steg 1 024, epoke-rekkefølge lik koordinatene, 150 Exit-eide felt
uendret, observasjoner ved 0/64/256/1 024 på TRAIN256 med identiske rader og
mål. Målt (TRAIN-tilpasning, memorering, ingen generalisering): 20-dagers
contrast-MSE 617 310 → 204 059, rank-IC 0,07 → 0,97; handlinger L/S/F
50/10/196 → 120/119/17, ingen ties; uoptimerte horisonter 1 d 0,08 → 0,23 og
5 d 0,16 → 0,37. Mål: snitt LONG +6,6 / SHORT −63,3 bps, std 391 bps.
Kjeden virker ende til ende; ingen påstand om edge. Windows-interop er slått
på av brukeren 14.10 (wsl.conf), så fangster, oppgave, omstart og
bekreftelse gjøres av agenten heretter. Neste: kurve (4 096 steg, 65 536
rader, CONTROL4096 ved 0 og 4 096) under samme sekvens.
Kurven er admittert i policy (entry_learning_study_run, run_id
ENTRY_D1_HORIZON_CURVE_20261012_002, 4 096 steg, én invokasjon, 65 536
unike TRAIN-rader fra epoke-0-rekkefølgen, CONTROL4096 observert ved 0 og
4 096); recipe bygges gjennom gatene og campaignen settes opp av agenten
(fangst, plan, oppgave, omstart, bekreftelse).
**Kurve forsøk 1 feilet i trener-preflight (14.10 15:40 UTC, 0 steg):** ved
første CONTROL-observasjon bandt studiens kontrollvisning (CONTROL4096) målet
med v1-binderen, som avviser horisont-scope
(ENTRY_OBSERVED_MARKET_DATASET_SCOPE_INVALID). Fit-fasen har ingen
kontrollobservasjoner og traff aldri stien. Rettet i f72a430b: hjelperen
dispatcher på scope; test med horisont-scope og replikering på ekte
VAL-bytes (4 096/4 096 admittert). Kurve forsøk 2 kjører under root
ENTRY_D1_HORIZON_LEARNING_20261012_003 (run_id _003), fit-resultatet står
under _002 (phase_roots i spec-en). Forsøk 1 bokført under attempts; oppgaven
deaktivert; campaign-evidensen beholdt.

Rebuild-/baselinemålet for begge generasjoner er ferdig; engangsplaner og
budsjetter er lukket (PREPARATION_CPU_002 i den aksepterte generasjonen,
PREPARATION_CPU_001 i NATIVE_D1-generasjonen). De feilede kjerneforsøkene
NATIVE_D1_REBUILD_20261011_001/_002 med event-røtter er foreldet evidens og
ryddes bare gjennom retention-eieren (plan → godkjenn → utfør) i bakgrunnen. Valgt sampler (batch 16, målt CPU-epoke 7679 s) bindes i
recipe først i den native bølgen. Native launch er stengt.
Ingen automatisk trening, forlengelse eller relansering av konsumerte planer.
training_enabled=false; TEST er forseglet. Ingen live/paper, broker, ordre, spending,
promotion eller opprydding. Bevar én agent, én tung CURRENT-jobb, alle genuine
features, åtte familier, kausalitet, vakter, originalkvitteringer og checkpoints.
Ingen fast tapsgrense eller maksimal holdetid.
