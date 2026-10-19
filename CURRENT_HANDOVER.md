# CURRENT HANDOVER — XAUUSD M5 Entry, kompakt førsteprofil

Autoritet: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les GX1_RULES.md og kjør `bash scripts/gx1_handover.sh --check`.
NEXT_RUN_POLICY.json eier aktuell arbeidsstatus og eksakte artefaktbindinger.
Én integrator, avgrensede agenter, én capped tung jobb og fryst kilde/policy.
Originalkvitteringer bevares; konsumerte engangsplaner relanseres aldri.

## Mål og brukerens siste profilvalg

M5 Entry skal lære LONG, SHORT og FLAT fra observerbar markedstilstand.
Retning/netto endepunktsretur, signert MFE og konservativ MAE før første MFE
måles separat fra ett minutt til åtte timer; dager er støtte. Ingen Exit-
vurdering eller Exit-trening. Horisonter er ingen maksimal holdetid.
Behold lukket M5/M15/H1/H4/D1 og de rikeste26månedene med GC MBP-1.

Brukeren presiserte19.10 at første læringsrunde skal bruke MBP-1, SMC,
EMA/ATR og nødvendig kontekst, med utvalgte candlestick-/chart-patternfelt
parkert. Hele biblioteket, dataene og fullmodellens åtte familier beholdes.
D1/makrokontekst beholdes. Dette åpner én navngitt kompakt profil, ingen
HPO, ekstra treningsarm eller ekstra epoke. Egen SMC/MBP-modell/ensemble og
M30 krever senere identisk kronologisk OOF-evidens for merverdi.
Ingen felt erklæres ubrukelige eller støy uten måling.

TRAIN er [2024-05-01,2026-01-01), VAL [2026-01-01,2026-04-01), senere
utviklingskontroll [2026-04-01,2026-07-01). Alle26m er gjenbrukt utvikling.
TEST fra juli er forseglet. Nøyaktig én tidligere bestilt full nativeepoke
står ved lag etter faktiske profilbundne porter. Training_enabled=false.
Ingen broker/live/paper/ordre/spending/promotion. Nøyaktig to godkjente
practice-leseoppslag er allerede fullført; ingen gjenværende brokeradgang.

## Ferdig og målt — gjenbruk, ingen rebuild eller refit

-66GC-råfiler/892100712records og9173893prisnivårader er bevart. Identiske
  genuine Tposter dedupliseres ikke;17PTPminutter/18bok-eventbøtter sensureres.
  MBP-1 er aggregert BBO/ordercount, ikke MBO/L2/ordreidentitet.
-1205genuine GCverdier+1205separate masker er publisert. Fysiske inputs er
  signal2725/ctx-cont2542/ctx-cat1 og253felt perTF. Alle femTF er beholdt.
  Individuell feltindeks i060 binder kilder/testscope og ærlige ubeviste
  unit-/warmup-/uavhengig referanse-/genuin-forwardfelt. Ingen full individuell
  formelPASS hevdes. Makroens komplette historiske originalvintager er ubevist.
-051fysisk metadata og begge uttømmende seq/snap-kilderekonstruksjoner bestod
  på118249TRAIN/17284VAL.053full input-liveness bestod alle rader/TFfelt.
  ATR/OOD SHIFT_OBSERVED beholdes diagnostisk; ingen VAL-fit.
-055wholeTRAIN Entry-only-normalisering COMPLETE02:53:02UTC,357.914s,
  exit0/uendretfed117b1. Context118249før targetmasker; lokal96bars-union
  118800kilderader; MTF-fit118720/39638/9995/2686/683. Ingen Exit/VAL/TEST-fit.
  NORMALIZATION.jsonSHAc02ca97967a314f7dbd6c8bebc341c9782acd9e999e67e4b730db7e99e318c62.
  Transformen gjenbrukes komplett før kompakt intern gather; ingen refit.
-057faktiske native design-/ordre-/target-/koordinat-/cohortbindinger COMPLETE
  04:10:24UTC,1015.514s, exit0/uendret350303f3. TRAIN118249 gir7391batcher,
  sistebatch9. Frosne256TRAIN/256CONTROL er valgt uten outcome-utvalg.
  RESULTSHA1a4208cc06c2fe357fccfa2348c3b1f0cf9404ed98c4e141f8ad6f53970a5e4c.
  Dette binder kilder/pointere/masker; ingen ferdig læring eller økonomisk edge.
-058full konstruktormetadata COMPLETE04:55:31UTC,3.606s, exit0/uendretc6010ab6.
  MetadataSHAae3f80b9969fde9c287db643d1a2e345ec08c8e721f9375e58d6fe7379be87a9.
  Dagens fullgeometri/vinduer/ruter er bundet uten vekter/forward/optimizer.
-059COMPLETE05:02:07UTC,11.361s, exit0/uendret2b0909d1. Nøyaktig to
  practiceGET er konsumert; ingen flere oppslag/relaunch. Prospektiv
  kostautoritetSHAe93ed40451a24344f94716e1f5ebe4cd348e6c7c3120e7509d6d55827bf40522.
  Frosne258fills/finansieringsobservasjoner er gjenbrukt. Aktuelle antakelser
  er kvalifisert; historisk kostsannhet og økonomisk PASS er ikke bevist.
-060generisk kildebølge er kvalifisert og committetdca7044c:242unike PASS.
  To trainerjoins og én checkpointjoin bruker observedPLAN→DESIGN057 korrekt;
  begge identiteter beholdes. Ekte CPUfreshstate-/initialization-/measurement-
  publisering er implementert i eksisterende full_train-eier uten Dataset/
  forward/step. Original186PASS/8FAIL og repair167PASS er bevart.
  060fullprofilens runtimeutkast er uclaimet og skal ikke startes.

Faktiske immutable bevis ligger i GX1_RUNS-røttene055/057/058/059/060 og
policyen. Tidligere056/058sourcekvitteringer beholdes og gjenbrukes for
uendrede kontrakter. De beviser ingen ny kompakt modellfunksjon.

## Aktuell:061profil og064import kvalifisert;NY064CPUfreshplan klar

Profil: `compact_mbp_smc_ema_atr_clock_macro_entry_v1`, eksplisitt referanse
`{name,contract_sha256}` i recipe, konstruktormetadata, modell og state.
Ingen implicit profilerstatning; None beholder full standardfunksjon.
Kildeeier er eksisterende entry_specialist_feature_groups_v1.py.

Aktivt2622signal/2525ctx/139MTFfelt,7lokale/5MTF-familier; full fysisk
2725/2542/253 og alle055-/finite-/domene-/alias-/maskevakter før gather.
Alle1205GCverdier+1205masker og61D1/makrofelt beholdes. Lokal candle-rute
beholder74ekte MBPkanaler; familienavnet brukes ikke til blind slukking.
Lokal96bars og M5/M15/H1/H4/D1-vinduer16/64/96/96/252 og lagdybde beholdes.
Modellen bygger/forwarder bare aktive encoder-/projeksjonsruter: lokal8→7,
MTF40→25. Ingen kunstige nullfamilier. Dette er beregningsgeometri; faktisk
latency/peakminne/treningstid er ennå ikke målt.

Tre avgrensede agenter har implementert canonical profil/architecture,
hybridmodell og recipe-/konstruktørkobling. Uavhengig review avdekket én
konkret rapporteringskobling i både dynamicvalidator og train_epoch.
Samme eksisterende gateeiere bruker nå profilbredde7/25/(5,139), med full-
None-defaults bevart. Loss, optimizer og Exit-matematikk er uendret.

061har380unike godkjente source-/syntetiske modellcaser. Original377PASS/
3FAIL/0ERROR/SKIP på85.810s er bevart; alle11kompakte modellcaser bestod.
To train_epoch-fixtures fikk fryst Exitparameter. En gammel contract-prep-
test forventet feilaktig totalauditPASS; nå bevises korrekt routing/arkitektur
og eksakt failclosed legacy LIVE-only-grense separat. Alle2411conditional-
felt er eksplisitte:1281strukturelle masker inkl75GCverdimasker og1130andre
observed-masked GCverdier. Binærfixture og JSON-representasjon er korrigert.
Tre smale repairs32PASS/1FAIL,26PASS/1FAIL og1PASS/0FAIL er bevart; alle
sju produksjonsfiler var identiske gjennom repairs.377opprinnelige PASS og
senere berørte caser er slått sammen per eksakt JUnit-ID, ingen uavklart feil.
SOURCE_QUALIFICATION_FINAL_RECEIPT.jsonSHAb623a3de5b6d570773febb5ad0b11c4437f2c37f5594180a4ff287b233500930.
SOURCE_REVIEW.jsonSHAaf35febdf458e53a217c931bf5763d1d1fa10fae41c167f3a852b2d7237bdb06.

062er faktisk avsluttet FAILED_NO_ADMISSION06:04:35UTC etter2.530s, kilde
uendret3a76a7b7. Ingen metadata eller modellstate ble publisert; ingen modell,
Dataset, forward, step, normfit, Exit/TEST/broker. Engangsplanen er konsumert
og skal aldri relanseres. TERMINAL.json er bindende i policyen.

Målt blokkering: dynamisk recipe-env er korrekt fasevalidert før full_train-
import, men trainerens importvakt1388–93 tillater bare legacy entry_policy_pnl
og avviser vedtatt fixed-final-ONLINE-monitor. Kun denne konkrete import-
faseregelen er rettet med kun de to sourceeide kjente enumverdiene. Strenge
legacy-CLI-/native-phase-/recipe-/checkpoint-/sessionporter er bevart.
064bestod55aktuelle recipecaser inkl6fresh-prosessimport-/feilfasecaser på
104.300s, exit0/uendret kilde. Legacy og dynamic importerer; ukjentmonitor,
begge feilphasekombinasjoner og dynamicmonitor i legacyCLI avvises faktisk.
Alle funksjoner/klasser og øvrige modultoppnoder er AST-identiske; kun én
import-If er endret.061s380uendrede funksjonscaser gjenbrukes med separat
FUNCTION_AST_REUSE.json. Ingen miljøomgåelse eller relaunch062.
NY064CPUfreshplan er klar og uclaimet med egen policy/sourcebinding; bruker
samme kvalifiserte data/profil og future063/BUNDLE. Ingen ny treningsarm.

Ingen genuin kompakt modellforward, fresh modellstate, initialbaseline,
optimizersteg eller nativeepoke er gjennomført. Ingen læring/generaliserings-
eller edgepåstand. Separate kilde-/modelltester bruker syntetiske inputs.

## Neste nødvendige steg

1.064importguard er kvalifisert med55PASS og061funksjonsbevis gjenbrukt.
   Fullfør obligatoriske hooks/syntaks/stalepath/diffcheck og commit/push.
   BindNY064CPUfreshplan til eksakt source/policy/owners og kjør én capped
   invokasjon. Ingen relaunch058/059/062 eller gammelt060fullprofilutkast.
2. Etter faktisk vellykket064terminal: bind063profilbundet initialbaseline på ekte
   data. Mål alle aktive felt-/value-mask-/MTF-stier, konsum/paritet og
   Entry-only optimizer/resume. Ingen Exit-grad/vektendring; samme frosne
   kohorter/queries/per-horisont masker. Mål faktisk latency og peakminne.
3. Først etter faktiske maskinvare-/input-/funksjonsporter og eksakt ny
   campaignbinding kan én full7391steps nativeepoke starte. Ingen ny epoke,
   VAL-refresh, top-K eller økonomisk checkpointutvalg. Siste ONLINE beholdes.

Valgte og avståtte LONG/SHORT/FLAT må rapporteres med kvalitet og dekning.
Gatevekter alene forklarer ikke beslutningen; alltidFLAT er ikke selektivitet.
Targetmasker og fremtidige sensurgrunner brukes aldri som modellinputs.

## PC og grenser

18.10-PC-omstart er utført: shutdown exit0, ny Windowsboot18:44:29.500UTC,
autentisert WindowsSSH/Ubuntu og12bevarte hasher.034/REBOOT_CONFIRMED.json,
SHA39623175a29098a78dc8599396d3efc7b24e178a91bcbc77da6826e49ec46c61,
overstyrer eldre PENDING. Ikke utfør samme omstart på nytt. Periodiske og
før-invokasjon-maskinvareporter gjelder fortsatt separat.

Teknisk ferdigstilling, læring, generalisering og økonomisk edge er separate.
DATA/RUNS ryddes bare av retention-eieren. Ingen Pythonfil er bevist slettbar
ved full referanse-/import-/CLI-/test-/eierkontroll. Historikk finnes i Git.
Metode/bruksanvisning: docs/NATIVE_LEARNING.md; feltbevis: docs/FEATURE_AUDIT.md.
