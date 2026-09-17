# GX1 — gjeldende overlevering, 2026-09-17

**Den frosne kandidaten forkastes for utvidelse. Begge juni256-målinger er ferdige.**
Baseline velger256FLAT og0Bps. Kandidaten velger212LONG/15SHORT/29FLAT og får
−15,7477Bps per inngangsmulighet, inkludert kostnader og fire åpne posisjoner.
Dette er uavhengige inngangsmuligheter, ikke porteføljeavkastning. En separat
én-posisjonsreplay gir−1305,0322Bps i fast notional over24 handler, med én åpen.

Kandidaten kopierer Entry-læreren bedre, særlig LONG, men Exit-feilen blir
større for begge retninger i alle fem ukene. LONG Exit-MSE1408,83→1625,02,
SHORT1410,80→1553,61; begge verre enn TRAIN-konstanten. Korrelasjon med frosset
referanseutfall er0,035 og−0,006. LONG-prediksjonens middel er+9,10Bps mot
målets−5,07; SHORT−5,23 mot+5,02. Det er fortsatt skjev verdsetting.

Entry LONG-MSE116,44→26,66, SHORT30,23→32,34. Lærerregret5,73→0,96Bps
betyr bedre lærerimitasjon, ikke bedre markedshandel. TRAIN-gevinst overføres
ikke robust til juni. Juni er gjenbrukt utviklings-VAL; TEST forblir forseglet.
Ingen lærings-, generaliserings- eller lønnsomhetsport er bestått.

Begge native målinger har guardPASS/trainer0/observer0 og null optimizersteg.
Baseline brukte89c; kandidaten866e2e33 etter en nødvendig rettelse av bare
batchpresisjonsvakten. Prediction/rollout-AST utenom vaktfunksjonen var identisk.
Vekter, mål, utvalg og lærerbindinger er uendret og kontrollert eksakt like.
Første kandidatforsøk på boot450 er bevart som feilet preflight; ikke et resultat.
Kandidat-retry på boot451 er ferdig. Windows-tasken er deaktivert; ingen aktiv jobb.

NEXT_RUN_POLICY har stengt de brukte evalueringsunntakene. Ingen ny trening,
modell-forward, tuning, full epoch/full VAL eller TEST er åpnet. Kontrollen av eksisterende TRAIN-cache mot native VAL er nå ferdig i
omfanget beskrevet nedenfor. Ingen nytt fit på juni.

Se docs/FROZEN_NATIVE_VAL_REVIEW_20260917.md og maskinmålingen
handover_snapshot/FROZEN_NATIVE_COMPARISON_20260917.json. Alle originale
checkpoints, planer, resultater og den mislykkede kjøringen er bevart.
GitHub-push venter fortsatt på det allerede stilte godkjenningsspørsmålet.

## Kontrollert etter avvisningen

Alle40 lagrede TRAIN-inputcacher er hashkontrollert. Metadata, gammel base-
normalisering, child-kontrakt, lifetime-normalisering, checkpoint, boundary-
lærer og frosne koeffisienter samsvarer med native VAL. Samme collator, forward-
eier og Q_mu-beregning brukes. Ingen mismatch er funnet i dette kontrollerte
omfanget; det er ikke kjørt en ny GPU-replay av identiske TRAIN/VAL-inputs.

State0 mangler ikke:275 trente og67 separate TRAIN-Entries er målt der.
Separat state0 LONG-MSE516,63→487,73; SHORT515,90→544,83. Den samlede TRAIN-
gevinsten skjulte svakere SHORT ved selve inngangen. Ingen nytt fit er gjort.

De allerede lagrede juni-prognosene er identiske før/etter readout-endringen,
og taper mot nullprognosen i både MSE og MAE ved5/25/60/120 nominelle minutter.
120-minuttersprognosen har korrelasjon0,0209, snitt+5,09Bps mot faktisk−9,11
og42,97% riktig retning. Svak overføring er derfor påvist også uten Exit-læreren.
Dette beviser ikke at alle mulige kausale modeller eller features mangler signal.

Tidskontrollen er nå ferdig:331/512 separate TRAIN-tilstander deler observerte
reward-overganger med fitted TRAIN. Dette berører118/128 Entries; alle512 deler
480-bars lokal inputhistorikk. Ulike Entry-ID-er ga ikke uavhengige forløp.
Blant35 state0-kontroller uten targetoverlapp taper kandidaten mot TRAIN-
konstanten på begge sider: LONG-MSE637,08 mot623,90; SHORT731,74 mot620,42.
Disse35 er fortsatt gjenbrukt TRAIN, ikke et nytt holdout eller tuningutvalg.
Overlapp beviser avhengighet, ikke alene årsaken til juni-svikten eller
framtidslekkasje i kausale modellinputs. Se docs/TRAIN_TEMPORAL_OVERLAP_20260917.md.

Initialiseringskontrollen er også ferdig. Den eldre V9-forgjengeren har brukt
77 312 shufflede rader fra alle60 TRAIN-måneder, frem til2026-05-29;16 016
fra siste treningsår. Bundet seed har fullført65 295 ettårsrader, og315 har
fullført hele313 399-raders TRAIN. Normaliseringen er også fittet på hele
TRAIN. Ingen undersøkt startvekt i den aktive kjeden gir en usett senere
TRAIN-periode. Å nullstille bare hodene eller velge nye Entry-ID-er løser ikke
dette. Se docs/INITIALIZATION_EXPOSURE_20260917.md.

Én forsøksdesign er nå frosset før fit:49 017 kalender-TRAIN-rader fra
juni2025 til februar2026 og16 278 senere kontrollrader fra mars–mai2026.
CONTROL256 er låst med eksisterende seed20260911/salt1:83/87/86 per måned.
Fem TRAIN-ankerfasiter krysser allerede datogrensen; kalenderutvalget er ikke
ferdig target-/state-eligibilitet. Budsjett256 oppdateringer/4096 Entries er
kun design, ingen kjøreautoritet. Se docs/CHRONOLOGICAL_LEARNING_DESIGN_20260917.md.

Den konkrete Entry-/Exit-målkjeden er nå rettet i eksplisitt opt-in-modus:
reference_policy og reference_cutoff_time_ns kreves gjennom eksisterende eiere.
Entry bruker samme observerte state0-Q_mu som Exit, pluss første likvidasjon;
FLAT er0. Gamma, masks, terminal/bootstrap, detach og gammel standard er bevart.
Fasit som krysser datogrensen avvises før modellinputs bygges; ingen kunstig
terminal eller holdetidsgrense.77 målrettede syntetiske CPU-tester består.
Første forsøk hadde én testkopieringsfeil; bare den og eksisterende adapterkjede
ble kjørt etter testrettelsen. Original logg er bevart. Dette er teknisk
målberegningsbevis, ikke ny læring. Ingen faktisk modell-forward, targetcache,
normaliseringsfit eller optimizersteg. Se docs/COHERENT_REFERENCE_ENTRY_20260917.md.
Kildekontrollen viste at tidlige Entry-rader alene ikke avgrenser senere
normaliseringsdata: lifetime-fit fulgte hele forløpet til TRAIN-slutt.
Lifetime-normaliseringen har nå en eksplisitt fit-populasjon før cutoff,
med uendrede fysiske successor-counts og Entry-ID-er.24 syntetiske kontroller
består, inkludert kraftig endring av alle framtidige priser uten endring i
prefix-utvalg, fit-verdier eller normaliseringsstatistikk. Se
docs/PREFIX_NORMALIZATION_20260917.md. Ingen faktisk normalisering er fittet.
Neste er tilsvarende binding for base/context/MTF-normaliseringen i dens
eksisterende eiere, deretter fersk native oppstart og kontrollbindinger.
Ingen faktisk fit, target-refresh, modell-forward eller kjøring er åpnet.

Fersk initialisering, prefix-normalisering og native kontrollbindinger er
fortsatt nødvendige før en kjørbar plan. Samme200 features/åtte familier/MTF,
ingen ny arkitektur eller separat runner. Ingen fit, trening, VAL eller TEST
er åpnet. Gamle utviklingsperioder blir ikke urørte av nye vekter. Ingen
holdetidsgrense eller automatisk forlengelse. Ikke gjenta ferdige auditer.

## Start her

Aktiv arbeidsmappe: /home/andre2/src/GX1_CURRENT, branch work/gx1-current.
Dette er et Git-worktree av GX1_ENGINE; felles Git-katalog er
/home/andre2/src/GX1_ENGINE/.git. Data ligger fortsatt i /home/andre2/GX1_DATA.
GX1_ENGINE-mappen står på audit/v9-premiere-20260905 og er ikke aktiv kjørevei.
Mac /Users/andrekildalbakke/Desktop/GX1 XAUUSD er overleveringskopi.
Kjør ./handover.sh --check på Mac, eller bash scripts/gx1_handover.sh --check
på Linux. Les current_work og next_run i JSON; de eldre kilde-/checkpointfeltene
er merket completed_run_history. Scriptet er kun lesing og starter ingen jobb.
Les AGENTS.md, GX1_ARBEIDSMAAL.md, NEXT_RUN_POLICY.json og
[beslutningsgrunnlaget](docs/LEARNING_GATE_20260916.md).

Gamle «gjeldende»-avsnitt er samlet i handover_snapshot/
HANDOVER_HISTORY_BEFORE_LEARNING_GATE_20260916.md, uttrykkelig historikk.
Ingen eldre startplan, COMPLETED_RUN.json eller historisk kildekopi skal brukes
som alternativ vei. Statusnotater må kontrolleres mot levende prosess/receipt.

## Historikk: fullført referanse og split

Treningskilde fcd03e884c2b402979624df767a359c57562c2e1 var frosset gjennom hele
den fullførte kontrollen nedenfor. Ny rettelse endrer kun klippegrupper og
bundet overgang; denne kontrollens gamle kilde er referanse, ikke aktuell kode.
Finn gjeldende HEAD med git rev-parse HEAD og ny kampanjes bundne plan.

Reference32 bestod på fysisk boot444. Split16+16 bestod på boot445 og446;
alle tre receipts har guardPASS, trainer0 og observer0. Siste splitreceipt er
fra2026-09-16T12:13:56Z. Windows-task GX1NativeLearningCalibration er deaktivert
etter terminal kontroll. Ingen videre trening er planlagt av disse planene.

Faktisk tilstandssammenligning består: alle14 komponenter er eksakt like,
inkludert modell, frossen lærer, Adam, EMA, scheduler, RNG, epochrekkefølge og
progresjon. Bare checkpointnummer og sessionidentitet er unntatt som forventet.
Begge armer endte på global5809, epoch_index1, next_batch_offset1728, phase train,
complete=false. Reference er checkpoint96, split er97; begge starter fra
original95/global5777/offset1696. Original95 og stoppet134 er bevart.

Reference-løkken tok170.835s for32 steg/512 TRAIN-rader,5.33859s/steg.
Med checkpointlagring174.233s; hele den bevoktede invokasjonen832s, hvor655.178s
var før treningsløkken. Windows-omstart/controllerforberedelse kommer i tillegg.
Peak8224MiB,58Ccore/62Cmem/206.03W. Ingen relativ samlet speedup eller full-epochETA
hevdes. Ny full VAL256-paritet er ikke målt i denne TRAIN-kontrollen; tidligere
native GPU256-bevis står i policy med sine opprinnelige kildebindinger.
Førstebatchdiagnostikk før oppdatering er ikke etter32-læringsbevis.

## Eksakte tilstander og bevis

Artifactrot (kalt BASE nedenfor):
/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912

- Original95: BASE/NATIVE_FQI_TARGET_REFRESH_F11C1DBE/
  .gx1-candidate-training-session.CANDIDATE_BUNDLE/candidate_training_state_slot_0.pt
  SHA ca18cdf27f1ddf22741c893456d2040e1e53ce0cf1ac8e9532cb780973fd5dbe.
- Reference96: BASE/NATIVE_FROZEN_TRACE_MEMORY_FCD03E88_REFERENCE/
  .gx1-candidate-training-session.CANDIDATE_BUNDLE/candidate_training_state_slot_1.pt
  SHA77294b3f04afca6c0acb1e078fb68f392f6ba3ee0a1858a669a8c91b12622d0c.
- Split97: BASE/NATIVE_FROZEN_TRACE_MEMORY_FCD03E88_SPLIT/
  .gx1-candidate-training-session.CANDIDATE_BUNDLE/candidate_training_state_slot_0.pt
  SHA94bbb8101036f25d0e27bf272b4b30a2eb1dfccff41a0e56466de0acfbd088ef.
- Resume: BASE/NATIVE_FROZEN_TRACE_RESUME_EQUIVALENCE_FCD03E88/RESULT.json,
  SHAcd52dda262aebd33a8197db25850fa54a3ac2c4dfe6c354fa856c8316ec61b18.
  OPERATOR.py og logger er bevart der. Første operatørforsøk manglet PYTHONPATH
  og stoppet før modellimport; korrigert kjøring bestod. Ingen trenerkode endret.
- Reference-fart/minne: BASE/NATIVE_FROZEN_TRACE_MEMORY_FCD03E88_REFERENCE/
  REFERENCE32_OBSERVATION_20260916.json,
  SHA09e539af550379a565e0d400bfb5181bbdc4f3d52a14033de0a6a3e7f9a09050.
- Receipts: /home/andre2/GX1_RUNS/
  NATIVE_FROZEN_TRACE_MEMORY_FCD03E88_REFERENCE_BOOT443/receipts og
  NATIVE_FROZEN_TRACE_MEMORY_FCD03E88_SPLIT_BOOT444/receipts.
  Runtime-navnet er forberedelsesboot, ikke nødvendigvis faktisk kjøreboot.
- Samlet oversikt og lokale kopier: handover_snapshot/
  NATIVE_FROZEN_TRACE_CONTROL_RESULT_20260916.json og
  NATIVE_FROZEN_TRACE_RESUME_EQUIVALENCE_FCD03E88.json.

## Læringsmåling95→96 ferdig — ingen utvidelse begrunnet

CPU-målingen gjenbrukte lagrede inputs/targets og før95-output:174.65s,
topp-RSS2 222 524KiB. Ingen ny materialisering, lærerberegning, backward,
GPU, VAL eller TEST. Før95-output på første batch ble reprodusert eksakt.

- Første trente batch,16 Entries/64 overganger, samme femstegsmål:
  Entry-MSE3.56205→3.40652 (4.37% ned); begge16FLAT, også læreren.
  Exit-MSE26.17057→26.16781 (0.01054% ned); MAE blir litt verre.
- Bred1024 TRAIN-kohort: Entry-MSE12.56542→12.66827 (0.82% opp), fortsatt
  1024FLAT. Læreren velger914FLAT/49LONG/61SHORT. Bare16 av disse1024 lå i
  siste32 oppdateringsbatcher; de andre1008 er fortsatt TRAIN, ikke holdout.
- Bred Exit bruker gamle faste ettstegsmål for sammenlignbar diagnose,
  ikke som påstått femstegs-loss. MSE9.07095→9.08508. LONG blir4096HOLD.
  94.70% av kvadrert LONG-prediksjonsendring skyldes en konstant forskyvning;
  korrelasjon med ettstegsmålet er nær null. Bedre situasjonsvurdering er ubevist.
- Bred forecast-MSE blir litt verre på alle fire horisonter.32 steg og én
  femstegsbatch avgjør ikke endelig lærbarhet, men åpner ikke større trening.

Bevis: handover_snapshot/FROZEN_TRACE_LEARNING95_96_REVIEW_20260916.json.
Fullt resultat og nye BATCH_00..63_AFTER96.json ligger under BASE/
NATIVE_FROZEN_TRACE_MEMORY_FCD03E88_REFERENCE/LEARNING95_96_20260916.
RESULT.json SHA47822d2d58f2fcfdab82c22065c3c393d8ef2f1073b409dccff57d8c052c071c.

Årsaksdiagnosen er nå ferdig; se neste avsnitt. Ikke gjenta32/1024-kontrollen.

## Årsaksdiagnose — målkjede og verdilesing

Se [full årsaksrapport](docs/VALUE_LEARNING_CAUSE_20260916.md). Én cacheanalyse
og en avsluttende hodeattribusjon ble kjørt på CPU med eksisterende audit-vakt;
ingen trening, nye targets, GPU, VAL eller TEST. Begge kilde-/checkpointbindinger
bestod. To evalueringer av én lagret batch reproduserte alle gamle outputs eksakt.

Entry får bare0,02456Bps videreverdi mot−5,85327Bps første likvidasjonsverdi;
1021/1024 lærerhandlinger er de samme uten videreverdien. Femstegsberegningen
endrer ikke denne Entry-supervisjonen. I den lagrede batchen stopper lærerpolicyen
LONG etter ett steg på59/64, mens SHORT bruker fem på64/64; alle successors finnes.
Dette er en målt sideavhengig målhorisont, ikke en holdetidsgrense.

På den samme lokale batchen kommer LONG-økningen0,15992Bps omtrent60% fra siste
verdihode og40% fra representasjonen. SHORT-fallet−0,03854Bps kommer fra
representasjonen. Felles biasendring er bare+0,00151Bps. Det er ikke dokumentert
at en biasrettelse løser problemet. Den brede konstante LONG-forskyvningen er
fortsatt94,70%; tilstandsavhengig læring er svak. Prognoser har TRAIN-signal,
men lærer andre mål og inngår ikke direkte i Entry-argmax.

Manglende tidlige Exit-eksempler ble avkreftet:497/4096 treningssamples er ved
tilstand0. Tidligere gradientbevis gir fortsatt ikke grunnlag for generell
backbone-/tapsvektendring. Den eksakte oppdelingen mellom upstream-moduler og
Adam-historikk er ikke målt; lærerens skjevhet er ikke alene bevis for retningen
på de32 ONLINE-oppdateringene. Dette var status ved cache-/hodeanalysen;
Adam-/klippebidrag er senere målt i rapporten om Exit-gradientklipping.

Planer, operatorer, logger og resultater ligger i
handover_snapshot/CACHED_VALUE_CAUSE_20260916/ og tilsvarende mappe under
BASE/NATIVE_REAL_TRAIN_TO_VAL_FDD70E5C/OPERATOR_OBSERVATIONS/.
RESULT SHAde50f50bf2d81817cbfa0a033aa9143c213a415b71d67b692972bf7ff2f266e2;
ATTRIBUTION_RESULT SHA98cbd1b223542f5a678265ba48f4910ed9596d1079f8937a5a5c5d12e0853559.
Cacheanalysen tok1,04s; hodeanalysen9,75s. Modell-/treningskode og checkpoints er bevart.

Neste: kandidat32 og faktisk512-måling er nå ferdige. Gjenbruk deres caches
for neste kontroll av observerte utfall ved Entry-ankeret; reward/stopp/bootstrap
er nå målt i TARGET_COMPONENT_CAUSE_20260916.md. Ingen nye
klippe-/tapsvekt-/modellforsøk, Adam-nullstilling eller større trening før én
målmekanisme er begrunnet. Se den oppdaterte klipperapporten.

Cachemappene under BASE/NATIVE_REAL_TRAIN_TO_VAL_FDD70E5C/OPERATOR_OBSERVATIONS/
FROZEN_POLICY_TRACE_NATIVE_BATCH_20260916_V2 og BROAD_TRAIN95_134_20260916 er
bevart. Videre analyse av nye output krever ikke nye forwards.

Bruk eksisterende eiere. CPU-modellfabrikken finnes i
run_unified_exit_random_access_val_v1; bruk av fabrikk til en CPU-måling er ikke
starttillatelse til separat VAL-runner. CPU-operator må ha
PYTHONPATH=/home/andre2/src/GX1_CURRENT og CUDA_VISIBLE_DEVICES='', kjøres via
gx1_capped_run.sh. Python er3.10; hashlib.file_digest finnes ikke. Eksisterende
file_sha256 og verify_candidate_checkpoint_resume_v1-eier skal gjenbrukes.

## Hva funnene betyr

TRAIN95→134 ga ikke bedre Entry/Exit-verdilæring i den brede målingen. Entry var
FLAT på1 024/1 024 i begge modeller. Noen lengre prognoser bedret seg på TRAIN,
uten at det beviser senere markedsfordel. Tidligere delvis VAL var bare HOLD i
de evaluerte hypotetiske sideforløpene. Lærerens verdiestimat er ikke fasit;
Entry-handlingen er fortsatt delvis avhengig av Exit-læreren.

Ny femstegstarget og minimal oppdeling av frosne target-forwards virker teknisk.
Det er ingen fast holdetidsgrense. Bare målbar læring og etterfølgende økonomisk
validering kan begrunne utvidelse. Full profitt omfatter alle valgte handler,
realisert cash, åpne verdier og kostnader. TEST forblir forseglet.

## Opprydding og bevaring

7 093 786 972 logiske bytes (7,09GB;6,61GiB) gamle midlertidige Entry-arrays er
slettet.7 093 800 960 allokerte bytes ble frigjort og kontrollert. Ingen levende
prosessreferanser fantes; eieren oppretter unike TemporaryDirectory-filer i w+
modus, ikke resume-input. Se handover_snapshot/DISK_CLEANUP_20260916.json.
Datasett, analyser/caches for neste måling, alle resultater/checkpoints og bundne
runtime-/kildeavhengigheter er bevart. En eldre9,2GiB Windows-backup inneholder
filer uten nåværende original; den er ikke dokumentert redundant og er beholdt.
Ingen WSL-diskkomprimering, reinstallasjon eller ekstra omstart er utført.

Overlevering, policytekst, statusskript, systemkart, Mac-speil og den eksisterende
timekontrollen skal være samordnet. Stabil langkjøring kontrolleres omtrent
hver time, én agent/én tung jobb. Ikke fyll ventetiden med arbeid.
