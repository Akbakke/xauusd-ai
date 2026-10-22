# CURRENT HANDOVER — XAUUSD M5 Entry

Autoritet: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current` på
`gx1-3090-lan`. GX1_RULES er bindende; NEXT_RUN_POLICY eier arbeidsstatus.
Prosesser og terminalkvitteringer overstyrer prosa. Én integrator, én capped
CURRENT-jobb og fryst kilde/policy gjennom hver kjøring. Originalmålet står i
[docs/ORIGINAL_PROJECT_GOAL.md](docs/ORIGINAL_PROJECT_GOAL.md).

## Aktivt mål og neste steg

Det kontinuerlige målet er fortsatt bedre kvalifikasjon av M5 LONG/SHORT/FLAT:
MBP-1/SMC/lukket M15/H1/H4/D1, retning og størrelse, liten MAE før første MFE,
kostnad, kvalitet og dekning. Bedre retning, generalisering og edge er ubevist.
Brukeren prioriterte22.10 en solid featuregrunnmur før videre trening;175 har
avstemt faktiske kilde-/input-/numerikkbevis. Ingen ny generatorfeil er påvist.

173s nye tilstands-/RNG-audit er COMPLETE/consumed:172fresh og173s durable
nullstegscheckpoint har eksakt lik modell/target/optimizer/EMA/scheduler og
lagret CPU/Python/NumPy RNG. Bevisgrensen er før målingen; CUDA er bare
skjemakontrollert. Full canonical OBS-validering bestod faktisk i176materializer.

178 er COMPLETE/consumed: teknikken bestod, men LONG/SHORT/FLAT-entry er fortsatt
ukvalifisert. Overamplituden fra167 er kraftig redusert; modellen velger nå
LONG0/SHORT0/FLAT256 på både TRAIN og CONTROL. AlltidFLAT er ingen selektivitet.

CONTROL nettkontrast-MSE (bps²), samme rader/masker:

| Minutt | Egen173initial | 167final | 177final |
|---|---:|---:|---:|
|1|57,459|572,260|61,678|
|5|228,358|686,241|227,369|
|15|772,101|1358,138|774,629|

Alle6primærpaneler/TRAIN+CONTROL forbedres mot167; alle12parvise dag-/uke-
intervaller for de seks MSE-endringene er under0.5av6 punkt-MSE forverres mot
173initial; CONTROL5min har liten bedring med intervaller over/under0. h1CONTROL
predstd2,461 mot16720,117 ogobservasjon7,504bps; corr0,0578. Dette er dempet
feil/amplitude, ingen robust egeninitialforbedring eller kvalifisert Entry.
Begge siders predikerte bedstehorizon er1 på alle512sluttrader. MFE er fortsatt
overestimert vedh1; ingen automatisk terskel-, score-, feature- eller tapsendring.

179 er COMPLETE/consumed:1forward/16reverse/0updates på16faste genuineTRAIN,
172constructor/frosset177ONLINE. Alle84640komponentceller har delta0bps;
1e-4bps/rtol0 og eksakteaction/horizonties bestod. Gammel151pakkeopprinnelse
beholdes separat.177TRAINreferansen ble lest én gang undercap; ferdig178peker-
bevis ble gjenbrukt uten storRESULTS-/rå-/cacheskanning.

699early/20019later ekteFLAT-inklusive celler gir begge målbare gradienter til
decoder/sharedlag. Early/later hidden-normratio2,861, hidden-cos+0,247 mot
entryprojection−0,477: ingen uniform senere dominans eller7391stegsårsak er
påvist. Maksimal rekomponeringsresidual1,064e−7. Dette gjelder aggregert1–15-
tap; separateLONG/SHORT-tapsgradienter ble ikke målt. Lokalh1-scorederivert
terminalminuspreMAE er negativ for begge sider. Dette er ikke featureverdi.

44fokuserte CPU-tester og obligatoriske commitvakter bestod. Bare audithelper/
executor/tester ble endret; native183roller/182kilder er bevart. Alle189bundne
kilder/policy og modell-/session-/RNG-/mode-/thread-/gradtilstand ble bevart i
målingen. Ingen ONLINE-funksjonsendring eller vektoppdatering.

Neste steg er180read-onlyruteforberedelse: undersøk gjenbruk av141/143/144/175-
bevis og177/172tilstand for faktisk trent inputforbruk på genuine fasteTRAIN-
inputs. Skill ren terminalretning1/5/15 fra LONG/SHORT-vsFLAT-/risikomarginer;
bind normalisering, allealiaser, masker og ekte støtte per familie/TF.
Sensitivitet er ikke kronologisk merverdi. Budsjett/DESIGN/PLAN/uavhengigreview
og ny binding må konkretiseres før nye modellkall. Ingen ny feature-/modell-/
taps-/scoreendring, råskanning,218F-kontrafaktiskjobb eller trening er valgt.
Forwards/gradientjobb/trening er nå stengt; kontinuerlig mål er aktivt.

Root175: `/home/andre2/GX1_RUNS/NATIVE_M5_FEATURE_FOUNDATION_AUDIT_20261022_175`.
Root177: `/home/andre2/GX1_RUNS/NATIVE_M5_TERMINAL_BALANCE_SINGLE_FULL_NATIVE_EPOCH_20261022_177`.
Root178: `/home/andre2/GX1_RUNS/NATIVE_M5_TERMINAL_BALANCE_DIRECTION_COMPARISON_20261022_178`.
Root179: `/home/andre2/GX1_RUNS/NATIVE_M5_TERMINAL_BALANCE_TRAINED_GRADIENT_WITNESS_20261022_179`.
Root180: `/home/andre2/GX1_RUNS/NATIVE_M5_TRAINED_FEATURE_CONSUMPTION_20261022_180`.

## Faktisk siste kjøring

179CUDAproducer20G/512M er COMPLETE: native51,847s/ytre55,477s,1F/16R/0U.
HEAD8e342e7a/policybc61d8fb var fryst; actualpeer/TERMINAL/DECISION er bevart.
Guard45samples,maks45C/46C/130,51W/1586MiB/observerte160Wlimit, child0;
65C/80C/170Wdraw/160Wlimit/12288MiB-vakter uendret. Prosjektlåsen var fri etterpå.
PrelaunchWindowsboot526/10:09:50Z, nativeWindowsoppgaveDisabled/Last0/0native.
Ingen campaign-/Windows-task-/effektprofilendring ble gjort av denne diagnosen.

178 CPU4G/512M er COMPLETE30,647s/0nyeF/updates/refits: seks lagrede kurver og
seks canonical dynamic_entry_m1_row-projeksjoner, eksakte targetstorage/masker/
rekkefølge/cohorts. Én eksisterende168aggregatrapport beholder151referansen;
ingen nye177–151intervaller eller151utfall på nye177grupper rekonstrueres.
HEAD4183a8cf/policy1bc88553 bevart. TERMINAL/RESULTS/SUMMARY/ACTUAL_DIRECTION_
COMPARISON_PEER_REVIEW/LOSS_AND_REPRESENTATION_INTERPRETATION/DECISION er bevart.

177s matchede epoch er consumed:118249unikeTRAIN/batch16/siste9/7391updates og
32sluttONLINEforwards/512rader/full480.172fresh/egen173initial; targetanker
2eaf88… frosset, sluttONLINEe7ca6b1f…, ingen176privatevekter/teacherrefresh.
Native5828,584s/ytre6120,611s, fysiskboot526 etter525, guard/trainer/observer
PASS/0/0,4843samples,maks57C/66C/143,38W/1804MiB/observerte160Wlimit. Windows
Ready→Disabled/Last0/restart0/0nativekandidater. Guardtak85C/80C/300Wlimit/
310Wdraw/12288MiB beholdes.183sourcebyte/stat og23smårefs var bevart; ingen ny
stor-payloadrehash. SOURCE_CLOSURE og eksisterende v1-metadata-projeksjon bevares.
RESUMABLE7391 er forventet lagringscursor, ingen ny engangsadgang/ekstraepoke.

176privat4gate og173egen32F/0updateinitial bestod. Original163FAILED beholdes;
ingen cross-parameterization-paritet påstås. Originalkvitteringer/checkpoints
bevares; ingen gammel engangsplan relanseres.

## Featuregrunnmur og konkrete begrensninger

175 er en read-only evidensavstemming, ingen ny dataskanning/modelljobb/refit.
FOUNDATION_DECISION og egne rekonsilieringer overstyrer første175reviews sine
for brede slutninger fra eldre SOURCE_REVIEW-flagg. Faktiske125/126/133/134-
terminaler finnes og gjenbrukes:1205registrerteGCverdier+1205observasjonsmasker,
164059545celler av hver/0F32bitfeil; alle695aktive kompakte MTF-utganger er
numerisk kvalifisert på femTF, inkludert133s470coreutganger. Dette er felt/
utganger, ikke like mange uavhengige prediktorer. Ingen duplikataudit trengs.

Swing/SMC bruker deklarert forsinket pivotbekreftelse, lukkedeTF/strict-priorD1
og ærlig warmup. Ingen konkret kildelekkasje/repainting-feil er funnet i scope.
GCprofiler er histogrammer av utførteT-handler; BBO/ordercount/OFI er egne ruter.
Ukjent aggressorvolum bevares. MBP-1 gir ikke ordreidentitet/kø/skjultdybde.
055normalisering er TRAIN-only118249rader/asinh uten amplitudeclipping; ingen
VAL/TEST-fit. Nåværende183-rollematrise er171 med digest9b17b25c…;170 endret
bare tre taps-/kontraktroller og bevarte de andre180.

34aktiveGCfelt har under100genuine observasjoner i minst én split; Q1-nullene
er10strictH1signedflow og14D1proxyOB-statefelt. Strict krever komplett gyldig
minuttvindu, ett kontraktforløp og0ukjent aggressorvolum. OB krever bekreftet
pivot/BOS/motsattlys og sensureres ved helt manglendeTFbøtte. Ingen egen måned/
år/splittreset finnes. Sparsitet alene er ikke feedfeil eller grunnlag for
syntetisk fyll, fjerning eller ny skanning; H1observed-volume har16568/17284støtte.

61makrofelt er markeds-/Treasurykurve-/GLDbeholdnings-/COTfelt, ingen CPI/GDP.
Gjeldende konservative lag og offisiell forsinket COT-kalender er bevart;
originale Treasury/GLDpublikasjonsvintager/korreksjoner er avgrensede bevisgrenser,
ikke påvist generell PIT-feil. Modellkobling/påvirkning er ikke prediktiv nytte.
144s218F/13VJP gjaldt tre genuine nullstegsTRAINvitner, ikke trent featureviktighet.
Kronologisk OOF-merverdi/optimal bruk er fortsatt ubevist.

Aktiv profil er `compact_mbp_smc_ema_atr_clock_macro_entry_v1`, SHA2ac4fb4d…:
7lokale/5MTFfamilier.19Oct-unntaket tillater dette første kompakte utvalget;
fullbibliotekets åtte spesialister og niende D1/makro beholdes. Parkert geometri
og andre felt er verken generelt numerisk-/nytte-kvalifisert eller forkastet.
Ingen featureaktivering eller generatorendring er valgt av175.

## Valgt tapsforsøk og resultat som motiverer det

167s ene epoke er consumed:118249unikeTRAINrader/7391steg/32sluttforwards.
168 avviser den som retningsforbedring. CONTROL nettkontrast-MSE1/5/15min var
572,260/686,241/1358,138 mot151s352,477/482,464/1142,099; alle seks parvise
dag-/ukeintervaller for MSE-endring var positive. h1korrelasjon0,011 mot0,050;
prognosespredning20,12 motfasit7,50bps. Valgt LONG/SHORT predterminal+13,57/
+22,12 motobs−5,03/−6,40. PreMAE var allerede overestimert; dette er komponent-
proksier, ingen realisertP&L.169 viser pooledterminalfeil0,726→0,566 samtidig
som1..15 ble0,729→1,603. Det beviser ikke historisk gradientdominans.

170 velger terminalobjective som snitt av observerte1..15- og16..480-gruppers
RMS-enhetsmidler; beggegrupper50:50, éngruppe egetmiddel, ingengruppe utelater
terminaltask/+s.15/50:50 er designkonvensjon, ikke optimum. Alle480, ekteFLAT-N,
øvrige tre tasks/111units/inputs/targets/masker/score og raw/pooleddiagnostikk
beholdes; nye optimizerstats er egne og ikke samlet7391stegshistorikk.
171 kvalifiserte91fokuserte kontroller,0feil/hopp,5,616s CPU.172 publiserte
fersk ONLINE=TARGET=EMA2eaf88…/tomoptimizer/EMA0,122,508s/0F/updates/refits.
173s aktuelle initialmåling/tilstandsaudit og176s private tekniske gate bestod.
177s ene matchede167-budsjettepoke og178lagret sammenligning er fullført.
Inputs var faste. Amplitude-/feilreduksjon er målt, men kort Entry er fortsatt
ukvalifisert; ingen ny taps-/featureendring eller trening er valgt.

## Varige grenser

Aktive beslutningsinputs: TRAIN[2024-05-01,2026-01-01), Q1-VAL[2026-01-01,
2026-04-01), altså23måneder. April–juni2026 er senere gjenbrukt utvikling som
venter separat fysisk binding; råkildemålet er26måneder. Ikke kall aktuell
CONTROL juni eller23måneder full26måneders inputdekning. MTF-cache har lengre
historikk. TEST fra juli er forseglet for tilpasning/utvalg; bare vedtatt månedlig
forwardjournal scorer den. Ingen Exit/broker/live/paper/ordre/spending/promotion.

Entry beslutter på lukketM5.1–480min er prognosedomene, aldri Exit, fasttap eller
maksimal holdetid. AlltidFLAT er ingen selektivitetssuksess. Rapportér teknikk,
TRAIN-læring, kronologisk sammenligning, generalisering og økonomi separat;
lange støttemål har forhåndsvalgt alltidLONG/kjøp-og-hold som referanse.
Ingen DATA/RUNS-rydding uten retention-eier. Historiske detaljer ligger i Git
og originalkvitteringer; les dem bare ved konkret avvik/nødvendig gatekontroll.
Metoden er [docs/NATIVE_LEARNING.md](docs/NATIVE_LEARNING.md).
