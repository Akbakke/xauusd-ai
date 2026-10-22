# CURRENT HANDOVER — XAUUSD M5 Entry

Autoritet: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current` på
`gx1-3090-lan`. Les GX1_RULES; NEXT_RUN_POLICY eier arbeidsstatus.
Faktiske prosesser og terminalkvitteringer overstyrer prosa. Én integrator,
én capped tung CURRENT-jobb og fryst kilde/policy gjennom hver kjøring.
Originalbestillingen er bevart i [docs/ORIGINAL_PROJECT_GOAL.md](docs/ORIGINAL_PROJECT_GOAL.md).

## Aktivt mål og neste steg

Brukeren har21.10 bestilt kontinuerlig arbeid til bedre kvalifikasjon av M5
LONG/SHORT/FLAT-retning er målt: MBP-1/SMC/lukketMTF, horisontavhengig
retning og størrelse, liten MAE før første MFE, kostnad, kvalitet og dekning.
Målet står i `current_work.autonomous_direction_qualification_goal_20261021`.
Det er aktivt; ingen forbedret retning, generalisering eller edge er ferdig bevist.

165 velger et eksplisitt nytt forsøk med uendret158querykode og162 som
selvstendig aktuell ferskbaseline.163s eksakte gamle horizonparitet forblir
avvist. Ingen ny buffer, score, terskel, features, targets, masker eller tapskode.
Sammenlign læringsendring fra egeninitial og151 med samme data/budsjett;
ikke påstå identisk F32-startfunksjon eller isolert optimizergeometri.
165/DESIGN.json og ACTUAL_BASELINE_SCOPE_PEER_REVIEW.json er bundet i policy.

166 er COMPLETE og consumed: fire paritetsforwards, fire private optimizersteg,
logisk cursor2. Alle215456sammenlignede flyttallsceller har maksavvik0 ved
uendret1e-4bps/rtol0; action/query/horizon-masker er eksakte. To sammenhengende
steg mot ett+durable-reload+ett ga eksakte åtte rapporterte statefelt. Private
vekter er kun audit, aldri hovedinitial. Original163-avvisning er bevart.

166 brukte fysiskboot522 etter521 og frystHEADa612d6c3/policy480bae7d.
Alle183sourcehash/stat/path og11resultatfilreferanser var bevart; guard/trainer/
observer PASS/0/0. Ytre3389,128s;2644telemetrisamples, topper55Ccore/64Cmemory/
136,71Wdraw/1776MiB; alleobserverte fysiskegrenser160W. WindowsTaskReady/Last0
ble eksplisitt deaktivert etter terminal og0identifiserte prosjektprosesser.
TERMINAL.json og ACTUAL_NATIVE_TERMINAL_PEER_REVIEW.json er bundet i policy.

Root166: `/home/andre2/GX1_RUNS/NATIVE_M5_STANDARD_QUERY_TECHNICAL4_20261022_166`.
167 er COMPLETE og consumed: én faktisk registrert epoke med118249unikeTRAIN-
rader, batch16/slutt9,7391steg og32sluttforwards på TRAIN256/CONTROL256.
Nativepausekvittering, sluttresultat, logg og18peer-kontroller er enige.
Ytre6109,949s/native6010,822s; guard/trainer/observer PASS/0/0.4936samples,
topper57Ccore/66Cmemory/134,05Wdraw/1804MiB; alleobserverte fysiskegrenser160W.
Alle183hash/stat/path og frystHEAD3ceaa171/policy8855de74 ble bevart.
Boot523 ble bekreftet; TaskReady/Last0 ble deretter eksplisitt Disabled etter
terminal og0identifiserte prosjektprosesser. Fersk161/egen162baseline og frossen
targetanker2eaf er bevart; ingen privat166vekt eller ekstraepoke er brukt.

Root167: `/home/andre2/GX1_RUNS/NATIVE_M5_STANDARD_QUERY_SINGLE_FULL_NATIVE_EPOCH_20261022_167`.
TERMINAL/ACTUAL_NATIVE_TERMINAL_PEER_REVIEW og FINALRESULT er bundet i policy.
`training_enabled=false/full_epoch_training_allowed=false`; ingen relaunch,
videreepoke eller automatisk forlengelse. FullVAL/Exit/TEST/broker er stengt.

168 er COMPLETE/consumed:33,873s CPU, seks OBSJSON én gang hver og seks
canonicalparentpointer-projeksjoner,0nye forwards/steg/refits. Data/masker/
storage/cohort/nullable-target-joins er rapportert eksakte og peer-kontrollert.
Root168: `/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_COMPARISON_20261022_168`.
TERMINAL/RESULTS/SUMMARY/ACTUAL_DIRECTION_COMPARISON_PEER_REVIEW/DECISION er bundet.

167 avvises som retningskvalitetsforbedring. CONTROL nettkontrast-MSE1/5/15min
ble572,260/686,241/1358,138 mot151s352,477/482,464/1142,099. Alle seks parvise
UTCdag-/ukeintervaller for MSE-endring er positive. h1korrelasjon0,011 mot0,050;
prognosespredning20,12 motfasit7,50bps (151:17,43). TRAINs korte MSE ble også
større, mens mye lengreTRAINretning læres bedre; det beviser ikke generalisering.
CONTROL-valg økte til35LONG/119SHORT/102FLAT fra29/53/174, fortsatt h1overalt.
Valgte LONG/SHORT predterminal+13,57/+22,12 motobs−5,03/−6,40; predpreMAE9,75/
9,29 er allerede≥obs9,46/6,74. Terminalovervurdering dominerer derfor positiv
kandidatoptimisme i disse gruppene; sterkere MAE-penalty beviser ingen retning.
Dette er proxykomponenter under deklarerte kostnader, ingen realisertP&L.

169 er COMPLETE/consumed:0,488s CPU, to bundne payloadJSON(168RESULTS/111RMS),
0nye modellkall/reverse/steg/refits. HEADafcbbe3b/policyd453500c var bevart.
Root169: `/home/andre2/GX1_RUNS/NATIVE_M5_LOSS_ALLOCATION_DIAGNOSIS_20261022_169`.
TERMINAL/RESULTS/SUMMARY/ACTUAL_LOSS_ALLOCATION_PEER_REVIEW er bundet i policy.
Terminalpoolednormfeil falt0,726107→0,566106 mens1..15 steg0,728875→1,602966;
16..480 falt0,726012→0,530303. Tidligfeil kan skjules i samletfit. Det beviser
ikke parametergradientdominans eller årsaken gjennom7391steg. PerkoordinatRMS
finnes allerede;1..15s hypotetiske likråresidualmasse31,14% mot121..480s28,81%.
FLATs114298nullceller inngår fortsatt i terminalN342894; forecast er4M5+6dag.

170 velger én hypotese for kildeforberedelse: terminaloptimizerobjective blir
aritmetisk snitt av observerte1..15- og16..480-gruppemidler i fryste111RMS-
enheter. Beggegrupper gir50:50; éngruppe gir bare sitt observerte middel;
ingengruppe utelater terminaltask/+s.15/50:50 er en fast designkonvensjon,
ikke målt optimum. Beggegrupper forblir Entry-prisforløp fra minutt til timer.
Alle480, ekteFLAT-N, åtte familier+makro, øvrige tre tasks, rawbps/pooledunit-
diagnostikk, inputs/targets/masker og score beholdes. Ny optimizerobjective og
gruppe-N/én-/togruppebatcher skal ha egne navn; ikke kall balancedmean pooled.
Root170: `/home/andre2/GX1_RUNS/NATIVE_M5_TERMINAL_HORIZON_BALANCE_DESIGN_20261022_170`.
DESIGN er bundet i policy.170 er implementert i tre navngitte kildefiler,
med ny fokusert test og minste reparasjon av gammel gatefixture/forventetlabel.
Modellarkitektur/RNG er uendret. Ny treningskontraktv15 binder terminalobjective;
111basekontrakt/pooledhelper/jointscalarizer er bevart. Ny optimizer-evidens er
per train_epoch-kall/én kurvevalidering, aldri påstått samlet7391stegshistorikk.
171 har én eksakt audit4G/512M/1800s-kontrollrunde admitted; ingen nativekall.
Alle183roller er bundet; nøyaktig tre roller er endret,180andre byte/stat er like.
Root171: `/home/andre2/GX1_RUNS/NATIVE_M5_TERMINAL_HORIZON_BALANCE_SOURCE_QUALIFICATION_20261022_171`.
Bruk NATIVE_SOURCE_DELTA_CORRECTED_AST.json; originalens posisjonelle AST-detalj-
telling var misvisende ved innsetting av funksjoner og er bevart som forberedelse.
Ingen kilde-/matematikkfeil eller modellkjøring følger av metadatarettelsen.
Ingen modellobservasjon, trening, fullVAL, Exit, TEST eller broker er admitted.
Fersk kildebundet CPU-publisering, aktuelle initial-/paritets-/resumegater og
eksakt ny engangsadgang kreves før et eventuelt matchet167-budsjettforsøk.

167/EVALUATION_PLAN var låst før trening:1/5/15min primærretning, separat
MAE/MFE/kostnad og valgte/avståtte, dag2000seed20260911/mandagsukesensitivitet.
Alle26måneder er gjenbrukt utvikling. Retning/kvalitet/dekning/generalisering/
edge er fortsatt ubevist; målet fortsetter. Ingen relaunch, nyepoke, Exit ellerTEST.

## Faktisk evidens som gjenbrukes

151 er COMPLETE/consumed:118249unikeTRAIN-rader,7391steg og32sluttforwards.
Alle seks rå komponent-MSE ble bedre på TRAIN; CONTROL fikk bedre MFE/MAE,
men dårligere terminalretning/forecast. Korte retninger1/5/15min ble dårligere
på begge faste prober. Valgte CONTROL-LONG29 hadde terminal+14,24pred/−2,89obs;
SHORT53 hadde+22,05pred/−6,04obs. Ingen handels-/P&L-påstand.

154: alle1024slutt-sidekurver har strengt fallende terminal-minus-preMAE-score
og unikt1minmaksimum. CONTROL-terminalens standardavvik17,27/17,60bps mot
fasit7,61/7,50; korrelasjon0,053/0,049. Retningskontrastens korrelasjon er
−0,094TRAIN/+0,050CONTROL. Overoptimisme er mer enn et felles nivåskift.
156: ett genuineTRAIN16-forward,22reversekall,0steg.84640komponentceller
og valgene er eksakt151. Lokal terminalhorisontrespons er svak mot MAE;
alle16×480deriverte av terminal-minus-MAE er negative per side. Queryruten er
aktiv; ingen generell taskblokkering eller årsak gjennom7391steg er bevist.

158: analytisk query=(h−240,5)/sqrt((480²−1)/12), med algebraisk matchet
initialisering. Samme funksjonsklasse/parametertall; bare én av183native
innholdsroller er endret.41unike syntetiske kildekontroller er kvalifisert.
Nåværende183roller har digest2d7fd42b7ae46082a08da4abd20e5195cd39f28b194541b57b920b02cb31ec1f.

161fresh CPU,162nativeinitial og163state/RNG/canonicalfull480 er faktisk utført.
162:1989,652s ytre/1731,547s native,32forwards/0steg, TRAIN256/CONTROL256.
Guard/trainer/observer PASS/0/0, sourceHEAD00b7c2e2/policy152e6a79 bevart;
ONLINE=TARGET=EMA2eaf88..., tomoptimizer/0EMA. Boot521 ble bekreftet;
Ready/Last0 ble deretter eksplisitt deaktivert uten å stoppe prosjektprosesser.
Målte topper55Ccore/64Cmemory/130,93Wdraw/738MiB, fysisklimit160W.
Uendrede nativeguardterskler er85Ccore/80Cmemory/300Wlimit/310Wdraw/12288MiB;
160W er faktisk keepergrense, ikke guardens stopthreshold.

163s kombinerte funksjonsport er FAILED og consumed: horizonsets avviker fra148.
Separat bestod lagret state/RNG og canonicalfull480/åttefeltsinitialaudit;
originalene bevares og gjenbrukes bare for disse faktiske påstandene.
164:6,58s CPU/4lagredeJSON/0PT/0modellkall/steg. Alle512action-winners er like,
forecast er identisk og komponentmaks er5,96e-7bps, under uendret1e-4/rtol0.
Åtte ulike horizonsets gjelder bare SHORT-kandidater på FLAT-rader; ingen valgt
action er berørt. Nærmaksforskjeller er1–3ULP. Ingen math-/target-/mask-/orderfeil.
Dette forklarer163-avvisningen, men gir ingen lærings- eller launchadgang.
Alle tidligere engangsplaner er consumed; ingen av dem skal relanseres.

## Varig mål, grenser og opprydding

Entry beslutter på lukkede M5-barer med lukket M15/H1/H4/D1. M1 er finere
etterfølgende prisfasit.1–480min er prognosedomene, aldri Entry-kadens, Exit,
fast tap eller maksimal holdetid. Ingen Exit i denne bølgen. Framtidsfasit/
masker brukes ikke som beslutningsinputs. Åtte familier og niende D1/makro bevares.

Aktiv kompaktprofil: `compact_mbp_smc_ema_atr_clock_macro_entry_v1`, SHA
2ac4fb4ddfac84f588f65363ac8959544233ace286d9648bfebf60b97153f125.
121–126 kvalifiserer1205aktiveGCverdier/femTF;127/128/129/133 kvalifiserer
139aktivecorefelt perTF.134råMBP-scan66filer/892100712records og144genuine
inputpåvirkningskontroller gjenbrukes; ikke gjenta råscannen. Parker­te
fullbibliotekfelt er ikke generelt numerisk-/nytte-kvalifisert eller forkastet.
055wholeTRAIN-inputnormalisering og111TRAIN-only lossRMS gjenbrukes uten refit.

TRAIN[2024-05-01,2026-01-01),VAL[2026-01-01,2026-04-01),senereutvikling
[2026-04-01,2026-07-01). Alle26måneder, også JuneCONTROL, er gjenbrukt utvikling.
TEST fra juli er forseglet; bare vedtatt månedlig forwardjournal scorer den.
Ingen broker/live/paper/ordre/spending/promotion. AlltidFLAT er ingen suksess;
retning, størrelse, preMAE/MFE, kostnad, dekning og tidsblokk-usikkerhet rapporteres
separat. Lange støttemål bruker forhåndsvalgt alltidLONG/kjøp-og-hold som referanse.

Ingen DATA/RUNS-rydding uten retention-eieren; originalkvitteringer/checkpoints
bevares. Ingen ny historikkmappe. Tidligere status og hele lesehistorikken
(32Markdown/4570linjer ved98068b67) finnes i Git og originale154-lesekvitteringer.
Metoden er [docs/NATIVE_LEARNING.md](docs/NATIVE_LEARNING.md).
