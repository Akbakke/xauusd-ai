# CURRENT HANDOVER — XAUUSD M5 Entry

Autoritet: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current` på
`gx1-3090-lan`. Følg GX1_RULES og NEXT_RUN_POLICY; faktiske prosesser og
terminaler overstyrer prosa. Én integrator og én capped tung jobb.

## Gjort nå

151 er COMPLETE/consumed: den eksplisitt godkjente TRAIN-RMS-epoken kjørte
118249 unike TRAIN-rader, batch16/slutt9,7391 optimizersteg. Faste sluttmålinger
på samme TRAIN256/CONTROL256 brukte32 ONLINE-forwards. Guard/trainer/observer
PASS/0/0. Checkpointet er bevart ved den avtalte optimizergrensen;
`RESUMABLE` er lagringsstatus, ingen tillatelse til flere steg eller ny epoke.
Ingen ferdig kandidatbundle ble publisert. Ingen fullVAL/Exit/TEST ble kjørt.

Root: `/home/andre2/GX1_RUNS/NATIVE_M5_TRAIN_RMS_SINGLE_FULL_NATIVE_EPOCH_20261021_151`.
- `TERMINAL.json`:9e59a2620e60c95421dc45735407dfc41f60b8b116a3c1e3716d5e580667c431
- `ACTUAL_NATIVE_TERMINAL_PEER_REVIEW.json`:b9fe3dbdabda8fbbae35706cde762f0fa59c94b2e69b76895743136873f4dcef
- `.gx1-candidate-training-session.BUNDLE/final_online_measurement/RESULT.json`:f0c6c8d042e0ac98a682dd201cb5789bee685c3ab9a2db4d968ba8c84c653eab
- `POST_NATIVE_SOURCE_PRESERVATION.json`:c7c58129c4ee3c8aee014b5bd9417a8fc47a0ab766c8bbdad1601d6f08d63873

Frosset kildeHEAD0fc3ec0e og183roller,13inputidentiteter/policy/rules ble bevart
gjennom terminal. Ny Windowsboot520 er fysisk bekreftet21.10T14:33:23.500Z.
Oppgaven er nå Disabled/Last0. Vaktens målte topper:57Ccore/66Cmemory/
134,68Wdraw/1804MiB; fysisk grense160W. Etterpå signert35/42C/21,97W/78MiB.
Gammel18.10-omstartsbestilling er oppfylt og skal ikke gjentas.

## Etterkontroll og neste steg

152 er COMPLETE/consumed:15,64s CPUaudit av lagrede initial-/sluttprognoser,
null nye modellkall/optimizersteg/refits. Eksisterende full480-validator bestod
med fire bundne parentpointer-kolonneprojeksjoner. Alle seks rå komponent-MSE
ble bedre på TRAIN; CONTROL fikk bedre MFE/MAE, men dårligere terminalretning
og samlet forecast. Korte retningshorisonter1/5/15min ble dårligere i begge prober.
Dette viser blandet komponentlæring; bedre Entry/selektivitet er ikke kvalifisert.

Diagnostiske valg på512M5-rader:39LONG/71SHORT/402FLAT. Begge sidekandidatenes
predikerte maksimum ligger ved1minutt for alle radene. Entry-kadensen er M5.
Valgte LONG-/SHORT-gruppers gjennomsnittlige observerte terminal-minus-preMAE
var negativt på begge prober; dette er komponentdiagnostikk, ikke handler/P&L.
Kandidatscoren er predikert terminal-netto minus pre-MFE-MAE, maksimert over
1–480min. MFE trenes aktivt, men inngår ikke direkte i scoren; ingen direkte
action-/rankingloss trener kandidatvalget. Årsaken til1min-mønsteret er ikke fastslått.

Root: `/home/andre2/GX1_RUNS/NATIVE_M5_TRAIN_RMS_PAIRED_ENTRY_REVIEW_20261021_152`.
- `TERMINAL.json`:d6d3a8f99c0e73f4f10c4515e8e19934e60d224f9847801db3667a569fd5d47b
- `SUMMARY.json`:94eb81adeee5c2831d8d366046666a29007ed6becfa6db4503a61cac01c866a2
- `ACTUAL_PAIRED_REVIEW_PEER.json`:defb019aad8d660f69addf8293efc2afbceb5a7595b9e031a67fb20f4991a6be
- `DECISION_PATH_SOURCE_REVIEW.json`:1fef61310337bbf30e23025a60abbc9df61a5d4dd7b1e0b2fd4ca5a4bcd3a873
- `FORECAST_DOMAIN_CORRECTION.json`:a40decaff07f60d39e913a2b93a68cf9cf14841731a097ad1de6e371f0ed1e71

Etterrapportens opprinnelige8/2 forecastdeling er feil; korreksjonen eier4M5-
basekoordinater/6dagstøttekoordinater og gruppemidlene. Alle10koordinatfeil,
samlede forecast-tall og Entry-resultater er uendret. Originalbevis bevares.
Neste arbeid: skill terminal-/preMAE-feil og deres differanse på felles observerte
celler i allerede lagret152før en konkret mål-/beslutningsrettelse velges.
Ingen ny score, terskel eller horisontgrense er vedtatt; ingen ny trening er åpnet.

Læring, Entry-kvalitet/dekning, generalisering og økonomisk edge vurderes
hver for seg. Alle26måneder er gjenbrukt utvikling, også CONTROL/VAL.
`training_enabled=false`; ekstra epoker/HPO/fullVAL/Exit/broker/live/paper/
ordre/spending/promotion er stengt. De to practice-leseoppslagene er konsumert.
Automatisk oppfølging er kort og hver60.minutt; ingen minuttvis modellpolling.

## Varig mål og definisjoner

Entry beslutter på lukkede M5-barer med lukket M15/H1/H4/D1. M1 er nå bare
finere etterfølgende prisfasit. Prognosehorisonter1–480min er ikke M1Entry,
Exit eller fast holdetid. M1Exit kommer i neste bølge etter Entry-edge.
Lær LONG/SHORT/FLAT fra kausale inputs. Retningsretur, signert MFE og
konservativ MAE før første MFE har separate framtidsmasker; dager er støtte.
Framtidsfasit/masker brukes aldri som beslutningsinputs. AlltidFLAT er ingen
suksess; kvalitet og dekning gjelder valgte og avståtte muligheter.

Førsteprofil: `compact_mbp_smc_ema_atr_clock_macro_entry_v1`, SHA
2ac4fb4ddfac84f588f65363ac8959544233ace286d9648bfebf60b97153f125.
MBP-1,SMC/struktur,EMA/ATR og nødvendig klokke/makro er aktive. Utvalgte
candle/chart-/overlappende felt er parkert; fullbiblioteket og åtte familier
pluss niende D1/makro bevares. Separate modeller/stacking/M30 krever
kronologisk OOF-merverdi. Ingen felt er erklært støy uten måling.

TRAIN[2024-05-01,2026-01-01),VAL[2026-01-01,2026-04-01),senereutvikling
[2026-04-01,2026-07-01). TEST fra juli er forseglet.

## Bestått — gjenbruk

- 147fresh/148nullstegsfullkurvebaseline/149state-RNG/full480gate kvalifiserer
  den eksakte151starten; ingen108/117treningsvekter ble adoptert.
- 055wholeTRAIN-inputnormalisering og111TRAIN-only lossRMS gjenbrukes.
  Native kildekjede183roller har digest27952ad6e1e669cd59d80c763ad0980a619e53725c06bef36b62d8791121b735.
- 121–126:1205aktiveGCverdier/femTF og faktiskM5-context kvalifisert.
  TRAIN dekker1205/1205,VAL1181/1205;24felt er korrekt manglende iVAL.
- 127/128/129/133:139aktivecorefelt perTF,695outputs numerisk kvalifisert.
  Øvrige parkerte fullbibliotekfelt er utenfor den aktive kompaktkontrollen.
- 134:66råMBP-filer/892100712records og sluttmaskjoin PASS; rådata ble
  lest/parset én gang. Ikke gjenta denne råscannen.
- 144:218forwards/13input-VJP/3445genuine inngrep bekrefter funksjonell
  inputpåvirkning på tre TRAIN-vitner. Dette beviser ikke lært featureverdi/edge.
- 146:29fokuserte full480observer-/validatorcases PASS. Kun observerfunksjoner
  ble endret;117s Entry-only optimizer/paritet/resume-mekanikk er uendret.
- 108 er historisk fullført epoke,ikke151adgang eller dagens læringsdom.

Aktivt mål er ikke ferdig: komponentlæring og diagnostisk sporbarhet er målt,
men LONG/SHORT/FLAT-kvalitet/dekning, generalisering og økonomisk edge gjenstår
hver for seg. Originaldata,kvitteringer og checkpoints bevares; DATA/RUNS
ryddes bare via retention-eieren. Tidligere status finnes i Git og originale kvitteringer.
