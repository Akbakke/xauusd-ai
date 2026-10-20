# CURRENT HANDOVER — XAUUSD M5 Entry, kompakt førsteprofil

Autoritet: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les GX1_RULES.md og kjør `bash scripts/gx1_handover.sh --check`.
NEXT_RUN_POLICY.json eier arbeidsstatus og eksakte artefaktbindinger.
Én integrator, avgrensede agenter, én capped tung jobb og fryst kilde/policy.
Originalkvitteringer bevares; konsumerte engangsplaner relanseres aldri.

## Mål og beholdt omfang

M5 Entry skal lære LONG, SHORT og FLAT fra observerbar markedstilstand.
Retning/netto endepunktsretur, signert MFE og konservativ MAE før første MFE
måles separat fra ett minutt til åtte timer; dager er støtte. Ingen Exit-
vurdering eller Exit-trening. Horisonter er ingen maksimal holdetid.
Behold lukket M5/M15/H1/H4/D1 og de rikeste26månedene med GC MBP-1.
Kvalitet og dekning skal måles for både valgte og avståtte entries.

Første profil er `compact_mbp_smc_ema_atr_clock_macro_entry_v1`: MBP-1,
SMC/struktur, EMA/ATR og nødvendig klokke-/makrokontekst. Utvalgte candle-/
chart-pattern- og overlappende felt er parkert. Hele biblioteket og fullmodellens
åtte familier med niende D1/makrokontekst beholdes. Ingen felt er erklært støy
uten måling. Separate SMC/MBP-modeller, ensemble og M30 krever senere samme
kronologiske OOF-evidens for merverdi.

TRAIN er [2024-05-01,2026-01-01), VAL [2026-01-01,2026-04-01), senere
utviklingskontroll [2026-04-01,2026-07-01). Alle26måneder er gjenbrukt utvikling;
TEST fra juli er forseglet. Den bestilte ene nativeepoken er FULLFØRT i108.
Ingen gjenværende treningsadgang: training_enabled=false/fullEpoch=false/fullVAL=false.
Ingen broker/live/paper/ordre/spending/promotion. De to godkjente practice-
leseoppslagene er konsumert. Ingen Exit-vurdering eller Exit-trening er utført.

## Aktuelt — hel epoke ferdig; Entry-kvalitet er ikke kvalifisert

108root: `/home/andre2/GX1_RUNS/NATIVE_M5_COMPACT_SINGLE_FULL_NATIVE_EPOCH_20261020_108`.
Faktisk én ordnet TRAIN-epoke:118249unike rader,7391oppdateringer,batch16,siste9.
Canonical kjøring20.10kl.13:16:01–14:57:31UTC,guardPASS/trainer0/observer0.
Kun ett av tre forhåndsbundne vinduer ble brukt; de to andre gir ingen ny adgang.
Generic RESUMABLE/epoch0/complete=false betyr ingen ekstraepoke eller fortsettelse.
Windows-task er Disabled/LastResult0, ingen bundne prosjektjobber eller ACTIVE.

ACTUAL_SINGLE_EPOCH_TERMINAL_REVIEW.json
SHA87cf966c9cbbecd3542cd16e9d374a45fb10b346474f27c9b5caf7bb7693e543.
PAIRED_LEARNING_REVIEW.json
SHA3b360860d7d80ba8de0d8f511c912afc3726570ebba206082a04f79323db4c9d.
POST_NATIVE_SOURCE_PRESERVATION.json
SHA46eb57e1ed81968a532aa9440089229ef50aacfb705ce0601fe9b355eca339ad.
WINDOWS_TASK_DISABLED.json
SHA335627df0caa981dafe1a76d1e70b0d6cb35c4b0bda8c621f13d17f69fd80fda.

Fast ONLINE-sluttmåling ble sammenlignet med105førmåling på nøyaktig samme
256TRAIN+256CONTROL-rader og seks uendrede target-/maskeflater. Ingen utvalg
eller tilpasning i reviewet. Alle seks samlede TRAIN-komponentfeil falt.
På CONTROL falt samlede MFE-/første-MFE-MAE-prognosefeil, mens endepunktsretur
og støtteforecast ble dårligere. Ettminuttsfeilene økte på begge probegrupper.
Alle512sluttprober velger diagnostisk1minutt som beste predikerte horisont
for begge sider. Kandidatfordeling LONG/SHORT/FLAT er TRAIN0/8/248 og
CONTROL0/108/148. Dette er kandidatdiagnostikk, ikke handler eller akseptert
native beslutningsautoritet. Delvis komponentlæring er målt; hovedmålets
Entry-kvalitet, generalisering og økonomisk edge er fortsatt ikke kvalifisert.

## Ferdig — gjenbruk data og beståtte kontroller

- 66GC-råfiler med892100712records og9173893prisnivårader er bevart. Genuine
T-poster dedupliseres ikke;17PTP-minutter/18bok-eventbøtter er sensurert.
MBP-1 viser BBO/ordercount, ingen skjult dybde eller ordreidentitet.
- 1205GCverdier+1205separate masker er publisert. Fysisk flate er
signal2725/ctx-cont2542/ctx-cat1 og253felt per TF.051kilderekonstruksjon og
053input-liveness bestod på118249TRAIN-/17284VAL-rader. Ingen ny rebuild.
- 055wholeTRAIN Entry-only-normalisering er ferdig, SHA
c02ca97967a314f7dbd6c8bebc341c9782acd9e999e67e4b730db7e99e318c62.
Hele transformen og domene-/finite-/alias-/maskevakter brukes før gather.
- 057binder utfallsblindt valgte256TRAIN-/256CONTROL-prober og ordnet full
TRAIN-populasjon.059kostvilkår er prospektive, ikke historisk kostsannhet.
- Kompaktprofil beholder alle1205GCverdier+masker og61D1/makrofelt.
Aktiv flate2622signal/2525ctx/139MTF,7lokale encodere og25familie×TF-ruter;
candle-ruten beholder74genuine MBP-kanaler. Lokal96bar og MTF16/64/96/96/252.
- 104fresh→105initial32Entryforwards/0updates→106strictsavedstate/RNGaudit
bestod på aktuell closure.107bestod fire private Entry-oppdateringer,
fire paritetsforwards og åtte eksaktlike resume-state-trær; outputavvik0.
Private107vekter ble aldri brukt i108.103rettelsen bestod148fokuserte caser.
- 108endret ONLINE-state til542cf139…; opprinnelig anker9c4cf514… ble bevart.
182kilderoller/181filer og11inputstatidentiteter er kontrollert bevart.
Funksjonsclosure dc1c6b627a2d42a9fea18fcd5dd6bdcfca493a840f32bbf01e1a765a919b785d.

## Målte utfordringer, neste steg og resterende grenser

Den rettede attention-dispatchen bruker samme deterministiske rute i trening
og måling. Dynamisk Entry hopper over ubundet Exit-epochsetter, legacy er
fortsatt strict. Avgrenset hashgjenbruk fjerner gjentatt kontroll innen sitt
kvalifiserte scope. Ingen nye modell-/featureendringer ble gjort under108.

Neste avgrensede diagnose er de fire faktisk lærte task_log_variances i
105initial- og108sluttcheckpoint, sammen med lagrede feilsummer/tall. Ny eksakt
CPU-only-plan/policybinding er nå klar i109; ingen modellbygging,
forwards, updates, refit, GPU eller rådatasettlesing. Kvantifiser bidrag fra
korte og lange støttehorisonter med hele forecast-nevneren; feilenergi er ikke
bevis på gradientdominans eller årsaken til ettminuttssvakheten.
109root: `/home/andre2/GX1_RUNS/NATIVE_M5_TASK_SCALE_AUDIT_20261020_109`. Bare audit_saved_task_scales_bound.py er kjørbar
etter eksakt commit-/policybinding, capped audit4G/512M/900s, én invokasjon.
Originalt ubundet scriptutkast er ikke kjøreautoritet.
Etterpå rettes bare en målt blokkering eller vedtatt designendring. Ingen
ny treningsrunde følger automatisk av108 eller av dette diagnoseforslaget.

Alle1205GCverdier+masker har kildekart for formel/enhet/domene/warmup;
23tekniske formler×5TF har115feltposter. Dette er ikke full uavhengig numerisk
kontroll eller dokumentert nytte av hvert felt. Konkrete gjenværende kontroller
står i docs/FEATURE_AUDIT.md. Native LONG/SHORT/FLAT-kvalitet, valgte/avståtte
entries, horisontkalibrering og beslutningsforklaringer må fortsatt kvalifiseres.
Makroens tidligere oppgitte13felt med manglende originalvintage er ingen full
point-in-time-kvalifisering. TEST er fortsatt forseglet og urørt.

## Maskin, effektivitet og opprydding

108bekrefter fysisk Windowsboot516/20.10kl.13:09:26.500UTC,
REBOOT_CONFIRMED.json SHAc63ea6791d64a1cae1689c63944c09dd405ec29d69653c3dcf2c1fca9a4bdc51.
18.10-omstartbestillingen er også allerede fullført. Ingen ny omstart nå.
Signerte108maksima var57°Ccore/66°Cminne/133.84W/1802MiB; vaktene bestod.

Hele108tok6090.176sinkludert oppstart/avslutning; sessionstart til siste
checkpoint3910.443s, inklusive loader/checkpointarbeid. Ingen ren GPU-benchmark.
Ferdig MTF-NPY, targets og normalisering ble gjenbrukt. Ingen tett sekvens-
memmap ble bygget. Sparse prosessmålinger viser små ekstra fysiske lesinger
under stabil trening; rchar er logiske lesinger, ikke fysisk hosttotal.
Kontrollert fartsgevinst er ikke målt. Full fysisk flate og historisk warmup
leses fortsatt ved oppstart. Signal-only retention av unødvendig context/cat-
backing er en konkret, foreløpig uimplementert I/O-kandidat.

DATA/RUNS ryddes bare av retention-eieren. Ingen Pythonfil er bevist slettbar
ved full referanse-/import-/CLI-/test-/eierkontroll. Git eier historikken.
Metode/bruksanvisning: docs/NATIVE_LEARNING.md. Status står bare her og i
NEXT_RUN_POLICY; regler i GX1_RULES.md, automatisk lesing i gx1_handover.sh.
