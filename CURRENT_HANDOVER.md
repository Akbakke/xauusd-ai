# CURRENT HANDOVER — XAUUSD M5 Entry

Autoritet: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current` på
`gx1-3090-lan`. GX1_RULES er bindende; NEXT_RUN_POLICY eier arbeidsstatus.
Prosesser og terminalkvitteringer overstyrer prosa. Én integrator, én capped
CURRENT-jobb og fryst kilde/policy gjennom hver kjøring. Originalmålet står i
[docs/ORIGINAL_PROJECT_GOAL.md](docs/ORIGINAL_PROJECT_GOAL.md).

## Aktivt mål og neste steg

Bedre M5 Entry LONG/SHORT/FLAT med MBP-1, SMC og lukket M15/H1/H4/D1:
retning, størrelse, liten MAE før første MFE, kostnad, kvalitet og dekning.
Brukeren prioriterer solid featuregrunnmur og generell robusthet, uten bittesmå
tilpasninger eller overtilpassing. MAE alene beviser ikke invers edge.

191 er faktisk COMPLETE/consumed for den avtalte ene TRAIN-passeringen og
sluttmålingen:118249unike TRAIN-rader,7391oppdateringer,B16/siste9/drop_lastfalse,
16+16ONLINEforwards på256TRAIN+256Q1CONTROL/full480. Canonicalreceipt d151155c…
har guardPASS/exit0 og planlagt RESUMABLE ved optimizer_step_ceiling7391.
Candidate complete=false/bundle_written=false; ingen ferdig trainerbundle eller
ekstraepoke er autorisert. Rootterminal aeee0a93… og actualpeer a92d2941…
har ingen blokkere. Retningsgevinst og aktuelle actioncounts er ennå ikke målt.

191 startet fra187fresh, brukte188owninitial og189savedstate-audit etter190privat
mekanikkvalifikasjon; aldri190private eller177trente vekter. Alle11inputs,
original111RMS,targets/masker/kost/score/arkitektur/optimizer er bevart.
184s eneste endring er MFE-gruppemiddel for1–15 og16–480; det er en eksplisitt
designkonvensjon, ikke et målt optimum.185s133beståtte fokuserte tester gjenbrukes.

FysiskWindowsboot529/22Oct23:09:15.5Z etter528 er bekreftet. Guard rapporterte
4932samples,peak57Ccore/66Cminne/134,83W/1806MiB. Keeper160W/SHA62b53… er bevart.
Controller8108 og native752 er avsluttet. Eksakt191task var Ready/LastResult0/
restart0 og er deretter Disabled; øvrige taskdefinisjoner er bevart.
CanonicalUID1000-prosjektlås er målt fri og ingen nativeworkload er observert.
183kilderoller/182filer har eksakte byte-/statbookends, finalRESULT=recipe;
POST_NATIVE_SOURCE_PRESERVATION71083566… bevarer faktisk HEAD5644cc53/policy8374d3bd
gjennom native. Originalkvitteringer, OBS og checkpoints bevares.

192 er NY, separat savedCPU-sammenligning under det aktive kontinuerlige målet.
Eneste admitted kandidat er192/PLAN_GUARDED.json213b07cf… og guardedworker1786b00d…;
DESIGNb83fcfb5…/preparation6e7d7c50…/staticpeerff6175f1… har ingen blokkere.
Tidligere uførte192prototyper er superseded og uadmitted. AST/compile og
avgrenset matematikk-/metadatareview bestod; numerisk runtimeguard er ikke kjørt.
Kun én audit4G/512M/éntråd/pids64/1800s+10TERM tillates etter clean docscommit og
fersk firefelt EXECUTION_BINDING. Trening/fullEpoch/fullVAL/native/reboot er stengt.

192 leser seks lagrede OBS én gang, én178aggregate, seks små canonical
dynamic_entry_m1_row-projeksjoner og én frossen111LOSS_SCALES JSON for RMS.
Dette er ikke zero-Parquet; ingen fullParquet/features/sekvens-/targetcache/PT/
råMBP eller refit. Samme188initial/191final/matchet177 på identiske rader, mål,
masker og klokker;151 beholdes bare som allerede målteALL-tall via178→177.
Ingen nye191minus151-parintervaller eller151-under191grupper kan utledes.

Vurder retning/amplitude, inklusive+closed MFE/preMAE, kost og faktisk
LONG/SHORT/FLAT-kvalitet/dekning separat etter191EVALUATION. Behold186s åtte
kalenderkvartaler og åtte faste bånd, ekte celle-N/SSE og entrykalenderbootstrap.
Faktisk dag-/ukestøtte og tomme paneler skal vises; under8blokker er deskriptivt.
Ingen favorittmåned/-minutt, terskel-/kostfit, inversregel eller automatisk trening.
Lavere pooledtap eller alltidFLAT fullfører ikke målet.

## Siste ferdige retningsanalyse og kostnadsforklaring

178 gjaldt forrige177funksjon og er consumed.177 valgte LONG0/SHORT0/FLAT256
på både TRAIN og CONTROL; alltidFLAT er ingen selektivitet eller handelsgevinst.
CONTROL nettkontrast-MSE(bps²) på samme rader/masker:

| Minutt | Egen173initial | 167final | 177final |
|---|---:|---:|---:|
|1|57,459|572,260|61,678|
|5|228,358|686,241|227,369|
|15|772,101|1358,138|774,629|

Alle6primærpaneler bedres mot167, men5av6 forverres mot egen173initial.
CONTROLh1 predstd2,461 motobservasjon7,504bps,corr0,0578. Dempet overamplitude
er ikke kvalifisert retning.191s læring/valg må vurderes i192, uten å arve dommen.

186s brede inversjonsreview er consumed,0nyeF/U/fit. Historisk167s faktisk valgte
signaler ga terminalrespons etter deklarerte kostnader:

| Periode | Valgte | Original bps | Motsatt bps | Ukeintervall for forskjell |
|---|---:|---:|---:|---:|
|TRAIN|58|−6,983|−5,115|[−2,183;8,072]|
|Q1CONTROL|154|−6,091|−6,263|[−1,952;1,714]|

Inversjonseffekten skifter fortegn mellom perioder. CONTROLpreMAE øker7,361→7,587
og MFE faller3,185→3,002bps ved inversjon. Ingen universell snuregel er kvalifisert.
Bid/ask-spread ligger allerede i quoteavkastningen; ytterligere4bps er to
deklarerte slippageledd à2bps,commission0. Historisk quote-til-fill/slippagetruth
er ukjent; ingen gunstig VAL-kosttilpasning. Scoped kildeavstemming fant ingen
fortegns- eller dobbelavgiftsfeil. Felles4bps kanselleres i retningskontrasten og
forklarer ikke svak retning alene. PreMAE-straff er en utilityproxy, ingen cashfee.
Disse prisbanemålene er ikke realisertP&L.

## Featuregrunnmur og ubeviste grenser

Aktiv profil er `compact_mbp_smc_ema_atr_clock_macro_entry_v1`:7lokale/5MTF-
familier etter19Oct-unntaket; åtte spesialister og9.D1/makro beholdes.175
kvalifiserte alle695aktiveMTF-felt139×5TF og aktiveGC-verdier/masker med0F32avvik.
SMC, EMA/trend, swing, støtte/motstand, volumprofil og MBP-1 er bevart kausalt.
Manglende observasjoner forblir ukjente. MBP-1 er aggregertBBO/ordretall/OFI,
ikke ordreidentitet, kø eller skjultdybde.

180 målte177s lokale inputavhengighet:16genuineTRAIN,1F/5VJP/0U,
84640paritetsceller delta0 og3317kanoniske koordinater regnskapsført. Alle32aktive
av48deklarerte familie/TF-laner og alle8flytende inputflater hadde lokal respons.
Dette viser konsum, ikke optimal featurebruk eller kronologisk merverdi.
Parkerte felt er ikke generelt kvalifisert eller forkastet. Metodevitner179–183
bevares; ingen ny arkitekturhypotese eller gjentatt diagnose er autorisert.

Aktive beslutningsinputs:TRAIN[2024-05-01,2026-01-01),Q1CONTROL[2026-01-01,
2026-04-01),23gjenbrukte måneder. April–juni er senere utvikling som venter separat
fysisk binding; råkildemålet er26måneder. Faste256+256prober er ikke heleTRAIN,
26måneders OOF eller urørt generalisering. TRAINproben dekker193dager/84uker;
CONTROL68dager/14uker. Korte/sparse perioder får ingen generell bevisverdi.

Entry beslutter på lukketM5.1–480min er prognosedomene, aldri fasttap eller
maksimal holdetid. Skill teknikk, TRAIN-læring, gjenbrukt utvikling,
generalisering og økonomi. Lange støttemål har alltidLONG/kjøp-og-hold som
forhåndsvalgt referanse. TEST fra juli er forseglet for all utvalg/tilpasning;
bare vedtatt månedlig forwardjournal scorer den uten refit.
Ingen Exit/broker/live/paper/ordre/spending/promotion, ekstraepoke eller HPO.

Originalkvitteringer/checkpoints bevares; DATA/RUNS-rydding krever retention-eier.
Historikk ligger i Git og originale kvitteringer. Ingen relansering av consumed
planer. Metoden er [docs/NATIVE_LEARNING.md](docs/NATIVE_LEARNING.md).
