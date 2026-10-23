# CURRENT HANDOVER — XAUUSD M5 Entry

Autoritet: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current` på
`gx1-3090-lan`. GX1_RULES gjelder; NEXT_RUN_POLICY eier arbeidsstatus.
Prosesser og terminalkvitteringer overstyrer prosa. Root er integrator; én capped
CURRENT-jobb og fryst kilde/policy under kjøring. Originalmål:
[docs/ORIGINAL_PROJECT_GOAL.md](docs/ORIGINAL_PROJECT_GOAL.md).

## Aktivt mål og konkret funn

Brukerens nye mål23.10 er bedre prognoser/retningsedge for M5 Entry
LONG/SHORT/FLAT, med liten MAE før første MFE og kvalitet sammen med dekning.
Målet er aktivt og ufullført. Teknisk konsum, TRAIN-fit, senere generalisering
og økonomisk edge er forskjellige bevis. Exit venter.

216 kildeaudit fant at alle1205 GC-verdier og1205 tilgjengelighetsmasker er med
på96 lokale M5-rader/currentcontext. Native MTF-vinduene16/64/96/96/252 har0
GC-felt i både full253- og kompakt139-flaten. GC har altså lang TF-tilstand og
historiske sammendrag, men ingen lange egne TF-sekvenser i disse vinduene.
Kildene har319/246/226/215/199 verdier for M5/M15/H1/H4/D1.
Dette er ikke bevist årsak til svake prognoser og ingen dom over alle features.

217s targetfrie fysiske audit er COMPLETE/consumed. BBO-støttens plassering
innen delvis observerte bøtter mangler som modellfelt. Positive trailinggap
TRAIN/CONTROL: M5 78/34, M15 331/69, H1 3746/609, H4 11355/799,
D1 65237/4698. D1 har bare346/52 unike observerte kildebøtter; foreldre er
ikke uavhengige observasjoner. Worker1,826s/ytre2,095s,peakRSS230604800B,
0 targetutfall/fit/nativeforward. Uavhengig actualpeer PASS5961 kontroller.
Ingen217-relaunch; eksakte kilde-/resultatbindinger står i NEXT_RUN_POLICY.

## 218 fast kandidat og neste gate

218 DESIGN912dfc57… velger én additiv temporal/support-pakke:

- Behold lokale inputs, base253/kompakt139, original055, targets/kostnader,
  åtte spesialister og niende D1/makro, eksisterende encoder/attention/readout.
- Legg egne GC-verdi/maskhistorier på de eksakt samme allerede valgte native
  TF-radene, før eksisterende PE/familieencoder. Ingen5×5-duplisering/padding.
- Behold alle1205 verdi/mask-par og rette eiere; legg én BBO trailinggapverdi
  perTF til med eksisterende strukturell BBO-validmaske. Bredder639/493/453/431/399.
- Nye normaliseringsflater `gc_mtf_*` er separate fra original055. Bare unike
  fysiske TF-rader konsumert av TRAIN-vinduene kan tilpasses; ingen CONTROL-fit.
- BareNone slår kandidaten av. Delvis metadata/input avvises. Kandidatprofilen
  binder uendret basisprofil samt layout- og normaliseringshash.

Kildeimplementasjon er skrevet i kontrakt/feature-/modell-/profil-/arkitektureier,
Dataset/batch og to fokuserte testfiler. Kildepeer fant og rettet ny historisk
join som ellers kunne bære en eldre GC-bøtte over en helt manglende bøtte, samt
1ns-endepunktfeil som float64 ellers kunne skjule. Gammel lokal GC-asof er bevart.
GC-builder og nativecache deler canonical fasene H4=02:00 og D1=22:00; exactjoin
er kildekonsistent, mens fysisk024/cache-paritet fortsatt må måles.

218-kvalifikasjon er FAILED/consumed:41cases,15PASS og26failure/error fra én
felles registerfeil (genererte featurelag ble brukt som modellspesialister).
Ingen markedspayload/genuinefit/nativeforward; kilden var uendret.219 åpner
kun kobling til canonical8-spesialistregisteret.15beståtte cases gjenbrukes;
26feilede kvalifiseres etter minste rettelse under ny plan, ingen218-relaunch.

Neste gate er én frosset, kildebundet CPU-kvalifikasjon med syntetiske tabeller,
masker, unike TRAIN-rader, Dataset→collate→forwardbinding og prognose-/gradient-
utslag på hverTF. Dette er mekanikkbevis, ikke lærings-/edgebevis. Aktuell
innrømmelse er nå gitt til41 cases én gang, audit4GiB/512MiB/64pids/éntråd/180+10s;
plan3e5ec726… og kildepeer d033f4cb… er bundet i NEXT_RUN_POLICY. Ingen gammel plan
relanseres. Egen fysisk produsent/normalisering, fersk initialbaseline for den
endrede ONLINE-funksjonen, optimizer/resume-paritet og eventuell én fast native-
sammenligning må bindes separat. Produksjons-/servingbundle er ikke åpnet.

Senere prognosesammenligning må bruke samme faktiske TRAIN/CONTROL-populasjon,
prisfasit, kostnader, seed og oppdateringsbudsjett. Alle480 horisonter og opprinnelig
1/5/15min-retning rapporteres med paired dag-/ukeusikkerhet, valgte og avståtte
muligheter, censoring og LONG/SHORT/FLAT-dekning. AlltidFLAT eller lavere pooled
MSE alene oppfyller ikke Entry-målet. En gevinst gjelder hele pakken; ingen
isolert årsakshistorie mellom ny historikk og støttens plassering er bevist.

## Gjenbruk og målte grenser

215s faste lineære snapshotreferanser er fullført og forkastet som kvalifisert
retningsforbedring. Begge forbedret211, men slo eget prefix-TRAIN-middel bare2/18
og3/18 kvartalspaneler; ingen pooledhorisont bestod. Det utelukker ikke temporal
eller ikke-lineær informasjon. Ingen ny lambda-/terskeljakt eller featurefjerning.
193s originale paring med191 er allerede utført i201; den gjentas ikke.

191 er siste genuine nativeepoke:118249TRAIN/7391oppdateringer, egen188-initial.
192: alle512 handlinger FLAT;5/6 primære retnings-MSE-paneler verre enn initial,
alle tilsvarende dag-/ukeintervaller krysser0. Delvis bedre tidlige path-punktfeil
kvalifiserer ikke Entry eller økonomi.175/180s32 aktive familieruter beviser
konsum/paritet, ikke prediktiv verdi eller32 GC-historieruter.

Fysisk utviklingsgrunnlag er23 gjenbrukte måneder: TRAIN mai2024–desember2025,
CONTROL januar–mars2026. April–juni venter fysisk binding; råkildemålet er26.
055s hel-TRAIN-statistikk brukes ikke som intern prefix-normalisering.
M5 er beslutningsklokke;1–480min prognosedomene er ingen taps-/holdetidsregel.
Lang støtte vurderes mot alltidLONG/kjøp-og-hold.

Spread ligger i bid/ask; ekstra4bps er2×2slippage, commission0. Ingen påvist
fortegns-/dobbelavgiftsfeil; quote-til-fill er ukjent. MBP-1 er aggregert
BBO/ordretall/OFI, ikke ordrekø eller skjult dybde.

## Lukkede porter og opprydding

TEST fra juli er forseglet for all tilpasning/utvalg; bare vedtatt månedlig
forwardjournal scorer uten refit. Ingen Exit/broker/live/paper/ordre/spending/
promotion. training_enabled/full_epoch/fullVAL/native_launch=false.
Ingen DATA/RUNS-rydding; retention-eier kreves. Originalkvitteringer/checkpoints
og mislykkede engangsplaner beholdes. Ingen forebyggende repoopprydding.
Avslutt med fokuserte tester, syntaks, stale-path-scan og git diff --check.
