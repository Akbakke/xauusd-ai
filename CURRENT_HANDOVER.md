# CURRENT HANDOVER — XAUUSD M5 Entry

Autoritet: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current` på
`gx1-3090-lan`. Les GX1_RULES; NEXT_RUN_POLICY eier arbeidsstatus. Prosesser og
terminalkvitteringer overstyrer prosa. Én integrator, én capped CURRENT-jobb,
fryst kilde/policy under kjøring. Originalmål:
[docs/ORIGINAL_PROJECT_GOAL.md](docs/ORIGINAL_PROJECT_GOAL.md).

## Mål, status og neste beslutning

Kvalifiser M5 Entry LONG/SHORT/FLAT med genuine MBP-1-/indikatorfeatures og
lukket M15/H1/H4/D1-kontekst: retning, størrelse, liten MAE før første MFE,
kostnad, kvalitet og dekning. Prosjektmålet er ufullført; Exit venter.

Brukerens 23.10-bestilte kapasitetsoppfølging er utført: kildekvalifikasjon214,
fast kronologisk sammenligning215 og uavhengige faktisk-resultatkontroller.
Begge komplette referansepakker forkastes som dokumentert retningsforbedring.
Alle aktuelle kjøre-/fitporter er stengt. Ingen engangsplan relanseres.

Featuretilkobling er teknisk bevist, men nyttig prediktiv informasjon og ønsket
Entry-kvalitet er fortsatt ikke kvalifisert. Neste utviklingsbeslutning må rette
seg mot denne forskjellen gjennom eksisterende native sekvens-/prisbanemål,
med kvalitet og dekning sammen. En eventuell ny, fast hypotese må begrunnes og
innrømmes separat etter [docs/NATIVE_LEARNING.md](docs/NATIVE_LEARNING.md).
215 gir ikke grunnlag for featurefjerning, ny lambda-/terskeljakt eller en
native modell-/lossendring. Ingen ny slik endring er valgt eller admittert.

## Fullført kode og kapasitetstest

Ny isolert kontrakt `entry_direction_snapshot_capacity_reference_v1.py` bruker
fast mean-SSE + 1·||beta||² og fri intercept; SUM-alpha er eget observert TRAIN-N
per horisont. Tidligere SUM-SSE + 1 tilsvarer mean-SSE + 1/N. Styrken er fast,
ikke optimalisert. Prediktoren avviser nå også lagret intercept som ikke er
identisk med target_mean − feature_mean@beta. Eget stateskjema bevares.

213 feilet før claim/pytest/payload fordi planen manglet eksakt binding til
reused_guard_worker212. Originalfeilen bevares; ingen213-relaunch.214 rettet
kun bindingen og kvalifiserte samme design/modul: 27/27 syntetiske tester PASS,
0 fail/error/skip, ytre 2,981s; uavhengig faktisk peer PASS. Ingen markedspayload,
genuine fit eller native forward/update i kildekvalifikasjonen.

215 brukte begge hele representasjoner (baseline3345 / relativ3347), alle seks
faste kvartaler 2024Q4–2026Q1, egne modne 1/5/15min-targets og210s12 kvalifiserte
prefixoppskrifter. Alle aktive felt/masker og samme parentrader beholdt.
12 fit-eierkall/36 regresjoner,24 ekte prognosekall,3 syntetiske,9 payloads,
1098400 kodede parentrader,18 overlappende egne TRAIN-paneler separat og75
senere paneler. Ingen ny normaliseringsfit/targets/OBS eller native forward/update.

Worker1036,286s/ytre1037,100s, peakRSS3741519872B under audit4GiB/512MiB swap,
én tråd/64pids og3600s+10s. KildeHEADffed1fcb og policy var uendret under kjøring.
TERMINAL og ytreterminal er COMPLETE/exit0; claimed=completed1, relaunch=false.

## Målt resultat og grenser

Begge armer forbedrer alle18 kvartalsmålinger mot sin lagrede211-referanse;
alle6 pooled dag-/ukeintervaller per arm har negativ øvre95%-grense mot211.
Mot eget prefix-TRAIN-middel vinner baseline bare2/18 og relativ3/18 kvartals-
paneler. Ingen pooled horisont består kravet mot middel.1min-intervaller krysser0;
5/15min-intervaller er positive. Tabellen viser pooled MSE-overskudd mot middel:

| Variant, bps² | 1min | 5min | 15min |
|---|---:|---:|---:|
| Baseline | +0,007120 | +0,083204 | +0,792482 |
| Relativ | +0,004806 | +0,068784 | +0,678507 |

Alle36 egne TRAIN-scorer slår middel; gevinstbaseline0,358–2,879% og
relativ0,397–3,003%. Dette er målt krymping/stabilisering, uten kvalifisert
senere retningsgevinst. Ingen representasjonsvinner er valgt: alle6 pooled
relativ-minus-baseline-intervaller krysser0.

Uavhengig faktisk execution-peer PASS:55 JSON-filer hashet,43 immutable
outputartefakter,300 eksakte lagrede211-metrikkpaneler og432 vektede metrikk-
avstemminger. Kvalitetsreview PASS for1650 aggregatkontroller og forkaster begge
pakker etter låst krav.270 sparsomme måneds-/ukekontraster er korrekt deskriptive.
NPZ-body og bootstrap er ikke uavhengig rekonstruert av reviewer.

Dette er lineære snapshotreferanser på gjenbrukt utviklingshistorikk. Resultatet
beviser ikke at features mangler all temporal/ikke-lineær informasjon eller
forklarer native191s manglende retnings-/Entry-kvalifisering. Det kvalifiserer heller ikke valgt Entry,
prisbanekvalitet, dekning, generalisering eller økonomisk edge.

## Gjenbruk og aktuelle bindinger

215s kvitteringer finnes under
`/home/andre2/GX1_RUNS/NATIVE_M5_SNAPSHOT_CAPACITY_SIX_QUARTER_REFERENCE_20261023_215`:
TERMINAL42befaa3…, ytre39da072d…, RESULTf8639e87…, SUMMARY55931a6e…,
ACTUAL_EXECUTION_PEER7ec4a371…, ACTUAL_DIRECTION_REVIEW349c2757… og
DECISION1117375d…. Eksakte stier/SHA/budsjetter står i NEXT_RUN_POLICY.

NATIVE_COMPARISON_SCOPE_RECONCILIATION435448a3… lukker215s neste scopekontroll:
193s opprinnelige paring med native191 på256 TRAIN+256 Q1CONTROL-prober er
allerede utført i201 (RESULT4012aa2e…, terminalexit0). Alle modeller og tre
horisonter finnes i84 TRAIN- og15 CONTROL-paneler. Ingen gjentakelse er nødvendig.
Dette er den opprinnelige201-referansen;215-versus-native er ikke utført.

206s fysiske relative koordinater og210s12 prefixoppskrifter gjenbrukes; alle
1205 GC-verdi/mask-par,2622 lokale felt og alle fem MTF-pakker bevares.
242 kildebeviste nivåer transformeres med log(P/A), øvrige963 GC-felt beholdes;
444 TRAIN-ankre er eksplisitt ukjente.33 F32-tickceller har ikke bitlik invers.
Alle37 price-action-USD-felt var med; ingen utelatt37-feltsgruppe forklarer211.
215s arvede scope-template-prosa er bare proveniens: eksakt metode/budsjett er
CAPACITY_DESIGN+PLAN+worker (mean-SSElambda1/fem sammenligninger/kvalifiserte210).

191 er siste native epoke:118249 rader/7391 oppdateringer,egen188-initial.
192:alle512 handlinger FLAT,5/6 primære retnings-MSE-paneler verre enn egen
initial; alle tilhørende dag-/ukeintervaller krysser0. Tidlige path-punktfeil
forbedres delvis; dette er ikke oppnådd MFE eller handels-P&L.
175/180 beviser paritet/konsum i32 aktive av48 deklarerte familier×TF-ruter,
ikke prediktiv verdi eller optimalitet. Profilen har7 lokale/5 MTF-familier
etter19Oct-unntaket; åtte spesialister og9.D1/makro beholdes.

## Varige grenser og opprydding

Fysisk grunnlag er23 gjenbrukte måneder: TRAIN mai2024–desember2025 og
CONTROL januar–mars2026. April–juni venter fysisk binding; råkildemålet er26.
055s hel-TRAIN-normalisering brukes ikke til interne prefixevalueringer.
M5 er beslutningsklokke;1–480min prognosedomene er ingen taps-/holdetidsregel.
Lang støtte bruker alltidLONG/kjøp-og-hold som referanse.

Spread ligger i bid/ask; ekstra4bps er2×2slippage,commission0. Scoped kontroll
fant ingen fortegns-/dobbelavgiftsfeil; quote-til-fill er ukjent. MBP-1 er
aggregert BBO/ordretall/OFI, ikke ordrekø eller skjult dybde.

TEST fra juli er forseglet for tilpasning/utvalg; bare vedtatt månedlig
forwardjournal scorer uten refit. Ingen Exit/broker/live/paper/ordre/spending/
promotion. training_enabled/full_epoch/fullVAL/native_launch=false.
Ingen native modell-/feature-/target-/kostkode er endret i kapasitetsoppfølgingen.

Oppryddingsgrense: ingen DATA/RUNS-rydding i denne bølgen; retention-eier kreves.
Ingen forebyggende repoopprydding. Originalkvitteringer/checkpoints og mislykkede
engangsplaner beholdes. Fire eldre udefinerte rank-IC-felt i policy er endret
fra ikke-standard NaN til JSON null; originalkvitteringer og beregninger er uendret.
