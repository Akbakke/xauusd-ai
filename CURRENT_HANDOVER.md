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

Neste steg er én NY177fullTRAIN-epoke for170s vedtatte terminalvekting, matchet
mot167:118249unikeTRAINrader/batch16/siste9/7391updates og32sluttONLINEforwards
på faste TRAIN256/CONTROL256, alle480horisonter. Global20feltmap og eksakt
currentpolicy/recipe/campaign/hardware/all-projectidle må stemme; ny fysiskboot
må følge525.172fresh er eneste initialeier via egen173initial/audit;176private
vekter er audit-only. Ingen ekstra epoke/HPO/fullVAL eller featureendring.
177EVALUATION_PLAN er forhåndsbundet: egen173initial/matchet167final og151beholdt,
primær1/5/15min, støtte60/120/240/480, retning/amplitude/komponentfeil, alle
valgte/FLAT-rader/kvalitet/dekning og parvise UTCdag-/ukeintervaller.

Root173: `/home/andre2/GX1_RUNS/NATIVE_M5_TERMINAL_BALANCE_INITIAL_20261022_173`.
Root175: `/home/andre2/GX1_RUNS/NATIVE_M5_FEATURE_FOUNDATION_AUDIT_20261022_175`.
Root176: `/home/andre2/GX1_RUNS/NATIVE_M5_TERMINAL_BALANCE_TECHNICAL_20261022_176`.
Root177: `/home/andre2/GX1_RUNS/NATIVE_M5_TERMINAL_BALANCE_SINGLE_FULL_NATIVE_EPOCH_20261022_177`.

## Faktisk siste kjøring

176 er COMPLETE:4privateupdates/4paritetsforwards/32genuineTRAIN/logiskcursor2,
åtte eksakte resume-statefelt.16rader per rolle/full480 paritet har maks0bps
numerisk avvik ved atol1e-4/rtol0; masker/action/ties og målings-RNG er eksakte.
Dette er samme funksjon, ingen ny cross-parameterization-paritet. Opprinnelig163
FAILED beholdes. Privatevekter beholdes kun som audit; hovedtrening er fortsatt
ikke utført for170. RESUMABLE er forventet stepceiling2, ikke ny engangsadgang.

Én GENESIS-invokasjon på fysiskboot525 etter524, guard/trainer/observer PASS/0/0,
ytre3373,874s.2577signerte samples, maks55Ccore/64Cmemory/138,15W/1776MiB;
observerte fysiske effektgrenser160W. Guardtak er85C/80C/300Wlimit/310Wdraw/
12288MiB. WindowsTaskLast0/Disabled/restart0 og0nativekandidater er bekreftet.
En etterkontroll feilet etter vellykket Disable på feil config-filnavn; original
logg beholdes og read-only kontroll med faktisk installersti bekreftet tilstanden.
Ingen nativekjøring ble repetert. SOURCE_CLOSURE verifiserte183sourcebyte/stat,
23små immutable metadatarefs og receiptbevis; stor input-/state-rehash påstås ikke.
TERMINAL/ACTUAL_TECHNICAL_PEER_REVIEW/SOURCE_CLOSURE_REVIEW/RESULT er bevart.

173s egen initial er fortsatt baseline:32F/512rader/full480/0updates, state2eaf88…
fra172fresh. Teknisk konsistens, featurekobling og numerikk er ikke bedre retning.
Alle originalkvitteringer/checkpoints bevares; ingen gammel engangsplan relanseres.

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
177 er neste nye matchede167-budsjettforsøk etter eksakt adgang. Ingen hovedtrening
er startet for170. Inputs holdes faste for å måle tapsendringen.

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
