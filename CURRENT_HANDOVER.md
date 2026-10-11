# CURRENT HANDOVER — 11.10.2026

Rebuild og baseline er akseptert. HEAD-avvik er rettet (modellfunksjon v10);
fersk nullstegsbaseline kreves før smoke. Swing stage A og B er målt 11.10 uten
bekreftet edge etter familiejustering; neste evidenskilde er et operatørvedtak.
Ingen dokumentert Entry-edge.
Autoritet: /home/andre2/src/GX1_CURRENT, work/gx1-current.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; prosess og terminal overstyrer prosa.

## Pågående klargjøring

Brukeren ba etter verifisert omstart om ny kjøreplan og nødvendige oppstartskontroller.
configs/research/ENTRY_CORRECTED_PREPARATION_20261010.json binder én ny TRAIN-only
CPU-samplermåling på korrigerte inputs. Eksisterende tre kandidater, batch16 og
18 timers absolutt budsjett beholdes; forrige måling brukte omtrent15 timer.
Ingen modell, optimizer, normaliseringsrefit eller TEST. START/prosess/terminal i
samme run-root under SAMPLER_BENCHMARK_001 avgjør faktisk kjøretilstand.
Deretter gjenstår valgt sampler, fysiske native koordinater og eksakt Entry-only
recipe for planlagt høyst256 steg/4096 Entries. Dette er ikke stortreningsautoritet.
Ferdig rebuild-/baselineplan og alle tidligere engangsplaner forblir lukket.

## Rettelser i HEAD 11.10.2026 (operatørbestilt)

Brukeren bestilte retting av de uløste HEAD-avvikene. Kode og tester ligger i
56d87f49; eksakte filhasher og begrunnelse per rettelse står i
NEXT_RUN_POLICY.current_work.head_deviation_fixes_20261011.

1. Objective-kontrakt v11 bærer `entry_training_phase`. Entry-only deklarerer
   observert utførbart nettomål uten Exit-modell og uten Exit-gradienter; joint
   beholder den frosne Exit-læreren. Treneren sender fasen den faktisk kjører.
2. Alle spesialist-, MTF- og akse-encodere avsluttes med samme parameterfrie
   LayerNorm som hovedencoderen (F12).
3. Vindus-readout er middel pluss siste lukkede bar på alle fire steder (F27).
   Arkitekturversjon v9 → v10; samme parametre og state_dict-nøkler, annen funksjon.
4. Vekt-EMA er bias-korrigert med min(decay, t/(t+1)); steg-telleren lå allerede
   i checkpoint-state, så resume følger samme skjema (F18).
5. Checkpoint-policy v6/v7/v8: en VAL-epoke med konstant Entry-handling kan aldri
   velges som beste checkpoint. Validering teller handlingsfordelingen; native
   full-VAL leser long/short/flat-count; manglende telling feiler lukket (F15).
   Ingen ny numerisk terskel er innført.
6. Forskningseieren merker en konstant (degenerert) referanse eksplisitt.

Ikke endret: hjelpehodene i Entry-only (operatørvedtak 09.10), slippage 2,0 bps
(krever eget kostvedtak og nye målrader), patience/min_delta (ingen oppfunnet tall).

Konsekvens: ONLINE-/TARGET-funksjonen er endret. FRESH_INITIAL_BASELINE_002 gjelder
videre som mål-, koordinat- og databevis, men ikke som funksjonsbaseline. En fersk
nullstegsbaseline på v10 ved samme 4096 TRAIN-/4096 CONTROL-koordinater kreves under
nytt eksplisitt budsjett før noen Entry-only-smoke. Gamle v9-checkpoints avvises ved
bundle-last. Samplerbenchmarkens terminal er observert: SAMPLER_BENCHMARK_001/TERMINAL.json, exit 0, source_unchanged=True på
20ab5fe5. Tre kandidater fullførte; valgt sampler batch 16 med målt
CPU-epoke 7679 s og null modellforwards. Kapasitetsmålingen er modelluavhengig og gjelder
fortsatt for v10; sampleren bindes i recipe først etter fersk v10-baseline.

Samme bølge binder operatørvedtaket om regel 1: navngitte instrumenter som
forskningsinputs og forskningsmål (GX1_RULES.md, siste avsnitt). Native forblir XAUUSD.

Verifisert: fokuserte tester for kontrakter, modell, trener, resume, native full-VAL
og forskningseiere er grønne på rettet kilde; syntaks og `git diff --check` består.
Ubevist: ingen optimizersteg eller ekte-data-forward er kjørt på v10; læringseffekten
av LayerNorm/readout er ikke målt. training_enabled=false; TEST er forseglet.

## Swing-programmets kilder 11.10.2026

Hentemoduser i research_ta_campaign_v1 (7636d0df): fetch-fred-current,
fetch-cftc-cot-archives, fetch-oanda-panel-daily, fetch-bounded-files, alle med
reuse_download for reléerte bytes under eksakt hash. Målt i dag:

- **COT (CFTC):** 18 årsarkiver 2009–2026 hentet, CRC-verifisert, ingen rad parset.
- **OANDA-panel:** 11 av 12 instrumenter som H1 2009-06 til 2026-06-30 (ca. 100 000
  rader hver); USD_CNH ga ingen candles fra startdato og feilet lukket. Forsøk 001/002
  er bevart (manglende seal-nøkkel; granularitet D støttes ikke av leseklienten).
- **FRED:** lesetidsavbrudd på alle serier ved 60 s og 300 s fra forskningsverten.
- **GLD:** arkiv-URL fra 30.09 gir nå PDF via API; ingen maskinlesbar lenke funnet.
- **CME daglig volum/OI:** 403 bot-sperre på alle fire sider.

Relé fra operatørens Mac kreves for FRED (seks CSV-URL-er i policyblokken) og GLD
(nettleser-nedlasting). Filene legges i /home/andre2/GX1_DATA/research/RELAY_20261011/
med sha256sum; import skjer med reuse_download-manifest. Ingen prediktor er
innrømmet, ingen fit kjørt, TEST urørt. Detaljer og hasher:
NEXT_RUN_POLICY.current_work.swing_program_sources_20261011.

## v10 nullstegsbaseline 11.10.2026: målt, ingen læring

FRESH_INITIAL_BASELINE_003 (83adab26, nytt eksplisitt 6-timers CPU-budsjett
PREPARATION_CPU_002) målte den ferske v10-funksjonen (parameterfri slutt-LayerNorm,
mean+siste-bar-readout) på nøyaktig samme korrigerte inputs, normalisering og
TRAIN4096/CONTROL4096-kohorter som FRESH_INITIAL_BASELINE_002; 9,637,663 parametre,
512 ONLINE-forwards, én TARGET-paritetsforward (alle hoder bitidentiske),
null optimizersteg, ingen normaliseringsrefit, 339 s i producer20G.
TRAIN: argmax LONG/SHORT/FLAT [1878, 733, 1485] (_002: [2532, 717, 847]),
side-MSE [916.66, 877.67] (_002 [917.38, 874.26]), 1046 av 4096
argmax endret mot _002. CONTROL: [1779, 830, 1487] (_002 [2317, 884, 895]),
side-MSE [1877.09, 1878.74] (_002 [1875.83, 1875.19]), 1139 endret.
Targets bitidentiske med forrige studie. Dette er en utrent initialtilstand: ingen
læring, ingen edge, ingen native treningsautoritet. Neste native steg er et
operatørvedtak om målhorisont og inputs (D1-evidensen legger effekten i kryss-panel
og makrodrivere, som native ikke har), ikke blinde epoker.

## Swing stage C oppfølging 11.10.2026: forward-sjekk støtter, ablasjon peker på panel + drivere

Eieren fikk blokkmatchet long-null som deklarert sekundærstatistikk (d4d9229f) og
deklarert gruppeablasjon (b07918b9). Forward-sjekk 010 på 2026H1 (108 gyldige h20-
rader, ubrukt av de ni foregående analysene, lest før TEST-forseglingen; fit på alt
før 2026): h20 swing_ridge rank-IC 0,44, treff 0,91 ved 10 % (11 rader, 5 episoder)
og 0,70 ved 25 % (27 rader) i et halvår der bare 36 % av 20-dagersbevegelsene var
positive; alltid-LONG på samme rader scorer likt (timing av long). Pris-alene-ridge
0,44 ved 25 %, HGB invertert (0,37). Den forhåndsregistrerte støtteregelen (10 % og
25 %: treff over shift-p95 og blokk-null-p for fanget mål < 0,05) ble IKKE oppfylt
(0,035 ved 10 %, 0,065 ved 25 %) → INKONKLUSIV etter deklarert lav styrke, men
støttende på alle metrikker. Episodeøkonomi: 5 episoder, +803 bps/episode (beskrivende).
Ablasjon (h20 swing_ridge 5 %, treff/familie-p): full 0,781/0,043; uten panel
0,647/0,83 (effekten borte); uten drivere 0,721/0,27 (sterkt svekket; 013 feilet
lukket i porteføljefamilien fordi den fullt dekkede ridge-porteføljen gikk til null
egenkapital fra 2017-04, selektive tall beregnet post hoc fra lagrede prognoser med
eierens funksjon); uten COT 0,746/0,14; uten kalender 0,776/0,050 (inert); uten
dollarindeks 0,831/0,011 (sterkere; informasjonen ligger alt i FX-panelet). 15
analyser på 2011–2025. Verdikt: long-timing-effekten bæres av kryss-panel + makro-
drivere, er foldrobust og retningsstøttet forward, men ikke regelbekreftet. Neste:
frys 70-feltsflaten (uten dollar) som eneste kandidat, månedlig forward-journal
som eneste bekreftelsesrute, v10 nullstegsbaseline under nytt eksplisitt budsjett.
Ingen prediktor innrømmet; TEST urørt; ingen native endring.

## Swing stage C 11.10.2026: long-timing-spor på kanonisk rute, ikke bekreftet

H.10-retry (_002) ga den daglige nominelle brede dollarindeksen 2009–2026 (pakke 1
med fra/til og uten lastobs); pakkene 3/4 er månedlige. Eieren bærer indeksen som
egen gruppe med ukentlig publikasjonslag (brukbar etter slutt på tirsdag uken etter)
og gjør D1-lane valgfri. Forsøk 007 feilet ved start (eieren krevde D1-lane) og er
bevart uten resultat. Forsøk 008 (73 felt: A, COT, panel, drivere, FOMC-kalender,
dollar; 15 årsfolds) og 009 (samme inputs, to-års holdouts): ingen celle oppfyller
regelen. XAU h20 swing_ridge er familiesignifikant på treff i 008 ved 5 % dekning
(0,781, familie-p 0,043) og sterkere i 009 ved 2/5/10/25 % (0,872/0,831/0,784/0,698,
familie-p 0,022/0,0085/0,007/0,026); fanget mål er aldri familiesignifikant under
sirkulær-shift-nullen. Diagnose (post hoc, ikke forhåndsregistrert): de valgte radene
er 87–90 % LONG; alltid-LONG på de samme radene treffer 0,79 mot modellens 0,78, så
retningsvalget tilfører ingenting — effekten er NÅR en 20-dagers long betaler.
Prisstate-ridge alene har ingen effekt, så posisjonering/panel/drivere/kalender/
dollar bærer den. Blokkmatchet long-null (samme antall og lengde episoder per år,
tilfeldig plassering): treff p < 0,0005 og fanget mål p 0,018/0,030 (008), 0,003/
0,004 (009) ved 5/10 % — enkeltceller, ikke familiejustert. Episodeøkonomi 10 %/20 d:
95 episoder +119 bps (008), 90 episoder +169 bps (009) ex finansiering (~30 bps per
long-episode). Ni analyser på samme 2011–2025-historikk. Verdikt: ubekreftet
long-timing-spor; neste er blokkmatchet null i eieren, én forhåndsregistrert
forward-sjekk på 2026H1 (ubrukt, før TEST-forsegling, lav styrke oppgitt) og
gruppeablasjon. Ingen prediktor innrømmet; TEST urørt; ingen native endring.

## Pooled instrumenter 11.10.2026: ingen konsistent signal på tvers

Operatørens «Ja, kjør og hent inn resten»: eieren run-swing-pooled (9717ffbd) kjørte
XAU, XAG, XPT, XPD, XCU og WTI som egne mål med egne COT-koder, delte drivere og
kryss-panel, 68 felt uten D1-lane, samme folds og kost, én felles Westfall–Young-
familie over 336 celler (7668f071). Ingen celle oppfyller regelen (h20, dekning
≤ 25 %, familie-p < 0,05 på både treff og fanget mål). Treffrate alene er
familiesignifikant for XAU swing_ridge h20 ved 2/5/10 % dekning (0,885/0,831/
0,764, familie-p 0,022/0,015/0,044) og XPT a_ridge h20 ved 25 %; XPD a_ridge h5
ved 2 % oppfyller begge metrikker, men ved h5. HGB-effekten replikeres ikke på
tvers (h20 10 % treff: XAU 0,68, de andre 0,43–0,56), og dens episodeøkonomi er
negativ for XPD, WTI og XAG. Eneste konkrete spor: XAU h20 ridge på 68 felt uten
D1-lane. Forbehold: XAU gikk på H1-ruten her, ikke kanonisk M5; familiejusteringen
dekker denne studiens celler, ikke dagens serie av sju analyser på overlappende
data. FOMC-sider 2009–2028 er hentet; BLS er bak bot-sperre; H.10-indekspakkene
kom tomme (bilaterale kurser hentet). Verdikt INKONKLUSIV.

## Swing stage B forsøk 006: familiejustert inferens, ikke bekreftet

Westfall–Young max-t over alle 56 selektive celler med 2 000 felles sirkulære
shift (5ca376e5, d92eb260): de sterkeste cellene, swing_hgb h20 ved 2 % og 10 %
dekning, har rå p 0,0085, men familiejustert p 0,39 og 0,23; minste familie-p i
hele familien er 0,23 for treff og 0,38 for fanget mål. Ingen celle under 0,05.
Den selektive h20-effekten er forenlig med utvalg over 56 celler. Stage A og B
lukkes INKONKLUSIV uten bekreftet edge. Mer uavhengig evidens kreves før ny
inferens: pooled instrumenter som egne mål, annen foldstruktur, Fed DDP for
dollarindeksen. Ingen prediktor innrømmet; TEST urørt; ingen native endring.

## Swing stage B 11.10.2026: drivere fra primærkilder, sterkere h20-effekt

FRED svarte ikke fra verten, så driverne ble hentet fra primærutgiverne under
egne manifester (cf14d394): Treasury par-/realrentekurver 2009–2026, Cboe VIX og
GLD-beholdning via Internet Archive-kopi 14.02.2026. Drivereieren (5c3d9cf9) gir
13 felt med tilgjengelighet etter slutt på dag D+1 New York; 4 533 komplette
dagsrader. Forsøk 005 (aaed881d, 237 felt): porteføljefamilien INKONKLUSIV for
begge. Selektivt ved h20, swing_hgb: treff 0,80/0,80/0,71/0,69 ved 1/2/5/10 %
dekning (18/21/28/51 episoder, alle 15 år), over shift-p95 på alle fire,
fanget mål over p95 på tre; rank-IC +0,006. h5 bekrefter ikke, så den
forhåndsregistrerte regelen er ikke oppfylt. Episodeøkonomi 10 %/20 d: 51
episoder, treff 0,67, +184 bps per episode mot +63 ubetinget, t ≈ 2,1; 2011 og
2023 står for 6 100 av 9 360 bps, resten gir +105 per episode over 31 episoder.
Verdikt: INKONKLUSIV, men dette er det sterkeste selektive funnet i prosjektet.
Før noe mer sies kreves familiejustert inferens over alle 56 selektive celler
med felles sirkulær-shift-null, foldstruktur-sensitivitet og ablasjon av
drivergruppen. DTWEXBGS er utelatt til Fed DDP-pakken er bundet.

## Swing stage A forsøk 004: seeds og episodeøkonomi

Seeds 1–3 (f481c993, da5df963) ga bit-identiske HGB-resultater: med early_stopping
av og uten subsampling er HistGradientBoostingRegressor deterministisk, så
seed-kontrollen er tom for denne læreren. Episodeøkonomi ved 10 % dekning, 20
dagers hold, én utførbar handel per episode, bid/ask og 2 bps slippage, uten
finansiering: swing_hgb h20 64 episoder på 15 år, treff 0,64, +91 bps per episode
mot +63 bps for ubetinget 20-dagers long med samme fyll; overskudd
+28 bps med t ≈ 0.6; 2011 (+2 752) og 2015 (−2 254) dominerer; verste
episode −845 bps; long-finansiering ved 5,4 % ville tatt ca. 1 800 bps. swing_ridge
124 episoder, +88 bps, t ≈ 0.7, uten at radnivå-treffen er over null. a_hgb
+29 bps, altså lik driften. h5-økonomien for swing_hgb er negativ. Verdikt:
INKONKLUSIV. Treffraten er over null, økonomien kan ikke skilles fra ubetinget
long med 64 episoder. Ikke GO; stage B-hypotese med behov for flere uavhengige
episoder (lengre historikk eller pooled instrumenter) og driverne fra reléet.

## Swing stage A forsøk 003: utvalg innen hvert år

Med utvalg innen hvert fold-år (2624b635, d9781e41) forsvinner ridge-effekten helt
(treff 0,55–0,58 ved 1–25 % dekning, under shift-null). swing_hgb ved h20 viser
derimot treff 0,73/0,73/0,69 ved 1/2/10 % dekning (45/78/403 rader, 18/22/64
episoder, alle 15 år, 87–96 % LONG), over sirkulær-shift p95 på de tre dekningene,
med fanget mål +2,0/+2,0/+1,6 ATR over p95. 11 av 15 år har treff over 0,5,
sterke år 2014/2016/2019/2022/2024/2025, svake 2013/2017/2018. h5 bekrefter ikke,
porteføljefamilien er INKONKLUSIV og bulk rank-IC er −0,04, så den
forhåndsregistrerte regelen er ikke oppfylt. Dette er prosjektets første positive,
ikke-refuterte selektive effekt og er INKONKLUSIV til stage B har bekreftet den
med HGB-seeds, FRED-drivere, episodenivå-økonomi med kost og familiejustert
inferens over de 56 selektive cellene. Ingen prediktor innrømmet; TEST urørt.

## Swing stage A målt 11.10.2026: ingen porteføljeedge

ENTRY_DIRECTION_SWING_D1_20261011_002 (eier run-swing-d1, 4c475e92): 224 felt
(7 A-felt, 4 COT, 44 panel over 11 instrumenter, 169 D1-lane uten candle), 15 årsfolds
2011–2025, h5/h20, ridge/HGB matched mot A. Porteføljefamilien ved h20: swing_ridge
NO_GO mot alltid-LONG (−0,67 bps/dag, simultant intervall [−2,28; +0,94]), swing_hgb
INKONKLUSIV (−0,40, [−3,46; +2,66]). Selektiv treffprosent for swing_ridge var høy ved
1–5 % dekning (0,93/0,86/0,76), men diagnosen viser at radene nesten utelukkende er
2011-folden (39 av 40, 133 av 200), nesten bare LONG, fra folden med minst fit-sett
under en ensrettet oppgang; h5 bekrefter ikke. Prognosestørrelser er ikke
sammenlignbare på tvers av folds, så pooled dekningsrangering velger den villeste
folden, ikke ferdighet. Verdikt: ingen selektiv edge dokumentert. Forsøk 001 feilet
lukket i evalueringen på instrumentenes helligdager og er bevart; eieren bærer nå
siste lukkede instrumentbar. Stage B krever relé (FRED, GLD), kalender og
fold-standardiserte selektive endepunkter. Ingen prediktor innrømmet; TEST urørt.

## Gjort og akseptert

Featureauditen på 569d0054 reviderte alle 254 lokale signalfelt, 71 eksakte
kontekstaliaser, ett sesjonsfelt og 190 MTF-felt. Fire funksjonsfeil og misvisende
metadata er rettet. 141 fokuserte tester besto; metode, hvert felt og avgrensninger
står i [docs/FEATURE_AUDIT.md](docs/FEATURE_AUDIT.md) og feltregisteret ved siden av.

Ny generasjon: HISTORY2009W_FEATURE_REPAIR_20261010. Korrigerte MTF-/M5-/M1-felt,
berørte labels og immutable kildebindinger er publisert gjennom eksisterende eiere.
Alle 5 523 147 komplette pre-TEST M1-rader og 392 143 437 aliasverdipar er kontrollert.
Alle 652 552 TRAIN-/70 880 VAL-sekvenser og snapshots er rekonstruert mot korrigert M5.
Kanonisk readiness, spesialistruting og hele fysiske TRAIN-/VAL-indeksen består.

Base-normalisering er tilpasset én gang på hele fysiske TRAIN før sampling:
652 552 Entry-rader, 955 670 unike M5-inputrader og 3 978 505 unike M1-inputrader.
Fersk TRAIN-summary-fit og splitbindinger er publisert uten ny basefit.
VAL-/TEST-fit er null. Faktiske normaliserte NumPy-/PyTorch-input, bitlike aliaser,
kausale MTF-ruter og Entry/Exit-førstetilstand består.

FRESH_INITIAL_BASELINE_002 fullførte 21:36 UTC med exit 0 på uendret 25cde83c.
Eksisterende komplette modell har 9 637 663 parametre, alle åtte familier,
96 lokale M5-steg, 254 signalfelt og originale native MTF-vinduer.
Fersk seed 20260911; ONLINE/TARGET har identiske vekter og samme aktuelle funksjon.
Alle aktive output-head-kontrakter og bitidentisk ONLINE/TARGET-forward besto.

4096 TRAIN- og 4096 CONTROL-rader er målt ved nøyaktig tidligere bundne tidspunkter;
ingen manglende tidspunkt eller erstatningsrader. Netto-targetene er bitidentiske
med forrige studie. Baseline ble målt på CPU gjennom producer20G/512M på 301 sekunder,
med 512 ONLINE-forwards og én TARGET-paritetsforward. Ingen optimizer ble opprettet,
null optimizersteg, ingen normaliseringsrefit, gamle vekter eller TEST-analyse.
Tre av 8192 argmax-handlinger endret seg fra gammel initialbaseline. Sammenligningen
inneholder også tidligere GPU mot nåværende CPU; den isolerer ikke kun featureeffekten.
Initialprediksjoner og feilmål er ikke læring eller dokumentert edge.

Samlet aksept og uavhengig kontroll av metrics, koordinater, tilstandsbytes og ti
vellykkede terminalkvitteringer:
/home/andre2/GX1_RUNS/FEATURE_REPAIR_REBUILD_20261010_001/ACCEPTANCE_REVIEW_001/RESULT.json
Alle kjørebevis og originale avvik ligger i samme run-root.

Brukerbestilt omstart etter ferdig baseline er verifisert: Windows boot
10.10.2026 21:51:28 UTC (23:51 norsk tid), ny WSL-boot og signert GPU-telemetri.
Maskinens prosjektjobber, GPU-beregninger og prosjektlåser var ledige før omstart.
Alle 160 native kildebindinger og 30 sentrale filer, inkludert initialtilstandens
bytes, er uendret. Fsync-publiserte kontrollkvitteringer er intakte. Native
treningsoppgave er deaktivert; omstarten startet eller autoriserte ingen trening.
Bevis: samme run-root, HOST_BOUNDARY_002/POST_REBOOT.json.

## Bevart avvik og grenser

Trygg Windows-/WSL-omstart ble gjennomført mellom core og komplett M1. Tre ferske
kontrollfiler uten fsync var tomme etter omstart; tomme originaler er bevart,
observasjoner gjenopprettet fra Mac og atomisk publisering brukt videre.
POSTBUILD_REVIEW_001 fikk OOM ved feil 4 GB-klasse etter beståtte komplette
sekvenskontroller. Kun uferdig spesialistaudit ble kjørt i eksisterende producer10G.
Første baselineoppstart manglet eksplisitt CPU-synlighet og stoppet før konstruksjon.
Etterfølgeren skjulte GPU før PyTorch-import; øvrige metoder og vakter var uendret.

Ukjente rådatagap er fortsatt ukjente; ingen imputering eller antatt markedslukking.
Deklarerte kostvilkår er ikke bevis for historisk kostsannhet. CONTROL er gjenbrukt
utviklingsdata. Ingen Exit-rollout, all-trades-/åpen-posisjonsøkonomi eller profitt
ble målt. Nye baselinekoordinater gir ingen native trenings-/sampler-/GPU-autoritet.
Tidligere benchmark- og treningsbevis tilhører sin gamle generasjon.

## Læring og neste beslutning

Forrige læringsstudie ga REJECT_ENTRY_QUALIFICATION: CONTROL4096 valgte 4095 FLAT
og én SHORT. Retnings-MSE 7357,64 mot konstantens 7360,93 ga ingen statistisk påvist
forbedring. Den nye nullstegsbaselinen endrer ikke denne konklusjonen.

Rebuild-/baselinemålet er ferdig; tilhørende engangsplaner og byggebudsjett er lukket.
Ny bestilt klargjøring har egen binding og CPU-grense ovenfor; native launch er stengt.
Ingen automatisk trening, forlengelse eller relansering av konsumerte planer.
training_enabled=false; TEST er forseglet. Ingen live/paper, broker, ordre, spending,
promotion eller opprydding. Bevar én agent, én tung CURRENT-jobb, alle genuine
features, åtte familier, kausalitet, vakter, originalkvitteringer og checkpoints.
Ingen fast tapsgrense eller maksimal holdetid.
