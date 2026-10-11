# CURRENT HANDOVER — 11.10.2026

Rebuild og baseline er akseptert. HEAD-avvik er rettet (modellfunksjon v10);
fersk nullstegsbaseline kreves før smoke. Ny avgrenset treningsklargjøring er bestilt.
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
