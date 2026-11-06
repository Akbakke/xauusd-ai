# CURRENT HANDOVER — GX1 markedstilstand før Entry

Eneste arbeidssted: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les `GX1_RULES.md` helt ved første overtakelse og kjør
`bash scripts/gx1_handover.sh --check`. `NEXT_RUN_POLICY.json` er eneste
arbeidsstatus. Terminalkvitteringer overstyrer denne teksten. Tidligere
statusnotater er bevart i Git og originale kvitteringer.

## Aktiv fase etter brukerpresisering 06.11

Definer og lær dynamisk OPP/NED/FLATT-markedstilstand på M5, M15, H1, H4 og
D1 først. M5 er prognosetakt; Entry-plassering, direkte Entry-Q, Exit og
handelsbeslutninger kommer senere. Ingen ny fit er innrømmet.

Én [målsensitivitetsplan](/home/andre2/src/GX1_CURRENT/configs/research/DYNAMIC_MTF_STATE_TARGET_SENSITIVITY_20261106_001.json)
SHA-256 `9aa37c8531de784968c9821960b43cd40ee7722a018827c00561ab1319543e2a`
er fullført og konsumert. [RESULT](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_TARGET_SENSITIVITY_20261106_001/RESULT.json) SHA-256
`b31f022ec3398a1a6400e202f513407830229d97e0e86d73063bf70bc1655ab2`; [TERMINAL](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_TARGET_SENSITIVITY_20261106_001/TERMINAL.json) SHA-256
`db33cf6136e1d020027f4bf4cfad9022f63d5eb6d613531148e900add87326ae`. Den gjenskapte alle 13 lagrede
0,5-arrays eksakt og sammenlignet 1,0 på 29 849 mai–september-foreldre
uten modellkall, TEST eller publisering av 1,0-serie. Exit 0, 10,3 s og
481,1 MB prosessmaks under 4 GiB-vakten; ingen ny kjøring er innrømmet.

| TF / native lead | 0,5 observert | 1,0 observert | FLATT av observerte, 0,5 → 1,0 |
| --- | ---: | ---: | ---: |
| M5 / +5 | 16 798 / 29 849 | 27 936 / 29 849 | 13,8 % → 63,9 % |
| M15 / +15 | 25 679 / 29 849 | 29 062 / 29 849 | 13,9 % → 61,2 % |
| H1 / +60 | 27 770 / 29 849 | 28 278 / 29 849 | 13,4 % → 60,5 % |
| H4 / +240 | 26 829 / 29 849 | 26 632 / 29 849 | 8,4 % → 54,7 % |
| D1 / +1440 | 18 799 / 29 849 | 16 640 / 29 849 | 3,4 % → 52,7 % |

Større grense løser mye av M5-observasjonsmangelen, men gjør FLATT til
majoriteten av de observerte etikettene og reduserer D1-dekningen. Dette
er en mål- og observerbarhetsavveining, ingen modellkvalifisering eller
automatisk beslutning om ny terskel. Gjeldende 0,5-kandidat er fortsatt
semantisk ulåst. Ny TRAIN-semantikk under viser at 0,5 gir mange kjente
aktive M5+5-vinduer med ukjent retning, mens 1,0 gir langt flere FLATT-
tilstander og kortere aktivitetsskifter. Neste målbeslutning må binde hva
vedvarende OPP/NED/FLATT betyr ved M5-prognosetakt, med separate aktivitets-
og retningsmasker, output, tap og måling. Den avgrensede normaliseringskontrollen
er fullført; uavhengig formelkvalifisering av 95 felt/TF gjenstår.

Kildegjennomgang viser en konkret delvis observert fasit som dagens tap ikke
bruker direkte ved senere native ledespørsmål: full lokal range kan bevise
ikke-FLATT selv når OPP/NED er ukjent. [Lagret dekningskontroll](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_BINARY_ACTIVITY_LABEL_COVERAGE_NATIVE_LEADS.json)
SHA-256 `963115cac4f3b7859a47f8155985c4305d361c9763a867b3db49b9575a862331`
fant 21 252 slike M5+5-rader i den eldre 47 602-raders TRAIN-populasjonen og
8 305 i 17 288 senere gjenbrukte utviklingsrader. Det aktive binære
tillleggstapet gjelder bare +1; native +5-tapet krever kjent treklassefasit.
Sekvensloss kan fortsatt gi indirekte gradient via nabotilstander.
Eksisterende målspørring kan hente binær aktivitet ved vilkårlig native lead,
og desember-scoreren gjør det allerede på alle fem TF. Ingen ny fasitmotor
eller scorer trengs nå. [Lagret binær skår](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_BINARY_ACTIVITY_SCORE_NATIVE_LEADS.json)
SHA-256 `c8bb833a3f0777f362108252a9b2c9dbcbb98aaa7cde8fea3df405ec2a451bb4`
viser at den eldre ettpassmodellen ved M5+5 hadde binær Brier 0,063594
mot Markov 0,071860 på gjenbrukt utvikling, mens D1 var verre.
Dermed er et ekstra binært tap en kontrollert hypotese, ikke en klar første
rettelse for OPP/NED. Bind eventuell delvis-observert native loss, maske,
enhet og vekt før fersk baseline; ikke tren den automatisk.

Målkontrakten er ennå ikke låst som meningsfull markedsdefinisjon. Faktoren
0,5 × ATR(14) per TF er en brukerakseptert kandidat, ikke endelig kalibrert.
02.11-gjennomgangen bestod teknisk regelkontroll på 61 TRAIN-spor og merket
bare dette avgrensede reviewet `target_rule_locked=true`; gjeldende
04.11-målvurdering og 05.11-serie har `false`. Ingen korrupt artefakt eller
mål-kodefeil er påvist av spriket. Før ny fit må bevegelse/FLATT, de
forhåndsvalgte framtidige gyldighetstidene, ukjent-maske, output, tap og
måling bindes sammen på hver TF. M1-OHLC gir observerbar M5-tilstand på
126 395 av 238 142 unike TRAIN-tidspunkter (53,08 %); oktober M5+5 har
3 359 av 6 323 (53,12 %). Ukjent er aldri FLATT. Den aktive
[bundne OANDA M1-kilden](/home/andre2/GX1_DATA/data/data/prebuilt/HISTORY2009W_BOOTSTRAP_20260927/artifacts/pretest_direct_m1_quotes_source/manifest.json)
har ett OHLC-tidsstempel per minutt og kan ikke ordne intraminutt høy/lav.
En spredt, umanifestert Dukascopy-tickcache dekker ikke aktiv 2024-TRAIN;
GC-futures er heller ikke fasit for XAU. Den avgrensede 0,5/1,0-ATR-
sensitiviteten over er fullført og konsumert; den valgte ingen ny terskel.
Neste målavgjørelse gjelder semantikken i delvis observerte tilstander,
uten modellscore, VAL eller TEST som terskelvelger. Første
[semantiske TRAIN-plan](/home/andre2/src/GX1_CURRENT/configs/research/DYNAMIC_MTF_STATE_SEMANTIC_CALIBRATION_20261106_001.json)
ble claimet og stoppet før prislesing fordi kalenderkvitteringens faktiske
nøkkel heter `source_reference`, ikke `calendar_source_reference`.
[TERMINAL](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_SEMANTIC_CALIBRATION_20261106_001/TERMINAL.json)
SHA-256 `0e9c683c308541f4766f3eb84822d2aa30ee4bdac1c58a588efebdcc0bab40ad`,
`FAILED_CONSUMED`; ingen RESULT, target-build, modell eller TEST. Den
opprinnelige planen er sperret og kvitteringene bevart.

Den separate [korrigerte semantiske engangsplanen](/home/andre2/src/GX1_CURRENT/configs/research/DYNAMIC_MTF_STATE_SEMANTIC_CALIBRATION_20261106_002.json)
SHA-256 `af921fdacd659f1c090259531ea70e2e9ac1644fcdb1ab1e71df79503184e55c`
er fullført og konsumert. [RESULT](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_SEMANTIC_CALIBRATION_20261106_002/RESULT.json)
SHA-256 `bcd9a82ff6e3218cf58539ec0c4d88cdbb80b1e7c497d2991d31aab071033bf7`;
[TERMINAL](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_SEMANTIC_CALIBRATION_20261106_002/TERMINAL.json)
SHA-256 `ac1873ec5b7ef3f07de335580eaa2f8c8facba850b40613cd84d2922130ee00c`,
`COMPLETE`, exit 0. Eksakt 0,5/1,0 treklassetall mot tidligere sensitivitet,
29 849 mai–september-foreldre, fem TF, 45 forhåndsbundne vitner, 0 modellkall,
0 TEST, 6,8 s og 511 MB prosessmaks. Bundet kilde ble lest fram til mars 2025;
rapport og vitner er bare før oktober 2024. Ingen faktor ble valgt.

| M5 / +5, mai–sep TRAIN | 0,5 × ATR | 1,0 × ATR |
| --- | ---: | ---: |
| Komplett lokal aktivitet (av 29 849) | 29 814 | 29 814 |
| FLATT ved komplett aktivitet | 2 313 | 17 845 |
| Kjent ikke-FLATT, ukjent OPP/NED | 13 017 | 1 878 |
| Månedlig median målgrense, USD | 0,69–0,82 | 1,38–1,64 |
| Månedlig median grense / sitert spread | 1,74–2,14 | 3,48–4,29 |
| Observerte FLATT/ikke-FLATT-skifter på unike M1-kilder | 8 910 | 21 401 |

Skiftene teller månedssensurerte, unike M1-kildeminutter, ikke uavhengige
M5-prognoser. Strengt observerte M5 1,0-retningsløp hadde månedlig median
3–4 minutter blant fullførte løp, under femminutters prognosetakt; ukjente
forløp er sensurert, så dette er ikke totalvarighet for alle retninger.
På D1 finnes også kjent treklasse uten komplett aktivitetsvindu (0,5:
3 105; 1,0: 906), så maskene må forbli separate. Historisk sitert spread
er bare fysisk målskala, ikke gjennomførbar økonomi. Det valgte vitneutvalget
er kontroll av målbetydning, ikke et representativt estimat alene. Neste
målavgjørelse må avklare fysisk meningsfull bevegelse og ønsket varighet
uten å velge faktor etter klassebalanse, dekning eller modelltreff.
På M15 og H1 er også de fullførte 1,0-aktivitetsløpenes månedsmedian kortere
enn henholdsvis +15 og +60 minutter. Ingen av de to grensene kvalifiserer
alene en vedvarende fem-TF-markedstilstand. Den eksisterende M5+5 binære
modellen scorer bedre enn Markov på gjenbrukt utvikling, så en ny binær loss
er ikke en påvist retningsrettelse. [Overgangsmekanismeplanen](/home/andre2/src/GX1_CURRENT/configs/research/DYNAMIC_MTF_STATE_TRANSITION_MECHANISM_20261106_001.json)
SHA-256 `8b7814d978ffc2cad3d410627d5d6239443934185271a3e5ee1adbb45e263b24`
er fullført og konsumert. [RESULT](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_TRANSITION_MECHANISM_20261106_001/RESULT.json)
SHA-256 `6becbc980f770c5029292f18104a9f2b36e81d3b9516d01bf0264c3386b5861a`;
[TERMINAL](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_TRANSITION_MECHANISM_20261106_001/TERMINAL.json)
SHA-256 `5b9543ed169b3eb073065aa4b400738163d914fe76ab81fdfc8c6dd7f3581fd1`,
`COMPLETE`, exit 0. Av 30 818 månedsinterne radnaboskifter med 1,0 ATR på
fem TF var 30 760 mellom sammenhengende hele M1-minutter; 58 krysset
kildegap. Ingen av de 30 760 krevde en oppdatert ATR-grense for å skifte.
Ved 0,5 ATR var 14 268 av 14 305 sammenhengende; 337 av de 14 268 krevde
ATR-endring. M5 alene: 1,0 ga 21 353 sammenhengende av 21 401, mot 8 904
av 8 910 ved 0,5. De raske 1,0-skiftene er dermed i hovedsak faktisk
rullerende range ved uendret grense, ikke et artefakt av markedsstengning.
Dette gjelder binær FLATT/aktiv, ikke OPP/NED-kvalitet. Ingen ny målserie,
faktorvalg, modell, TEST eller Entry. Neste målbeslutning er én fysisk
begrunnet kontrakt for en vedvarende OPP/NED/FLATT-episode, med separate
aktivitets- og retningsmasker. En ren M1-sluttprisretning ble vurdert,
men kan skjule en stor motgående high/low-bevegelse inne i samme minutt og
kan ikke oppgradere `{OPP, NED}` til sikker fasit. Behold den konservative
OHLC-masken; en close-proxy må være separat. Varighet må kontrolleres både i
observerte M1-minutter og ved eksakt prognosetid i UTC, uten å omskrive
eldre etiketter med framtidig varighet.

Fullprofilen ruter 253 felt per TF til modellen. ATR14, EMA20-avstand og rå
støtte-/motstandsavstand har kontrollert formel, variasjon, TRAIN-skala og
kausalt valgt lukket bar i målt scope. 95 tidligere parkerte felt per TF
inngår nå i fullprofilen. `close_position_range20` har full uavhengig fem-TF
formelkontroll. Fem generiske felt har nå en avgrenset 64-barers TRAIN-kontroll
per TF (nedenfor); 89 av de 95 mangler fortsatt en selvstendig numerisk
formelstikkprøve. Dette er ikke bevis for prediktiv verdi.
Råaudit beviser ikke verdier etter normalisering, kvaliteten på de observerte nivåene
eller prediktiv bruk. Første [postnorm-plan](/home/andre2/src/GX1_CURRENT/configs/research/DYNAMIC_M5_OCTOBER_NATIVE_POSTNORM_20261106_001.json)
ble claimet og feilet i ren feilmeldingsformatering: `np.argwhere` indekserte
en tom avviksliste selv om første TFs 8 × 253 verdier lå innen toleransen.
[TERMINAL](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_POSTNORM_20261106_001/TERMINAL.json)
SHA-256 `31460991b2ed2008bd03c38c93176f1fc806b8514486bf8840391075a5837249`,
`FAILED_CONSUMED`; ingen RESULT eller påstand om de fire øvrige TF. Ingen
modell-forward, fit eller TEST ble startet. Originalkvitteringene er bevart.

Den separate [korrigerte engangsplanen](/home/andre2/src/GX1_CURRENT/configs/research/DYNAMIC_M5_OCTOBER_NATIVE_POSTNORM_20261106_002.json)
SHA-256 `172907a002cb8ce808f0836e3166cc8cec9c109550799660e97a84ed30fb92c2`
er fullført og konsumert. [RESULT](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_POSTNORM_20261106_002/RESULT.json)
SHA-256 `dfce8d60a1f9fb9439ca3aa58c66b187c705732d443ba81ecec6878354ebf829`;
[TERMINAL](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_POSTNORM_20261106_002/TERMINAL.json)
SHA-256 `03e1883098699b6cd4d9c0b325160a1ac2ddb0d7fb1099e9435595b1878bf18e`,
`COMPLETE`, exit 0. På hver av M5/M15/H1/H4/D1 matchet alle 8 × 253
normaliserte cache-verdier NumPy-referansen eksakt (`maks |diff| = 0,0`):
10 120 sammenligninger, men bare 22 unike fysiske TF-rader. Binær-/kategorimasker
og tilstedeværende/fraværende nivåverdier passerte. Én CPU-konstruksjon,
0 forward/bakover/optimizer, 0 TEST; 14,5 s og 713 MB prosessmaks under 4 GiB.
Eksakt kildeavvik i en tapsfunksjon ble bundet uten fit-/forward-adgang.
Stikkprøven beviser ikke all-253 gather, indikatorenes formel eller verdi.
Neste indikatorsteg er numerisk og kausal kontroll av de 89 ennå ikke
stikkprøvde feltene per TF, samt bredere kontroll av de fem avgrenset testede;
bidrag må deretter måles på kronologisk utvikling. Den opprinnelige
95-listen er [128-scope](/home/andre2/GX1_RUNS/NATIVE_M5_TREND_EVENT_NUMERIC_REFERENCE_20261021_128/REMAINING_MTF_SCOPE.json)
SHA-256 `fcbc3731d084ef097e44ddc3fb6732100692964f4f3403e0e8a606bc16a9879a`
minus de allerede kvalifiserte feltene i [133-indeksen](/home/andre2/GX1_RUNS/NATIVE_M5_REMAINING_ACTIVE_CORE_NUMERIC_REFERENCE_20261021_133/QUALIFIED_MTF_NUMERIC_INDEX.json)
SHA-256 `f5022afbdcde059440a327d8ea2b9b06cb5bc07a077b482b45b61302232a292f`:
trendlinje 31, candle 21, momentum 15, lokale 16, generiske 6, volum 3 og
squeeze 3 per TF. Den [uavhengige close-position-kontrollen](/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_CLOSE_POSITION_NUMERIC_20261028_001/OUTPUT/RESULT.json)
SHA-256 `7f381bdf8d876471fd67ae0b2d8289737b3bdc42d7d40140bc2b43ba11f9598a`
matchet dagens cache- og formelkilde på fem TF, 0 avvik. De fem øvrige
generiske feltene har nå avgrenset numerisk PASS nedenfor; 89 felt gjenstår
uten selvstendig formelstikkprøve. Trendlinjene prioriteres på grunn av tilstandslogikk og
støtte/motstand. Den tidligere 94-felts revisjonen tok 1 326,7 av 1 800 s;
én ny 95-felts totaljobb er ikke ressurskvalifisert. Revider produsentvis,
med uavhengig formelreferanse og eksisterende datakilder, uten å gjøre
produksjonskoden til sin egen fasit.

Fem tidligere trendlinje-replayer matcher produsent/cache på alle 31 felt og
alle TF i sitt før-01.07.2025-scope; dette beviser transport, ikke at
linjevalg og livssyklus er matematisk riktig. En
[trendlinjeplan](/home/andre2/src/GX1_CURRENT/configs/research/TRENDLINE_CONDITIONAL_PREFIX_20261106_001.json)
SHA-256 `bc4469117ef476250e50ed308941b790da67ed30b134b5bf350e100240f17491`
for 12 betingede geometri-/klokkeverdier og alle 31 representasjonsfelt er
innrømmet én gang, ennå ikke claimet eller kjørt. Direkte capped `--help`
røykprøve bestod etter en importrettelse før innrømmelse. Den numeriske kontrollen bruker bare
før-01.07.2025-prefikset og fysiske TRAIN-vitner fra mai–desember 2024;
helfil-sjekksum leser også senere cachebytes, uten statistisk TEST-inspeksjon.
De 19 livssyklusfeltene og selve linjevalget står fortsatt åpne. En separat
[plan for de fem gjenværende generiske feltene](/home/andre2/GX1_RUNS/NATIVE_M5_GENERIC_SCALAR5_NUMERIC_20261106_001/PLAN.json)
SHA-256 `abbd91e0db3b445fb52583d7a880f151bf4672f609ebd330ceef22987bbca360`
var statisk forhåndskontrollert på 64 faste TRAIN-barer per TF, men eksakt
capped direkte skriptstart for `_001` stoppet før worker-CLAIM eller data:
`ModuleNotFoundError: No module named gx1` ved top-level import.
[FAILED_PRECLAIM](/home/andre2/GX1_RUNS/NATIVE_M5_GENERIC_SCALAR5_NUMERIC_20261106_001/FAILED_PRECLAIM.json)
SHA-256 `5a9986673049d415cb59eec0e04fffa554a4263157b3fe4360cd5ab070cfc92c`;
ingen OUTPUT, RESULT eller numerisk påstand. Planen er konsumert og skal ikke
relanseres. Separat [plan `_002`](/home/andre2/GX1_RUNS/NATIVE_M5_GENERIC_SCALAR5_NUMERIC_20261106_002/PLAN.json)
SHA-256 `8cdfc239032fac94d6affd524e955f9fce27693dd4d8e6311e6e7e395609a711`
endrer kun direkte import/start og planidentitet, ikke formler, utvalg eller
sammenligning. Eksakt capped direkte `--help` røykprøve bestod før claim:
[STARTUP_SMOKE](/home/andre2/GX1_RUNS/NATIVE_M5_GENERIC_SCALAR5_NUMERIC_20261106_002/STARTUP_SMOKE.json)
SHA-256 `85f86ec446dc0b5716978257685732576a5e909d38f13f09cbbcb037efb3bce6`.
Engangsporten er fullført og konsumert: [RESULT](/home/andre2/GX1_RUNS/NATIVE_M5_GENERIC_SCALAR5_NUMERIC_20261106_002/OUTPUT/RESULTS/RESULT.json)
SHA-256 `36f80f529362aa9abe32a66c24820f7b9815c644a63d57d79664544231cd8c6b`;
[TERMINAL](/home/andre2/GX1_RUNS/NATIVE_M5_GENERIC_SCALAR5_NUMERIC_20261106_002/OUTPUT/TERMINAL.json)
SHA-256 `6c7fbee7c35c4bb47a46ee2cec6940d68b572f741b3a38435619b6a3c21c2490`,
`COMPLETE`, exit 0. Alle 5 × 5 × 64 = 1 600 forhåndsvalgte TRAIN-verdier
lå innen toleranse; maksimalt absolutt avvik `2,2081e-7`, null feilrader.
Dette er avgrenset formelsamsvar, ikke full cache, nytte, generalisering
eller økonomi. Begge planene gjenbrukte eksisterende råkilder og cache;
ingen ny modell eller trening.

Vi har ikke kjørt omfattende trening gjennom flere år. Tidligere fem-TF-fit
var ett pass på 47 602 TRAIN-foreldre (2 976 steg); oktoberparet var ett pass
per arm på 29 849 foreldre (1 866 steg hver), med senere score uten refit.
26 måneder beskriver datadekning, ikke treningsmengde. Den lagrede
M5-retningsrangeringen på senere utviklingsrader falt fra AUC ca. 0,845
ved +1 minutt til ca. 0,498 ved +5; på foreldre med begge observerte
retningsfasiter skiftet 48,09 % klasse mellom disse tidene. Det avgrenser
hvilket mål og tidspunkt som faktisk ble lært, ikke alt mulig signal.
Først etter låst mål og kvalifiserte inputs kan én forhåndsbundet
tilstandsfit sammenlignes med enkle referanser på senere utviklingsrader.
TEST og Entry-fasen forblir stengt.

## Gjort og målt

- Begge armers pre-oktober TRAIN-fit er avsluttet på 29 849 foreldre og
  1 866 optimizersteg hver. De egne oktober-scorekjøringene er konsumert:
  6 323 senere foreldre, 396 forwards per arm, null nye optimizersteg og
  signert vakt PASS. Oppstartsoppgaven er deaktivert. Ingen TEST ble brukt.
- [Kontrollens terminalreview](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001/CONTROL_SCORE_003/ACTUAL_ROOT_REVIEW.json)
  SHA-256 `d37c387736b8578fc3ff228aff7eba09d75b8380dcd102624061c6f4d4e9d932`;
  [kandidatens terminalreview](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001/CANDIDATE_SCORE_002/ACTUAL_ROOT_REVIEW.json)
  SHA-256 `02043546ea5da35437843164724d9107cdc0a994042e99b8b07a332b0e71c47d`.
- [Paret oktoberutviklingsrapport](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_PAIRED_DESCRIPTIVE_20261106_001/REPORT.json)
  SHA-256 `82988bb594c31c5d77ac1108b79befb6497dbaac65277b3e6babc8770f17e258`.
  M5 +5 har 3 359 observerte av 6 323 foreldre. Brier for kandidat /
  kontroll / fryst TRAIN-prior / Markov er 0,614672 / 0,615091 / 0,612714 /
  0,615325. Kandidat-minus-kontroll har radvektet 95 % UTC-dag-intervall
  [−0,000861, +0,000032]. OPP/NED-AUC er 0,4898 for kandidat og 0,4897
  for kontroll. M5 fullgrid Brier er 0,686345 mot kontroll 0,685498 og
  prior 0,624951. De andre fire tidsrammene er sekundære i rapporten.
- Av 2 964 ukjente M5 +5-fasiter har 2 950 årsakskode 24:
  intraminutt-rekkefølge ukjent og retning uavklart under M1-OHLC. Ukjent
  er ikke FLAT. Dette er målkontraktens konservative kildegrense, ingen
  påvist kodefeil.

## Bevarte utviklingsmålinger og senere Entry-spor

Aldersinput kvalifiserte ikke M5-retningsverdi. Oktober er utviklingsdata,
og den opprinnelige planen manglet en konkret beslutningsregel bundet før
scoring. Rapporten er derfor deskriptiv og gir ingen retrospektiv PASS.
Ingen uavhengig retningsedge eller økonomisk edge er påvist.

[Entry-path TRAIN-kvitteringen](/home/andre2/GX1_RUNS/NATIVE_M5_ACTUAL_NATIVE_BINDINGS_20261019_057/PHYSICAL_TARGET_RECEIPT.json) SHA-256
`dd95d9ecb43f309635e61ba268ec4db6518a3ee23550aa0c94330ff303a5063d` og [oktoberaksen](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_INPUT_ONLY_20261105_003/OCTOBER_AXIS.json) SHA-256
`2aa5bad2804d6635211ceac21eb9935639198f147d7d1a821188d3a07d6c17c5` har samme 118 249-raders M5-barstartklokkehash
`e5d528ff8fcec2b23f0099834b1c5b42a12489dadfada0a6fc4aeab3b6ff4c5d`. Oktober er radene [29 849, 36 172), 6 323
foreldre. Beslutningstid er barstart pluss lukket M5-bar i den eksisterende
eieren. Dette kvalifiserer rad-/klokkematch, ikke observerbarheten av
framtidige executable endepunkter eller MFE/MAE-stier; tette framtidsbaner
er ikke lagret. [Dekningsrapporten](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_ENTRY_MASK_COVERAGE_20261106_001/REPORT.json) SHA-256
`17943a62ef36020120878c3e72a124c99304fe2f60ea7d7c638926b5175a0b37` er fullført og engangsplanen konsumert. Den målte alle
tette 1..480 minutters horisonter innen oktobercutoff med 0 modellkall,
0 oppdateringer, 121,6 sekunder og maksimalt 2,94 GiB RSS under 4 GiB.
Ved +5 var 6 322 rader foldmodne (én høyresensur); 6 297 hadde executable
terminal-open og observerbar terminal-open bane på begge sider, og 6 322
hadde observerbar closed-prefix-bane. Dette er andre mål enn tilstandsfasiten
på 3 359 observerte rader; bedre dekning er ikke lært retningsedge.
Ved +480 hadde 5 607 av 6 240 foldmodne executable terminal og 4 079
observerbar terminal-open bane. Den eksisterende kildeleseren åpnet hele
pre-2026 TRAIN-kilden ved oppsett, men rapporterte bare spørringer før
oktobercutoff; ingen TEST. Den forhåndsbundne, modellfrie M5 +5-joinen er nå fullført én gang.
[CLAIM](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_ENTRY_DIRECTION_JOIN_20261106_001/CLAIM.json) SHA-256
`03dbcfb1e5836502ed02bc129cbc2b4cc6e98251eb38369642090fca3bf12c34` og
[REPORT](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_ENTRY_DIRECTION_JOIN_20261106_001/REPORT.json) SHA-256
`078f9ea44a096a64d5001799773b9ae8d6e508756c332868fa74dce236101ad0`.
Av 6 323 foreldre var 6 322 innen cutoff og 6 297 hadde executable
terminalutfall på begge sider. Ingen eksakte LONG/SHORT-ties. Kandidatens
radvektede AUC var 0,490519 mot kontrollens 0,490249. Paret forskjell
hadde 95 % UTC-dag-bootstrapintervall [−0,001306, +0,001950]. Ved lik
vekt per UTC-utfallsdag var AUC 0,477911 mot 0,477474. Kandidat mot
0,5 hadde lik-dag-intervall [−0,043802, −0,004016]; radvektet
mot 0,5 omfattet null. 27 utfallsdager. 0 modellkall, 0 oppdateringer,
0 TEST, 0 seleksjon; 116,0 sekunder, 2,72 GiB maksimal RSS under 4 GiB.
Erklærte historiske kostnader er ikke sannhetskvalifiserte. Dette er bare
utviklingsdiagnostikk og gir ingen Entry-gate, uavhengig eller økonomisk edge.
Planen er konsumert og kan ikke kjøres på nytt.

Stans +5-tilstandsscoren som surrogat for Entry-sidegate. Det løpende
OPP/NED/FLATT-prognosemålet på fem TF består som egen fase. Kontraktene viser
at tilstand ved beslutning+5 gjelder lokal M5-retning fra lukkede M1-midtbars,
mens Entry-sideutfallet er netto executable BID/ASK-endring fra beslutningen
til terminal-open. De er ikke samme target. En lesende gjennomgang fant at
direkte native 191/192 valgte FLAT på alle 256 TRAIN + 256 gjenbrukte CONTROL;
177/178 gjorde også bare FLAT, mens 151/152 valgte 39 LONG og 71 SHORT av
512 uten kvalifisert valgt kvalitet. Ingen bred kronologisk direkte Entry-OOF
er lagret i aktivt scope. Gamle rene retningsreferanser 247/249/252/259/215
er konsumert eller avvist; ikke gjenta dem.

Den fysiske Entry-kvitteringen dekker 118 249 TRAIN- og 17 284 VAL-foreldre,
men `dense_future_paths_materialized=false`; den lagrer M1-radpekere og
første masker, ikke en bred LONG/SHORT/FLAT-fasit. Eksisterende
`DynamicM5TargetProvider.targets_for_rows` beregner terminal og path
on-demand. Oktoberrapporten måler bare dekning; 512-raders native panel er
ikke bred kronologisk Entry-OOF. En bred modellfri målfordeling ville bare
telle kjente framtidsmuligheter, ikke kausal forutsigbarhet. Den endrer ikke
Entry-beslutningen og er ikke neste kjøring; en slik diagnose får bare egen
plan ved et nytt, konkret spørsmål som krever den. Skjult maksimum av
framtidsfasit over horisonter er forbudt som Entry-Q.

[Kostautoriteten](/home/andre2/GX1_RUNS/NATIVE_M5_CURRENT_COST_TERMS_20261019_059/COST_POLICY/parameter_authority.json)
SHA-256 `e93ed40451a24344f94716e1f5ebe4cd348e6c7c3120e7509d6d55827bf40522`
er en komplett prospektiv parameterpolicy, men `historical_cost_truth_qualified=false`.
[Brokerevidensen](/home/andre2/GX1_RUNS/NATIVE_M5_CURRENT_COST_TERMS_20261019_059/BROKER_EVIDENCE.json)
SHA-256 `d0dd8b19a4aa8aed82aaa3c20cadc0dae211d14f6edcc797db0e0c2084ee951c`
har 258 fills, men mangler full historisk finansieringsrenteserie,
beslutningsquote-til-fyll-klokke og eksplisitt historisk prisplan. M1
BID/ASK-candles ved barstart kan ikke gjenopprette disse faktaene; 2 bps
slippage per utførelse er en prospektiv antakelse med 1/2/4-sensitivitet.
Rapporter deklarert kost og sensitiviteter, aldri historisk kostsannhet
eller realisert økonomi fra disse bytene.

Mål-mot-mål-kontrollen er fullført én gang etter [engangsplanen](/home/andre2/src/GX1_CURRENT/configs/research/DYNAMIC_M5_OCTOBER_TARGET_ALIGNMENT_20261106_001.json) SHA-256
`90915641b504d18603fd76562a97461fa2872c399283f36014bb0fd1f84e3d39`.
[CLAIM](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_TARGET_ALIGNMENT_20261106_001/CLAIM.json) SHA-256
`b6ee022fc6d75860abe89f862cdf6e30f3a4c760510888a5e39b17fafd20fee0` og
[REPORT](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_TARGET_ALIGNMENT_20261106_001/REPORT.json) SHA-256
`3377e7d092abcccc90705e064031d6c545ec63ca7329978e4b43682c0566f090`.
Felles observerte oktober M5+5-rader: 3 344. DOWN hadde 262 LONG-bedre og
1 067 SHORT-bedre; UP 1 212 LONG-bedre og 317 SHORT-bedre; FLAT 263/223.
P(LONG-bedre | UP) = 0,7927 mot 0,1971 ved DOWN; forskjell 0,5955 med
beskrivende 95 % UTC-dag-intervall [0,5662, 0,6252] over 27 dager.
Ytterligere 2 953 executable rader hadde ukjent tilstand; 15 hadde kjent
tilstand, men ukjent Entry-side; 11 hadde begge ukjent. Tilstandsmålet
samsvarer dermed sterkt med sideutfall på den observerte delmengden,
men fullpopulasjons-samsvar og forutsigbarhet fra kausale inputs er ubevist.
Samtidig rangerte de lagrede modellprognosene M5+5 nær tilfeldig på oktober.
0 modellkall, 0 optimizersteg, 0 TEST, 0 action-seleksjon; 110,5 sekunder,
2,75 GiB maksimal RSS under 4 GiB. Kostsannhet er fortsatt ukvalifisert.
Ingen Entry-gate eller økonomisk edge følger av denne utviklingsrapporten.
Planen er konsumert og kan ikke kjøres igjen.

Lagret sluttmåling på 16 798 observerte TRAIN M5+5-rader har kandidat-Brier
0,604215 mot fryst prior 0,609071. Oktober har 0,614672 mot prior 0,612714.
På hele M5-horisontgridet var kandidaten verre enn prior også i TRAIN:
0,680811 mot 0,618665. Det native +5-endepunktet har eksplisitt tapsvekt
1,0 og faktisk gradient; ren 5×1440-tapsfortynning er ingen påvist forklaring.
Kandidat-only TRAIN-rangmålingen er fullført én gang etter [engangsplanen](/home/andre2/src/GX1_CURRENT/configs/research/DYNAMIC_M5_OCTOBER_TRAIN_RANK_20261106_001.json) SHA-256
`1da8201cc59a5e69308b0b6c8c3bef6a311a9f52711d779df2254f050b29e553`.
[CLAIM](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_TRAIN_RANK_20261106_001/CLAIM.json) SHA-256
`f75d5382c858b4a693e971ef9af10f1b652de487e73b665c6a4f8be8d676a94b` og
[REPORT](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_TRAIN_RANK_20261106_001/REPORT.json) SHA-256
`a1518c6ae33705ce1229ee9eb77e87c9079f227bb0b48f3dd089fe2f5462ec3b`.
På 16 798 observerte TRAIN M5+5-rader var 14 485 OPP/NED. Kandidatens
in-sample OPP/NED-AUC var 0,521784; beskrivende 95 % UTC-dag-intervall
for AUC minus 0,5 var [+0,012716, +0,030792] over 131 utfallsdager.
Lagret Brier ble gjenskapt eksakt innen avrunding (0,6042147567).
Senere oktoberutvikling hadde AUC 0,489817 på 3 359 observerte rader.
Dette viser svak in-sample rangering uten målt overføring til oktober;
periodene er ikke en paret eller uavhengig test. Den lagrede 2,58 GB-filen
ble SHA-verifisert, M5+5 ble minnekartlagt, og kjøringen brukte 0 modellkall,
0 optimizersteg, 0 TEST, 7,35 sekunder og 3,04 GiB maksimal RSS under 4 GiB.
Ingen Entry-handling, uavhengig edge eller økonomisk edge er påvist.
Planen er konsumert og kan ikke kjøres igjen.

Lesende feature-/maskerevisjon fant ingen konkret input- eller
normaliseringsfeil. Kandidatens fit- og scoreoppskrifter binder samme
feature-view, prefix-normalisering, kausale input og målskilder; faktiske
kjøringer gjorde 1 866 optimizersteg og 396 score-forwards. Den fire-raders
[batchkvitteringen](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_BATCH_BINDING_20261106_002/RESULT.json) SHA-256
`57608c0cab08605fd6ac4095090f80ee56ac90037e89887503210a1230a80aa0`
viser de 17 forventede inputnøklene. Koden ruter fem TF og åtte spesialister,
med 61 D1/makrofelt til kontekst. Teknisk rute og liten batchparitet beviser
ikke at alle felt er informative eller at modellen lærte bidragene deres.
M5+5-observasjonsdekning var 56,28 % i TRAIN og 53,12 % i oktober;
DOWN/FLAT/UP-fordelingen var henholdsvis 41,29/13,77/44,94 % og
39,71/14,59/45,70 %. Kandidatens Brier forverret seg 0,01046 mot priorens
0,00364. Mask-/klasseforskyvning alene forklarer ikke målt fall, og ingen
kausal feilårsak er påvist.
Lagret [kandidatfit](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001/CANDIDATE_FIT_002/RUNTIME/prefix-fit/window-0001/RESULT.json) SHA-256
`a42712db365c1bb4f53669c4564c09db7cf3e56c9feb2f6ff6257d43f53f11a5` viser
1 866 fysiske AdamW-steg. Siste batch hadde ikke-null gradient i alle 135
rapporterte grupper, inkludert åtte spesialistprojeksjoner, MTF/GC,
`entry_q_joint` og tilstandshodene. Første batch hadde null i de åtte
spesialistprojeksjonene; det vedvarte ikke. Kvitteringen gjelder samlet tap
og måler ikke isolert endpoint-gradient eller vektendring per parameter.
Ingen vedvarende gradientbrudd er påvist; dette innrømmer ingen kodeendring
eller ny fit.

Ny feilhypotese fra brukerens EMA/ATR/nivåinnvending: oktoberkandidaten
predikerer M1-midbasert markedstilstand, ikke executable Entry-Q. I den
historiske dynamiske M5-Entry-implementasjonen er `entry_action_q_loss` inaktivt;
Entry-kvalitet er predikert terminal-netto minus predikert tidlig MAE fra
to separat trente komponenter. Dette er en mål-/beslutningsmismatch,
ikke en bevist featurekodefeil. Den eldre samme-rad-kontrollen i
`docs/DYNAMIC_MTF_STATE_NO_EDGE_DIAGNOSIS_20261103.md` fant motsatt
M5-candlefortegn AUC 0,5185 mot modell 0,4979 på 7 812 senere
utviklingsrader, men ingen Entry- eller økonomisk edge. EMA20/ATR14 og rå
støtte-/motstandsavstand har varierende TRAIN-normaliseringsstatistikk på
alle fem TF; faktisk rå tensorparitet ble kontrollert nedenfor.

Én [rå inputaudit](/home/andre2/src/GX1_CURRENT/configs/research/DYNAMIC_M5_OCTOBER_NATIVE_FEATURE_NUMERIC_20261106_001.json) SHA-256
`b5a6785ab4666d03663ce4ad3878f6c84a839821d4c3f2598cc6d2f2e01ef0d9`
er fullført én gang. [RESULT](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_FEATURE_NUMERIC_20261106_001/RESULT.json) SHA-256
`62f37fc0ae50d967b2b8a449a54d72ed66b2fc5a08ad11ae0a381f68f92388e7`
og [TERMINAL](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_FEATURE_NUMERIC_20261106_001/TERMINAL.json) SHA-256
`3f9abd4bf5c822cf62e119abd17fa1dab7004a53436a39c460428c794e70cec9`.
Alle 36 172 foreldrene valgte siste lukkede cachebar på fem TF; M5-valget
hadde eksakt samme barstart som forelderen. På åtte forhåndsbundne rader
var rå ATR14, EMA20-avstand og støtte-/motstandsavstand bitlike mellom
cache og modellens gather-tensor. Alle fire felt var finite og ikke-konstante
på alle fem TF over foreldrene. 0 modellforwards, 0 optimizersteg, 0 TEST;
15,92 s, `ru_maxrss=4 388 908 KiB` og verifisert `memory.max=4G`.
RSS-tallet overstiger 4G-grensen; faktisk cgroup-topp ble ikke bevart, så
regnskapene må ikke omtales som samme mål. Kjøringen avsluttet 0 uten OOM.
Kontrollen beviser ikke post-normalisering, per-felt modellbidrag, faktisk
nivåobservasjon eller Entry-verdi. Planen er konsumert og kan ikke kjøres igjen.

Parkert hypotese for en senere Entry-fase: ett direkte tap på
`terminal-netto − MAE`-kvalitet. Den er ikke gjeldende neste steg og gir
ingen fit-adgang. Tidligere 191s 256 CONTROL-rader hadde negativ observert
sidekvalitet i snitt; det kvalifiserer heller ikke Entry.

En siste metadataavklaring fant ingen lagret, foreldrejustert 2D-matrise
med hele den native inputflaten i gjeldende bundne kandidat-artefakter. [CAUSAL_INPUTS](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_FEATURE_DELIVERY_20261106_001/CAUSAL_INPUTS/CAUSAL_INPUTS.json) SHA-256
`2fdea20ebeb90ec8a027f067726f1dd488139e4a1662ee61a75920b1ab764ae6`
har bare justert nåtilstand/alder/masker. [M5-cachemanifestet](/home/andre2/GX1_DATA/data/data/prebuilt/XAU_GC_NATIVE_26M_20261018_003/MULTI_TF_V4_CACHE/manifest.json) SHA-256
`3f4c3b4b54b08931850b716e0b7cf4b5de0b54c42b0fee813cde6f1324371da4`
har en kausal M5-matrise på 1 140 982 × 253 på baraksen, men den må først
kobles as-of til beslutningene og mangler native signal-/kontekst-/GC-flaten.
Eksisterende eier `gx1/features/htf_features.py:slice_multi_tf_v4_window`
velger siste lagrede M5-barstart ≤ `issue−5 min`. Ny rå inputaudit
har bevist likhet med alle de 36 172 foreldrebarene i dette bundne
pre-november-scope. Denne M5-only-baselinen er
ikke innrømmet: den kan ikke avgjøre full native input- eller Entry-verdi.

Ingen ny native eller enkel modellfit er innrømmet. Gjeldende neste steg
er tilstandsmålets semantikk og observerbarhet, deretter indikatorenes
normaliserte konsum og kvalifisering. En senere tilstandsfit krever én
forhåndsbundet kronologisk hypotese, målmaske, referanse og akseptregel.
Oktober er undersøkt utvikling; TEST er forseglet. Senere retningsverdi
er ikke påvist.

Eksisterende data er nok til å undersøke læring videre: fysisk native XAU/GC
TRAIN har 118 249 foreldre fra 2024-05-01 til 2026-01-01, og VAL har
17 284 fra 2026-01-01 til 2026-04-01. GC MBP-1, genuine indikatorer, fem TF,
åtte spesialister og D1/makro er ikke en manglende datakilde. Ny historisk
kronologisk TRAIN-OOF kan planlegges med foldspesifikk fit på tidligere,
målmodne rader uten nye data; dette åpner ikke dagens `training_enabled=false`
eller en ny kjøring uten særskilt hypotese og eksakt policyadmisjon. Alle
26 månedene er allerede undersøkt utvikling, så intern OOF er ikke en urørt
sluttbekreftelse. Nye uavhengige data er ikke en forutsetning for å undersøke
læring nå; de trengs først for en endelig uavhengig edgepåstand.

En fryst lineær avlesning av kandidatens siste hidden er ikke en gyldig
kronologisk OOF-test av representasjonen: backbonen ble fit på alle 29 849
pre-oktober TRAIN-foreldre. Lagrede [TRAIN-arrays](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001/CANDIDATE_FIT_002/RUNTIME/prefix-fit/window-0002/FINAL_TRAIN_PREDICTIONS/ARRAYS.json)
SHA-256 `77abe66d40db906fdb8266edfdb532b2dbffa161db83994ef644107f1f973b38`
og [oktober-arrays](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001/CANDIDATE_SCORE_002/RUNTIME/prefix-fit/window-0001/OOF_PREDICTIONS/ARRAYS.json)
SHA-256 `d8bd67c9c471f4ef93978ca8ab8367f50197d9a82694c1d0eee5dd9b6f557bcc`
inneholder bare rad/tid, sannsynligheter og sekvenstap, ikke hidden. Et nytt
uttrekk ville kreve ca. 2 261 full-native batcher over 36 172 foreldre.
Direkte +5-join måler bare relativ LONG/SHORT terminal-markout, ikke FLAT,
MAE før første MFE, valgt kvalitet eller økonomi. Den lineære proben er
høyst komponentdiagnostikk og er ikke innrømmet som Entry-test.

Zero-refit-skåring av en senere utviklingsmåned er fysisk mulig med bevart
checkpoint, men dagens scorer og mål/akser/tape er bundet eksakt til oktober.
En ny måned krever ny avgrenset plan og kvalifisert full-native input; to
oktoberlignende scorearmer tok samlet omtrent 35 minutter og 1,1 GB output.
Dette ville fortsatt måle tilstand, ikke full Entry-beslutning, og alle
26 historiske måneder er utvikling. Forward-journalen i `GX1_RULES.md`
gjelder D1-swing, ikke M5. Ingen slik kjøring er innrømmet. Neste
kvalifiserende hypotese gjelder fem-TF-markedstilstand med forhåndsbundet
kronologi, dekning, mål og enkle referanser. Direkte Entry,
path, kost og økonomi hører til den senere fasen.

Aktivt mål er fem-TF-markedstilstand med M5-prognosetakt, kausal lukket
M15/H1/H4/D1-kontekst, åtte spesialister og niende D1/makro.
`training_enabled=false`. TEST er forseglet; bare forward-journalen scorer
sin periode på forhåndsbundet måte. Exit,
broker, ordre, live/paper, spending og promotion er stengt. Ingen konsumert
engangsplan relanseres. Stabil langkjøring kontrolleres ca. hver 15.–30.
minutt, gjerne sjeldnere ved stabil drift.

## Bevarte eldre artefaktankere

Disse stistrengene sto bare i tidligere overlevering og beholdes for
retention-eierens navngitte sti-vern. De gir ingen arbeids- eller kjørestatus.

/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2
/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2/DIRECTION_BPS_HEAD_RANK_DESIGN_20261027_001.json.
/home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19.
/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_INPUT_ONLY_PREP_20261105_003/TEST_TERMINAL.json
/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001
/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001/CONTROL_PRIVATE_001/CAMPAIGN/CAMPAIGN_PLAN.json
/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/
/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_CURRENT_INPUT_LINEAGE_PARITY_REVIEW.json
/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_LOSS_COMPOSITION_REVIEW.json
/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63
/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/ORDER_BATCH_GEOMETRY_ROOT_REVIEW_20261028_002.json.
/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/ORDER_NATIVE_B16_SUPPORTED_QUALIFICATION_20261028_003/ACTUAL_ROOT_REVIEW_001.json.
/home/andre2/GX1_RUNS/XAU_INPUT_PHYSICAL_CLOCK_20261105_003/CAPTURE/INPUT_TIMING.json
