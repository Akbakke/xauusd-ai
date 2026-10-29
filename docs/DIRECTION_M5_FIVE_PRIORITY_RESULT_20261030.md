# M5-retning: 10 ganger vekt på fem minutter

Den vektede modellen forbedret femminutters treffprosent mot den tidligere M5-modellen på samme gjenbrukte Q1 2025-utviklingsrader, men lå fortsatt under alltid-opp-referansen. AUC-endringen er for usikker til å fastslå bedre rangering, og ingen av 60 horisonter har begge forhåndsdefinerte simultane nedre AUC-bånd over 0,5. Dette er ikke dokumentert retningsedge eller handelsverdi.

## Sammenligningen

Begge modellene bruker M5-entry med lukkede høyere tidsrammer, samme 47 602 TRAIN-foreldre, 60 korrigerte mål, 256 byteidentiske startprediksjoner, identisk initialmodell/buffere, seed, batchstørrelse og hash for stokket TRAIN-rekkefølge. Den eneste vedtatte objektivendringen er at femminutters BCE-celler vektes 10 mot 1 for hver av de 59 andre, med normalisering etter summen av kvalifiserte vekter i hver batch. Det betyr samtidig at de øvrige cellenes relative vekt endres. Begge modellene er trent ett pass.

Femminuttersmålingen bruker 17,192 modne, observerte og ikke-null retninger etter eksakt kontroll av foreldre, beslutningsklokker og alle 60 målfelt. Q1 er brukt tidligere og er ikke en uberørt bekreftelse. Tidsutfall og høyere tidsrammer overlapper.

| Måling, 5 minutter | Tidligere M5 | 10x fem minutter | Enkel referanse |
|---|---:|---:|---:|
| Treffprosent | 48.616 % | 50.250 % | Alltid opp: 51.582 % |
| AUC | 0.496961 | 0.500775 | Konstant skår: 0,5 |
| Logloss, lavere er bedre | 0.694381 | 0.693230 | TRAIN-prior: 0.692738 |
| Brier, lavere er bedre | 0.250616 | 0.250041 | TRAIN-prior: 0.249796 |

AUC-forskjellen kandidat minus gammel modell er +0.003814; parrede 98,75 %-intervaller er [-0.008555, +0.016147] med dagsblokker og [-0.005745, +0.017847] med ukeblokker. Begge inkluderer null. Logloss og Brier ble forbedret mot den gamle modellen i begge blokkintervallene, men punktscorene er fortsatt svakere enn den faste TRAIN-prioren.

Treffprosentforskjellen mot gammel modell er +1.634 prosentpoeng; eksplorative 98,75 %-intervaller er [+0.412, +2.846] på dager og [+0.270, +2.931] på uker. Mot alltid opp er forskjellen -1.332 prosentpoeng; disse eksplorative intervallene ligger under null. Blokkuavhengighet/utbyttbarhet er ikke bevist.

På fem minutter sa gammel modell opp i 3,226 av 17,192 tilfeller; ny modell sa opp i 9,151. Ved seks minutter er den nye modellens treffprosent 50.817 %, lik alltid-opp-referansen; dens AUC 0.512414 er en beskrivende nabohorisont, ikke en ny valgt hovedmåling.

## Alle 60 fremtidige horisonter

Intervallene er 97,5 % simultane maksimumsavviksbånd innen hver av de to blokkordningene. Kolonnene viser kandidatens AUC-bånd; disse kan ikke brukes til å velge en ny horisont på de samme dataene.

| Minutter | Kvalifiserte | Gammel treff % | Ny treff % | Alltid opp % | Gammel AUC | Ny AUC | Ny simultan dag | Ny simultan uke |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17211 | 49.980 | 50.044 | 49.695 | 0.499136 | 0.503456 | [0.473340, 0.533572] | [0.472232, 0.534680] |
| 2 | 17239 | 49.864 | 50.450 | 50.914 | 0.499827 | 0.507689 | [0.477573, 0.537806] | [0.476465, 0.538914] |
| 3 | 17237 | 49.690 | 50.705 | 51.111 | 0.502942 | 0.498428 | [0.468312, 0.528544] | [0.467204, 0.529653] |
| 4 | 17253 | 49.719 | 50.658 | 51.301 | 0.501122 | 0.500951 | [0.470835, 0.531068] | [0.469727, 0.532176] |
| 5 | 17192 | 48.616 | 50.250 | 51.582 | 0.496961 | 0.500775 | [0.470658, 0.530891] | [0.469550, 0.531999] |
| 6 | 17189 | 50.794 | 50.817 | 50.817 | 0.487936 | 0.512414 | [0.482298, 0.542530] | [0.481190, 0.543638] |
| 7 | 17192 | 51.408 | 50.012 | 51.326 | 0.506821 | 0.494800 | [0.464683, 0.524916] | [0.463575, 0.526024] |
| 8 | 17196 | 51.146 | 50.204 | 51.204 | 0.495583 | 0.498851 | [0.468735, 0.528967] | [0.467627, 0.530075] |
| 9 | 17194 | 50.663 | 50.506 | 51.861 | 0.503281 | 0.497080 | [0.466963, 0.527196] | [0.465855, 0.528304] |
| 10 | 17133 | 50.762 | 50.978 | 51.912 | 0.496512 | 0.514218 | [0.484102, 0.544335] | [0.482994, 0.545443] |
| 11 | 17123 | 50.581 | 50.885 | 51.562 | 0.501697 | 0.512043 | [0.481927, 0.542159] | [0.480819, 0.543268] |
| 12 | 17134 | 50.706 | 48.763 | 52.072 | 0.508351 | 0.500044 | [0.469928, 0.530160] | [0.468820, 0.531268] |
| 13 | 17137 | 49.945 | 51.351 | 52.010 | 0.506806 | 0.511263 | [0.481147, 0.541380] | [0.480039, 0.542488] |
| 14 | 17139 | 51.316 | 52.080 | 52.418 | 0.513002 | 0.520915 | [0.490798, 0.551031] | [0.489690, 0.552139] |
| 15 | 17077 | 48.949 | 50.056 | 52.392 | 0.507430 | 0.496575 | [0.466459, 0.526691] | [0.465351, 0.527799] |
| 16 | 17071 | 50.706 | 50.706 | 52.282 | 0.513384 | 0.501294 | [0.471178, 0.531410] | [0.470070, 0.532519] |
| 17 | 17075 | 50.102 | 49.710 | 52.463 | 0.510389 | 0.517260 | [0.487144, 0.547376] | [0.486036, 0.548484] |
| 18 | 17076 | 50.404 | 48.472 | 52.594 | 0.506161 | 0.501531 | [0.471415, 0.531648] | [0.470307, 0.532756] |
| 19 | 17080 | 51.013 | 52.465 | 52.611 | 0.513308 | 0.487135 | [0.457018, 0.517251] | [0.455910, 0.518359] |
| 20 | 17012 | 51.440 | 49.982 | 52.645 | 0.510734 | 0.510493 | [0.480377, 0.540609] | [0.479269, 0.541717] |
| 21 | 17014 | 51.434 | 51.328 | 52.704 | 0.506158 | 0.504398 | [0.474282, 0.534514] | [0.473173, 0.535622] |
| 22 | 17017 | 50.902 | 50.914 | 53.047 | 0.513403 | 0.513697 | [0.483581, 0.543813] | [0.482473, 0.544921] |
| 23 | 17015 | 52.195 | 51.854 | 52.918 | 0.501725 | 0.518012 | [0.487896, 0.548129] | [0.486788, 0.549237] |
| 24 | 17009 | 53.219 | 52.790 | 53.125 | 0.515883 | 0.512222 | [0.482105, 0.542338] | [0.480997, 0.543446] |
| 25 | 16954 | 48.313 | 50.513 | 53.120 | 0.507650 | 0.509995 | [0.479879, 0.540111] | [0.478771, 0.541220] |
| 26 | 16956 | 50.678 | 51.321 | 53.108 | 0.508982 | 0.500895 | [0.470779, 0.531011] | [0.469671, 0.532119] |
| 27 | 16957 | 50.581 | 49.897 | 53.476 | 0.504940 | 0.504868 | [0.474752, 0.534984] | [0.473644, 0.536093] |
| 28 | 16952 | 52.377 | 52.130 | 53.386 | 0.512687 | 0.510573 | [0.480457, 0.540689] | [0.479348, 0.541797] |
| 29 | 16947 | 50.310 | 49.501 | 53.473 | 0.516652 | 0.499937 | [0.469821, 0.530053] | [0.468712, 0.531161] |
| 30 | 16887 | 50.743 | 50.009 | 53.556 | 0.512529 | 0.494439 | [0.464323, 0.524556] | [0.463215, 0.525664] |
| 31 | 16889 | 52.904 | 52.028 | 53.479 | 0.508542 | 0.510326 | [0.480210, 0.540442] | [0.479102, 0.541551] |
| 32 | 16892 | 52.179 | 50.562 | 53.587 | 0.510198 | 0.509002 | [0.478885, 0.539118] | [0.477777, 0.540226] |
| 33 | 16884 | 51.581 | 51.558 | 53.749 | 0.506303 | 0.515046 | [0.484930, 0.545163] | [0.483822, 0.546271] |
| 34 | 16889 | 51.862 | 53.893 | 53.692 | 0.514959 | 0.517636 | [0.487519, 0.547752] | [0.486411, 0.548860] |
| 35 | 16829 | 51.637 | 50.223 | 53.651 | 0.512624 | 0.513193 | [0.483076, 0.543309] | [0.481968, 0.544417] |
| 36 | 16831 | 49.534 | 53.413 | 53.532 | 0.513533 | 0.497487 | [0.467371, 0.527603] | [0.466263, 0.528711] |
| 37 | 16828 | 52.104 | 53.952 | 53.999 | 0.519836 | 0.520720 | [0.490603, 0.550836] | [0.489495, 0.551944] |
| 38 | 16824 | 51.296 | 50.458 | 53.941 | 0.521267 | 0.519895 | [0.489779, 0.550012] | [0.488671, 0.551120] |
| 39 | 16823 | 52.024 | 53.902 | 53.873 | 0.521637 | 0.522935 | [0.492819, 0.553051] | [0.491711, 0.554159] |
| 40 | 16761 | 52.628 | 52.604 | 53.905 | 0.525846 | 0.519376 | [0.489260, 0.549492] | [0.488152, 0.550600] |
| 41 | 16765 | 50.820 | 53.296 | 53.922 | 0.521948 | 0.523525 | [0.493408, 0.553641] | [0.492300, 0.554749] |
| 42 | 16763 | 51.417 | 52.198 | 54.101 | 0.519919 | 0.521110 | [0.490994, 0.551227] | [0.489886, 0.552335] |
| 43 | 16767 | 52.890 | 52.037 | 53.951 | 0.511826 | 0.515067 | [0.484951, 0.545183] | [0.483843, 0.546291] |
| 44 | 16765 | 51.262 | 52.293 | 54.381 | 0.517269 | 0.515610 | [0.485494, 0.545726] | [0.484386, 0.546835] |
| 45 | 16704 | 52.472 | 52.874 | 54.185 | 0.514531 | 0.520975 | [0.490859, 0.551092] | [0.489751, 0.552200] |
| 46 | 16697 | 51.848 | 54.207 | 54.201 | 0.516519 | 0.506926 | [0.476810, 0.537042] | [0.475701, 0.538150] |
| 47 | 16697 | 50.380 | 48.919 | 54.159 | 0.521428 | 0.520836 | [0.490720, 0.550952] | [0.489611, 0.552060] |
| 48 | 16704 | 52.209 | 51.503 | 54.274 | 0.520771 | 0.517759 | [0.487643, 0.547876] | [0.486535, 0.548984] |
| 49 | 16695 | 52.393 | 52.842 | 54.429 | 0.517131 | 0.517488 | [0.487372, 0.547604] | [0.486264, 0.548712] |
| 50 | 16640 | 51.424 | 52.933 | 54.405 | 0.517706 | 0.516870 | [0.486754, 0.546986] | [0.485646, 0.548095] |
| 51 | 16638 | 53.216 | 52.098 | 54.616 | 0.516809 | 0.515785 | [0.485668, 0.545901] | [0.484560, 0.547009] |
| 52 | 16638 | 51.424 | 50.457 | 54.418 | 0.514184 | 0.524446 | [0.494330, 0.554562] | [0.493222, 0.555670] |
| 53 | 16638 | 52.669 | 51.785 | 54.568 | 0.520023 | 0.519009 | [0.488893, 0.549125] | [0.487784, 0.550233] |
| 54 | 16639 | 52.389 | 51.860 | 54.474 | 0.515920 | 0.520496 | [0.490380, 0.550612] | [0.489272, 0.551721] |
| 55 | 16574 | 53.011 | 54.024 | 54.706 | 0.522508 | 0.522909 | [0.492793, 0.553026] | [0.491685, 0.554134] |
| 56 | 16574 | 51.177 | 53.216 | 54.634 | 0.519985 | 0.523433 | [0.493317, 0.553549] | [0.492209, 0.554658] |
| 57 | 16577 | 51.819 | 51.903 | 54.588 | 0.519188 | 0.524485 | [0.494369, 0.554601] | [0.493261, 0.555709] |
| 58 | 16573 | 51.795 | 51.638 | 54.565 | 0.516845 | 0.506537 | [0.476421, 0.536653] | [0.475313, 0.537761] |
| 59 | 16577 | 52.241 | 49.502 | 54.606 | 0.508427 | 0.513910 | [0.483794, 0.544026] | [0.482686, 0.545135] |
| 60 | 16512 | 52.332 | 51.986 | 54.797 | 0.518947 | 0.515809 | [0.485693, 0.545925] | [0.484584, 0.547033] |

## Verifikasjon og begrensninger

Én fysisk oppstart, 2,976 optimizersteg og 47,602 TRAIN-foreldre ble fullført med signert vakt PASS. Alle 60 skårer og den eksplorative treffprosent-bootstrapen ble uavhengig gjenspilt fra lagrede prediksjoner; største skårforskjell var 2.22e-16. AUC/logloss/Brier-bootstrapintervallene er beholdt fra den opprinnelige scoreren og ble ikke uavhengig gjenskapt. Ingen nye modellkall eller trening ble utført under sammenligningene.

For de seks kontrollerte GC-feltene er D1-historikkens dekning reelt ulik: TRAIN-vinduene hadde i snitt 83 av 252 dager observert, Q1 hadde 199. Maskene fungerer i den kontrollerte data- og NumPy-normaliseringsveien, men vi har ikke målt hva denne forskjellen gjør med den nye modellens retning. M1-åpningsverdiene er en candle-basert fasitproxy uten dokumentert faktisk quotehendelse eller lokal ankomsttid. Spredning, forsinkelse, utvalg av handler, åpne posisjoner og gjennomførbare fills er ikke kvalifisert her. TEST, broker, ordre, live/paper, Exit og promotion er stengt.

Original aggregat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_M5_FIVE_PRIORITY_ONEPASS_FIT_20261029_001/FIVE_PRIORITY_FIXED_COMPARISON_001/RESULT.json`, SHA-256 `75748b24a98523916c2ef1ae5a8b2405084ab83f7613af5da9cb7cf96b7b3cd1`.
Uavhengig gjenspilling: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_M5_FIVE_PRIORITY_ONEPASS_FIT_20261029_001/FIVE_PRIORITY_FIXED_COMPARISON_001/INDEPENDENT_REPLAY.json`, SHA-256 `f97a0adf70c177af099cad1d2acaae1d50228453b52c42f0e7db87b811e7fdfe`.
