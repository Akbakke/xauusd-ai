# Entry lærer gjennom egne representasjoner: ett fast TRAIN-forsøk

TRAIN16-gradientmålingen påviser at dagens V4-detach hindrer Entry-feilen i
å trene upstream feature-/MTF-representasjon. Den åpne forbindelsen har endelige
ikke-null gradienter med samme outputs og hodegradienter. Dette er begrunnelsen
for én konkret modellrettelse, ikke en dokumentert læringsgevinst.

Bare Entry-Q-kildens detach fjernes. Hele øvrige modell-AST er identisk med
4f6a64ec, inkludert Exit-tokenets detach. Parameteroppsett, initialisering,
alle features/familier/tidsrammer, kostnader, losses og kontrollparametere
beholdes. Sju relevante eksisterende gradienttester består.

Gjenbruk eksisterende native campaign og lagret fersk initialmodell/optimizer/
EMA/scheduler/RNG. Tren eksakt samme4096 Entries i opprinnelig rekkefølge,
256 oppdateringer av16. Uendret frossen targetmodell, ingen refresh eller
best-checkpoint-valg. Originalt detached-forsøk og alle artefakter bevares.

Eksisterende sluttmåler avgrenses med ett eksplisitt kilde-/policybundet
TRAIN-only-valg.13 unike scope-/målekontroller består;7 scope-cases ble kjørt
etter siste avvisningsrettelse, ikke telt som nye unike tester. TRAIN256,
256 ankerstates og1024 samplede states brukes med identisk lagret fasit.
Ingen ny CONTROL-forward, økonomi, full epoch/VAL eller TEST.

Krev at både LONG/SHORT Entry-MSE og sentrert feil forbedres mot lagret initial
og detached256 og slår TRAIN-konstanter. Rapporter alle ni TRAIN-måneder,
korrelasjon, målregret og LONG/SHORT/FLAT-fordeling. Konstant forskyvning eller
klassesammenbrudd er ikke tilstrekkelig. Exit rapporteres på begge sider og
stateflater mot samme baselines; Entry alene består ikke samlet læringsport.

Dette er en paret mekanismekontroll på gjenbrukt TRAIN med overlapp til fitted
rader. Den beviser ikke senere generalisering eller lønnsomhet. Budsjett,
mål og vurdering er bundet før fit. Ingen søk, automatisk ny kandidat eller
utvidelse. Plan og tester finnes i
handover_snapshot/ENTRY_CONNECTED_FIXED256_PLAN_20260917.json.

## Fullført resultat: mer Entry-fit, fortsatt ingen samlet læringsport

Kilden2c5ddb45 kjørte2026-09-17T12:11:47–13:26:28UTC (14:11:47–15:26:28Oslo).
256 oppdateringer, samme4096 Entries og eksakt samme targets/masks/cohort/lærer.
GuardPASS/trainer0/observer0. Ingen CONTROL-forwards. Task og unntak er stengt.

| TRAIN MSE i Bps² | Initial | Detached256 | Connected256 | TRAIN-konstant |
|---|---:|---:|---:|---:|
| Entry LONG |246,12|240,24|234,50|240,64|
| Entry SHORT |237,22|235,34|220,73|237,64|
| Exit anker LONG |629,65|623,29|624,21|625,04|
| Exit anker SHORT |627,77|619,96|620,39|622,61|
| Exit samplet LONG |864,79|860,95|861,98|864,71|
| Exit samplet SHORT |859,50|854,49|855,70|859,42|

Entry sentrert feil bedres238,64→230,77LONG og235,00→219,95SHORT. MSE bedres
mot detached i5/9LONG- og8/9SHORT-måneder; sentrert feil i6/9 og5/9.
Alle månedene og opprinnelige baselines er med i snapshotet. Baselineberegninger
er kontrollert numerisk identiske med forrige rapport, MSE=varians+bias².

Den eksakte dekomponeringen MSE_LONG+MSE_SHORT=2*MSE_felles+0,5*MSE_kontrast
viser at88,63% av forbedringen gjelder fellesverdien(LONG+SHORT)/2.
Kontrastens korrelasjon er0,101, mot initial0,101 og detached0,075. Sentrert
kontrastfeil635,61 er fortsatt litt verre enn initial635,09. Dette forklarer
hvorfor bedre absolutte verdier ennå ikke gir klart bedre retningsvalg.

Entry velger222LONG/19SHORT/15FLAT. Referanseregret8,5074 slår detached8,5952,
men taper mot alltidLONG8,4556. De34 avvikene fra LONG gir samlet−13,2529Bps
i referanseverdi mot konstantvalget. Dette er gjenbrukt TRAIN og Q_mu-fasit,
ikke faktisk handelsprofitt. Kjent FLAT=0 endrer to valg og øker regret til
8,5654; ingen modell- eller terskelendring er gjort på dette grunnlaget.

Exit taper svakt mot detached i alle fire MSE-/sentrert-feilceller. LONG er
fortsatt1024/1024HOLD; SHORT1004/1024EXIT. Kandidaten oppfyller ikke samlet
Entry/Exit-læring eller beslutningskravet. Ingen senere kvalitetsmåling,
generalisering, økonomi, utvidelse eller automatisk ny trening er godkjent
av dette resultatet. Entry-rettelsen bevares som en upromotert arbeidskandidat.

Kildekontroll: online Exit-markeds-/path-/summary-fusjonen har ingen tilsvarende
Entry-Q-detach. Exit bruker egne exit_episode_family_tf_context/token_gate;
Entry-routingens tidligere `unused` Exit-gradient er ikke bevis for at Exit-
ruteren er frakoblet. Cache-detach tilhører frossen evaluering. Dette er
kildebevis; nye faktiske Exit-gradienter er ikke målt.

Neste avklaring skal bruke eksisterende TRAIN-bevis til å skille svak
retningsinformasjon/representasjon fra en konkret læringsbegrensning før
ny kode eller trening. Ingen ny CONTROL-tilpasning, tapsvektsøk eller TEST.
