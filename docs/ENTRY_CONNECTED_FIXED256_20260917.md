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
