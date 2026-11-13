# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier status; prosess/terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_ORDER_ACCURACY_READOUT_001_ADMITTED_NOT_STARTED. Edge-målet er uoppnådd.
Fullført native ordre001 på kildea31e2448/boot649: 3698updates,59148presentasjoner,
29574unike TRAIN-rader i hvert av2pass,0skip. TerminalCOMPLETE/exit0; vaktPASS;
Windowsoppgave Disabled/LastResult0. RNG-passparitet og INITIAL-restaurering eksakt.
RESULTa3ff7bdb, COMPLETION_VERIFICATION160cb198;0dekodinger i avslutningskontrollen.
training_enabled/native_training_enabled=false. Planen er konsumert og skal ikke relanseres.

Neste er brukerens konkrete spørsmål om faktisk treffprosent versus AUC:
én ny admittert CPU-avlesning av8beholdte prediksjonsarkiver, uten ny modell/trening.
15syntetiske fokustester PASS; opprinnelig faktoriseringsfunksjon og kvalifisert scorer/
bootstrap gjenbrukes. Eksakt metadata-preflight før én avlesning, maksimalt240sek/4G.
Alle10S/F×5TF på TRAIN og senere utviklingspanel. Fast pass2 primær, pass1 sekundær;
fullprefix002 er beholdt kontroll. Beregn binær accuracy/balansert accuracy, confusion,
klassefordeling, aktivitet og faktoriserte DOWN/FLAT/UP. Faste M5F-sannsynlighetsgrupper
vises alle, uten terskeltilpasning eller beste-gruppe-utvalg. Tidligere0,51 var AUC,
ikke modellens treffprosent; known_side_accuracy var alltid-LONG-referanse.
Integrator eier privat avlesning/policy/launch; composition_audit eier kun liten ren
accuracy-hjelper og fokustester. Ingen endring i modell, features, mål eller tapsfunksjon.

Ordre001 bruker samme bef35-INITIAL,29574mai–september2024-rader,2pass og3698updates
som fullprefix002, med faste43×43-transponerte originale batcher og tail6 sist.
Rekkefølge omfatter også dropout-til-rad-tilordning og optimizerbane; ikke separat årsak.
Pass1 viste mindre sannsynlighetsfeil enn gammel rekkefølge i alle10senere celler;
fullført pass2/rangering/accuracy er ennå ikke skåret. Ingen ny edge-konklusjon.

Tidligere fullpopulasjonsordre ER testet: FULL_REVERSE20261110_002 RESULTbe7a403c/
TERMINAL40a284e6, alle29574én gang, eldre ae772-INITIAL. TRAIN-tap2,021683→1,883423;
senere192 M5F Brier0,248827→0,243230, men M5S0,235061→0,238067. Samlet gateFAIL.
Eldre16×16-paring RESULTb11a1932 brukte4096rader/256updates og ga også M5-avveiinger.
Disse gjenbrukes; nytt forsøk er ingen første test av ordre eller generell interleavinggevinst.
CurrentCOVERAGE RESULTd5ba8fa9 endret innhold/repetisjon/ordre samlet. Første4096
spenner allerede over hele mai–september; ingen utvidet kalenderperiode foreslås.

Siste moduskontroll20261113_001 er COMPLETE/CONSUMED: RESULT6275bbfe,
TERMINAL2cb8e04a, MODE_EFFECT_REVIEW873044be;224forwards/0updates.
FINAL non-M5F train-mode meanP er bare0,160–0,178prosentpoeng under eval;
INITIAL→FINAL-driften består med+21,04–22,22prosentpoeng på256TRAINrader.
Ren modusforskjell forklarer ikke den store skjevheten på dette utvalget.
Gradientobservasjon003 RESULTd5efba0d/TERMINAL81432923 viste at FINAL totalgradient
og lagret siste Adam-retning allerede peker ned for non-M5F. Historisk læringsbane
forblir uisolert; ingen generell vedvarende oppgavekonflikt ble påvist her.
Begge observasjoner, tidligere001/002 og numerikkontroll001 er konsumert.
Ingen modell-/feature-/tapsrettelse eller ny edge er etablert av disse observasjonene.

## Nytt resultat

Samlet kilde: docs/TA_FULL_PREFIX_LEARNING_REVIEW.json.
Native root: /home/andre2/GX1_RUNS/MARKET_STATE_TA_FULL_PREFIX_LEARNING_20261112_002.
Frossen kilde2f825e27; fysisk boot648; RECIPE993b9f1d; RESULTcc5d1ba1;
TERMINAL72d686c5; COMPLETION_VERIFICATION2630485c. Windows Disabled/LastResult0.
To komplette kronologiske pass av29574TRAIN-foreldre fra mai–september2024, inkludert tail6.
3698updates/59148presentasjoner/0skip;5234forwards;1modell/optimizer/checkpointdekoding;
0checkpointskrivinger,1privat evidence-kapsel. Ingen kanonisk resume eller promotion.
Samme bef35-INITIAL/kontrollmodell;4096TRAIN/LATER var bitlike kontrollen før updates.
Måling0/1/2, pass2 primær. Native8283,25s; vakt exit0, peak61C/62C/168,26W/2207MiB.

Readout root: /home/andre2/GX1_RUNS/MARKET_STATE_TA_FULL_PREFIX_READOUT_20261113_001.
Frossen kilde0d36ca3f; RESULTf1b0321e, TERMINAL440db398, REVIEW_VALIDATION6bf3c18f.
6arkiver,0modeller/checkpoints/forwards/updates,8,125s. Kvalifisert scorer/bootstrap gjenbrukt.
Alle publiserte proper scores replayet innen1,12e-15; månedsaggregering innen2,23e-16.
70dager/14uker,2000parede trekk per gruppering; punktvise utviklingsintervaller.

TRAIN-panelets totaltap1,864995→1,960094→1,909092. Senere native-tap2,350915→2,639348→2,645637,
mot prior2,316114. Alle10senere AUC-endringsintervaller inkluderer0 i begge grupperinger.
Alle8ikke-D1 S/F-celler har dårligere Brier/log i begge intervaller. D1-endringene er usikre.
M5F AUC0,511290→0,509071; meanPup0,515273→0,600113 mot fasitUP0,519242.
M15/H1/H4/D1 F meanPup blir0,806739/0,816133/0,818379/0,814294, mot fasit0,491060/0,5/
0,539045/0,575089. S/M5 ogS/D1 rangerer over konstantprior på dette gjenbrukte panelet;
S/M15 ligger under. Alle5F-rangeringer mot konstantprior er usikre. Dette er ikke økonomi.

Månedstabellene støtter ikke generell siste-måned-glemsel som tilstrekkelig forklaring:
F/H1 forverres mai–juli og bedres august–september; F/M15/H4 forverres også i september;
F/D1 TRAIN-Brier bedres i alle5måneder. UP-nivået øker bredt, også på senere desemberdata.
TARGET_WEIGHT_REUSE_REVIEW3ed18d86 gjenbruker eksisterende BATCH_COUNTSf51405c6 som JSON.
18inputreferanser,8target-/loss-eiere, numerisk recipe, queryseed og begge1849batchpass matcher.
Ingen nye arrays/modeller. Fullpopulasjonens konstante F-UP-optima er0,511126/0,531247/
0,537731/0,568059/0,641793. Siste256 er UP-rike, men siste16har23kjente F/D1, alleDOWN.
Verken samlet klassevekting eller «de aller siste etikettene var UP» forklarer alene≈0,81.
Kumulativ ordre-, representasjons- og Adam-effekt er fortsatt uisolert.

## Grenser og gjenbruk

Alle genuine features/8familier/5TF/D1 beholdes. TRAIN sampled-S og LATER native-S er ulike
oppgaver; sammenlign endringer innen hvert panel. Kjent side i fasit er ingen valgbar Entry-
dekning. AUC er rangering, ikke treffrate eller økonomi. Alltid-LONG native-F-referansen er
publisert som kjent-side-treffandel; ingen avkastning er målt. Ingen uavhengig generalisering,
Entry-kvalitet eller økonomisk edge er etablert. To pass på fem måneder dømmer ikke full
kvalifisert historikk og30–40epoker. INITIAL-backbone arver2tidligere fulle pass over resetgrenser;
hoder har annen historie. Alternative FINAL-armer er ikke arvet. Combined-variant er ikke promotert.
Feilet klargjøring001 skal ikke relanseres. Tidligere generelt interleaving-forsøk gjenbrukes;
det ga kalibrerings-/M5-avveiing og er ikke grunnlag for enda en generell ordretest.
TEST forseglet; GC på pause; ingen broker/handel/spending/promotion. Originalkvitteringer bevares.
Opprydding: ingen DATA/RUNS-sletting eller generell repo-opprydding i denne bølgen.
