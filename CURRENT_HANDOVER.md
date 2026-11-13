# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier status; prosess/terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_CURRENT_MODE_OBSERVATION_001_COMPLETE_UP_DRIFT_PERSISTS. Edge-målet er uoppnådd.
Training/native/full_epoch/fullVAL er false. Ingen aktiv jobb. Moduskontroll001, gradientobservasjon003,
tidligere001/002 og numerikkontroll001 er konsumert; ingen av planene skal relanseres.

Fullført: /home/andre2/GX1_RUNS/MARKET_STATE_TA_CURRENT_UPDATE_OBSERVATION_20261113_003.
Frossen kilde3019998a, PLANea82d0a0, RESULTd5efba0d, TERMINAL81432923,
REVIEW_VALIDATIONcc4d547f, MECHANISM_REVIEWfce77fa3.344,101s; guardexit0;
peak45C/50C/143,05W/2068MiB. Én modell,2faktiske endepunkterbef35/1542,
16fasteB16TRAINgrupper perendepunkt,alle590aktive tensorer i8disjunkte grupper.
96forwards/32grad-forwards/256autogradkall;0optimizer/updates/backward/RNG-restores/checkpointskriving.
Alle32nativekontroller var bitlike sine beholdte referanser; alle32grad-forwards var bitlike egne
nye dynamiske no_grad-referanser.256gruppeclosure-kontroller PASS,maxrelativt1,52782e-7.
Parametre/buffere/flagg/RNG/features var uendret per fase; source/input-referanser verifisert.
Uendrede toleranser. Kryssprofil avviker fortsatt: maxPdiffINITIAL9,51715e-7,FINAL6,21928e-6.
Dette reparerte observasjonens referanse; ingen ONLINE-, taps-, feature- eller treningsfunksjon ble endret.

F/non-M5: geometri av middelgradienter gir totalSGD-retning+0,967994INITIAL til-13,457736FINAL.
FINAL future-side-ledd-13,467711; øvrige tap netto+0,009975. Nominell siste Adam-retning-0,013459,
negativ i16av16grupper; totalSGD negativ i10av16. Dette støtter ikke generell vedvarende UP-drift
fra øvrige oppgaver ved FINAL. M5 S/F har positiv gjennomsnittlig batch-dot i alle fem delte grupper
og samlet12positive/4negative batcher ved begge endepunkter. Ingen generell konflikt er påvist her.
Gjennomsnittlig batch-dot er ikke dot mellom middel-S/F-vektorer; sistnevnte ble ikke beholdt.

Faktisk FINAL-INITIAL parameterdelta har positiv lokal logitprojeksjon ved begge endepunkter.
FINAL non-M5-bidrag: query_projection+0,660965,remaining_encoder+0,602346,shared_side_heads+0,216385;
samlet+1,484738. Dette er lokale J·delta-koordinater, ikke kausale andeler av faktisk historisk endring.
Adam-målet er siste lagrede momenters nominelle gradientledd uten decay, ikke et predikert neste steg.
Eval-mode TRAIN-utvalg rekonstruerer ikke historisk dropout, batchrekkefølge eller treningsbane.

Gjenbruk av eksisterende pass0/1/2 viser at den store UP-forskyvningen oppstår innen første pass:
senere M15F meanP0,578201->0,823873->0,806739; H1F0,574586->0,828949->0,816133.
Andre pass reduserer deler av non-M5-skjevheten. M5F TRAIN-AUC0,537196->0,535205->0,546849,
men senere AUC0,511290->0,510127->0,509071. Bedre TRAIN-fit er ikke bedre senere prognoser.
To pass på fem måneder avgjør ikke kapasiteten med hele historikken og30-40epoker.

Nyest fullført: /home/andre2/GX1_RUNS/MARKET_STATE_TA_CURRENT_MODE_OBSERVATION_20261113_001.
Frossen kildeaa032b60,PLAN62e3a004,RESULT6275bbfe,TERMINAL2cb8e04a,
REVIEW_VALIDATION1b1c4375,MODE_EFFECT_REVIEW873044be.255,383s;guardexit0;
peak45C/48C/137,38W/2034MiB.224forwards/192grad-enabled/0autograd/updates,
1modell/2states/2TRAINarkiver/6spanarkiver.2privateNPZ=800280bytes;3584capturerader,
256unikeTRAINrader,16B16grupper og7profiler perendepunkt.10fokustester PASS.
Alle32nativekontroller bitlike referansen; alle32train-zero-dropout bitlike dynamic eval/grad.
34MHA- og105Dropout-innstillinger gjenopprettet; øvrig state/features uendret. Egen seed20261113,
énFINAL-RNGrestore;32batcher med7routegrensers RNG-kjede parer eksakt. Ingen historisk-maskepåstand.

FINAL F/non-M5 meanP i train medfire dropouttrekk ligger bare0,160–0,178prosentpoeng under eval.
INITIAL→FINAL-driften er fortsatt+21,04–22,22prosentpoeng;15av16batcher øker for hver non-M5F.
Originaltap øker1,875017→1,912449 i eval og1,875272→1,910833 i train-middel; øker i allefiretrekk.
Sannsynlighetene gjelder alle256rader, ogsåFLAT/UNKNOWN: ingen treffrate eller kjent-side-skår.
Store lærte UP-forskyvninger består i train-mode. Dette avviser en stor ren modusforklaring på
utvalget, ikke dropouts påvirkning under historisk læring eller andre perioders modusrespons.

Neste: gjenbruk den faktiske matchedCOVERAGE-planen/koden og dens INITIAL-/targetbindinger
for å avgjøre om den bredere kalenderdekningen kan omfatte alle29574aktuelle TRAIN-foreldre
med uendret mål og oppdateringsbudsjett. Sammenhold tidligere ordrekontroller før én konkret
læringsintervensjon klargjøres; ingen ny trening er admittert. Ikke gjenta moduskontrollen.
EksisterendeCOVERAGE ga bedre senere native-F Brier/log enn matchedFULL for alle5TF i begge
punktvise dag-/ukeintervaller. M5F AUC0,528606 mot INITIAL0,511290 har positive intervaller,
men motmatchedFULL0,513672 inkluderer begge0. Innhold/repetisjon/ordre endres samlet;
non-M5bias består. Combineddecoder ga ingen etablert AUC-merverdi. Dette er gjenbrukt utvikling,
ikke uavhengig bekreftelse, ferdig modell eller økonomisk edge.

Bevar tidligere kontrollkvitteringer: numerikkontroll001 sourceb2d2073d/RESULT1c1a92d6/
TERMINALb74ed6b8/validationc0936a6b; den målte énINITIALB16gruppe og0autograd/updates.
Observasjon001 feilet før modell på parent_order-metadata;002 stoppet ved feil profilreferanse.
Siste-grense-revisjonen RESULTbbda929f/validationb2c96178 og parameter-replayRESULT3545cbf2/
validation0d6be484 gjenbrukes. Historikk og full metode finnes i Git og aktuelle årsaksdokumenter.

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
