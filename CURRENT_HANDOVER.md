# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier status; prosess/terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_M5_SIDE_BOUNDARY_GRADIENT_ADMITTED_NO_MODEL_EXECUTION. Edge-målet er uoppnådd.
Training/native/full_epoch/fullVAL er false. Native002, readout001 og parameter-replay002 er
konsumert. Ny engangsscope MARKET_STATE_TA_M5_SIDE_BOUNDARY_GRADIENT_20261113_001:
2eksakte tilstander +2beholdte TRAIN-paneler,0affine replay,2048finitt-differanse-kontroller.
Metadata-preflight før CLAIM; capped4G/512M med timeoutKILL240s.0modeller/modulkall/optimizer/
autograd/CUDA/updates/checkpointskriving. PLAN og REVIEW_BOUNDARY.py er hashbundet i policy.

Aktuelt M5-only-forsøk har sammebef35-INITIAL,15inputs og numerisk oppskrift som full-prefix002,
men beholder både S/M5- og F/M5-side. Det avgjør ikke om de motarbeider hverandre. Gammel
16-batch attentionmåling gjelder c171 og gammel aktivitetspressmåling delte aktivitetsrader;
verdiene kan ikke tilskrives dagens modell. Ingen aktuell S-only/F-only-arm er dokumentert.
S er lokal tilstand ved framtidig gyldighetstid; F er netto prisendring fra prognosestart.
Kilden viser null direkte aktivitetsgradient mot sideloggiten, men felles projeksjon/enkoder.

Neste kontroll beregner dL/dh = known·q/(5N)·(sigmoid(z)-y)·w for begge M5-sidegrener.
Bevar intakte B16-grupper og skill frie radvise representasjoner fra én felles translasjon av
alle representasjoner; middel av batch-dot er ikke dot av middelgradienter. TRAIN har ikke
beholdt M5-h. Felles funksjonskoordinat bindes til kilde og faktisk query1; ingen numerisk h-paritet
hevdes. FP64-deriverte ved beholdte FP32-logits er ikke native autograd eller historiske treningsgradienter,
query-/encoder-Jacobian, Adam-virkning eller bevist nyttig inngrep. Ingen ny native-/treningsplan.

Siste fullførte parameter-replay002: sourcecb16098e, RESULT3545cbf2, TERMINAL5d990762,
REVIEW_VALIDATION0d6be484.3,149s;2checkpoints/2arkiver/4affine endepunkter,0modeller/updates.
Alle590aktive parametertensorer endret seg og har Adam-steg3698.179inaktive og alle buffere
uendret. Dette beviser oppdateringer, ikke informative gradienter i hvert steg eller god læring.
Siste output-bias forklarer0,0303% av M5F-logitøkningen og0,322–0,364% for øvrigeF-tidsrammer.
Tidligere bias-/projeksjonsendringer kan fortsatt inngå i representasjonsleddet.
M5F,1689kjente siderader: meanΔlogit0,350501. Symmetrisk regnskap gir vektledd0,160746 og
representasjonsledd0,189650; begge ordnede regnskap er også publisert. Dette er algebraisk
endepunktfordeling, ikke entydige årsaksandeler. Samspillsleddet Δw·Δh er0,160961.
INITIAL/nytt hode+gammel representasjon/gammelt hode+ny representasjon/FINAL meanPup:
0,515273/0,535136/0,542344/0,600113. Begge blandingene har dårligere Brier/log enn INITIAL;
ingen frysestrategi eller ny kandidat er kvalifisert. Score-replayfeil0; maks algebrafeil3,33e-16.
Non-M5 mangler beholdte representasjoner; kun output-bias mot samlet rest kan skilles der.

Neste steg er den avgrensede analytiske kontrollen over. Hele gradientforløpet er ikke beholdt.

Parameter-replay001 stoppet etter2checkpointdekodinger,0arkiver/0modeller, på feilaktig krav om
lik view-proveniens. Original feilterminal bevares og001 relanseres ikke.002 validerte hver kapsel
mot eget originale view og nøyaktig fem metadataforskjeller. Constructor/normalisering er
byteidentiske; overlay-endringen tilføyer kvalifisert senere-dekning og bevarer TRAIN-cutoff.
Alle øvrige viewfelt er identiske. Dette var en rettelse av diagnosekontrollen, ikke treningskoden.

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
