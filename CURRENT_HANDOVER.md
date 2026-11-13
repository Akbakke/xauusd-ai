# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier status; prosess/terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_CURRENT_SIDE_BOUNDARY_AND_UPDATE_EXPOSURE_REVIEW_COMPLETE_CAUSE_UNRESOLVED. Edge-målet er uoppnådd.
Training/native/full_epoch/fullVAL er false. Alle tidligere planer er konsumert; ingen aktiv jobb.
Siste root: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_SIDE_BOUNDARY_GRADIENT_20261113_001.
Source47f7700a; RESULTbbda929f, TERMINALb7416725, REVIEW_VALIDATIONb2c96178.3,096s capped4G/512M,
timeout240s;2checkpoints/2TRAINarkiver,0modeller/modulkall/autograd/updates/affine replay.
2048finitt-differansekontroller max5,15e-12; M5-loss-replay max2,40e-7; score-replay0.
TRAIN har ikke beholdt M5-representasjoner. Felles funksjonskoordinat er kilde-/query1-bundet;
FP64-derivert ved faktiske FP32-logits er ikke målt h-paritet eller native autograd-historikk.

Aktuelt M5-only-forsøk deler faktiskbef35-INITIAL,alle15inputrefs og numerisk oppskrift med
full-prefix002, men beholder begge S/F-sideledd. Gammel c171attention-/aktivitetsrevisjon kan
ikke tilskrives dagens funksjon. S er framtidig lokal tilstand; F er netto prisendring fra utstedelse.
Ny kontroll:979M5rader med begge sider kjent;952enige/27motsatt. Hodecosinus0,7493→0,8506.
Alle256intakteB16grupper har ikke-negativt radvist S/F-dot ved begge endepunkter. Ved én felles
translasjon er71INITIAL/65FINAL av249støttede grupper negative, men begge aggregeringer positive.
Ingen generell M5-S/F-motvirkning er påvist ved denne siste grensen. Upstream-Jacobian er umålt.

JSON-only tillegg: CLIP_WEIGHT_REUSE_REVIEW069e74e6, ADAM_BIAS_MOMENT_REUSE_REVIEWb9775fc0,
METADATA_REUSE_VALIDATION1a892ffb.0nye array-/checkpointdekodinger/modeller/updates.
3693av3698preclip-normer>1. Nominell skala min(1,1/(norm+1e-6)); fullpopulasjonens F-UP-
referanser øker0,46–2,11prosentpoeng under disse faste skalaene. Sluttfasen skiller seg:
siste16F/H1 0,4161→0,5684;M15 0,5728→0,6787. Siste256D1 0,8218→0,8146. Dette er betinget
klasseeksponering, ingen øvre prediksjonsgrense, faktisk gradientrekonstruksjon eller årsaksandel.
Faktiske siste bias-exp_avg er negative forM5S/F, positive for non-M5S/F. Adam-gradientleddet
presser dermed de delte non-M5sidebiasene NED. Decoupled decay og øvrige parameterbidrag er egne.
Nominell β1=.9-historikkrekonstruksjon: F/M5 vektetP0,5183 motUP0,5715; non-M5 vektetP0,6085
motUP0,4889. Dette gjelder nylige faktiske treningsinputs/vekter, ikke FINAL/LATER-prognoser.
Direkte og rekursiv momentberegning matcher innen2,23e-16; native FP32-historikk er ikke replayet.
Ingen grunn til å fjerne S-målet, endre klippevakten eller gjenta et generelt ordre-/arkitekturforsøk.

Neste: klargjør minste aktuelle observasjon som kobler tapsgradient/lagret Adam-retning til
UP-endring i delt projeksjon/enkoder. Gjenbruk eksisterende tapsfrø/VJP-eier og dagens inputs,
men bind bef35/1542,aktuell8trådsruntime og kilde. Gammel c171/R1-worker og cap-rute er uforenlige;
ingen port omgås eller gammel scope relanseres. Ingen ny nativekjøring eller trening er admittert.

Forrige parameter-replay002 er bevart: RESULT3545cbf2,TERMINAL5d990762,validation0d6be484.
Alle 590 aktive parametertensorer fikk 3698 Adam-steg; 179 inaktive og alle buffere var uendret. Siste output-bias forklarte
0,0303%avM5F-logitøkningen og0,322–0,364%for øvrigeF. M5F-endepunktene viser samspill mellom
vekt og representasjon, ikke entydige årsaksandeler. Begge fikserte blandinger tapte INITIAL-Brier/log.
Parameter-replay001 feilet på feilaktig likhetskrav mellom view-provenienser; kvitteringen bevares,
001 relanseres ikke.002 verifiserte egne view-identiteter, fem metadataforskjeller og like normbytes.

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
