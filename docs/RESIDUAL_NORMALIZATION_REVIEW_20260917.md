# Residualnormalisering — fullført og avvist

Én native TRAIN16-prøve på88310075 fullførte256 oppdateringer og slutt-ONLINE.
Kjøring20:13:15–21:27:47UTC /22:13:15–23:27:47Oslo, boot461. GuardPASS,
trainer/observer0. Topp59°C kjerne,66°C minne,162,12W,8019MiB. Task Disabled;
ingen native prosess. Entry-/Exit-fasit og frossen lærer er eksakt bevart.

Paret analyse bruker samme256 TRAIN-observasjoner,256 Exit-ankere og1024
samplede Exit-states, begge sider og alle ni måneder. Gamle outputs er
gjenbrukt; null nye forwards eller trening. Baselineberegningene er kontrollert
mot tidligere rapporter, inkludert kausal256. Resultatet er ikke generalisering.

| MSE | Kausal256 | Residual256 | TRAIN-konstant |
|---|---:|---:|---:|
| Entry LONG | 612,813 | 613,846 | 608,803 |
| Entry SHORT | 607,468 | 606,286 | 603,814 |
| Exit-anker LONG | 625,789 | 626,062 | 625,037 |
| Exit-anker SHORT | 622,351 | 622,144 | 622,612 |
| Exit-samplet LONG | 864,707 | 865,095 | 864,710 |
| Exit-samplet SHORT | 858,718 | 858,597 | 859,420 |

Entry velger fortsatt FLAT256/256. Exit velger fortsatt HOLD for alle LONG og
EXIT for alle SHORT, både anker og samplet. Valg og referanseverdi/regret er
identiske med kausal256. Små feilendringer er blandet mellom sider/måneder;
begge Entry-sider taper mot konstanter. Ingen samlet læringsport er bestått.
LONG–SHORT-prediksjonsstandardavvik er0,104Bps mot target49,239Bps; dette
beskriver svak variasjon, men beviser ikke alene hva som er predikerbart.

CPU-kontrollen av lagrede vekter viser analytiske normgrenser1,568 /3,474 /
1,753 for de tre endrede residualkorreksjonene etter fusjonsskala. Dette er
ikke målte globale representasjonsnormer eller Bps. Det dekker ikke hoved-
fuse eller rå MTF-representasjon som også går inn i Entry-Q. Skalakontroll ved
de tre portene alene ga altså ikke bedre beslutninger; gjenværende årsak er
ikke lokalisert. Ingen ny normalisering eller tapsendring er begrunnet ennå.

Artefakter: BASE/NATIVE_RESIDUAL_NORMALIZED_FIXED256_20260917/
PAIRED_TRAIN_REVIEW.json,VERDICT.json,RESIDUAL_BOUND_AUDIT.json og
COMPLETION_REVIEWED.json. BASE er
/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912.
Alle checkpoints og originalobservasjoner er bevart. Brukt scope er stengt.
Ingen ny kjøreplan, full epoch/VAL, CONTROL/TEST, live/paper eller spending.
