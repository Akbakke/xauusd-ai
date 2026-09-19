# Hovedencoder — målt gjenværende fusjonsproblem

Native TRAIN16-måling er fullført: guard PASS, to forwards, null optimizer/backward,
eksakt samsvar med lagrede prediksjoner og valg. Checkpoints/RNG bevart.
Windows-task Disabled; brukt scope stengt. Ikke gjenta måling eller review.

| Måling | Initial | Final |
|---|---:|---:|
| seq_pool L2 | 8,2469 | 11,0903 |
| main_fuse L2 | 0,8085 | 25,8532 |
| main_fuse cosine | 0,51210 | 0,99727 |
| joint-normalisert RMS standardavvik | 0,33687 | 0,10385 |
| Entry-hidden RMS standardavvik | 0,28158 | 0,10704 |
| Retningsdel av MSE | 308,1568 | 308,7949 |

Fusjonsinputenes RMS-spredning er nesten uendret: 0,5653 mot 0,5601.
Fusjonsmatrisenes største singularretning har bare 5,31/4,09 prosent av energien.
Problemet er ikke påvist rank-1 i fuse-vektene. Residualprojeksjonene har derimot
97,999–99,304 prosent i største retning. Inputnormalisering er uendret;
tapsvekter nær 1. Checkpoint har ikke batchvise tap eller full gradientattribusjon.

Neste: én parameterfri normalisering etter hoved-fuse i Entry. Exit har egen hovedbane og kan påvirkes via Entry-tokenet.
Målt felles aktiveringsvekst begrunner en kontrollert kandidat, ikke en påstått kur.
Bevar gammel lærerfunksjon, vektnøkler, RNG, features og kostnader. Ny initialbaseline
må måles før separat bundet fixed256-prøve. Ingen full epoch/VAL/CONTROL/TEST,
reskalering til targetvarians eller terskelsøk. TRAIN256-porten er fortsatt ikke bestått.

Bevis: BASE/NATIVE_MAIN_ENCODER_REPRESENTATION_20260919/REVIEW.json og
SAVED_SCALE_CAUSE/RESULT.json. Første saved-state-rapport traff ubrukte best-VAL-
felters inf ved JSON-skriving; bare metadataformat ble rettet, feilrapport bevart.
BASE=/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912.
