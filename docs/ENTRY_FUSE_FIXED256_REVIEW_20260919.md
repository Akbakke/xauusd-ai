# Entry fuse256 — bedre verdiestimat, uendrede handelsvalg

Prøven avsluttet 19. september 2026 kl. 13:12:16 UTC / 15:12:16 Europe/Oslo.
256 optimizersteg, 4096 TRAIN-rader, guard PASS, task Disabled, ingen native
prosess. Paret CPU-review er fullført uten nye forwards, fits eller optimizersteg.
ONLINE- og lærerhash, optimizer/EMA256, cursor, receipts og identiske korrigerte
targets/masker/cohort er kontrollert. Alle historiske aggregater ble gjenskapt.

| MSE, Bps² | Ny initial | Hovedencoder256 | Fuse256 | TRAIN-konstant |
|---|---:|---:|---:|---:|
| Entry LONG | 630,787 | 610,234 | 607,208 | 608,803 |
| Entry SHORT | 693,007 | 604,074 | 602,660 | 603,814 |
| Exit-anker LONG | 629,516 | 626,710 | 626,304 | 625,037 |
| Exit-anker SHORT | 627,921 | 622,628 | 621,883 | 622,612 |
| Exit-samplet LONG | 864,747 | 864,918 | 864,703 | 864,710 |
| Exit-samplet SHORT | 859,520 | 858,564 | 857,879 | 859,420 |

Begge Entry-sider slår alle sammenlignede modeller og TRAIN-konstanter på samlet
MSE og sentrert MSE. Alle seks Entry/Exit-celler forbedres mot hovedencoder256.
Forbedringen er liten og ujevn per måned: sentrert Entry-feil slår hovedencoder
på LONG i 5/9 og SHORT i 8/9 måneder. Entry LONG−SHORT-korrelasjon er 0,248;
innen månedene er den 0,238 mot 0,231 for hovedencoder og 0,060 for ny initial.
Alle ni måneder har positiv kontrastkorrelasjon. Dette er gjenbrukt TRAIN med
fitted-overlapp; de små utvalgene beviser ikke generalisering.

Entry velger fortsatt FLAT256/256. Beste LONG-minus-FLAT er −4,462 Bps;
beste SHORT-minus-FLAT er −5,292 Bps. LONG/SHORT-prediksjonenes standardavvik
er 0,380/0,372 Bps. Exit velger fortsatt HOLD for alle LONG og EXIT for alle
SHORT, både på 256 ankre og 1024 samplede states. Handlinger og referanseverdi/
regret er identiske med hovedencoder256. På samplede Exit-states taper begge
faste sidevalg mot de respektive TRAIN-konstantvalgene.

Konklusjon: REJECT_EXPANSION_ENTRY_FUSE_VALUE_FIT_IMPROVED_ACTIONS_UNCHANGED.
Målbar verdiestimat-forbedring er ikke bedre handelsbeslutninger eller profitt.
Samlet læringsport er ikke bestått. Ingen ny kjøring er bundet; brukt scope er
stengt selv om checkpoint/cursor bruker complete=false/RESUMABLE.

Rettelsen var én parameterfri normalisering i Entry-fuse. Exit har separat
hovedbane og bare en indirekte kobling via Entry-token; bedre Exit-fit er ikke
bevis på at Exit har fått en direkte arkitekturrettelse. Ingen model/trainer-
endring er gjort etter dette forsøket. Bevar den målte kandidaten og originalene.

Neste avklaring gjelder den faktiske felles treningsoppdateringen, før nye
modellendringer. First-batch Entry-rutingsgradient18,151 dekker bare fire
familie-/tidsramme-gateparametre. Den dokumenterer ikke gradientkonflikt eller
full Adam-oppdatering gjennom fuse og begge verdi-hoder. Les eksisterende
checkpoint/optimizer og målerens dekning før eventuell separat native binding.
Targetvarians er ikke nødvendigvis predikerbar; ikke skaler opp eller flytt
terskler for å tvinge handler. Ingen brede søk eller automatisk ekstra trening.

Artefaktrot: /home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_ENTRY_FUSE_NORMALIZED_FIXED256_20260919.
PAIRED_TRAIN_REVIEW.json, DECISION_GAP_AUDIT.json, VERDICT.json og
COMPLETION_REVIEWED.json er fullførte bevis. Ikke kjør operatorene om igjen.
