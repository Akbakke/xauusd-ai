# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier status; prosess/terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_PREPOOL_READOUT_001_COMPLETE_PERIOD_TRANSFER_NEXT.
Edge-målet er uoppnådd. Alle treningsflagg er false. Ingen aktiv jobb eller ny fit er admittert.
Samlet siste rapport: docs/TA_PREPOOL_PREDICTIVE_REVIEW.json; eksakt SHA står i policy.

Neste arbeid gjelder overføring fra TRAIN til senere perioder: gjenbruk eksisterende
kronologiske fold-/dekningsbevis og månedlige native prognoser for å skille nivåforskyvning
fra svekket eller reversert rangering innen periodene. Ikke gjenta poolingvarians, separate
hoder eller samme prepool-fit. Høy TF-korrelasjon alene begrunner ingen arkitekturendring.
En retrospektiv splitt er ikke uavhengig OOF hvis backbone allerede har sett scoreradene.
Full kvalifisert historikk/30–40 epoker er fortsatt senere retning, ikke en åpnet kjøring.
COT2024 release/vintage forblir UNKNOWN; ingen konkret lekkasje er påvist.

## Siste målte resultat

En matchet fast lineær metode er kjørt på dagens FINAL c34. Fem F-mål × to128D-armer:
faktiske TF-vektorer etter cross-TF attention og før pooling, mot felles hidden etter
sammensetting/projeksjon. Samme TRAIN-rader, kjentmasker, like observasjonsvekter og prior.
Seks koordinatsett bruker bare TRAIN. Eksisterende Newton-fit med alpha0,01, også på bias.
Dette er eksplisitt standardiserte koordinater; gamle råkoordinatfits ommerkes ikke.
Alle ti fitter ble låst før LATER ble dekodet. Ingen native modellvekt eller output er endret.

M5 F senere AUC: PREPOOL0,503209 / SHARED0,501529 / NATIVE0,507776.
Alle fem PREPOOL−SHARED AUC-/accuracy-kontraster inkluderer null med både dag-/ukeintervall.
PREPOOL gir lavere M5/M15 Brier enn SHARED (−0,008094/−0,031590; begge intervaller under0),
men begge readouts er dårligere enn fast TRAIN-prior på M5/M15 Brier/log i begge intervaller.
SHARED passer TRAIN bedre enn PREPOOL på alle fem TF. SHARED TRAIN→senere AUC:
M5 0,6630→0,5015; M15 0,7567→0,4954; H1 0,8953→0,5502;
H4 0,9554→0,5056; D1 0,9989→0,5958. Overføringsgapet er målt; årsaken er ikke isolert.

Bevar unntakene: H1 SHARED−NATIVE AUC+0,069778 utelukker0 i begge intervaller,
men H1-readout mot0,5 er usikker. D1 PREPOOL AUC0,626675 har brede intervaller mot
prior/native/shared. Ingen generell påstand om at informasjon eller markedssignal mangler.

Native månedlig AUC oktober/november/desember:
M5 0,565820/0,511835/0,505090; M15 0,607200/0,497404/0,499639;
H1 0,647872/0,483441/0,445580; H4 0,468590/0,629468/0,483199;
D1 0,608163/0,557749/0,574633. Alle måneder beholdt, ingen månedsseleksjon.
M5 samletAUC0,507776 består av par innen samme måned (AUC0,530995) og par på tvers
(AUC0,496164). Dette er en eksakt parvektet oppdeling, ingen ny accuracy/prognose,
kalibrering eller bekreftet edge. M5/M15/H1 svekkes også innen måneder etter oktober.

## Utført og verifisert

PREPOOL_CAPTURE001: /home/andre2/GX1_RUNS/MARKET_STATE_TA_PREPOOL_CAPTURE_20261113_001.
Kilde2146fb5a; PLANcf0ed75f, RESULT360e5616, TERMINAL72f839be, VALIDATION0997f253.
Én FINAL-checkpoint, 512 forwards/8192rader, 0updates, 577,78s. TRAIN807801df og
LATERd15575c2 er private aktiveringer. Alle512S/F-batcher bitlike opprinnelige prognoser;
modell/buffere/moduser/RNG/hooks/inputs uendret. Vakt exit0, peak51C/54C/136,54W/552MiB.
Eksisterende head-input-observatør ble utvidet; produksjonsmodell-/treningskode uendret.
Seks fokuserte syntetiske tester, sju cap-rutetester og eksakt CLI-preflight PASS.

PREPOOL_READOUT001: /home/andre2/GX1_RUNS/MARKET_STATE_TA_PREPOOL_READOUT_20261113_001.
Kilde186a4071; PLAN137afc0b, SCRIPTff04cf52, RESULT7b94f2c5, TERMINALe50e7dbe,
VALIDATION372ae8af. Ti CPU-fits, seks koordinatsett, fire arkiver, 5,68s under4G/512M/1tråd.
0modeller/checkpoints/forwards/nativeupdates. Maks sluttgradient8,02e-9; alle konvergerte.
40 scoreceller kontrollert; native replayfeil0, månedsreaggregering≤2,23e-16 og
confusions eksakte. Uavhengig kilde-/JSON-/parregnskapskontroll PASS. Alle5TF og alle
måneder er beholdt; 2000parede trekk på70dager/14uker, punktvise intervaller.
Begge engangsplaner er COMPLETE/CONSUMED, ingen relaunch eller promotion.

## Gjenbruk og ubeviste grenser

Dagens funksjon har 317 signal,132 kontekst,255 felt per TF, åtte familier og D1/makro.
FullTA-konsum er kontrollert via tidligere8×5bevis107c9531, span8878d17a/4476c63b,
current inputdrift2c87d24c og native paritet. Ingen ny feltcensus trengs.
29 574 TRAIN-beslutninger er mai–september2024; D1-historievindu går til2023-05-09.
Current D1-EMA var konstant på TRAIN-beslutninger, men eldre D1-historikk har variasjon.
Juni2026 er gjenbrukt utviklings-VAL; denne sammenligningen bruker oktober–desember2024.

Tidligere hodedelingsforsøk cb9c02a6/66c08aa5 og transfer34de21d6 brukte modell7c86 og
samme post-query-vektor i begge armer; de tester ikke prepool. Temporal-capture b89942e5
brukte gammelc171 og256+256rader. Ingen av disse skal relanseres eller ommerkes.
Nære TF-rangeringer: docs/TA_CURRENT_TF_SEPARATION_REVIEW.json.
UP-driftens endepunktsregnskap: docs/TA_HEAD_REPRESENTATION_REVIEW.json.
Seksfelts kontekstintervensjon: docs/TA_CONTEXT_DRIFT_REVIEW.json; liten målt virkning.
Eksakt accuracy/klasseforveksling og ordre: docs/TA_ORDER_ACCURACY_REVIEW.json.
Input-/queryavvik: docs/TA_QUERY_INPUT_DRIFT_REVIEW.json. M15S native15min var svak
allerede påTRAIN; pooled sampled-S er ikke samme oppgave som senere native-S.

Denne lineære metoden måler ikke all tilgjengelig informasjon eller isolerer pooling alene.
INITIAL-backbone arver eldre trening; den er ikke en helt utrent modell. Korte forsøk dømmer
ikke full historikk/kapasitet. Kjent-side-nevner er ikke valgbar Entry-dekning. Ingen uavhengig
bekreftelse, Entry-/Exit-kvalitet eller økonomi er etablert. GC er på pause, genuine mål består.
TEST forseglet; ingen broker, handel, spending, promotion eller DATA/RUNS-opprydding.
