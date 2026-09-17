# Én læringsprøve med korrekt Entry-fasit

Entry-fasiten valgte første HOLD/EXIT-handling etter observerte framtidsutfall.
Rettelsen på ad464322 bruker forventningen under den allerede bundne kausale
referansepolicyen. Exit-HOLD-mål, lærer, kostnader, kausale inputs, features,
arkitektur og hyperparametere er uendret. Tidligere Entry-gradientrettelse beholdes.

Den kanoniske økonomieieren er brukt for første likvidasjon på TRAIN256.
Alle 512 gamle sidemål gjenskapes eksakt før korreksjon. Lagrede prediksjoner
gjenbrukes uendret. Baseline er eksplisitt avledet, med original native receipt
og separate filbindinger. Ingen ny modell-forward eller normaliseringsfit.
Første operatør stoppet på semantisk hash mot filhash før økonomiberegning;
den er bevart. Riktig validering bruker begge hashtypene på sine egne vilkår.

Ny targetmiddel er −3,7500 LONG og −7,9601 SHORT Bps. Gamle connected256 har
MSE 649,51/681,49, mot konstante TRAIN-midler 608,80/603,81. Valgene har
referanseverdi −3,5522 Bps, mot FLAT 0. Dette viser manglende kvalitet i gamle
prediksjoner under korrekt fasit; ikke utfallet av den nye læringsprøven.

Minste native endring er en eksplisitt avledet-baseline-binding. Den validerer
original observasjon, uendrede prediksjoner/Exit-mål, kilde/data/policy og
korrekt Entry-beregning. Sluttmålingen krever eksakt targetlikhet som før.
TRAIN-only leser ikke CONTROL-observasjoner. 13 berørte syntetiske tilfeller
består, inkludert endrede targets/prediksjoner/kilde/policy og bevart session.
Tre ikke-anvendelige derived+CONTROL-kombinasjoner hoppes over. Faktisk
baseline og scope er også validert. Ingen bestått fullsuite gjentas.

Én prøve: samme lagrede ferske initialisering, samme 4096 Entries og rekkefølge,
TRAIN16, 256 oppdateringer, uendret frossen lærer, slutt-ONLINE. Ingen best-
checkpoint-utvelgelse. Mål TRAIN256, Exit state0 og 1024 samplede states.
Sammenlign initialmodell, tidligere connected256 og konstante TRAIN-baselines
med identiske korrigerte targets. Rapporter begge sider og alle ni måneder,
MSE, sentrert feil, korrelasjon, LONG−SHORT-kontrast, handlingsfordeling og
referanseverdi/regret. Biasflytting eller all-FLAT/all-HOLD er utilstrekkelig.
Entry alene består ikke samlet Entry/Exit-port.

Dette er gjenbrukt TRAIN med fitted-overlapp og Q_mu/V_mu-referanseutfall,
ikke generalisering eller realisert handelsprofitt. Ingen CONTROL-forward,
økonomireplay, søk, automatisk gjentakelse, full epoch/VAL eller TEST. Alle
native vakter og opprinnelige checkpoints/resultater bevares. Plan og bevis:
handover_snapshot/CAUSAL_ENTRY_TRAIN_BASELINE_20260917.json.

## Teknisk fullført — historisk status før analysen

Kjøring955abf19 fullførte 256 oppdateringer og TRAIN-only sluttmåling
2026-09-17 kl.15:59:49UTC /17:59:49Oslo. GuardPASS/trainer0/observer0;
maskinvaretopper59C/66C,156,89W,8014MiB. Ingen aktive native prosesser;
Windows-tasken er deaktivert. Det brukte launch-unntaket er stengt.

Sluttresultatet binder korrekt avledet Entry-baseline, samme targetmodell og
eksakt bevart fasit. Mål og observasjoner er lagret; læringskonklusjon er
**ikke gjort**. Neste er den på forhånd beskrevne parete analysen. Se
VEIEN_VIDERE.md og handover_snapshot/CAUSAL_ENTRY_FIXED256_COMPLETION_20260917.json.

## Paret analyse fullført

Læringsporten er ikke bestått. Entry velger FLAT256/256 og slår ikke
TRAIN-konstanter; Exit-handlinger er konstante per side og alle fire MSE er
verre enn connected256. Se docs/CAUSAL_ENTRY_FIXED256_REVIEW_20260917.md.
Ingen ny trening er åpnet; avsnittet over er kun tidsstemplet historikk.
