# GX1-status — 17. september 2026

Kausal Entry-prøve på kilde955abf19 er fullført: 256 oppdateringer og native
TRAIN-only sluttmåling, guardPASS/trainer0/observer0. Ingen native prosess;
Windows-tasken er deaktivert. Launch-unntaket er stengt. Ingen ny trening nå.

Korrekt Entry-baseline og uendrede Exit-targets er bekreftet eksakt bevart.
**Paret analyse er ferdig: læringsport ikke bestått. Entry FLAT256/256 og
MSE dårligere enn TRAIN-konstanter; Exit alltid HOLD for LONG / EXIT for SHORT.
Alle fire Exit-MSE er dårligere enn forrige kandidat. Neste er årsaksdiagnose
fra eksisterende bevis, ingen ny trening. Generalisering/profitt er ikke bevist.**

Les CURRENT_HANDOVER.md og VEIEN_VIDERE.md. Kjør ./handover.sh --check på Mac
eller bash scripts/gx1_handover.sh --check i Linux for fersk status.

## Prediksjonsavviket er målt — ingen aktiv jobb

Fire forwards er fullført på e53645d7, samme cachede TRAIN16 og frosne initial/
finalmodeller. Native sluttkvittering: 18:34:13 UTC / 20:34:13 Oslo, guard PASS,
trainer/observer 0. Faktisk boot459, forberedt runtime-navn BOOT458. Windows-task
deaktivert, controller avsluttet, null optimizersteg og originale checkpoints bevart.

Inferens matcher begge lagrede prediksjoner eksakt. Gradientmodus avviker med
maks 0,0002992153 Bps initialt og 0,0001640320 Bps til slutt; ingen handlingsbytter.
Den gamle 0,0001 Bps-kontrollen mellom ulike modus består altså ikke. Toleransen
er uendret. Dette forklarer signaldiagnosens måleblokkering, ikke læringssvikten.
Det er ikke isolert hvilken enkeltkernel som gir forskjellen.

Den testede rettelsen gjelder bare diagnostikken: verifiser vanlig inferens mot lagret
inferens med samme toleranse, og beregn/rapporter gradientmodus separat. Bevar
handlingskontroll og synlig numerisk avvik; ikke påstå eksakt samsvar mellom modus.
Deretter kan en særskilt bundet signaldiagnose finne hvor variasjon/gradientsignal
går tapt. Rettelsen er implementert og fem målrettede tester består.
En ny signalplan må bindes før kjøring; ingen ny trening er åpnet.
Den brukte parity-planen er stengt og skal ikke relanseres.

Bevis: `handover_snapshot/ENTRY_FORWARD_PARITY_{RESULT,REVIEW}_20260917.json` og
`BASE/NATIVE_ENTRY_FORWARD_PARITY_20260917/REVIEW.json`. Her er BASE den vanlige
prebuilt-roten i GX1_DATA. Entry/Exit-læringsporten er fortsatt ikke bestått.
