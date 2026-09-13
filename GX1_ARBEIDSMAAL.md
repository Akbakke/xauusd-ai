# Gjeldende GX1-mål — 2026-09-13

Tren GX1 mot positiv kostnadsjustert netto Bps med hele feature-settet og samarbeid mellom timeframes og familier. Ett års TRAIN og full juni-VAL er fullført. Hovedløpet bruker hele femårsgrunnlaget, opptil 30 epocher, juni-VAL etter hver og early stopping med patience 5. TEST er forseglet; ingen live-/papirhandel eller ekstern spending.

Første femårs-TRAIN er fullført og bevart: 313 399 rader, 19 588 optimizersteg, checkpoint 309. Kilden 0e81f5b8 fullførte et godkjent VAL-segment med 1 344 280 tilstander og lagret fremdrift. Brukeren har godkjent effektiviseringstiltak 1–4 uten kvalitetsreduksjon.

Ny frosset og pushet kilde 39bdb3ce fjerner gjentatt markedsinput, samler økonomibehandlingen, bruker fire CPU-arbeidere innen samme jobb og tillater WSLs 19 CPU-tråder med normal prioritet. Alle features, tidsrammer, familier, juni-data, begge retninger, kostnader og FP32 beholdes. Foreldrens numeriske trådtall er fortsatt åtte. Ny oppskrift binder det eksisterende EMA-checkpointet og lagret VAL; fullført TRAIN gjentas ikke.

Nitten målrettede kontroller og obligatoriske Git-kontroller består. Neste handling er å bekrefte faktisk oppstart, videreført VAL og målt fart gjennom hele batchbehandlingen. Ingen pålitelig sluttid er fastslått; tidligere lineære anslag er trukket tilbake.

Én agent og én tung jobb. Kontroller omtrent hvert 15. minutt. Endre bare observerte blokkeringer og uttrykkelig bestilte tiltak. Stående autorisasjon gjelder. Bevar frosne kilder, fullførte resultater og lagret fremdrift.

Ressurser: 20 GiB RAM, 512 MiB swap, 128 oppgaver, CPU 0–18; 300 W fysisk grense, 85 °C kjerne, 80 °C minne, 12 GiB VRAM. Keeper senker til 200 W ved 80 °C kjerne. Eksakte bindinger står i CURRENT_NATIVE_RUN.json. Positiv samlet Bps, nyttig bidrag fra alle ruter og liveklarhet er ikke dokumentert.
