# Gjeldende GX1-mål — 2026-09-13

Tren GX1 mot positiv kostnadsjustert netto Bps med hele feature-settet og samarbeid mellom timeframes og familier. Ett års TRAIN og full juni-VAL er fullført. Hovedløpet bruker hele femårsgrunnlaget, opptil 30 epocher, juni-VAL etter hver og early stopping med patience 5. TEST er forseglet; ingen live-/papirhandel eller ekstern spending.

Første femårs-TRAIN er fullført og bevart: 313 399 rader, 19 588 optimizersteg, checkpoint 309. Brukeren har godkjent videre effektivisering etter tiltak 1–4 uten kvalitetsreduksjon.

Pushet kilde 03592fe6 i /home/andre2/src/GX1_VAL_PAUSE_ENVELOPE_V40 retter faktisk pauseformat og viderefører samme VAL-kontrakt. 18 353 548 vurderinger, hele TRAIN og samme EMA-checkpoint er bevart. Fem målrettede kontroller og obligatoriske Git-kontroller består. V39 feilet pausekvitteringen etter lagring; dette er ikke en fullført ytre kontroll.

Målt stabil fart økte fra 1 077 til 1 528 vurderinger/s etter Windows-klokkeinnstilling (+42 %), ved identiske modellbeslutninger. Alle features, tidsrammer, familier, FP32, kostnader og ressursgrenser beholdes. Videreføring er startet. Klokker settes bare når GPU-arbeid er lastet, og tilbakestilles ved tomgang. Faktisk videreføring består: alle 18 353 548 vurderinger ble gjenbrukt, og VAL har passert 18,44 millioner. GPU-kontroll har null Q-avvik. Automatisk klokking ved arbeidsstart og fravær av tomgangsblokkering er bekreftet. Første nye pausekvittering har PASS, og ordinær omstart til boot 410 med automatisk videreføring og klokking er bekreftet. VAL har passert 26,06 millioner vurderinger. Ingen pålitelig sluttid er fastslått.

Én agent og én tung jobb. Kontroller én gang i timen, etter brukerens presisering 2026-09-13. Endre bare observerte blokkeringer og uttrykkelig bestilte tiltak. Stående autorisasjon gjelder. Bevar frosne kilder, fullførte resultater og lagret fremdrift.

Ressurser: 20 GiB RAM, 512 MiB swap, 128 oppgaver, CPU 0–18; 300 W fysisk grense, 85 °C kjerne, 80 °C minne, 12 GiB VRAM. Keeper senker til 200 W ved 80 °C kjerne. Eksakte bindinger står i CURRENT_NATIVE_RUN.json. Positiv samlet Bps, nyttig bidrag fra alle ruter og liveklarhet er ikke dokumentert.
