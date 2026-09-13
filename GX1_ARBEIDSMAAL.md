# Gjeldende GX1-mål — 2026-09-13

Tren GX1 mot positiv kostnadsjustert netto Bps med hele feature-settet og samarbeid mellom timeframes og familier. Ett års TRAIN og full juni-VAL er fullført. Hovedløpet bruker hele femårsgrunnlaget, opptil 30 epocher, juni-VAL etter hver og early stopping med patience 5. TEST er forseglet; ingen live-/papirhandel eller ekstern spending.

Første femårs-TRAIN er fullført og bevart: 313 399 rader, 19 588 optimizersteg, checkpoint 309. Brukeren har godkjent videre effektivisering etter tiltak 1–4 uten kvalitetsreduksjon.

Ny pushet kilde 1548dd7c i /home/andre2/src/GX1_VAL_HOTPATH_V39 gjenbruker faste opplysninger og identisk posisjonshistorikk mellom LONG/SHORT i VAL, og unngår gjentatt bygging av hele aktivitetslisten. TRAIN, alle features, tidsrammer, familier, begge retninger, kostnader og FP32 beholdes.

Forrige økt lagret 13 176 595 VAL-vurderinger, men feilet deretter ved kolliderende pausekvitteringer. Ny versjon identifiserer også lagret VAL-fremdrift i kvitteringsnavnet. Disse lagrede beregningene og samme EMA-checkpoint er bundet i ny oppskrift. Femten målrettede CPU-kontroller og obligatoriske Git-kontroller består. Neste handling er faktisk videreføring, GPU-kontroll og måling av fart. Ingen pålitelig sluttid er fastslått.

Én agent og én tung jobb. Kontroller omtrent hvert 15. minutt. Endre bare observerte blokkeringer og uttrykkelig bestilte tiltak. Stående autorisasjon gjelder. Bevar frosne kilder, fullførte resultater og lagret fremdrift.

Ressurser: 20 GiB RAM, 512 MiB swap, 128 oppgaver, CPU 0–18; 300 W fysisk grense, 85 °C kjerne, 80 °C minne, 12 GiB VRAM. Keeper senker til 200 W ved 80 °C kjerne. Eksakte bindinger står i CURRENT_NATIVE_RUN.json. Positiv samlet Bps, nyttig bidrag fra alle ruter og liveklarhet er ikke dokumentert.
