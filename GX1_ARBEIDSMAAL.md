# Gjeldende GX1-mål — 2026-09-13

Tren GX1 mot positiv kostnadsjustert netto Bps med hele det avtalte feature-settet og samarbeid mellom timeframes og familier. Ett års TRAIN og full juni-VAL er fullført. Hovedløpet bruker hele femårsgrunnlaget, opptil 30 epocher, juni-VAL etter hver og early stopping med patience 5. TEST er forseglet; ingen live-/papirhandel eller ekstern spending.

**Beslutning og status kl. 09:17 norsk tid:** Brukeren beordret gjenstart for bedre utnyttelse. Gammel VAL/task er stoppet. Første femårs-TRAIN er bevart ved checkpoint 309, 19 588 optimizersteg og 313 399 rader. Ny kilde f2b597f8ff9a62a81dfb7106afd23b4ae74c894b er pushet og kampanjen er klargjort. Verifiser faktisk ny PID, restaurerte steg og VAL-fremdrift med handoverscriptet; denne daterte teksten påstår ikke at oppstart allerede er fullført.

Exit-VAL-batch økes fra 16 til 128. TRAIN og Entry-VAL beholdes på 16. Modell, target, optimizer, EMA, scheduler, RNG, rekkefølge og modellvalg videreføres fra lagret TRAIN. Juni-VAL starter på nytt; gammel del-VAL blandes ikke inn. Alle data/features, økonomi, presisjon og lært Exit beholdes. Første produksjonsbatch krever nære Q-verdier og samme handlinger mot 16-batcher og logger målt hastighet. Ingen fart eller positiv Bps påstås før faktisk måling.

Fire målrettede kontroller, faktisk checkpoint-kontroll og Git-hooks består. Ingen fullført TRAIN eller full smoke er gjentatt. Ingen ny agent. Ressursgrensene er fortsatt 300 W, 85 °C kjerne, 80 °C minne og 12 GiB VRAM; keeper senker til 200 W ved 80 °C kjerne. CPU-tallet alene er ikke mål på effektivitet; sammenlign tilstander per sekund og faktisk GPU-utnyttelse.

Fryst kilde: /home/andre2/src/GX1_VAL_THROUGHPUT_V34
Runtime: /home/andre2/GX1_RUNS/UNIFIED_EXIT_FULL_TRAIN_VAL128_F2B597F8_BOOT398
Recipe: /home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_FULL_TRAIN_RECIPE_VAL128_V5.json
Recipe-fil-SHA: 1af1bec61ad9f794fc97964da19665f9de5f665dff6897e6488f09855f47f5fd
Plan: /home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/CAMPAIGN_NATIVE_VAL128_F2B597F8_BOOT398/CAMPAIGN_PLAN.json
Plan-fil-SHA: 6e990eed1a6d8a3d4013fb0689ddbec92c05caa62d2a4d9f0f1caa046a0bcc46

Ta over via CURRENT_HANDOVER.md, SYSTEM_MAP.md og CURRENT_NATIVE_RUN.json. handover_snapshot/VAL128_PREPARED.json inneholder klargjøringskvitteringen. Gamle native/Windows-observasjoner i samme mappe beskriver uttrykkelig forgjengeren. Checkpointet er kontrollert på Mac; Git er ikke en full rådatabackup. Første nye native start og VAL-resume må bekreftes i aktuell runtime.

Neste arbeid er å måle faktisk fart og følge TRAIN/VAL gjennom eksisterende kontroller, ikke nye analyser eller fullsuiter. Kontroller lange kjøringer omtrent hvert 15. minutt og vær stille ved vanlig fremgang. Brukeren har autorisert denne konkrete gjenstarten; ikke be om samme godkjenning igjen. Bevar begge fryste kilder og begge checkpoint-/VAL-historikker.

Kvalitetsbegrensninger og fullført tre-agent-gjennomgang står i docs/audit_20260912/. Positiv samlet Bps, nyttig bidrag fra alle ruter og liveklarhet er ikke dokumentert. Gammel smoke ga minus på lukkede shortforløp og åtte åpne valgte posisjoner, derfor ingen full-policy Bps.
