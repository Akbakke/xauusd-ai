**Siste klargjøring kl. 09:55 norsk tid:** Første batch-128-forsøk f2b597f8 stoppet ved Q-sammenligningen før noen VAL-forward ble registrert. Ingen treningssteg er tapt. Ny kilde 267bb0c8 slår også av cuDNN TF32, som manglet i det deklarerte FP32-oppsettet. Fem målrettede kontroller og Git-hooks består. Ny faktisk sammenligning/fart gjenstår; gammel numerikk påstås ikke å ha vært kontrollert under denne innstillingen.

# Gjeldende GX1-mål — 2026-09-13

Tren GX1 mot positiv kostnadsjustert netto Bps med hele det avtalte feature-settet og samarbeid mellom timeframes og familier. Ett års TRAIN og full juni-VAL er fullført. Hovedløpet bruker hele femårsgrunnlaget, opptil 30 epocher, juni-VAL etter hver og early stopping med patience 5. TEST er forseglet; ingen live-/papirhandel eller ekstern spending.

**Beslutning og status kl. 09:17 norsk tid:** Brukeren beordret gjenstart for bedre utnyttelse. Gammel VAL/task er stoppet. Første femårs-TRAIN er bevart ved checkpoint 309, 19 588 optimizersteg og 313 399 rader. Ny kilde 267bb0c8bfa573c4553a75a0789ca4568a0af4e2 er pushet og kampanjen er klargjort. Verifiser faktisk ny PID, restaurerte steg og VAL-fremdrift med handoverscriptet; denne daterte teksten påstår ikke at oppstart allerede er fullført.

Exit-VAL-batch økes fra 16 til 128. TRAIN og Entry-VAL beholdes på 16. Modell, target, optimizer, EMA, scheduler, RNG, rekkefølge og modellvalg videreføres fra lagret TRAIN. Juni-VAL starter på nytt; gammel del-VAL blandes ikke inn. Alle data/features, økonomi, presisjon og lært Exit beholdes. Første produksjonsbatch krever nære Q-verdier og samme handlinger mot 16-batcher og logger målt hastighet. Ingen fart eller positiv Bps påstås før faktisk måling.

Fire målrettede kontroller, faktisk checkpoint-kontroll og Git-hooks består. Ingen fullført TRAIN eller full smoke er gjentatt. Ingen ny agent. Ressursgrensene er fortsatt 300 W, 85 °C kjerne, 80 °C minne og 12 GiB VRAM; keeper senker til 200 W ved 80 °C kjerne. CPU-tallet alene er ikke mål på effektivitet; sammenlign tilstander per sekund og faktisk GPU-utnyttelse.

Fryst kilde: /home/andre2/src/GX1_VAL_FP32_V35
Runtime: /home/andre2/GX1_RUNS/UNIFIED_EXIT_FULL_TRAIN_VAL128_FP32_267BB0C8_BOOT399
Recipe: /home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_FULL_TRAIN_RECIPE_VAL128_FP32_V6.json
Recipe-fil-SHA: f19dbd19adb389acc859278daa483c3f3df3959ce6ab09ed33a083f2645ffde7
Plan: /home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/CAMPAIGN_NATIVE_VAL128_FP32_267BB0C8_BOOT399/CAMPAIGN_PLAN.json
Plan-fil-SHA: c723e57d159baa111ddb3745a577bbd58155572997856a044a3361cf8ac6cc0d

Ta over via CURRENT_HANDOVER.md, SYSTEM_MAP.md og CURRENT_NATIVE_RUN.json. handover_snapshot/VAL128_PREPARED.json inneholder klargjøringskvitteringen. Gamle native/Windows-observasjoner i samme mappe beskriver uttrykkelig forgjengeren. Checkpointet er kontrollert på Mac; Git er ikke en full rådatabackup. Første nye native start og VAL-resume må bekreftes i aktuell runtime.

Neste arbeid er å måle faktisk fart og følge TRAIN/VAL gjennom eksisterende kontroller, ikke nye analyser eller fullsuiter. Kontroller lange kjøringer omtrent hvert 15. minutt og vær stille ved vanlig fremgang. Brukeren har autorisert denne konkrete gjenstarten; ikke be om samme godkjenning igjen. Bevar begge fryste kilder og begge checkpoint-/VAL-historikker.

Kvalitetsbegrensninger og fullført tre-agent-gjennomgang står i docs/audit_20260912/. Positiv samlet Bps, nyttig bidrag fra alle ruter og liveklarhet er ikke dokumentert. Gammel smoke ga minus på lukkede shortforløp og åtte åpne valgte posisjoner, derfor ingen full-policy Bps.
