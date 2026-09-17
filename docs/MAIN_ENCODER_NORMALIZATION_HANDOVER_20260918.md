# Hovedencoder — historisk overlevering før målebinding

Måleporten er senere rettet og én nullstegsmåling bundet. Se
MAIN_ENCODER_INITIAL_MEASUREMENT_20260918.md. Restarbeid omtalt nedenfor
beskriver det tidligere stoppunktet.

Bekreftet 18. september 2026. Modellrettelsen er en hypotese støttet av
representasjonsmålingen, ikke et nytt læringsresultat.

## Kode som er bevart

- `gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py`: én ny
  `norm=nn.LayerNorm(d_model, elementwise_affine=False)` i hovedencoder.
  Ingen nye parametere, state-nøkler eller RNG-trekk. Den eksisterende
  memory-bounded encoder bruker final norm i både trening og inferens.
- `gx1/models/entry_v10/entry_v10_ctx_train_v3.py`:
  `_copy_frozen_prefix_reference_model` kopierer modellen og fjerner bare
  denne final norm fra læreren. Prefix-bindingen navngir ONLINE/target-
  funksjonene; fresh/restore/epoch-target-konstruksjon bruker samme hjelper.
- `gx1/scripts/run_unified_exit_random_access_full_train_v1.py`:
  initial-/sluttmåling bruker lærerhjelperen og registrerer modellfunksjonene.
- `tests/test_entry_main_encoder_normalization.py`: normgrense, gradienter,
  radisolasjon, train/inference og komplett original lærerparitet for Entry,
  Entry-token og Exit. Tre eksisterende prefix-testfiler har tilpassede Linear-fixtures.

ONLINE: `main_encoder_final_layernorm_no_affine_v1`.
Target: `main_encoder_no_final_norm_v1`.
For endelig aritmetikk begrenses encoder-token/pool til L2 ≤ √128 ≈ 11,31.
Dette begrenser ikke senere fuse-vekter eller andre grener. Ingen målformel,
tapsvekt, feature eller handelsregel er endret.

## Gjennomførte kontroller

Første kombinerte testkjøring: 49 bestått, 11 fixture-feil, 3 utelatt.
De 11 feilene skyldtes at Linear-testkopien ikke ble satt eval/frozen.
Etter fixture-rettelsen besto alle 11 berørte tester; tre ugyldige kombinasjoner
var utelatt. Ingen kjent testfeil står uløst; ingen fullsuite ble gjentatt.
Dette er samlet evidens fra to kjøringer, ikke påstand om én feilfri fullkjøring.
Se `handover_snapshot/MAIN_ENCODER_TEST_REVIEW_20260918.json` for hashes.

Faktisk produksjonsconstructor ble kontrollert
17. september 22:56:11 UTC / 18. september 00:56:11 Oslo:
9 617 497 parametere; identiske online-/lærervekter og RNG mot original
constructor. ONLINE-funksjonen endres, lærerfunksjonen bevares. Constructor-
audit inneholder ingen native dataforward; outputparitet er separat syntetisk test.

Artefaktrot: `/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/MAIN_ENCODER_NORMALIZATION_20260918`.
Bevar `INITIALIZATION_AUDIT.json`, `AUDIT_INITIALIZATION.py`,
`INITIALIZATION_AUDIT.log`, `TESTS.log`, `TEST_RECHECK.log` og `TEST_RECHECK.xml`.
`TRAIN_ONLY_TESTS.log` finnes ikke. Ingen TRAIN-only-kontraktsendring ble lagt
inn før overleveringen, selv om et tidligere arbeidsnotat hevdet dette.
Autoritativ kode og artefakter ble kontrollert; se VEIEN_VIDERE.md for restarbeid.

## Eksakte gjenbruksreferanser

`BASE=/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912`.

- Original initialisering: `BASE/PREFIX_FRESH_INITIALIZATION_20260917/RESULT.json`,
  SHA256 `0c3566aac54b866d4c18d148a725cd1c66e9a135f36d94da1391f14151fba4af`.
- Samme mappe, `INITIAL_STATE.pt`, SHA256
  `f25206898ff121d78f6d042c897886aacd4ec95f2d861b2cf2abbe01dfb28bfe`.
  Modell-/lærertensordigest `f1e8691dd92c133860100bbca5853065ea02088ab762d2772c9f2bb735b3656d`.
- Ny modellfil SHA256 `9ac05e5c79279e7f02da66ff0fd0f98bf591ad7966d80f1c4dce2f726e1e1541`.
  Original funksjon: commit `6f149e37423ae122155818d86ca7701a9946e264`,
  modellfil SHA256 `cd09eb6bc798b9335c55f7b6d8c011abe5565e737bae29199edf308678a3c35a`.
- Ferdig representasjonsdiagnose:
  `BASE/NATIVE_RESIDUAL_REPRESENTATION_20260918/REVIEW.json`, SHA256
  `754fe0677135733f83c2c8e47426ab4ca2dbf100da7da137e22760e3af584bf1`.
- Avvist residual256: `BASE/NATIVE_RESIDUAL_NORMALIZED_FIXED256_20260917`,
  `PAIRED_TRAIN_REVIEW.json` og `VERDICT.json`. Originalt checkpoint5/offset256,
  modell `c940a3b855e95067e16882558daffeb25f896e22b6ab23d425afc80db40f4db1`.
- Kausal Entry-baseline: `BASE/CAUSAL_ENTRY_TRAIN_BASELINE_20260917/RESULT.json`.
- Historisk native nullstegsmåling: `BASE/NATIVE_PREFIX_INITIAL_MEASUREMENT_20260917`.
  Bruk struktur/koordinater som referanse, aldri gamle ONLINE-prediksjoner som
  ny baseline. Originale Entry-targets før kausal rettelse er heller ikke fasit.

Gamle ikke-null checkpoints kjørt gjennom ny ONLINE-kode er endrede funksjoner.
Gjenbruk lagrede historiske prediksjoner for historiske sammenligninger.
Ingen native kjøring, CONTROL/VAL/TEST eller 256-prøve er nå autorisert.

## Overleveringskontroll

Alle aktive inngangsdokumenter og de tre lesende shell-inngangene er oppdatert.
Collector viser rettelsen og eksakt stoppunkt under `current_work`.
Handover-testene ga først 29 bestått / 5 feil fordi en eldre testfixture arvet
referansepolicyfelt fra dagens policy. Bare disse to fixture-feltene ble fjernet.
Alle 11 berørte tester besto deretter; samtlige fem feil er verifisert dekket.
Ingen produksjonsport eller modellkode ble endret under overleveringsoppdateringen.
Se `handover_snapshot/HANDOVER_CHECKS_20260918.json`.
