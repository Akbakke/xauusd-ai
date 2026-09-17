# Veien videre — ny startmåling før læringsprøve

## Nøyaktig stoppunkt for neste agent

Modellrettelsen og bevaringen av lærerfunksjonen er implementert og kontrollert.
Det er ikke gjort noen native forward med den nye ONLINE-funksjonen. Ingen jobb
eller ny plan er aktiv. Fortsett ved kontrakten nedenfor; ikke start diagnosen på nytt.

1. Rett bare den observerte måleblokkeringen i
   `gx1/contracts/unified_exit_native_candidate_campaign_v1.py`:
   `require_native_run_scope` avviser nå TRAIN-only ved initialmåling;
   `require_chronological_initial_measurement` mangler denne policybindingen;
   `require_chronological_learning_measurement` krever initialobservasjoner
   for både TRAIN og CONTROL og kontrollerer ikke ny modellkilde som baseline.
   Tillat eksplisitt bundet TRAIN-only ved nullsteg, og avvis initialbaseline
   fra annen ONLINE-modellkilde. Bevar nullsteg/256-grensene, eksakt policybinding
   og øvrige vakter. CONTROL er fortsatt stengt. Kontroller bare disse endringene
   i eksisterende initial-/learning-measurement-tester. Endringen er IKKE lagt inn.
2. Lag en ny avledet initialiserings-RESULT under en ny artefaktmappe. Gjenbruk
   original `INITIAL_STATE.pt` uendret, med faktisk constructor-audit som bevis,
   gjeldende kildehash, de to modellfunksjonene og referanse til original RESULT.
   Ikke overskriv originalen eller bare bytt hash uten proveniens.
3. Bind deretter én ny native nullstegs TRAIN-only startmåling via eksisterende
   campaign, `gx1_capped_run.sh` og vakter. Først ferdig kode/commit og eksakt
   NEXT_RUN_POLICY/recipe/plan-binding; ingen oppstart på dagens policy.
   Samme TRAIN 256 Entry/256 Exit-ankre/1024 samplede states, original lærer,
   vekter/RNG/optimizer/EMA og eksisterende målekoordinater. Ingen CONTROL/VAL/TEST.
4. Verifiser nye ONLINE-startprediksjoner, uendrede lærer-/targetverdier og
   uendret optimizer/EMA/RNG. Entry-targets sammenlignes med korrekt kausal
   avledet baseline; gamle opprinnelige Entry-targets hadde etterpåklok klipping.
   Bruk originale Exit-targets. Lik vekthash betyr ikke lik ONLINE-funksjon.
5. Først deretter vurder og bind separat én fast 256-stegs læringsprøve.
   Krev bedre tilstandsavhengig Entry OG Exit mot frosne sammenlignbare baselines,
   begge sider og måneder. Lavere bias, større representasjonsvariasjon eller
   all-FLAT/all-HOLD er ikke PASS. Ingen videre trening er nå åpnet.

## Gjenbruk og avgrensning

`BASE=/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912`.
Detaljer, filbindinger og teststatus står i
[MAIN_ENCODER_NORMALIZATION_HANDOVER_20260918.md](docs/MAIN_ENCODER_NORMALIZATION_HANDOVER_20260918.md).
Gjenbruk representasjonsdiagnosen, constructor-audit og beståtte tester.
Ingen ny normalisering-/loss-/terskel-/modellrunde. Ingen gammel kampanje kan
relanseres; `RESUMABLE` og `complete=false` er historiske checkpointfelt.
TRAIN-fit, senere kronologisk kvalitet og samlet økonomi er separate porter.
