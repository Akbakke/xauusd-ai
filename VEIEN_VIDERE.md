# Veien videre — lokaliser gjenværende variasjonstap

Siste native256-prøve og paret analyse er fullført og avvist. Brukt scope er
stengt; ingen modelljobb eller ny kjøreplan er aktiv. Følg CURRENT_HANDOVER.md.

1. Gjenbruk den ferdige vurderingen og RESIDUAL_BOUND_AUDIT.json. Ikke gjenta
   trening, baselinetester eller tidligere parity-/signaldiagnoser uendret.
2. Les bevart TRAIN16-cache og ny native TRAIN_OBSERVATION/checkpoint på CPU.
   Kontroller eksakt rad-/input-/target-/modellbinding før en ny måling.
   Tidligere cachede sluttprediksjoner tilhører kausal256, ikke ny kandidat.
3. Avklar hva eksisterende native diagnose faktisk kan måle med disse
   bindingene. Velg ett avgrenset representasjonsforsøk på samme inputs for
   å lokalisere eventuell svekkelse mellom rå local/MTF/context og Entry-hidden.
   Bind en ny eksplisitt plan før forwards. Ingen ny optimizer eller fit.
4. Endre modellkode først ved en konkret ny måling som begrunner rettelsen.
   Ikke stable flere normaliseringer, endre tapsvekter eller fjerne features
   på håp. Sammenlign videre mot frosne baselines med samme korrekte targets.

Gjeldende modellkode inneholder den avviste kandidaten for reproduserbar
diagnose. Gamle ikke-null-vekter gjennom denne koden er ikke gamle modell-
outputs. Gjenbruk opprinnelige prediksjoner og bevar kildesamsvar.

Artefakter under BASE/NATIVE_RESIDUAL_NORMALIZED_FIXED256_20260917.
BASE=/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912.
Ingen ny kjøring/full epoch/VAL/CONTROL/TEST, handel eller spending er åpnet.
TRAIN-fit, senere kronologisk kvalitet og samlet økonomi vurderes separat.


## Nøyaktig stoppunkt for neste agent

Handover er kontrollert mot ren, pushet kilde1634c002. Nyere dokumentasjons-
commit endrer ikke treningskilden88310075. Prosesslisten var tom og Windows-
task Disabled. Kjør handover på nytt ved overtakelse for fersk driftsstatus.
Kun filenes eksistens og SHA er kontrollert i denne overleveringen: cache-
skjema, radkobling og ny måleplan er fortsatt ugjort. Ingen forwards er kjørt.

Bruk BASE ovenfor og disse bevarte filene:

- Korrigert TRAIN16-cache:
  `NATIVE_ENTRY_SIGNAL_INFERENCE_CHECK_20260917/entry_gradient_diagnostic/TRAIN16_INPUTS_AND_TARGETS.pt`
  SHA256 `99079cf43cd423686edc1197d3070a6f6e26a4e5e37f81ce8e70ea06731dc02d`.
  Lagrede sluttprediksjoner her tilhører kausal256, ikke residualkandidaten.
- Ny sluttmåling:
  `NATIVE_RESIDUAL_NORMALIZED_FIXED256_20260917/.gx1-candidate-training-session.CANDIDATE_BUNDLE/final_online_measurement/TRAIN_OBSERVATION.json`
  SHA256 `5015339d2cf4cc16802b22a1daedc1741e8aa6cac36789365664c31fd62a7abf`.
- Ny sluttstate: samme CANDIDATE_BUNDLE, `candidate_training_state_slot_0.pt`.
  SHA256 `d00e1107282ffb1c1d1b5878403a7f0cdc85e8517042c73266bcd8049928feaa`.
  Modell-SHA `c940a3b855e95067e16882558daffeb25f896e22b6ab23d425afc80db40f4db1`.
  Den kontrollerte completion-filen binder også lærer, pointer og receipt.

Første arbeid er en lesende CPU-kontroll via eksisterende
`bash scripts/gx1_capped_run.sh --class audit -- .venv/bin/python ...`.
Sammenhold cache-skjema og identiteter med de bevarte TRAIN-observasjonene;
ikke anta at de samme radene inngår eller at en gammel prediksjon gjelder ny modell.
Ingen modellforward inngår i denne første kontrollen.

Les deretter `require_entry_gradient_diagnostic` i
`gx1/contracts/unified_exit_native_candidate_campaign_v1.py`, samt
`_run_entry_gradient_diagnostic`, `_entry_signal_pair` og `_entry_signal_losses`
i `gx1/scripts/run_unified_exit_random_access_full_train_v1.py`.
Tidligere parity-bevis gjelder kausal256. Det er ikke automatisk gyldig for
ny sluttstate. Avklar dette før eventuell planbinding; ikke gjenta to GPU-
diagnoser bare fordi tidligere operatører gjorde det. Minste neste leveranse
er dokumentert rad-/input-/target-/modellbinding og avgjørelse om eksisterende
native måler kan brukes uendret. Først deretter kan en avgrenset plan åpnes.
