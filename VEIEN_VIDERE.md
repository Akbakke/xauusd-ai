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
