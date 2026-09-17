# Veien videre — én kandidat mot residualenes skalavekst

Start med vanlig handover og CURRENT_HANDOVER.md. Ingen aktiv modelljobb eller
kjøreplan finnes etter den fullførte signaldiagnosen. Arbeid kun i
/home/andre2/src/GX1_CURRENT på work/gx1-current; én agent/én tung jobb.

1. Gjenbruk signalresultatet og SAVED_SCALE_CAUSE-auditen. Ikke gjenta dem.
2. Lag minste kandidat som normaliserer inputen til specialist_out,
   cross_tf_out og family_tf_cooperation_out. Bevar alle features/familier/
   tidsrammer og eksisterende state_dict-nøkler. En parameterfri normalisering
   inne i en gjenbrukt lineær projeksjon kan dekke alle Entry-/Exit-kallsteder.
   Ikke endre tapsvekter,targets,clipping eller andre modeller samtidig.
3. Bevis bevart fersk initiering og frossen lærer før start. Alle tre residual-
   vekter og biaser er målt null i f1e8691d-startlæreren. Ny normalisering må
   ikke endre dens outputs eller RNG. Entry-hidden/Q inngår i Exit-tokenet;
   sammenlign derfor relevante Entry-/Exit-koblinger og frosne mål.
4. Først etter kontroll: bind én eksisterende native fixed256-plan med samme
   4096 TRAIN-rader i samme rekkefølge, fersk state, TRAIN16, frossen lærer
   og slutt-ONLINE.
   Gjenbruk korrekt baseline og opprinnelige checkpoints. Ingen full epoch.
5. Sammenlign alle TRAIN256,Exit-ankere256 og samplede Exit-states1024 med
   initialmodellen, kausal256 og konstanter. Rapporter MSE/sentrert feil,
   LONG–SHORT-kontrast,valg,referanseverdi og alle ni måneder/begge sider.
   Kandidaten består bare med tilstandsavhengig forbedring, ikke felles bias.

BASE=/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912.
Nye bevis: BASE/NATIVE_ENTRY_SIGNAL_INFERENCE_CHECK_20260917/REVIEW.json og
SAVED_SCALE_CAUSE/RESULT.json. Forrige fullførte treningsplan,recipe,initiering
og paret vurdering: BASE/NATIVE_CAUSAL_ENTRY_FIXED256_20260917.
Korrekt Entry-baseline: BASE/CAUSAL_ENTRY_TRAIN_BASELINE_20260917/RESULT.json.
Gamle planer og operatører er historikk; tilpass og bind en ny plan før bruk.

Den foreslåtte normaliseringen er en begrunnet hypotese, ikke en bevist årsak
eller kur. Om forbedring uteblir, bruk forsøket til å avgrense årsaken; ikke
start bredt søk, ny epoch eller nye features på håp. Ingen CONTROL/VAL/TEST,
handel eller spending er åpnet. Senere kronologisk kvalitet/økonomi krever
egen læringsport og inkluderer alle handler/åpne posisjoner med kostnader.

Bruk bare etablerte native vakter og source bindings. Commit/push ferdig arbeid
når kildefrys tillater det; stående autorisasjon gjelder uten ny bekreftelse.
