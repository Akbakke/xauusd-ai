# GX1 — overlevering 17. september 2026

## Nåstatus og autoritet

**Signaldiagnosen er fullført og det brukte omfanget stengt. Ingen modelljobb
kjører. Læringsporten er fortsatt ikke bestått. Neste er én kontrollert kandidat
mot den målte svekkelsen av tilstandsvariasjon; kandidaten er ikke implementert
eller bundet ennå.**

Kode: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Data: `/home/andre2/GX1_DATA`. Mac er overleveringskopi.
Start med `./handover.sh --check` på Mac eller
`bash scripts/gx1_handover.sh --check` i Linux. `current_work` er nåstatus;
øvrige gamle checkpoint-/VAL-felt er historikk. Ingen handover starter trening.
NEXT_RUN_POLICY.json har training_enabled=false og ingen aktiv kjøreplan.

## Siste bevis

Native `NATIVE_ENTRY_SIGNAL_INFERENCE_CHECK_20260917` på kilde812e4e66 kjørte
19:05:27–19:17:59 UTC / 21:05:27–21:17:59 Oslo. Fire forwards, samme cachede
TRAIN16, frosne initial-/sluttmodeller og korrigerte mål; null optimizersteg.
Guard PASS, trainer/observer0. Topp47 °C kjerne,50 °C minne,130,17 W,1932 MiB.
Faktisk boot460; runtime-navnet BOOT459 viser forberedelsesboot. Task Disabled,
siste resultat0, ingen native prosess. Originale checkpoints er rehashet/bevart.

I denne TRAIN16-prøven kommer98,18 % av MSE-bedringen fra fellesnivået og bare
0,36 % fra LONG–SHORT-komponenten. Retningskomponenten utgjør99,97 % av slutt-
feilen. Variasjonen i Entry-hidden faller til9,93 % av startnivået. M5/fused
middelradnorm vokser134,6×/155,9× og blir nesten lik mellom radene.

Inputene varierer; normaliseringsbufferne er uendret. Numeriske asinh-inputs
har maksimum5,69 i de målte flatene. De tre residualprojeksjonene konsentrerer
henholdsvis99,03 %,92,35 % og92,48 % av vektenergien i én retning. Dette
lokaliserer et reelt problem med skalavekst og svekket tilstandsvariasjon;
det beviser ikke at en bestemt normaliseringsendring vil løse læringen.

Hjelpetapene reduserer den nyttige retningsprojeksjonen på Entry-private
rutingparametre til47,74 %, men snur ikke nettoretningen i målt eval-gradient.
Dette er ikke en rekonstruksjon av full klippet Adam-oppdatering eller Exit.

Vanlig inferens gjenskaper lagrede verdier eksakt. Det lille numeriske avviket
i gradientmodus er nå korrekt rapportert separat. Fem målrettede tester og
Git-kontroller består. Ikke gjenta fullført parity-/signaldiagnose.

## Neste handling

Følg VEIEN_VIDERE.md. Én kandidat: normaliser inputen til specialist_out,
cross_tf_out og family_tf_cooperation_out. Bevar alle features, familier og
tidsrammer, eksisterende parameternøkler og samme initiering. Dette er en
konkret hypotese mot den observerte skalaveksten, ikke en bevist kur.

Den frosne startlærerens tre vekter OG biaser er målt eksakt null. Bruk dette
til å kontrollere at kandidatens ferske modell og frosne lærer gir uendrede
outputs før en separat native256-plan bindes. Entry-output inngår i Exit-
tokenet: ikke anta at andre Entry-endringer lar Exit-fasiten være uendret.

Sammenlign deretter eventuelt samme ferske initiering,4096 TRAIN-rader i samme
rekkefølge,TRAIN16/256 steg og frosne TRAIN256,256 Exit-ankere og1024 samplede
Exit-states. Bruk eksisterende korrigert baseline og kausal256-kandidat.
Alle ni måneder og begge sider skal med. Bare biasbedring eller all-FLAT/
sidekonstant Exit er ikke læring. Ingen større trening uten læringsbevis.

## Bevarte resultater og grenser

BASE er `/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912`.
Siste signalresultat: BASE/NATIVE_ENTRY_SIGNAL_INFERENCE_CHECK_20260917/
entry_gradient_diagnostic/RESULT.json. Samme mappe har REVIEW.json,
SAVED_SCALE_CAUSE/RESULT.json, recipe/plan og operatører. De er fullførte.

Den siste trente kandidaten ligger i BASE/NATIVE_CAUSAL_ENTRY_FIXED256_20260917.
PAIRED_TRAIN_REVIEW.json og VERDICT.json avviser læringsporten: Entry FLAT256/256,
MSE dårligere enn TRAIN-konstanter, Exit alltid HOLD for LONG og EXIT for SHORT;
alle fire Exit-MSE er dårligere enn forrige kandidat. Checkpoint5,256 steg,
offset256,epoch0 er bevart. complete=false/RESUMABLE gir ikke resume-tillatelse.

To tidligere feil er rettet: Entry-Q-detach og etterpåklok klipping av Entry-
fasiten. Korrekt mål bruker første likvidasjonsverdi+(119/120)*Q_HOLD; negativ
fortsettelsesverdi beholdes. Bevar Exit-mål,kostnader,bootstrap og kausalitet.
Den fullførte korrekte baselinen og tidligere tester skal gjenbrukes.

Én agent/én tung jobb, bare eksisterende native campaign og vakter. Ingen full
epoch/full VAL, CONTROL/TEST, live/paper eller spending. TEST er forseglet;
mars–mai og juni er allerede utviklingsdata. Ingen fast taps-/holdetidsgrense.
Generalisering/profitt er ikke bevist. Stående offentlig push-autorisasjon
gjelder kode,dokumentasjon og aggregater; ikke rådata,vekter eller hemmeligheter.
