# Veien videre

## Nå: ny trening med eksisterende v38-indikatorer

Brukerens prioritering 07.10.2026 setter GC/order-flow og nye order-block-/
footprint-utvidelser på pause. Arbeidet og kildebevisene bevares for senere.
Dagens v38-indikatorer, SMC-primitiver, åtte modellspesialister, tidsrammer,
TRAIN/development-VAL-perioder endres ikke. Den nyere bestillingen krever
ferskt datasett og ny whole-TRAIN-normalisering, deretter trygg retention og
liten smoke før større trening. Ingen relabeling av gammel v38-generation.

Ny kjøreplan er configs/research/NATIVE_V38_REBUILD_AND_SMOKE_20261007.json:
repo-/mismatch-kontroll → fersk capped native build → obligatorisk komplett
M1-state-klokke, fysiske visninger, normalisering og genuine audits →
exact-target retention → liten teknisk/avgrenset læringssmoke → kun betinget
større trening. Se docs/NATIVE_LEARNING.md for detaljene og NEXT_RUN_POLICY
for faktisk fremdrift. Før sletting må nye outputs være akseptert og alle
transitive rå-/kalibrerings-/kostforeldre og aktive stier være beskyttet.
Ukjent closure/TEST-nektelse stopper sletting, aldri håndlaget unntak.

Bind planen til ren/committet/pushet kilde før start. Den avbrutte
ATTEMPT_003 er konsumert; ny samplerbenchmark trenger egen finite plan og
budsjettautoritet, ikke gammel engangsgodkjenning. Ingen modelltrening er startet.
Følg docs/NATIVE_LEARNING.md: full samplerbenchmark → valgt sampler/fryste
koordinater → fersk initialmåling → avgrenset læringsprøve → separat
TRAIN/CONTROL256-review → kun betinget, endelig utvidelse. Full epoch/full VAL
er ikke åpnet. Manglende GC-data blokkerer ikke det eksisterende v38-oppsettet.

## Senere: GC-sporet ved uttrykkelig gjenopptakelse

Alle fire ufullførte GC-trinn bevares i GC_ORDER_FLOW_RESEARCH_001.
docs/GC_ORDER_FLOW_RESEARCH.md og den kildebundne configen beskriver opplegget.
Kildeaudit er implementert og mekanikktestet. Ingen fil er bundet til GC-strategitesten;
ingen A/B/C-effekt, v38-læring eller lønnsomhet er målt av denne bølgen.

Den særskilt bestilte gratisundersøkelsen er dokumentert i GC-protokollen og
policyens free_source_investigation. AlgoSeek US6011 er den mest lovende større
gratis kandidaten fra Q1-2023/full-univers-katalogen, ikke en verifisert GC-fil.
Ved gjenopptakelse er den eksterne avhengigheten brukbar Sandbox-tilgang/evidens som
viser GC-utløpskontrakter/datoer, faktisk gjenværende gratisgrense og demoens
rett til lokal offline trening/backtest. Bind deretter ett lite eksakt GC-uttak.
Ingen konto, abonnement, lisensaksept eller gebyr autoriseres av undersøkelsen.
Den faktisk inspiserte Databento MBP-1-filen har kun ESZ5; Portaras gullprøve
har bare 5m39s og mangler aggressor. Ingen av dem dekker strategitesten.
Forsknings-/TEST-periodene skal aldri flyttes eller krympes for gratisutvalget.

1. Bind en ekte pre-TEST outright-GC-prøve, aggressor-side og event-BBO,
   tidssemantikk, kvittering/mapping, lisens og prisestimat før bruk.
   TBBO gir handelsdelta, men ikke bokoppdateringene mellom handler; OFI
   trenger MBP-1 eller en tilsvarende kvalifisert hendelsesflate.
   Kjør capped lokal kildeaudit og kvalifiser faktisk dekning. Struktur-PASS
   er ikke modelladgang; manglende kvalitet/evidens stopper.
2. Bygg kausale delta-/OFI-/GC-pris-/basis-features. Bind tilgjengelighet,
   kontraktsrulling, ordrebokprefiks, warmup og GC/OANDA-synkronisering.
   Test prefix-/framtidsmutasjon og mål paritet på genuine kvalifiserte rader;
   manglende eller ukjent flow blir ikke null-flow.
3. Bind eksakt snapshot-/ridge-recipe, kronologiske folder, targets,
   usikkerhet og kostnader før empirisk fit. Samme rader i A=254 felt,
   B=A+GC-pris/basis, C=B+delta/OFI. Primært C−B, separat B−A.
4. Test senere LOCATION × FLOW × STATE med eksisterende eiere, deretter
   absorption/failed-breakout/profilinteraksjoner. Egen niende familie og
   kontrollerte indikatorablasjoner krever målt tilleggseffekt/egen beslutning.

Hvert trinn har leveranser og ferdigkriterier i configs/research/GC_ORDER_FLOW_RESEARCH_001.json.
Fremdrift står bare i NEXT_RUN_POLICY/current_work.gc_goal_progress.
Første konkrete avhengighet er genuine lisensavklarte GC-files eller brukbar
leverandørtilgang: avgrenset DATA/RUNS-inventar fant ingen kandidat og ingen
DATABENTO_API_KEY ble funnet i de kontrollerte miljø-/konfigurasjonsflatene.
Ikke relanser tom kildeplan som gjentatt framdrift. Ingen konto/lisens/kjøp
opprettes av målet alene. Nye resultater publiseres immutabelt og gjenbrukes.
Ikke gjenoppta på eget initiativ mens brukerens pause gjelder. Ved en senere
bestilling kreves genuine filer eller faktisk leverandørtilgang som er
kontrollert. Delvis kode og
mekanikk-PASS kan ikke erstatte manglende markedsdata eller empiriske resultater.

Repo-oppryddingen etter operatørens benchmarkstopp er fullført og verifisert;
eksakte bevis står i docs/REPO_REVIEW.md. Genuine familier og vakter bevares.

Full makro-B (DFII10, DTWEXBGS, T10YIE, GLD, COT, VIX) består som separat mål.
Historiske dataversjoner/tilgjengelighetsklokke er uavklart for deler av kjeden.
MACRO_CORE erstatter ikke B og har ingen GO/promotion. Ingen nye hentinger eller
forskningsfits er åpnet av oppryddingen eller den tomme GC-kildeplanen.

TEST, broker, live/paper, handel og spending forblir stengt.
