# Familiesammenligning: inputpilot30.10.2026

Pilot001 er fullført og konsumert. Dette er input-/mekanikkbevis, ikke indikatorverdi eller retningsedge.

- 128 forhåndsvalgte TRAIN- og128Q1-foreldre; alle14 råflater.
- 20 intervensjonsvitner: ti varianter på første16foreldre i hver del. Baseline endret ingen verdier.
- Kun tre akse-arrays dekodet; null målarrays, modeller, optimizersteg og TEST.
- Original terminal og outer hadde exit0; kilde- og inputbookends var uendret.

## Målte begrensninger

Dagens kompakte profil parkerer71/253XAU-MTF-kolonner per tidsramme og16/2542kontekstkolonner. Dette følger det uttrykkelige kompaktprofilunntaket i GX1_RULES; biblioteket er bevart. Parkerte indikatorer kan ikke bedømmes fra denne modellens resultater.

I pilotens aktive GC-D1-verdiceller var22,849% observert i TRAIN og60,244% senere. Tallene teller gjentatt historikkeksponering; de er ikke uavhengige observasjoner. I nåkonteksten manglet10aktive verdifelt helt i TRAIN-utvalget og27senere;72/69observerte felt var konstante. Årsaken til manglende verdier er ikke fastslått.

Prislysarmen endret nullverdier i første TRAIN-vitne, men hadde støtte i det bredere TRAIN-utvalget og endringer i senerevitnet. Dette beviser derfor ikke frakobling. Full støtteaudit gjenstår.

## Kostnad og minste rettelse

Hele piloten tok173,59s. Providerforberedelse tok74,63s; collector58,58s, fordelt på gather27,16s og summering31,03s. De20intervensjonskontrollene tok30,86s.

Perkolonne binær-/maske-/aliaskontroller ble samlet uten å endre assignment, modell, profiler, begge valideringer, kontrakthash eller beskyttede bytekontroller.37fokuserte tester bestod. På identisk syntetisk16-raders input falt gjennomsnittet fra0,685725s til0,236135s; alle420flateoutputs (10armer×14flater×3repetisjoner) var byteidentiske med førversjonen. Dette er syntetisk timing; genuin ettermåling følger fullauditen.

Prosessens peakRSS var4567416832bytes, mens cgroupgrensen var4294967296bytes. Disse målene har ulik regnskapsavgrensning; pilotkvitteringene forklarer ikke avviket. Fullauditen registrerer cgrouppeak/events i tillegg. Ingen grense er økt.

## Originale bevis

Studierot: /home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19

- Pilot: INPUT_PILOT_001/RESULT.json, TERMINAL.json og OUTER.json.
- Fokusert kontroll: SOURCE_QUALIFICATION_003/RESULT.json og JUNIT.xml.
- Timing/paritet: INTERVENTION_SYNTHETIC_COST_BEFORE.json og INTERVENTION_SYNTHETIC_COST_AFTER.json.

NEXT_RUN_POLICY.json binder eksakte filhasher og neste engangsplan. All ny modell-/treningskjøring er fortsatt stengt.
