# Samlet gjennomgang av aktiv femårstrening — 12. september 2026

Gjennomgått av hovedagenten og tre uavhengige underagenter etter brukerens uttrykkelige bestilling. Kilde: `/home/andre2/src/GX1_FULL_TRAIN_NATIVE_DISPATCH_V33`, ren og fryst commit `2959cd09f34b20fe82f351f97f7b051b49918999`. Oppskrift: `NATIVE_FULL_TRAIN_RECIPE_DISPATCH_V4.json`. Runtime: `/home/andre2/GX1_RUNS/UNIFIED_EXIT_FULL_TRAIN_NATIVE_2959CD09_BOOT388`.

**Konklusjon:** Vi har ikke påvist en feil som gjør de pågående modelloppdateringene ugyldige eller begrunner å forkaste denne kjøringen. Vi fant tre konkrete, begrensede svakheter og flere vesentlige forbehold for senere resultatgodkjenning. Ingen fryst kilde, data eller treningsstate ble endret under gjennomgangen. Ingen ekstra tung jobb eller modelltest ble startet.

Dette er en gjennomgang av faktisk kode, bundne metadata, eksisterende målinger og levende trening. Hele femårsdataproduksjonen er ikke kjørt på nytt. Første ordinære pause/gjenopptakelse ble senere verifisert17:45UTC som beskrevet nedenfor. Første kandidat-VAL i den nye hovedkampanjen har ennå ikke funnet sted. Fravær av alle mulige feil er ikke bevist.

## Konkrete funn og håndtering

| Funn | Faktisk konsekvens | Håndtering |
|---|---|---|
| Første minutt med LONG-finansiering mangler | Fill er ved M1-open, første Exit-beslutning60sekunder senere. Nåværende cash-akkumulering starter ved første beslutning. LONG får omtrent **0,001026694 Bps for optimistisk nettoresultat per handel**; SHORT påvirkes ikke med gjeldende0-finansiering. | Tas med ved endelig netto-Bps-vurdering. Senere rettelse må belaste første intervallet én gang og samordne TRAIN Entry-target med VAL. Størrelsen begrunner ikke å forkaste pågående vekter eller skrive om eksisterende resultater. |
| Epochsluttkontroll bruker bare siste gjenopptatte del | HOLD/EXIT-tellerne nullstilles ved hver invokasjon, men sluttsjekken krever begge i gjenværende del. En gyldig full epoch kan derfor få en falsk avvisning ved en ensidig siste del. | Kontrollér ordinær epochslutt. Siste optimizercheckpoint lagres før denne sjekken. Ved faktisk feil skal fullført trening bevares og kontrollen rettes til riktig omfang; ikke tren året om igjen. |
| Entropiberegning gir en liten positiv verdi også for nøyaktig én aktiv rute | Nullsannsynligheter erstattes med1e-12 før p·log(p). Nøyaktig én aktiv av8ruter gir ca.1,934e-10 i stedet for0. Positiv gjennomsnittsvekt og entropi er dessuten ikke et mål på praktisk nyttig samarbeid. | En slik PASS skal ikke tolkes som bevis på nyttig bruk av alle familier. Korrekt formel beholder0-koeffisientene og begrenser bare logaritmens argument. Kontroller målt bruk og input-influence på valgt kandidat før godkjenning. Ingen påvist korrupsjon av dagens trening. |

Finansieringsfunnet forklarer ikke seedens negative all-SHORT-resultat. Det første minuttets risikokost er0 med gjeldende parametere. Entry-targets uttrykkelige første-tilstands-konvensjon utelater også et kapitalhurdle-ledd på omtrent1,81e-7 relativt; dette er et deklarert tidsvalg med svært liten virkning, ikke en uoppdaget stor kostnad.

## Vesentlige kvalitets- og databegrensninger

- **Kandidaten kan fortsatt bli ubrukelig selv om beregningen er korrekt.** Seedens Entry-ruting var nesten helt konsentrert om én rute, og0,8083% av observerte Entry-feature-gateverdier nådde øvre grense2. Hvis dette vedvarer, avvises kandidaten av gjeldende helsekrav. De delte gateparametrene får også ikke-mettede Exit-gradienter, så det er ikke påvist at de er permanent fryst.
- **Juni-sluttsensurering kan hindre checkpointvalg.** Én faktisk valgt handel som fortsatt er åpen ved månedsslutt gir utilgjengelig full-policy Bps. Slike epoker bruker patience, og fem kan avslutte beregningen uten valgt checkpoint. Dette er eksisterende avtalt oppførsel. Ingen kunstig månedsslutt-EXIT eller erstatning med bare lukkede handler skal innføres for å skape grønt resultat.
- TRAIN har **3136 ukjente kildegap**, hvorav1985 i2021. Disse sensurerer forløp. Hele Entry-utvalget er med, men alle historiske utfall er ikke sammenhengende observerte. Juni-VAL har0ukjente gap.
- Kostnadene er et eksplisitt prospektivt scenario, og deler av markedskalenderen er utledet. Resultatene blir derfor ikke automatisk dokumenterte historiske brokerresultater.
- Juni er brukt i tidligere utvikling/validering. Den er ikke en urørt slutt-test. TEST er fortsatt forseglet.
- Faktisk modell bruker bevart **v7 base-normalisering**, ikke den nye v8-statistikken som også finnes i child-pakken. Begge er tilpasset femårs-TRAIN uten VAL/TEST-fit. Ny full-TRAIN lifetime-summary-normalisering brukes. Dette er eksplisitt seedbevaring, ikke skjult lekkasje.
- Fullpopulation betyr alle313399Entry-par én gang per epoch, med **fire samplede Exit-overganger per Entry** på begge sider og første-tilstandsanker. Det betyr ikke at alle mulige Exit-tilstander enumereres under TRAIN. Full juni-VAL ruller derimot de faktiske policyforløpene.

## Hva som er undersøkt

| Del av kjeden | Evidens og resultat |
|---|---|
| Datadekning og split | Faktisk konstruktør og indeks krever313399TRAIN,5508juni-VAL, komplette parent/child-koordinater og kunTRAIN/VAL. Context-raden førjuni inngår ikke som VAL-entry. |
| Klokke og kausalitet | M5[t,t+5min) er kjent før fill ved M1-open(t+5min). Første Exit-beslutning er t+6min. Closed-bar MTF-cutoff og pivotbekreftelse er undersøkt; ingen påvist framtidslekkasje. |
| Featurebredde og familier |238lokale signaler,71kontinuerlige og1kategorisk kontekstfelt;176MTF-felter fordelt eksakt over8familier. Alle kontraktsindekser har eiere. Entry32ogExit40familie/tidsramme-tokens, medM5lokalt påEntry. |
| Seed, normalisering og lærbarhet | Ferdig ettårs-EMA er bundet til målt seed; parametere er lærbare. Fitted-Q-target, online-nett og EMA har adskilt oppdatering og riktig gradientflyt. |
| Tap, targets og sampling | Faktisk lifecycle-v2-kjerne, Bellman-targets, actionmasker, fire Exit-samples, Entry-targetbro og felles Entry-gradienter er gjennomgått. Ingen påvist dobbel loss-gradient, manglende hovedtap eller skjult Entry-cap. |
| Optimizer og determinisme | AdamW, gradientklipping, targetoppdatering, EMA, scheduler, RNG og eksakt epochrekkefølge lagres. Source/metadata støtter korrekt resume; ny hovedkampanjes første faktiske resume er fortsatt utestet. |
| Varig checkpoint og kampanje | Inaktivt checkpoint-slot skrives/fsync før atomisk pointer. Windows/Linux binder kilde, oppskrift, startmarkør, boot og forrige kvittering. Historiske kvitteringer bruker uforanderlige snapshots slik at senere mutablepointer ikke ugyldiggjør historikken. |
| Validering og Bps | Samme bundne EMA brukes for Entry-valg og Exit-forløp. Netto kontant-Bps beregnes over alle5508muligheter, medFLAT=0. Bare faktisk valgt side styrer policy-score; ingen dobbel HOLD-mark-to-market. |
| Sensurering og ressursvinduer | Naturlige grenser skilles fra beregningsavkorting.70minVAL-vinduer kan gjenopptas. Ingen tvungen512-terminal eller skjult redusert juni-cohort funnet. |
| Helse, selection og completion | Helse kommer fra faktiske observasjoner, ikke hardkodetPASS. COMPLETE betyr avsluttet beregning og kan ha selected_checkpoint=None; bundle_written=False. Det betyr ikke positivBps, ferdig mål eller livegodkjenning. |
| Driftsramme | Én tung jobb.300Wfysiskgrense,85°Ckjerne/80°Cminne, automatisk200W ved80°Ckjerne. Kilde er ren. LevendePID725 og lagret checkpointfremgang er bekreftet under gjennomgangen. |

En mulig konflikt om epoch_index30 ble undersøkt og avkreftet for den normale kjørebanen: normal fullføring lagrer siste epoch med indeks29 før inkrementering. Ingen forebyggende rettelse er nødvendig der.

Kampanje-/driftskildene hovedagenten undersøkte inkluderer `run_unified_exit_native_candidate_window_v1.py:30–159`, `run_unified_exit_random_access_full_train_v1.py:68–163,172–252,445–560`, `local_random_access_campaign_v2.py:1090–1145,1219–1299`, `GX1-RandomAccessCampaignV2Controller.ps1:530–565,626–670` og `GX1-RandomAccessCampaignV2Progress.ps1:47–99`.

## Detaljbevis

- [Data, tidsjustering, normalisering og inputdekning](DATA_CAUSALITY.md)
- [Treningskjerne, gradienter, sampling og resume](TRAINING_KERNEL.md)
- [VAL, økonomi, gatehelse og checkpointvalg](VAL_SELECTION.md)

## Neste handling

La fryst hovedtrening fortsette. Følg omtrent15min-kontroller og kontroller første ordinære pause/gjenopptakelse uten ekstra testkjøring. Bruk første planlagte komplette kandidat-VAL til å måle nettoBps, faktisk gatehelse og om noen checkpoint kan velges. Regnskapssvakheten og de andre funnene skal følge resultatvurderingen; de må ikke skjules av et genereltPASS eller et tekniskCOMPLETE.

## Senere faktisk resume-bevis —17:45UTC

Første invokasjon avsluttet medRESUMABLE,guardPASS og begge prosesserexit0 etter2624lagrede optimizersteg. Fysisk Boot390 startet invokasjon2 med eksakt foregående cursor-SHA73904dd9… og TRAIN-pointer-SHA48a13061…. Treningsloggen bekrefter resumed=1,batch_offset2624 og batch2625 fullført; ny pointer45viser2816steg. Dermed er første ordinære hoved-resume observert uten ekstra treningstest. Dette er korrekt gjenopptakelses- og fremgangsbevis; ingen separat numerisk sammenligning med et uavbrutt GPU-løp er utført. Detaljer: ../NATIVE_FIRST_RESUME_PASS.json.
