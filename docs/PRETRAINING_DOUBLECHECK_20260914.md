# GX1: kontroll før neste trening — 14. september 2026

**Beslutning: Ikke klar for en ny epoch. Gjør avgrenset arbeid på de kjente lærings- og oppstartsproblemene; ingen generell optimaliseringsrunde.**

Kontrollert på ren `cb67cc719c25778ae9a6bdc2313e7965c003f536` i `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`. Handover 09:56:47 UTC viste ingen native prosess, checkpoint 315 / 19 908 steg, og manglende GPU256-, totalfart- og resume-bevis. Risiko er avklart: ingen fast tapsgrense eller maksimal holdetid. `operator_stop_not_resolved` i kontrollen er navnet på `training_enabled=false`; det betyr ikke at brukeren fortsatt skylder et risikosvar.

Maskinbevis: [PRETRAINING_DOUBLECHECK_20260914.json](../handover_snapshot/PRETRAINING_DOUBLECHECK_20260914.json). Begge checkpointfiler er hashet på nytt og stemmer. Seneste relevante SHA fra de tre testbevisene stemmer for 19 kilde-/testfiler; testloggene er uendret. Tidligere runder med 42, 57 og 23 tilfeller er gjenbrukt, ikke kjørt om igjen eller summert som uavhengige tester. Ingen modellforward, bakoverpass, GPU, optimizersteg eller TEST-tilgang i denne kontrollen.

| Spørsmål | Hva koden faktisk gjør | Vurdering |
|---|---|---|
| Er nye mål tatt i bruk? | V3-økonomi og v5-checkpointvalg finnes og er testet. Ingen ny bundet kjøring er aktivert; de bevarte vektene kommer fra gammel økonomi. | Implementert er ikke innlært. |
| Har Entry fått beskjed om kvalitet fremfor kvantitet? | Entry velger største LONG/SHORT/FLAT-Q. FLAT-target er 0; handelsfasiten er det frosne Exit-nettets første verdi. Ingen ny tillitskalibrering eller avståelsesmekanisme er innført. | Selektivitet er mulig, men ikke dokumentert lært. Juni ga 0 FLAT. |
| Hindres en ny handel på hver bar? | Ny resultatmåling/checkpointscore spiller av én fast posisjon om gangen. Dette gir ikke nye treningsmål for opptatt kapital eller verdien av å vente. | Begrenser overlapp; beviser ikke bedre Entry. På lagret juni ville første LONG binde posisjonen måneden ut. |
| Lærer Entry markedet selvstendig? | Eksisterende forecast-hode lærer faktiske prisreturer ved 5/25/60/120 minutter. Handelsvalget styres fortsatt av Exit-avledet Q; Exit-gradienter går også inn i Entry-representasjonen. | Samarbeidet er reelt, men Entry er ikke dokumentert robust. Ingen kausal skade fra deling er bevist. |
| Har Exit riktig økonomisk prinsipp? | Nytt mål sammenligner videre verdi med gjennomførbar lukkeverdi, med kostnader og tidsdiskontering. Åpne tap kan ikke forsvinne fra checkpointscoren. | Regnskapsidentiteten er testet. Modellen har ikke trent på endringen. |
| Betyr målet «mest mulig penger»? | Forventet kostnadsjustert verdi, med bundet tidsdiskontering. Eksisterende særskilt hold-risikostraff er 0; ingen ny MAE-straff eller fast grense er lagt til. | Ingen egen garanti for tidlig Exit før et krasj. Det avhenger av prognosen. Modellen kan holde et tap hvis den forventer bedring. |
| Snakker alle familier og tidsrammer sammen? | Attention, lærte porter og delte representasjoner kobler dem sammen. Hele feature-settet er bevart. | Tilkobling er bekreftet; nyttig bidrag fra alle er ikke bevist. Entry-poolingen er nesten ensidig. |
| Er data, kostnader og måling bevart? | Samme bundne TRAIN/VAL-grunnlag, kausale eiere og normalisering. Bid/ask, slippage og finansiering beholdes. Åpen verdi rapporteres uten å fabrikere Exit. | Ingen ny full rekonstruksjon av historiske features. De kjente 3 136 TRAIN-kildegapene og prospektive kostnadsforutsetningene består. TEST er forseglet. |
| Er drift og gjenopptakelse klare? | Bare 256 / 8 arbeidere / 3 timer er tillatt videre. Eksisterende VAL-overføring krever fortsatt batch 128 og uendret treningskontrakt. | Den gamle overføringen kan ikke brukes som resume av nytt mål. Ny overgang og faktisk GPU-/resume-/fartsevidens mangler. |

De tidligere resultatene trenger ingen ny analyse: begge handels-Q var positive på alle 5 508 junirader. Selv FLAT satt nøyaktig til 0 endrer derfor ingen valg. Valgt Entry ga −8,3884 netto Bps etter 60 minutter, uavhengig av Exit-tiden. Forecast-korrelasjonen var svak. Sluttvektingen i Entry la 99,9977 % på D1 × SMC/likviditet, og featureporter traff øvre grense. Attention før sluttvektingen kan fortsatt formidle andre inputs; dette beviser ikke null bidrag fra dem. Et generelt «alt samarbeider godt» er likevel ikke dekket av målingene.

**Nytt funn om læringssignalet:** Native trening bruker ett neste steg i Bellman-målet og oppdaterer target-nettet etter hele epochen. Korrigeringen på SHORT-HOLD i v3 er `(1-gamma) × neste lukkeverdi`. Med bundet rho = 0,09531018 per år og ett minutt blir `1-gamma = 0,0000001812118`. Ved −100 Bps er korrigeringen −0,0000181212 Bps.

En eksplisitt regneillustrasjon med konstant lukkeverdi −100, null SHORT-kostnad og en optimistisk startverdi +1 gir fortsatt +0,99945093 etter 30 eksakte tabulære oppdateringer. Dette viser at korreksjonen alene kan fjerne slik optimisme svært langsomt. Det er **ikke** en simulering av GX1-nettet eller bevis på konvergensfarten med delte nevrale parametere, andre priser og andre treningsstater. Det er grunn til å måle læringssignalet før flere epocher regnes som løsningen.

Den konkrete sammenligningen jeg anbefaler er å la eksisterende Exit-hoder lære videreverdi utover kjent, gjennomførbar lukkeverdi. Da kan HOLD lære prisverdiendringen direkte, mens kjent kontant-/lukkeverdi forankres i økonomiberegningen. Samme totale nettoverdi må føres tilbake til Entry og sammenlignes med FLAT. Dette er et forslag til mål-/verdirepresentasjon, ikke en implementert eller målt forbedring. Flere steg i kredittildelingen er et alternativ dersom målingen viser at dette er flaskehalsen. Ingen tilfeldig tapsgrense, holdegrense eller juni-tilpasset terskel trengs for disse sammenligningene.

Prioritert videre arbeid:

1. Avgrenset måling av Entrys selvstendige prognose, metning og relevante oppgavegradienter, sammen med Exit-targetenes læringssignal. Rett påvist mål-/skalaproblem i eksisterende eiere. Ingen nye modeller eller bred regeljakt.
2. Bind riktig overgang fra bevarte vekter til det endelige økonomimålet. Gammel optimizer-/targethistorikk må ikke ommerkes som eksakt resume. Samordne Entry-selektivitet, én-posisjonsmåling og hva FLAT faktisk representerer.
3. Dokumenter GPU256-paritet, samlet fart og korrekt gjenopptakelse gjennom den tillatte native profilen. Handover leser NEXT_RUN_POLICY; selve launcheren bruker egne kampanje-/recipebindinger. De nye risikobevisene og kapasitetskravene er ikke direkte lest fra NEXT_RUN_POLICY ved native start, så håndhevingen må kontrolleres når neste kjøring bindes.
4. Deretter full femårstrening med VAL etter hver epoch, høyst 30 og patience 5. Mål Entry-kvalitet separat fra økonomisk totalresultat. Positiv lukket-vinnerstatistikk er fortsatt ikke bevis på samlet lønnsomhet.

Vi trenger altså konkret forbedring og måling av noen få kjente forhold. Det finnes ikke grunnlag for å si at alle regler og læringsvalg allerede er optimale, eller at modellen nå har forstått selektivitet og Exit slik brukeren ønsker.
