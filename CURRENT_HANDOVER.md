# GX1 overtakelse — 2026-09-14

## Native kalibreringsvei forberedt — ingen GPU startet

Ny eksplisitt økonomiovergang godtar bare den bevarte v2-origin checkpoint315 /
19 908 optimizersteg, og skriver til en ny privat v4-session. Modell, target,
Adam, EMA, scheduler og RNG beholdes. Ny full femårsdekning begynner ved
epoch0/offset0/global0; gammel steghistorikk og interne tellere bevares i et
uforanderlig overgangsbevis. Gammelt best/top-k/patience/VAL nullstilles for
nytt mål. Dette er kontrollert initialisering, ikke v2-v4 resultatlikhet.

79 unike målrettede tilfeller består i denne runden, inkludert reell restore-
kode med syntetisk state og beholdte interne tellere19 908. En eksisterende
session-testfixture trengte dagens checkpoint-policy (fem linjer); ingen
produksjonsomgåelse. Bevis:
[NATIVE_CALIBRATION_CONTRACT_VERIFICATION_20260914.json](handover_snapshot/NATIVE_CALIBRATION_CONTRACT_VERIFICATION_20260914.json).

Første v4-batch i en begrenset TRAIN-invocation kan nå måle faktiske vektede
forecast-/Entry-Q-/Exit-gradienter på de samme Entry-ruterparametrene, norm og
cosinus, samt Entry-target/prediksjonsfordeling/bias og HOLD-targets. Målingen
bruker eksisterende grafer, bevarer .grad/RNG og følger logg/pause-statistikk.
Den er en avgrenset læringskontroll, ikke bevis for fremtidig lønnsomhet.

NEXT_RUN_POLICY er nå håndhevet i native materializer/window/fulltrain. Når
stor trening er av, tillates bare deklarert kalibrering med kumulative16/32
optimizersteg. Dette er operative målegrenser, ingen handelsgrenser. TRAIN16,
VAL256/8 arbeidere/10 800 sekunder og alle maskinvarevakter er bevart.
Manglende origin kan ikke falle tilbake til gammelt smoke-seed. Oppstart av
full trening krever også bundne overgangs-/kalibreringsbevis; feltene er NULL.

Neste handling er å binde faktisk v4-readiness/recipe og gjøre overgangen og
den begrensede TRAIN-målingen i godkjent native kampanje etter fersk fysisk
boot. Ingen recipe, kampanje, checkpointovergang eller GPU-kjøring er ennå
aktivert for dette. Handover BLOCK gjelder fortsatt full trening.

Etter at TRAIN-målingen er tolket, gjenstår eksplisitt rapporterende native
VAL-kapasitetsfase:16/32 TRAIN når ingen hel epoch/VAL og kan ikke produsere
GPU256-/totalfartbevis alene. Deretter må uavbrutt v4 sammenlignes med gjenopptatt
v4 fra samme tilstand. Endring av den bundne NEXT_RUN_POLICY endrer recipe-
identitet; kalibreringssessionen bevares som kontrollforsøk. Full v4-trening
kan starte fra samme uforanderlige gamle checkpoint. Ingen overføring av32
prøvesteg eller produksjonsresume er ennå dokumentert.

## Læringsrettelse v4 verifisert — stor trening fortsatt sperret

Exit lærer nå HOLD-fordelen over kjent gjennomførbar lukking. EXIT-fordelen er
eksakt 0; HOLD-target inneholder faktisk prisendring, finansiering og risiko.
Entry får første lukkeverdi etter kostnad + forventet positiv videreverdi mot
FLAT=0. Ingen nye modellparametere, tapsterskler eller maksimal holdetid.
Fysisk cash/MTM-regnskap er uendret fra v3; v2/v3 beholder gammel semantikk.
Den nye økonomien er eksplisitt v4 og er ennå ikke aktivert i en recipe.

143 unike målrettede CPU-tilfeller består etter én liten reparasjon av en
eldre AST-testfixture. Ingen produksjonskode ble endret for fixturefeilen.
Ny læring, kostnader, kausalitet, faktiske modelloutputs og syntetisk native
VAL256 pause/resume er dekket. Se
[LIQUIDATION_RELATIVE_LEARNING_VERIFICATION_20260914.json](handover_snapshot/LIQUIDATION_RELATIVE_LEARNING_VERIFICATION_20260914.json).
Dette er ikke dokumentasjon på GPU-paritet, produksjonsresume eller lønnsomhet.

Uavhengig forecast er faktiske brutto M5-close-returer K1/K5/K12/K24 med L1.
Nye mål skiller horisontene (nominelt 5/25/60/120 min; markedsgap gir lengre
veggklokketid). Forecast lærer delt representasjon, men er ikke en uavhengig
handelsverdi-lærer. Entry-Q lærer fortsatt fra fryst Exit. Exit-gradient kan
også nå Entry-Q gjennom Entry-token; skade eller nyttig balanse er ikke målt.
Task-log-variance-gradient er ikke bevis for gradient til features/rutere.
M5 har direkte/residuale veier, så D1-gate alene beviser ikke fravær av M5.
Ingen TF er tvunget opp/ned. De gamle VAL-aggregatene inneholder ikke rå
forecast-arrays til nye horisontmål; ingen gammel analyse er kjørt om igjen.

Neste observerte blokkering: eksisterende origin godtar bare gammel epoch-1
VAL16→128 / CPU128→128. Pointer315 står i TRAIN med 19 908 optimizersteg og
kan ikke brukes med nytt mål eller batch256 via denne overføringen. Native
wrapper forbyr legacy resume-probe. Minste videre arbeid er en eksplisitt
hashbundet økonomiovergang og begrenset TRAIN-måling i samme native kjørevei.
Mål targetfordeling/prediksjonsbias og reelle forecast/Entry/Exit-gradienter
før ytterligere modelljustering. Resume sammenligner v4 uavbrutt mot v4
pauset fra samme tilstand. Gamle 320 steg teller ikke som v4-femårsdekning.
Nytt modellvalg må få tom best/top-k/patience/VAL-akkumulator; gammel historikk
bevares. Ingen checkpoint er lastet om eller endret i denne rettelsen.

Brukeren har nå tillatt underagenter. To avgrensede arbeidsstrømmer er brukt;
kun root kjører tunge verifikasjoner, én jobb om gangen. TEST er forseglet.

Brukeren har svart «Ingen fast grense». Risiko-/holdemålet er nå fastsatt i
[docs/RISK_OBJECTIVE_20260914.json](docs/RISK_OBJECTIVE_20260914.json) og hash-bundet
i NEXT_RUN_POLICY.json: ingen absolutt tapsgrense eller maksimal holdetid.
Exit sammenligner forventet videre nettoverdi med lukking nå.
Videre arbeid er gjenopptatt. Økonomi-/målerettelsene er verifisert til 2938eab3.
En fast posisjon om gangen følger den tidligere anbefalingen; Entry vurderes
ved ledig posisjon. Treningsfasit er fortsatt per handelsmulighet, og global
optimalisering av fremtidig kapitalbruk er ikke bevist.
Checkpointmål v5 er implementert og verifisert nedenfor. GPU256-, totalfart-
og faktisk resume-bevis samt korrekt overgang til ny økonomi gjenstår;
training_enabled=false beholdes til alle porter er klare.
Målet om lønnsom trening er ikke oppnådd. Ubesvarte risikospørsmål i tidligere
trinn nedenfor er historikk og gjelder ikke lenger.

## Siste dobbeltkontroll: fortsatt ikke klar for ny epoch

Brukeren har bedt om å kontrollere treningsklarhet, selektivitet, Exit-prinsipp
og samarbeid på nytt. Samlet svar og prioritering står i
[PRETRAINING_DOUBLECHECK_20260914.md](docs/PRETRAINING_DOUBLECHECK_20260914.md),
med [maskinbevis](handover_snapshot/PRETRAINING_DOUBLECHECK_20260914.json).
Begge checkpoints er hashet på nytt, og 19 siste kilde-/testhasher stemmer.
Ingen beståtte tester ble gjentatt. Ingen modellforward, GPU eller trening.

Selektiv Entry er ikke innlært av den nye én-posisjonsmålingen. Targets er
fortsatt Exit-avledet handelsverdi mot FLAT=0; den uavhengige prisprognosen er
fortsatt svak. Koblingene mellom familier/TF finnes, men Entry-konsentrasjon og
metning består. Den nye økonomien er ikke brukt til å trene bevarte vekter.

Ny konkret risiko for læringen: ett-stegs bootstrap og target frosset per epoch,
mens v3 sin HOLD-korrigering ved −100 Bps bare er −0,0000181212 Bps per minutt.
Den lagrede skalare illustrasjonen er ikke målt nevralt treningsutfall. Vurder
direkte læring av videreverdi over kjent lukkeverdi eller bedre kredittildeling
bare gjennom avgrenset sammenligning; ikke legg på vilkårlige regler.

Eksisterende VAL-overføring krever fremdeles batch 128 og lik treningskontrakt.
Dette er ingen overgang til nytt mål / batch 256. GPU-, fart- og resumeporter
består. Handover sjekker NEXT_RUN_POLICY; native start har egne kampanjeporter,
men leser ikke disse nye bevisrollene direkte. Kontroller binding/håndheving
som del av neste native kjøring. Se rapportens prioriterte rekkefølge.

## Checkpointvalg følger nå hele den kronologiske nettoverdien

Den observerte blokkeringen var at åpne valgte handler gjorde den gamle
checkpointscoren utilgjengelig. Eksplisitt MTM-økonomi v3 er nå koblet til
checkpoint-policy v5: kontantresultat pluss gjennomførbar åpen verdi etter
kostnader, kronologisk med én fast posisjon. Patience 5 og høyst 30 epocher
er beholdt. Gamle økonomikontrakter beholder gammel score.

Recipe, native kampanje og trener binder samme monitor til økonomikontrakten.
Native kontroll rekonstruerer målingen fra samtlige 11 016 sideforløp før
bruk; verken endrede summer, ufullstendig VAL, ikke-endelige tall eller en
gammel resultatschema med innskutt ny score aksepteres. Resultatfilen er
fortsatt et uforandret målebevis; kopien i treningens utvalgsstatistikk angir
eksplisitt used_for_early_stopping=true. Ingen modell-EXIT fabrikkeres.

23 målrettede CPU-tilfeller besto i én capped jobb. En syntetisk hel måned
med åpen valgt tapsposisjon gir −30 Bps og sperrer 5 507 overlappende innganger.
Den eksisterende durable koordinatoren ble prøvd med pause/resume og patience
5 for begge monitorer, og velger den kronologiske vinneren selv når gammel
score foretrekker en annen epoch. Dette er styringsbevis, ikke modellresultat
eller produksjonsresume. Se
[MARKED_SELECTION_VERIFICATION_20260914.json](handover_snapshot/MARKED_SELECTION_VERIFICATION_20260914.json).

Ingen GPU eller ny epoch er startet. Neste arbeid er en eksplisitt, bevart
overgang fra gammel checkpointøkonomi, deretter de avtalte GPU256-, totalfart-
og resume-målingene. Entrys uavhengige markedsprognose og gate-kollaps er fortsatt
uløste kvalitetsfunn; ny økonomi alene beviser ikke at disse er rettet.

## Native måling av åpen verdi og posisjonsbruk er forberedt

Den inaktive MTM-varianten har nå et eget native resultatformat v3. Ved naturlig
avkorting måles gjennomførbar lukkeverdi ved siste observerte beslutningsklokke,
inklusive kostnader. Denne legges til kontantbokføringen i en separat
valuation, uten å opprette EXIT, endre HOLD-status eller ommerke cash-ledgeren.
Diskontert utility får tilsvarende sluttverdi. Gammel juni-score forblir uendret.
Ukjent kildegap og ufullstendig beregning gir fortsatt ingen komplett NAV-score.

Ny marked_policy_evaluation rapporterer både uavhengige muligheter og en
kronologisk kontroll med én fast posisjon om gangen, uten pyramidering eller
rentesrente. EXIT behandles før ny inngang ved lik klokke. Den skiller modellens
FLAT fra handler som ikke får plass mens en posisjon er åpen. Kontrollen endrer
ikke Entry-targets eller faktisk opptaksregel; position_rule_trained=false og
used_for_early_stopping=false er eksplisitt. Avtalen må fullføres før trening.

En konkret cachefeil fra det nye inngangsavhengige økonomimålet er rettet:
ferdigberegnede HOLD-steg deler nå bare intervallkostnader. Inngangens mark
beregnes på nytt uten å øke cache per handel. 400 syntetiske forespørsler over
to intervaller ga to cacheoppføringer med eksakt scalar-/slice-likhet, også
ved omvendt rekkefølge. Dette er ikke en måling av total GPU-/CPU-fart.

57 målrettede tilfeller besto i én capped audit-jobb. De omfatter den endrede
økonomikomposisjonen, cache, tap i åpne posisjoner, overlapp, samtidige klokker,
datagap, compute-guard og syntetisk native VAL med batch 256 og pause/resume.
Resultatvalidatoren rekonstruerer også den nye metrikken og avviser endrede summer.
Bevis: [MTM_VAL_VERIFICATION_20260914.json](handover_snapshot/MTM_VAL_VERIFICATION_20260914.json),
med bevart logg og eksakte kildehasher. Tidligere 42-testbevis gjelder sin angitte
kilde; denne runden dekker de nå endrede eierne. Ingen fullsuite eller faktisk
GX1-modellanalyse, GPU, ny juni-VAL eller trening ble kjørt.

Fortsatt uavklart: brukerens risikovalg er ubesvart (markedsstyrt Exit alene
eller også en brukeroppgitt absolutt tapsgrense). Ingen risikogrense er valgt.
Før videre kampanje må risiko-/posisjons-/resultatavtalen fastsettes, Entrys
selvstendige kvalitet undersøkes, og eksplisitt overgang fra gamle checkpoints
til nytt mål samt GPU256-paritet, samlet fart og faktisk resume dokumenteres.
NEXT_RUN_POLICY.json er uendret. Ingen ny recipe/kampanje er aktivert.


## Verifisert økonomirettelse, ennå ikke aktivert

I GX1_CURRENT er en eksplisitt kandidat for markedsverdibasert økonomi nå
implementert i eksisterende økonomieier, provider, readiness-validator og bygger.
Velges som reward_accounting=liquidation_value_increments_v1, kontrakt v3.
Ingen ny recipe/kampanje er opprettet, NEXT_RUN_POLICY.json er uendret og
trening er fortsatt stoppet. Risikospørsmålet er ubesvart.

Q-feltene beholdes. Med L som gjennomførbar lukkeverdi og V som neste valgte
Q-verdi er HOLD-target = -finansiering - risikostraff + (1-gamma)*L_neste
+ gamma*V_neste. Relativt til lukking nå er dette endring i L, fratrukket
kostnad/risiko, pluss diskontert videreverdi. Ved konstant pris og null kostnad
beholder ubestemt HOLD tapets negative verdi; det gamle målet ga null.
Dette gir likegyldighet ved helt flat pris, ikke en påtvunget Exit eller
vilkårlig holdetid. Første minutt LONG-finansiering er med én gang i v3.

42 målrettede CPU-tilfeller besto i to sekvensielle capped audit-jobber (39 + 3).
De dekker regnskapsidentitet med markedsgap, begge sider, scalar/vector FP32-bytes,
Entry-avhengig VAL-cache, naturlig manglende successor, native Bellman til Entry,
separat kontant-/utility-regnskap i VAL, ny bygger/VAL-binding og parent-offset.
Bevis med eksakte kilde-/logghasher:
[MTM_OBJECTIVE_VERIFICATION_20260914.json](handover_snapshot/MTM_OBJECTIVE_VERIFICATION_20260914.json).
Loggene er bevart ved siden av beviset. Ingen modellforward, GPU eller trening.

Dette er ikke bevis på lært Exit-forbedring, godt Entry-signal eller lønnsomhet.
V3 er ikke eksakt resume av det gamle optimaliseringsproblemet. Gamle bindings-
kontrakter og standard v2-adferd er bevart for historisk etterprøvbarhet.
Dette første 42-testtrinnet omfattet ikke åpen NAV eller kronologisk score.
Neste verifiserte trinn øverst i handover har disse målingene implementert, men
ikke aktivert som opptaks-/early-stopping-regel. Gammel juni-score er uendret.
Neste arbeid er å fullføre denne evaluerings-/posisjonsavtalen, avklare risiko
og bevare Entrys selvstendige kvalitetskrav før ny kilde-/checkpointovergang
og de påkrevde GPU256-/totalfart-/resume-målingene. Ingen beståtte tester gjentas
uten ny relevant endring eller feil.


## Aktivt lønnsomhetsmål og uavhengig Entry-kontroll

Brukeren har bestilt videre arbeid etter anbefalingen og presisert at Entry må
kunne velge riktige innganger selvstendig. Ny aritmetikk fra allerede lagrede
epoch-1-forløp er bevart i
[ENTRY_INDEPENDENT_QUALITY_20260914.json](handover_snapshot/ENTRY_INDEPENDENT_QUALITY_20260914.json).
Ved 60 minutter ga valgte innganger −8,3884 netto Bps i gjennomsnitt; valgt side
var best av LONG/SHORT på 47,9484 % av 5 508 rader. Ved 240 minutter var tallene
−18,4727 Bps og 42,2162 % på 5 171 felles gyldige rader. Dette er faste
diagnosehorisonter uavhengig av Exit-tid, ikke modellens handelsresultat,
holdetidsgrenser eller en fasit fra etterpåklok optimal Exit.

Kodekontroll på ren 2b6b7b51 bekrefter at Entry-Q-target er første tilstands
frosne Exit-verdi for begge sider, med FLAT=0. Exit sender også gradienter inn
i Entry-representasjonen. Det separate forecast-hodet lærer observerte
close-til-close-returer ved 5/25/60/120 minutter; det velger ikke handel direkte.
Exit-avledet Q er derfor ikke selvstendig bevis på markedsretning. Svakt
Entry-signal må undersøkes før nye modeller/hoder, og før stor trening kan
begrunnes som løsning. Ingen kausal feil i gradientdelingen er ennå påvist.

Neste risikoavklaring er sendt brukeren: markedsstyrt Exit uten absolutt
tapsgrense, eller en brukeroppgitt Bps-grense. Ingen svar er mottatt ennå.
Økonomi-/targetkode og NEXT_RUN_POLICY.json er uendret; ingen ny epoch er startet.
Seneste handover --check, 08:21:14 UTC, viste ingen native prosess og uendret
checkpoint 315 / 19 908 steg. GPU256-, totalfart- og resume-bevis mangler fortsatt.

## Ny målt prosjektgjennomgang — 2026-09-14

Brukeren har bestilt en bred, målt gjennomgang og prioriterte forbedringsforslag:
selektiv Entry, markedsstyrt Exit og nyttig samarbeid mellom åtte familier/TF-er.
Rapport: [docs/PROJECT_REVIEW_20260914.md](docs/PROJECT_REVIEW_20260914.md).
Maskinbevis: handover_snapshot/PROJECT_AUDIT_METRICS_20260914.json.
Analysegrunnlaget er første epochs uforanderlige EMA, ikke epoch 2.

Nye funn: Entry-poolingen har 99,9977 % gjennomsnittsvekt på D1×SMC/likviditet;
lokal familievekt har 99,7495 % på sesjon/regime. Attention blander signaler før
vektingen, så dette er ikke bevis på null påvirkning fra øvrige ruter. 51 av 704
Entry-featurekoordinater har minst én øvre metning; eksisterende gatekrav består
ikke. Exit har langt bredere rutebruk og ingen observerte mettede featureporter.
Begge handels-Q er positive på alle 5 508 innganger. Eksakt FLAT=0 ville ikke
endret ett valg. Uavhengige valgte forløp overlapper med opptil 3 287 posisjoner;
dagens mulighetsmetrikker er ikke en kronologisk kontostrategi.

Anbefalt retning er konsistent markedsverdibasert økonomi med uttrykkelig risiko,
selektivitet og avtalt posisjonsbruk. Ren flytting av belønning tidligere er ikke
bevis på at HOLD-insentivet er rettet. Tallfestet risiko, eventuell posisjonsgrense
og ny resultatmetrikk er fortsatt uavklart; NEXT_RUN_POLICY.json er uendret.
Ingen modell-/runtimekode, vekter eller checkpoints er endret. Ingen ny trening,
modellforward eller GPU-kjøring. Bare dokumentasjon og nye beregninger fra lagrede
resultater; SHA-verifisert epoch-1-parameterinspeksjon ble gjort på CPU.

Gjeldende arbeidskopi: /home/andre2/src/GX1_CURRENT, branch work/gx1-current.
Kjør bash scripts/gx1_handover.sh --check. Scriptet viser verifisert historisk
resultat, lagret checkpoint, faktisk prosessstatus og hvorfor neste løp er blokkert.
Det starter ingenting og har ingen gammel reservevei.

## Fullført kjøring og resultater

Full femårs-TRAIN epoch 1 og full juni-2026 VAL er ferdig. Brukerstopp gjelder.
Windows-kampanjeoppgaven er fjernet; ingen videre epoch startes automatisk.
Epoch 2 startet før stoppbeskjeden og rakk 320 lagrede steg: checkpoint 315,
epoch_index 1, totalt 19 908 steg. Første epochs EMA ved 19 588 steg er uforanderlig.
Siste kjøring var fra frosset kilde 03592fe6; denne skal ikke redigeres.
COMPLETED_RUN.json binder nøyaktige kilder, checkpoints og hele resultatfilen.
Invokasjon 7 ble manuelt avsluttet; den skal ikke omtales som en ytre PASS.

VAL: 57 845 748 tilstandsvurderinger / 467 371 forwards. 7 472 av 11 016
hypotetiske sideforløp ble lukket; 3 544 nådde månedsslutt med HOLD.
Entry valgte 4 180 LONG / 1 328 SHORT / 0 FLAT. Av 5 508 valgte handler ble
2 227 lukket og 3 281 avkortet ved månedsslutt. Full-policy netto Bps er utilgjengelig.
Hypotetisk månedssluttslukking av åpne valgte handler gir −312,90 Bps per handel;
dette er diagnostikk, ikke lært Exit eller porteføljeavkastning. Verste netto MAE −1 369,27 Bps.

## Hva analysen betyr for neste steg

72,42 % av valgte månedssluttåpne handler hadde først +20 netto Bps og senere
under −20 uten lukking; rundt 89 % hadde største MFE før verste MAE.
Høyeste 10 % Q-margin ga fortsatt −218,62 Bps i samme hypotetiske sluttberegning.
Entry-MAE-hodet gjelder 95 minutter og har ingen direkte beslutningsmyndighet.
Exit mottar faktisk akkumulert MFE/MAE. Entry-Q-MSE var 2,66 % svakere enn konstant baseline.
SHORT-HOLD har null løpende finansiering og risikostraff. Innen dagens mål kan
ubestemt HOLD derfor dominere frivillig tapsrealisering. Flere epocher alene er
ikke dokumentert som løsningen. Avklar ønsket økonomisk holdetid og risikoramme,
og gjør så den minste nødvendige endringen i eksisterende økonomimål.
Ingen terskel, risikogrense eller modellmål er foreløpig endret.
Hele analysen og presiseringene står i docs/ENTRY_EXIT_REVIEW_20260914.md.

## Én vei for neste kjøring

NEXT_RUN_POLICY.json krever VAL-batch 256, åtte CPU-arbeidere og 10 800 sekunders
VAL-vinduer, 12 000 sekunders native budsjett og 13 800 sekunders ytre vakt.
Bevar market-cache, metadatacache, delt prissti for sider, batched økonomi,
FP32 uten TF32 og den laststyrte GPU-clock-launcheren. Alle features/familier/TF-er beholdes.
Målt CPU-trinn: +17,36 % med åtte arbeidere/batch 128, +24,65 % med batch 256;
identiske tilstandsbytes, fem repetisjoner, CPU 0–7. Dette er ikke GPU-/totalfart.
Før trening kreves risikoavklaring, faktisk GPU-paritet på 256, målt samlet fart
og eksakt resume-likhet. Profilen er ikke aktivert; ingen ny kampanje er laget.
Eksisterende kilde-/data-/checkpoint-/guardporter gjelder i tillegg.

30 epocher med patience 5 er fortsatt sluttmålet når dette er klart. TEST er
forseglet, ingen live/papir. Én tung jobb og timesvis modellobservasjon.
300 W / 85 C kjerne / 80 C minne / 12 GiB VRAM / 20 GiB RAM / 512 MiB swap.

## Opprydding og bevaring

28 gamle arbeidskopier er fjernet. Gjeldende kilde er GX1_CURRENT.
Elleve historiske kilde-/runtimekopier er beholdt fordi checkpoint- og datahistorikken
binder dem, eller fordi de eier Git/kjøremiljøet. De er ikke alternative oppstartsveier.
Ingen treningsdata, modellvekter, fullførte resultater eller checkpoints er slettet.
Gamle oppstartsnotater, ferdigbrukte forberedelsesbindinger og alternative
treningsgrener i den felles kjøreinnpakningen er fjernet fra gjeldende kilde.
Importerte moduler med gamle navn beholdes når feature- eller modellkoden bruker dem.

Gjenoppretting av fjernet kildemateriale: /home/andre2/GX1_ARCHIVE/CLEANUP_20260914/ALL_REFS.bundle.
Unike lokale endringer er lagret privat utenfor aktivt prosjekt.
Kvitteringer står i handover_snapshot/WORKTREE_CLEANUP_20260914.json,
CLEANUP_CODE_REMOVALS.json og TASK_RETIREMENT_20260914.json.
Arkivet skal ikke brukes som en alternativ treningskilde.

Mac-arbeidsområdet er også ryddet: 85 gamle toppnivåfiler/-mapper, 798 filer
og 252 889 123 byte er fjernet fra aktivt prosjekt etter kontrollert arkivering.
Hele trade_review_20260914 med VAL-resultat og analyser er beholdt.
Det midlertidige Mac-arkivet er slettet etter brukerpresisering. Bare tidligere
fullført resultatrapport er beholdt utenfor prosjektet. Den dupliserte checkpoint-
filen var SHA256-identisk med bevart original på treningsmaskinen.
På Mac brukes bare ./handover.sh (--check eller --verbose); den leser GX1_CURRENT
via SSH og starter aldri trening. Scriptets versjonerte kilde er scripts/macos/gx1_takeover.sh.

Windows: 108 gamle overføringsfiler, engangsscript og oppstartsvarianter er slettet.
Resultatduplikater var identiske med bevarte resultater. Gammel kampanje-XML er
også slettet. Aktive vakter og refererte runtimefiler er beholdt.
