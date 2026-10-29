# CURRENT HANDOVER — XAUUSD, retning og bps

Eneste repo: /home/andre2/src/GX1_CURRENT, work/gx1-current, på gx1-3090-lan,
WSL Ubuntu-22.04. Ingen oppdrag fra
andre chatter. Den tidligere lesende retningsrevisjonen er fullført.
Les GX1_RULES helt ved første arbeid; kjør scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY er arbeidsstatus. Originale terminaler overstyrer prosa.

Programrot:
 /home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2

## Aktuell status

29.10 — FUSJONSKONTROLL 002 FEILET UNDER PYTORCH-IMPORT: TEST-navnsperren rammet installert torch.testing, ikke forseglet markeds-TEST. Begge mislykkede kontroller hadde allerede lest checkpointfilenes bytes for hashkontroll, men hadde ikke deserialisert vektene eller kjørt modell. 002 er konsumert med claim/feilkvittering. Ny 003 importerer PyTorch før runtime-sperren; syntetisk test tillater bare installerte pakker og eksakt egenfil, mens forseglet TEST og ubundne RUNS-filer fortsatt avvises. Se NEXT_RUN_POLICY.

29.10 — FUSJONSKONTROLL 001 FEILET FØR CHECKPOINT-DESERIALISERING: Den nye lesesperren avviste kontrollskriptets egen kildefil under PyTorch-import. Claim og feilterminal er bevart, forsøket er konsumert. Ny kontroll 002 tillater kun den eksakt hashbundne egenfilen i tillegg; syntetisk test bekrefter at andre RUNS-filer og TEST fortsatt avvises. Ingen modell eller trening er kjørt. Se NEXT_RUN_POLICY.

29.10 — DIREKTE SPOR: FUSJONSVEIEN FOR SVAKT PRISSIGNAL: Kilden viser at både den nye momentumprojeksjonen og de to videre residualutgangene cross_tf_out og family_tf_cooperation_out starter med null vekter og bias. Konstruktormetadata oppgir faste skalaer 0,5 og 0,25. Det er en konkret mulig startbarriere, ikke bevis for at dagens trente modell har null gradient eller at dette forårsaker manglende markedsedge. Én hashbundet 4 GB CPU-kontroll er åpnet for opprinnelige initial- og sluttvekters aggregerte normer og skalaer; ingen modellforward, optimizersteg, markedspayload, TEST eller handel. Se NEXT_RUN_POLICY.

29.10 — M5-PRISPROJEKSJONENS SVÆRT LAVE BIDRAG UAVHENGIG BEKREFTET: Separat capped CPU-replay gjenskapte nøyaktig originalprediksjonene og kontrollerte 600 AUC/logloss/Brier- og 600 påvirkningspaneler. Ingen av RSI14, MACD-linje/-histogram eller ret_1, enkeltvis eller samlet nullstilt ved XAU-momentumprojeksjonen på fem TF, endret en eneste binær beslutning i kvalifiserte TRAIN-/Q1-utvalgsrader på noen av de 60 horisontene. Femminutters Q1, 1 726 rader: samlet gjennomsnittlig absolutt logitendring 0,0000011575, original/null AUC 0,510522/0,510519. Observatoren registrerte 99,603–100 % ikke-null tidsrute-celler i Q1; tellingen og bootstrapintervallene er ikke uavhengig etterregnet. Andre ruter og lokale aliaser er bevart, nulling kan være unaturlig og dette beviser ikke edge eller årsaken til all svak retning. Alle engangsporter er lukket. Aggregert 60-horisontsrapport og kildebevis: docs/DIRECTION_M5_MOMENTUM_INTERVENTION_RESULT_20261029.md. Neste direkte spor er hvorfor denne prisrutens signal blir så lite gjennom projeksjon/fusjon/utgang. TEST/handel/Exit stengt. Se NEXT_RUN_POLICY.

29.10 — M5-INDIKATORINTERVENSJON FULLFØRT; UAVHENGIG REPLAY ÅPNET: 256 TRAIN- og 1 736 senere Q1-foreldre, 750 modellevalueringer, byteidentisk lagret baseline, null optimizersteg og uendret modell/kilde. I første resultat endret nulling av alle fire prisfelter ingen retningsbeslutninger på noen av de 60 horisontene i de observerte kvalifiserte radene; ved 5min var gjennomsnittlig absolutt logitendring 0,0000011575 på 1 726 rader. Inngangsobservatoren rapporterer ikke-null verdier for feltene i alle fem TF, men disse observasjonsantallene er ikke uavhengig gjentatt. Dette måler den ene prisprojeksjonen, mens GC og andre ruter beholdes; forklarer ikke alene all svak retning eller beviser edge. Signert vakt PASS med maks faktisk trekk 147,48 W, kjerne 48 °C og minne 52 °C; 250 W beholdt og målt idle 22,93 W. Engangsporten er lukket. Én ny avgrenset CPU-kontroll av lagrede skårer er åpnet, ingen modell/trening/TEST. Samtidig er feilaktig påstand om umålt Q1/Q2-minuttgap presisert mot allerede fullført audit; effekt på score er fortsatt ukjent. Se NEXT_RUN_POLICY.

29.10 — M5-MOMENTUMKONTROLL 003 BUNDET: Én ny plan er bundet til den ferdigtrente M5-modellen, den opprinnelige forhåndsdefinerte populasjonen og intervensjonen, historisk fit-kildekontroll og den testede nye runneren. Eksakt 250 W konfigurert maksimum, fortsatt 170 W stopp for faktisk trekk og eksisterende temperatur-/kapasitetsvakter. Maks 750 forwardkall og ingen optimizersteg. De to avviste planene kan ikke relanseres. Ingen ny retningsscore ennå. Se NEXT_RUN_POLICY.

29.10 — 250 W-PRODUSENTRETTELSE KONTROLLERT: 15 fokuserte tester bestod, inkludert syntetiske sperrer ved faktisk trekk 171 W, konfigurert grense 251 W, kjerne 66 °C og minne 81 °C. Kun en eksakt hashbundet M5-intervensjonsplan 003 kan få 250 W konfigurert; øvrige produsenter beholder eksisterende grense. Historisk fit-runner og ny aktiv runner har hver sin kontrollerte kildebinding. Ingen modell er kjørt; neste steg er å binde én ny plan til testet kilde. Se NEXT_RUN_POLICY.

29.10 — SNEVER 250 W-PRODUSENTRUTE UNDER KILDEKONTROLL: Kun eksakt hashbundet M5-intervensjonsplan 003 kan få 250 W konfigurert grense i CUDA-produsenten. Vakten for faktisk trekk 170 W, 65/80 °C, signert telemetri, GPU-minne, tids- og kapasitetsgrenser beholdes. Den fullstendige historiske fit-kildevalideringen og den opprinnelige Git-bloben kontrolleres separat fra den nye kjørende runnerens hash; de øvrige 62 fit-kildeeierne må fortsatt være byteidentiske. Koden må testes før en ny plan eventuelt bindes. Ingen modellkjøring er startet og ingen ny edge er målt. Se NEXT_RUN_POLICY.

29.10 — M5 FIT-KILDEVALIDERING FULLFØRT OG KONSUMERT: Den eksisterende fit-validatoren passerte med alle 63 originale fysiske kildeeiere. Opprinnelig capped-runner og GPU-vakt samsvarte også bytevis med faktisk fit-commit 6b199953; 4 GB capped CPU, ingen modellforward, optimizer, checkpoint eller TEST. Kvitteringen er hashbundet og skal bevare historisk proveniens ved en eventuell separat, snever runnerendring. Ingen ny retningsscore. Se NEXT_RUN_POLICY.

29.10 — KILDEFORKONTROLL 002 FEILET I STIKONTROLL: Ved ombinding ble referansen til den fullførte fit 001 feilaktig endret til en ikke-eksisterende fit 002. Ingen modell, checkpoint eller TEST ble åpnet. Forsøket er konsumert. I ny kontroll 003 er alle faste stier, fitreferansene og de 63 fysiske kildehashene forhåndskontrollert; kontrollen skal fortsatt validere hele fit-oppskriften. Se NEXT_RUN_POLICY.

29.10 — KILDEFORKONTROLL 001 FEILET FØR PROSJEKTIMPORT: En CPU-jobb utenfor repoets kildekatalog fant ikke gx1-pakken. Ingen prosjektdata, modell, checkpoint eller TEST ble åpnet; forsøk 001 er konsumert og bevart. Ny kontroll 002 har eksplisitt kanonisk importsøkesti, kontrollert på forhånd, og samme avgrensning. Se NEXT_RUN_POLICY.

29.10 — HISTORISK FIT-KILDEKONTROLL ÅPNET: Én 4 GB CPU-metadatajobb skal kontrollere den fullførte M5-fitens opprinnelige oppskrift og 63 fysiske kildeeiere mens runneren fortsatt har treningens hash, samt kontrollere Git-kilde ved faktisk fit-commit. Dette er nødvendig før en eventuell snever justering av produsentens effektvakt; ingen modell, checkpoint, TRAIN/VAL-markedslesing, TEST eller handel er åpnet. Se NEXT_RUN_POLICY for eksakt engangsadgang.

29.10 — INTERVENSJON 002 AVVIST FØR MODELL/DATA: CUDA-produsentens eksisterende signerte preflight målte 250 W konfigurert grense og stoppet med exit 75 fordi denne ruten fortsatt har 160 W-grense, selv om brukerens gjeldende policy er 250 W. Ingen claim, outputmappe, modellforward eller dataåpning; logg og avvisningskvittering er bevart. Plan 002 er lukket og konsumert. Produsentens separate faktiske effektgrense er 170 W; temperatur-, telemetri- og kapasitetsvakter skal beholdes. Å endre capped-runner berører den historisk hashbundne fit-kilden og krever derfor en korrekt kildeovergang før ny engangsplan. Ingen ny retningseffekt er målt. Se NEXT_RUN_POLICY.

29.10 — INTERVENSJONSSTART RETTET FØR DATAÅPNING: Første start ble avvist av eksisterende capped-runner med exit 75 fordi kommandoen brukte relativ Python-sti; ingen modell, datapayload, guardlogg eller claim ble opprettet. Den første planen er lukket og erstattet av egen 002-plan med samme forhåndsdefinerte populasjon/intervensjoner og kanonisk absolutt Python-sti. Ingen nye retningsresultater følger av avvisningen. Se NEXT_RUN_POLICY.

29.10 — EFFEKT AV RSI/MACD/RET_1 PÅ FERDIG M5-MODELL ÅPNET: Én avgrenset guarded CUDA-producer er bundet til det fullførte price_only-checkpointet. Først må alle lagrede prediksjoner gjenskapes byteidentisk på 256 faste TRAIN-rader og hver tiende opprinnelige Q1-batch (1 736 senere foreldre). Deretter nøytraliseres RSI14, MACD-linje, MACD-histogram, ret_1 og alle fire separat ved den normaliserte XAU-projeksjonsinngangen på M5/M15/H1/H4/D1; GC, lokale aliaser og øvrige ruter beholdes. Alle 60 målehorisonter rapporteres, med justerte parrede dag-/ukebånd ved 5/60min. Høyst 750 forwardkall og ingen optimizer, TEST, Exit eller handel. 18 målrettede syntetiske tester, én separat test gjennom faktisk native arkitektur og metadataforkontroll med 59 referanser bestod. Dette er en intern intervensjon på gjenbrukt utvikling, ikke bevist tradingedge eller effekt av retrening. Se NEXT_RUN_POLICY for plan og vakter.

29.10 — PRISRUTENS FAKTISKE CHECKPOINT KONTROLLERT: Den nye RSI14/MACD/ret_1-projeksjonen startet med 512 nullvekter; etter én TRAIN-pass var alle 512 ikke-null, med samlet RMS 0,002574. Alle fire felt hadde 128 ikke-null projeksjonsvekter, og alle fem tidsrammegatenes 512 vekter var ikke-null. Faste TRAIN-normaliseringsmetadata har positive skalaer og varierende transformerte min/max for hvert av feltene på alle fem TF. Dette avviser ikke effekt av manglende input eller lavt reelt bidrag; parameterstørrelser og metadata er ikke en kausal ablasjon og forklarer ikke de nesten identiske senere prediksjonene. Begge egne checkpointfiler og kildens hasher er kontrollert, uten markedspayload, modellforward, optimizer eller TEST. Engangsadgangen er lukket og konsumert. Neste direkte kontroll er en avgrenset fixed-model intervensjon av de fire feltene på TRAIN og gjenbrukte senere rader, ikke ny trening. Se NEXT_RUN_POLICY for bevis.

29.10 — NESTE KONKRETE SPOR, NULLINITIALISERT PRISRUTE: Den hashbundne modellkilden oppretter de fire nye momentumfeltenes felles projeksjonsvekter og bias som null, og de tilhørende TF-gatene som nøytrale. Dette var designet for å bevare den tidligere initialmodellen; den nye og tidligere korrigerte modellens lagrede 256 startprediksjoner har samme oppgitte filhash. Privat kvalifisering viste gradientkontakt på de fire feltene og på gatene etter ett steg i et 32-foreldres utvalg. Dette forklarer ikke hvorfor fulltrente prediksjoner fortsatt er så like. Én avgrenset CPU-kontroll er åpnet for de to hashbundne egne checkpointene og eksisterende TRAIN-normaliseringsmetadata: mål faktisk parameterbevegelse og feltskala, uten markedspayload, modellevaluering, ny trening eller TEST. Se NEXT_RUN_POLICY.

29.10 — M5-SAMMENLIGNING UAVHENGIG GJENSKAPT OG KONSUMERT: En separat Mann–Whitney-rangberegning gjenskapte kandidatens og referansens 120 AUC-punkter på alle 60 horisonter, og 5min-treffene. På samme 17 192 gyldige 5min-rader ga den nye modellen en annen OPP/NED-beslutning på bare 28 rader: 15 nye riktige mot 13 gamle riktige; logit-korrelasjonen var 0,999998. Nye/gamle prediksjoner er derfor svært like på denne gjenbrukte utviklingsfolden, uten at dette identifiserer årsaken. Ny/gammel AUC 0,496961/0,496976 og treff 48,616/48,604 %, alltid OPP 51,582 %; parrede dag-/ukebånd for AUC-differansen inkluderer null, og ingen av 60 samtidige dag-/ukebånd dokumenterer kandidat-AUC over 0,5. Prisfelter og projeksjon/gater endres sammen; isolert indikatorverdi er ikke målt. Alle trenings- og engangsporter er lukket, engangstask deaktivert, 250 W beholdt og TEST/Exit/broker/handel stengt. Neste relevante undersøkelse er effektivt indikatorbidrag og inputskala, uten å anta hvorfor forskjellen er liten. Se NEXT_RUN_POLICY for eksakte bevis.

29.10 — M5-SAMMENLIGNING FULLFØRT, UAVHENGIG REPLAY ÅPNET: På samme 17 192 gyldige femminuttersrader i gjenbrukt Q1 er ny/gammel AUC 0,496961/0,496976 og treff 48,616/48,604 %, mot alltid OPP 51,582 %. Parrede dag-/ukeintervaller for AUC-forskjellen inkluderer null. Ingen av de 60 kandidatbåndene dokumenterer AUC over 0,5 i både samtidig dag- og ukeintervall. Alle 60 fasit-/klokkeakser er kontrollert like; prisfelter og projeksjon/gater endres sammen, så enkeltindikatorer kan ikke isoleres. Én avgrenset CPU-kontroll skal uavhengig gjenskape de 120 lagrede AUC-punktene og måle hvor like 5min-prediksjonene er; to syntetiske rang-/tie-tester bestod. Ingen ny modellforward, trening, TEST eller handel. Sammenligningsporten er lukket og konsumert. Se NEXT_RUN_POLICY.

29.10 — REN M5-ENTRY FIT FULLFØRT; FAST SAMMENLIGNING ÅPNET: Én fysisk boot 580, 47 602 TRAIN-foreldre, 2 976 oppdateringer og 17 288 senere Q1-utviklingsforeldre ble fullført. Original sluttkvittering, signert vakt PASS, uendret kilde og kandidatens egne 256 startprediksjoner er kontrollert; private vekter ble ikke brukt. Maks faktisk GPU-trekk 137,71 W; engangsoppgaven er deaktivert, 250 W beholdes og målt tomgang etterpå var 27,62 W. All trening er lukket. Én CPU-sammenligning av de hashbundne lagrede prediksjonene er åpnet på forhåndsbestemt felles populasjon og korrigert fasit; den sjekker lik klokke og fasit i alle 60 hoder før femminutters paret måling. Tre syntetiske join-/negativtester bestod. Prisfelter og projeksjon/gater er endret sammen. Ingen ny retningseffekt eller økonomisk edge er konkludert ennå. TEST, Exit, broker og handel er stengt. Se NEXT_RUN_POLICY for bevis og eksakt adgang.

29.10 — REN M5-ENTRY ETT PASS ÅPNET: Den fullførte private price_only-kontrollen og egen 256-raders startprediksjon er bundet til én ny fullfit med 47 602 TRAIN-foreldre, maks 2 976 oppdateringer og opptil fire vinduer bare som fortsettelse av samme løp. Metadatakontroll verifiserte 59 referanser og målfilhasher, med price_only og ingen M1-referanser; 137 relevante tester samt 10 etter siste protokollpresisering bestod. Kildeovergang godtar bare angitte endringer i orkestrering; modell, features, tapsfunksjon og øvrige kildeeiere er de samme som privatkvalifiseringen. RSI14, MACD-linje/-histogram, ret_1, åtte spesialister, D1/makro og 60 korrigerte mål beholdes. Første måling sammenligner 5min paret mot eksisterende korrigert modell på gjenbrukt Q1; øvrige horisonters modellforskjeller er beskrivende, og simultane AUC-bånd gjelder kandidaten. Ingen prediktiv forbedring eller økonomisk edge er påvist ennå. Fysisk restart før hvert vindu; 250 W er grense også etterpå, temperaturvakter gjelder. TEST, Exit, broker, ordre og handel er stengt. Eksakt autoritet og bevis i NEXT_RUN_POLICY.

29.10 — REN M5-ENTRY, PRIVAT KONTROLL FULLFØRT OG KONSUMERT: På fysisk boot 579 bestod kandidatens egen initialkontroll på 256 foreldre og fire private oppdateringer på 32 faste TRAIN-foreldre. RSI14, MACD-linje/histogram og ret_1 fikk gradientkontakt i begge forløp, alle 60 mål/priorer ble kontrollert og gjenopptak var byteidentisk. Ingen separate M1-verdier eller M1-masker er entryinput. Signert vakt PASS, maksimal faktisk GPU-effekt 135,29 W; grensen er fortsatt 250 W, målt tomgang 32 W. Windows-engangsoppgaven er deaktivert. Denne lille mekanikkontrollen viser ikke bedre retning eller økonomisk edge. Treningsporter er lukket mens egen fersk ettpasskjøring bindes og kontrolleres; private vekter brukes ikke. TEST, Exit, broker og handel er stengt. Se NEXT_RUN_POLICY.

29.10 — REN M5-ENTRY: ET EGET EKSAKT PRIVAT LØP ER ÅPNET etter brukerens stopp/omstart. Den faktiske metadatakjeden passerte for price_only uten M1-referanser; 59 metadatareferanser og begge originale/korrigerte fysiske målhasher ble kontrollert, uten modellforward, oppdatering eller TEST. Fire fokuserte price_only-/momentumtester bestod. Kun egen 256-foreldres initialkontroll, 32 faste TRAIN-foreldre og inntil fire private oppdateringer er åpnet, én gang etter maskindekkende idle-sjekk og fysisk restart. Full fit er fortsatt stengt til egen genuin kvalifisering og begrenset binding er verifisert. 250 W beholdes også etterpå; eksisterende temperaturvakter gjelder. Ingen Exit, broker, ordre, paper/live eller TEST.

29.10 — HISTORIKK: REN M5-ENTRY KLARGJORT, SENERE KVALIFISERING ÅPNET: En ny prisbasert privat kvalifiseringsoppskrift er bundet til den eksisterende price_only-ruten uten M1-verdi eller M1-maske som entryinput. Kompaktprofilen beholder RSI14, MACD-linje/histogram, ret_1, åtte spesialister, D1/makro, M5/MTF og korrigert 1–60-fasit. Forhåndsdefinert sammenligning er mot den eksisterende korrigerte ettpassmodellen på samme gjenbrukte Q1, med 5min primær og alle 60 hoder; prisfelter og projeksjon/gater endres sammen, så en enkelt indikator kan ikke isoleres. Ingen ny genuin kvalifisering, fit, TEST eller handel er gjennomført. Oppskrift er ennå ikke admittert; eksakt status i NEXT_RUN_POLICY.

29.10 — BRUKERSTOPP OG M5-ENTRY: Brukeren presiserte at entry skal vurderes på M5 med lukkede høyere tidsrammer, mens M1 og senere tick er fremtidig oppfølging etter åpen posisjon. Brukeren ba om å stoppe den påbegynte utjevnede M1-armen og starte på nytt med riktig scope. Egen trenervakt mottok SIGTERM; prosessene er borte og Windows-engangsoppgaven er Disabled. Siste varige checkpoint er 1 024 oppdateringer / 16 384 TRAIN-foreldre; faktisk arbeid ved stoppsignalet er ukjent. Originalt ACTIVE_INVOCATION-merke er bevart: normal terminal/progresskvittering ble ikke opprettet etter signalet. Kampanjen er avbrutt/konsumert og skal aldri gjenopptas. Ingen sluttprediksjoner eller retningsresultat. GPU-konfigurasjon og faktisk grense 250 W, målt tomgang 26,26 W; vakt og telemetri Running. M1-ordrearmen er ikke kjørt og er erstattet av M5-scope. Ny ren M5-prisrute finnes syntetisk, men mangler genuin kvalifisering og egen fullfit-binding; alle treningsporter er lukket inntil disse er konkrete og kontrollert. Bevar åtte spesialister, D1/makro, 1–60 korrigerte mål, samt TEST-, broker- og handelsforsegling. Eksakte bevis i NEXT_RUN_POLICY.

29.10 — HISTORIKK: MATCHET M1/M5-MODELLTEST FORBEREDT, SENERE STOPPET/ERSTATTET: Brukeren har bedt om å kjøre testen uten gjentatte godkjenningsspørsmål. De beståtte private kontrollene er nå bundet til en separat full-fit-rute: én TRAIN-gjennomgang per arm, 47 602 foreldre og høyst 2 976 steg, sekvensielt med korrigert 1–60min-fasit. Begge beholder RSI/MACD, M5-ordreflyt og samme M1-masker; kontrollarmen bruker M1-bøttegjennomsnittet, kandidatarmen hvert minutt. Kun kontrollarmen er åpnet, med ny faktisk initialparitet før første steg og egne finite kampanjevinduer. Kildeovergang er AST-kontrollert mot begge kvalifiseringene, faktiske inputmetadata og målhasher er verifisert, og fokuserte negative tester avviste feil bindinger. Dette er ingen målt modellforbedring. TEST, handel og bruk av private kvalifiseringsvekter er stengt. Se NEXT_RUN_POLICY.

29.10 — FAKTISK M1 GITT LIK M5, 1–60 MINUTTER: På 17 288 gjenbrukte Q1-foreldre hadde 17 135 komplett M1-dekning. Ved 5 min hadde 17 039 gyldig ikke-null fasit og 16 464 inngikk i strata med både OPP/NED; siste M1-OFI ga betinget AUC 0,492097 på 140 651 par innen eksakt samme fysiske M5-OFI. Samtidige dag-/ukebånd var [0,472665, 0,511530] og [0,471310, 0,512884]; begge omfatter 0,5 for alle 60 horisonter. Dette finner ikke en stabil positiv rangering for denne ene fremskutte M1-skåren. Det tester ikke en trent M1-modell, andre M1-sammensetninger eller alle modellinputs, og gjenbrukt utvikling er ikke TEST. Original terminal, hasher og separat lagret aggregatkontroll er bevart. Ingen modellforward, trening eller handel. Se NEXT_RUN_POLICY.

29.10 — M1 GITT SAMME M5: Den forhåndsdefinerte CPU-retningstesten er åpnet én gang etter brukerens M1/M5-bestilling. På gjenbrukt Q1 2025 sammenlignes siste avsluttede M1-OFI innen identisk M5-OFI mot korrigert 1–60-minuttersfasit, med blokkbånd per dag/uke. Bare komplett M1-dekning inngår; dette er ikke en trent modell eller økonomisk edge. All trening, TEST og handel står stengt. Se NEXT_RUN_POLICY.

29.10 — BEGGE MATCHEDE M1/M5-KONTROLLER FULLFØRT OG KONSUMERT: Begge varianter fikk gradientkontakt for RSI14, MACD-linje/histogram og ret_1 samt M1-verdiprojeksjonen i begge andre steg, alle 60 retningshoder hadde gradient på de faste batchene, og gjenopptak var byteidentisk. Kontrollvarianten gjentar M1-gjennomsnittet; separat minuttinformasjon måles først ved senere fit. Uavhengig sammenligning fant identiske 256 foreldre, klokker og 256×60 startlogits, samme 32 faste TRAIN-foreldre og samme 60 TRAIN-priorer. Dette er teknisk kvalifisering, ikke retningsscore eller edge. Minuttarmen brukte fysisk boot 577; signert vakt PASS, maks faktisk GPU-trekk 127,55 W, tilbake til 160 W-grense og 25,21 W tomgangstrekk. Begge engangskjøringene er stengt; ingen full fit, TEST eller handel. Se NEXT_RUN_POLICY.

29.10 — FØRSTE M1/M5-ARM FULLFØRT: Kontrollvarianten med M1-gjennomsnitt og M5-ordreflyt bestod på 256 initialforeldre og 32 faste TRAIN-foreldre, med fire private oppdateringer og byteidentisk gjenopptak. Alle 60 målhoder hadde gradient på hver batch; RSI, MACD og ret_1 hadde kontakt, og M1-verdiprojeksjonen hadde gradient i begge andre steg. Dette er modellmekanikk, ikke målt retningsedge. Fysisk boot 576, signert vakt PASS, maks faktisk GPU-trekk 131,15 W, tilbake til 160 W-grense og 23,47 W tomgangstrekk. Første arm er konsumert; den matchede M1-minuttarmen er åpnet for én tilsvarende kontroll etter ny fysisk omstart. Se NEXT_RUN_POLICY.

29.10 — M1/M5 KVALIFISERING AUTORISERT AV BRUKER: Begge matchede oppskrifter og fysiske SHA-er på de opprinnelige og korrigerte målarkivene er kontrollert. Kun kontrollvarianten er åpnet for kampanjeforberedelse og én privat kjøring på inntil fire fysiske oppdateringer; minuttvarianten står stengt til kontrollvarianten er fullført og gjennomgått. Fysisk omstart før hvert vindu, 250 W-grense under kjøring og verifisert 160 W etterpå. Ingen full fit, TEST eller handel i disse kvalifiseringene. Se NEXT_RUN_POLICY for eksakte porter.

29.10 — BPS OG EKSTRA TRENING: KILDEVERSJONER AVGRENSER ÅRSAK: Originalens og begge kandidatenes faktiske source-head stemmer med oppskriftenes hasher, men 13 felles kildeeiere har endret hash i hver kandidat. Endringene omfatter både utvidet arbeiderrute og kontrollkode og beviser ikke i seg selv feil. Sju utvalgte funksjoner er AST-identiske, blant dem retnings-BCE, batch- og GC-historikk-gather; fire andre er endret. Bps-forsøket har identiske 256 startprediksjoner og samme talloppskrift utenom målmodus/-identitet. Trepass-forsøket startet fra originalens slutt-checkpoint; full state og 256 TRAIN-prediksjoner matchet før videreføring, og nullstegskontrollens 51 kildehasher matchet senere kjøring. Dette kvalifiserer ikke all felles numerikk på alle rader/steg. De tidligere parpunktene er gyldige for de lagrede modellene, men kan ikke tilskrives bare bps eller treningsmengde med full sikkerhet. Korrigert evaluering av faste gamle prediksjoner er ikke retrening på korrigert fasit. Ingen ny markeds-/prediksjonspayload, modellforward, trening, TEST eller handel. Se NEXT_RUN_POLICY.

29.10 — GC-FERSKHET: MODELLIDENTITET OG FAKTISK GRADIENTKONTAKT: Den tidligere fersk-mot-forrige-GC-målingen brukte 60-hodersmodellen trent på original fasit, men skåret den mot korrigert desimalfasit. Den målte derfor verken modellen retrent på korrigert fasit eller den ennå utrente ruten med separat M1 og kompakt RSI/MACD. På 1 747 gjenbrukte Q2-rader var 5min-AUC 0,490082 med fersk og 0,499387 med erstattet/skjult GC; parrede dag-/ukebånd omfatter null. Begge traff 859 av de 1 747 radene ved 0,5-grensen. Den lagrede oppsummeringen oppgir 86 endrede retninger, men definerer ikke eksplisitt om disse er avgrenset til de gyldige radene; ingen parvis treffdeling er derfor utledet. Dette er eksisterende aggregat, ikke ny skåring. Uavhengig kontroll gjenskapte poeng og intervaller, men inngrepet kan gi unaturlige inputs og beviser ikke fravær av GC-informasjon. I den separat korrigerte 60-hoderskjøringen har de lagrede første og siste TRAIN-batchene ikke-null parametergradienter i både GC-projeksjons- og GC-gategrupper på M5, M15, H1, H4 og D1. Dette er summer på tvers av spesialister/bias under felles 60-hoders tap, ikke en test av hver indikator eller av isolert 5min-effekt. Begge modelloppskrifter, målidentiteter, faktisk tapsarbeider og aggregater er hashverifisert; ingen ny markeds-/prediksjonspayload, modellforward, trening, TEST eller handel. Se NEXT_RUN_POLICY.

29.10 — NOMINELL MINUTTHORISONT OG KILDEKLOKKE: Den faktiske korrigerte målbyggeren bruker bid/ask-åpning ved M5-beslutningens M1-lysstart og ved lysstarten n minutter senere. terminal_time_ns er lysstart, ikke et lagret tidspunkt for selve kursobservasjonen. Dersom lysene er usmoothede, åpningsquoten ligger i [lysstart, lysstart+60 sekunder) og hvert bid/ask-par er synkront, kan faktisk tid mellom de to kursene for nominell 1/5/60min ligge innen henholdsvis (0,2)/(4,6)/(59,61) minutter; dette er teoretiske grenser, ikke målte forsinkelser. Historisk smooth-default, synkronitet og faktisk quote-/ankomsttid er fortsatt ukjent. Streng cutoff utelukker i dette betingede tilfellet første observasjon etter en minuttjustert TRAIN-grense. Oppslaget krever eksakt start- og sluttlys uten fallback, men kontrollerer ikke mellomliggende lys; ingen faktisk gapfrekvens er målt, og sluttpunktsmål trenger ikke i seg selv en full prisbane. Hashbundne produsent-/målkilder og syntetiske grense- og missing-tilfeller er kontrollert. Ingen ny markedspayload, modellforward, trening, TEST eller handel. Se NEXT_RUN_POLICY.

29.10 — TRAIN-BLANDING KONTROLLERT MOT FAKTISK KJØRING: Originalens hashbundne kilde permuterte bare de 47 602 kronologiske TRAIN-foreldrene med fast seed; senere foreldre startet etter TRAIN-grensen, og tapsmasken godtok bare observerte ikke-null-fasiter med slutt før grensen. Dette beviser ikke alle andre feature-/tidskausalitetsforhold. Den allerede fullførte parvise 1/5/15-sammenligningen ga på 17 197 gjenbrukte Q1 5min-rader AUC 0,503021 blandet og 0,498501 kronologisk; dag-/ukebånd for forskjellen omfatter null. Samme oppskrift, input og 256 startprediksjoner, men 13 felles kildefiler var endret, full post-update-paritet ikke målt og siste B2-gruppering ulik; forskjellen kan derfor ikke tilskrives rekkefølgen alene. På disse faste skårene kan korrigert desimalfasit fjerne 2 OPP- og 3 NED-rader uten fortegnsskift eller nye ikke-null-rader; konservativ pargrense gir kronologisk-minus-blandet AUC høyst -0,003937. Dette er en matematisk grense, ikke nye skårer eller konfidensintervaller, og gjelder ikke ny trening av korrigert 60-hodersmodell. Status for det konsumerte løpet er rettet til at parrapporten er fullført; originale kvitteringer er uendret. Ingen ny markeds-/prediksjonspayload, modellforward, trening, TEST eller handel. Se NEXT_RUN_POLICY.

29.10 — RSI/MACD-GRADIENTSPERRE RETTET: En syntetisk kvalifisering kunne gi PASS uten gradientbevis for noen av de fire kompakte prisfeltene. Etter rettelsen må RSI14, MACD-linje, MACD-histogram og ret_1 hver vise ikke-null gradient i sin delte prisprojeksjon minst én gang i begge de to-stegs forløpene; ellers feiler kontrollen med bevart diagnostikk. Negative tester gjenskapte feilen før rettelsen, og referanse- og gjenopptaksbanene er testet etterpå. 47 fokuserte syntetiske tester bestod; den faktiske native-arkitekturen ga syntetisk gradientkontakt for alle fire. Dette beviser ikke verdi, alle tidsrammer eller retningsedge. Den tidligere ventende forespørselen er erstattet på grunn av nye kilde- og oppskriftshasher. Begge engangsplaner er ubrukte og stengt; oppdatert metadatakjede passerte uten markedsdata, med to målfilhasher utsatt til faktisk kjøring. Ingen trening på markedsdata, TEST eller handel. Se NEXT_RUN_POLICY.

29.10 — KONKRET MANGLENDE M1-GRADIENTSPERRE RETTET: En syntetisk kontroll gjenskapte at privat kvalifisering kunne gi PASS selv når alle fem M1-verdikolonnene manglet gradient. To negative tester feilet før rettelsen for både gjennomsnitts- og minuttvarianten. Etter første nullinitialiserte oppdatering krever nå begge M1-variantene ikke-null gradient i minst én M1-verdikolonne i andre logiske steg, både i referanse- og gjenopptatt bane; ellers feiler kontrollen med bevart diagnostikk. Dette kvalifiserer ikke alle fem minutter eller retningsedge. Forrige ventende forslag er erstattet av nye kilde- og oppskriftshasher; ny uttrykkelig godkjenning kreves. Begge engangsplaner er stengt og ubrukte, full fit, TEST og handel er stengt. 41 fokuserte og 59 kampanjevalidator-tester bestod; sperren er også kontrollert mot faktiske gradienttensorer i en syntetisk native-modell. Den oppdaterte metadatakjeden passerte for begge; to målfilhasher ble utsatt og må verifiseres ved faktisk kjøring. Se NEXT_RUN_POLICY.

29.10 — 5MIN FORTEGNSKJEDE OG REN SNUIING: Kildeversjonen i den korrigerte treningen er hashverifisert: positiv desimalmidtprisendring gir sign +1, 5min bruker ankerhodets rad 1 i outputkolonne 4, BCE bruker +1 som positiv klasse, og rapporten sier OPP for positiv logit. Ingen enkel fortegnsinversjon er funnet i denne kjeden. På 17 192 gjenbrukte Q1-rader uten null-logit traff modellen 8 356 (48,604 %); å snu alle faste prediksjoner ville gitt 8 836 (51,396 %), mot 8 868 (51,582 %) for alltid OPP. AUC snus matematisk fra 0,496976 til 0,503024, og eksisterende samtidige dag-/ukeintervaller transformert som [1-high,1-low] inkluderer fortsatt 0,5. Etteranalyse av faste prediksjoner, ikke ny bootstrap eller bevis for handelsedge; andre feil og mål-/featureverdier er fortsatt åpne. Ingen ny markeds-/prediksjonspayload, modellforward, optimizersteg eller TEST. Se NEXT_RUN_POLICY.

29.10 — RETTET STATUS, HELE TRAIN 1 MOT 3 PASS: Forrige status var misvisende: den separate, hashverifiserte parrapporten sammenlignet alle 47 602 TRAIN-foreldre, hvorav 47 308 hadde gyldig ikke-null 5min-fasit. Samme trehodersmodell ga TRAIN-AUC 0,515888 etter én pass og 0,533534 etter tre; på 17 197 senere Q1-rader var punktene 0,503021 og 0,500950, med dag-/ukeintervaller for differansen som inkluderer null. Dette er bedre tilpasning på TRAIN uten dokumentert overføring til senere data, ikke sikker årsak. For bps-varianten gjelder 256 TRAIN-diagnoserader i den sammenligningen. Fasitrettingen fjerner høyst 12 OPP/14 NED på TRAIN og 2 OPP/3 NED i senere fold; ved faste prediksjoner og konservativ pargrense er TRAIN-forskjellen minst +0,016563 AUC, mens senere 3minus1 er høyst -0,001486 og bps-minus-sign høyst -0,005616. Grensene er ikke nye skårer eller konfidensintervaller og sier ikke hva ny trening av 60 hoder ville gitt. Ingen ny markeds-/prediksjonspayload, modellforward, trening eller TEST. Se NEXT_RUN_POLICY.

29.10 — M1-OFI OG FASITRETTING: På samme 16 977 gyldige Q1-rader ga fortegnsforfining med siste M1 AUC-differanse -0,003156 mot M5, og utforskende kontinuerlig femdobling -0,006309; eksisterende dag-/ukeintervaller inkluderer null. Den eksakte fasitrettingen gjør høyst 2 OPP- og 3 NED-rader til null i hele folden, uten nye ikke-null eller fortegnsskifter. Med faste M1-/M5-skårer og verste mulig overlapp kan høyst 0,0594 % av opp/ned-parene bortfalle; øvre grense for korrigert AUC-differanse er henholdsvis -0,002564 og -0,005720. Dette er matematiske grenser, ikke nye skårer eller intervaller, og avviser ikke andre M1-uttrykk eller modellintervensjon. Kildebunden M1-reduksjon grupperer på Databentos ts_recv; leverandøren definerer dette som mottak på sin capture-server, ikke vår maskin. Derfor beviser den nominelle tilgjengeligheten ved M5-close ikke fysisk ankomst innen da; faktisk forsinkelse og effekt er ikke målt. Ingen rå markeds-/prediksjonspayload, ny modellforward, optimizersteg, TEST eller handel. Se NEXT_RUN_POLICY.

29.10 — FASITENS LYSKLOKKE OG AVGRENSNING AV MIDTPRISFEIL: Hashbundne kontroller bruker samme foreldre/kilder; originalmålene har 1/5/15-byteparitet, korrigert mål bevarer klokker/masker, og begge tegnmatrisene har bare ikke-null-til-null-endringer mot originalen. De separate aggregatene gir maksimalt 112/80/36 mulige tegnforskjeller mellom native M1-midtlys og korrigert bid/ask-midt for henholdsvis 1/5/15 minutter, uten direkte OPP/NED-flipp; eksakt radvis overlapp er ukjent. For gjenbrukt Q1 5min kan høyst 40+5=45 rader skifte inkludering. Med uendrede lagrede modellprediksjoner var modellen 512 treff bak alltid OPP på 17 192 rader; en konservativ verstefallsgrense gir fortsatt minst 2,709 prosentpoeng bak. Dette gjelder ikke ny modellfit, AUC, andre minutter eller faktiske quote-tider. OANDA angir lysstart i candle-time; usmoothet open er første pris innenfor intervallet, men individuelle pristider er ikke i lyskontrakten. Den hashbundne historiske produsenten satte ikke smooth eksplisitt; effektiv historisk default er ikke fastslått. Alder/forsinkelse og reell tick-horisont er ukjent og må ikke sluttes fra lysklokken. Ingen nye rå markeds-/prediksjonspayload, modellforward, trening, TEST eller handel. Se NEXT_RUN_POLICY.

29.10 — FAKTISK 5MIN-GRADIENT OG SISTE BATCH: I den korrigerte 60-hoderskjøringen var 5min-hodet i AdamW-gruppen. Både første og siste lagrede batch hadde ikke-null gradient på alle 128 hodevekter og på bias; dette viser kontakt i disse to batchene, ikke at signalet var sterkt eller nyttig. 47 602 foreldre ga 2 976 steg med batchstørrelse 16; siste steg brukte bare to foreldre og 120 gyldige målceller, mot 959 i første steg. Ved lik logit-residual er koeffisienten per målcelle 959/120 = 7,99 ganger større i siste tapsgjennomsnitt; dette er ikke forholdet mellom optimizeroppdateringer eller bevis for skjevhet. Begge de to målte gradientnormene var under klippgrensen 1,0. Evalueringen brukte sluttmodellen etter siste steg uten vektsnitt; effekten av den lille sluttbatchen på resultatet er ikke målt. Kilder er verifisert mot de faktiske treningshashene og eksisterende aggregatkvittering. Ingen ny modellforward, treningssteg, markeds-/prediksjonspayload eller TEST. Se NEXT_RUN_POLICY.

29.10 — 5MIN-EVALUERING MOT 60-HODERS TRENINGSTAP: Den korrigerte modellens hashbundne tapsfunksjon gir alle observerte, modne ikke-null minuttutfall lik BCE-vekt innen hver batch, mens den faste evalueringens primærhorisont er fem minutter. Av 2 792 676 gyldige TRAIN-mål over 1–60 minutter var 47 282 femminuttersmål, 1,693 % ved samlet celletelling; 1–15 minutter utgjorde 25,39 % og 16–60 minutter 74,61 %. I første og siste lagrede treningsbatch var femminuttersandelen henholdsvis 16/959 og 2/120. Disse andelene er ikke målt gradientbidrag eller eksakt samlet per-oppdateringsvekt, og de beviser ikke årsaken til svak retning. Dette er et konkret mål-/prioriteringsspor: behold alle 60 hoder, og krev en separat forhåndsbestemt, matchet test før eventuell endring av tapet. Ingen ny modellforward, payload-lesing, fit eller TEST; kilde og aggregater er hashkontrollert.

29.10 — M1/M5 ORDREFLYT, PRESIST OMFANG: Den hashbundne M5-layouten og kompilert kandidatprofil ruter kjente kjøp, kjente salg, ukjent volum, kjent handelsdelta og bok-OFI med egne masker til momentum_flow_encoder. Dette er kilde-/profilkontroll, ikke ny måling av verdi eller læring. Den nye separate M1-modellruten leverer derimot bare fem bok-OFI-verdier og masker. Den tidligere hashbundne M1-reduksjonens eier skriver også buy/sell/unknown/volume, men disse er ikke med som separate minuttserier i kandidatens M1-input. Den foreslåtte testen avgjør dermed ikke verdien av M1-handelsflyt; maskekvalitet, faktisk modellkonsum og tilført verdi for slike minuttserier er ikke kvalifisert. I reduksjonens hele kalendergrid hadde 589 198 minutter gyldig volum, hvorav 272 174 (46,2 %) hadde noe ukjent aggressorvolum og dermed ikke bestod den strenge signed_valid-masken. Dette er ikke dekning på modellens beslutningsrader. M5-eieren bevarer separat kjente kjøp/salg og ukjent volum innen gyldig observert volum; en eventuell M1-utvidelse må ikke gjette ukjent side eller forveksle delvis kjent med full signert flyt. Ingen ny markeds-payload, modellforward, fit eller TEST.

29.10 — FAKTISK METADATAKJEDE FOR M1-KANDIDATER KONTROLLERT: Begge de foreslåtte kvalifiseringene passerte sine faktiske kilde-/metadata- og målreferansekoblinger med 59 metadatareferanser hver. De to kontrollene av originalt og korrigert målarkivs filhash var eksplisitt stubbet: ingen markeds-/mål-payload ble åpnet, og faktisk kjøring må verifisere dem selv. En innledende testvakt avviste Python sitt forsøk på å åpne .pyc ved dynamisk import; dette ble gjenskapt, og begge aktuelle cachefiler var fraværende. Kontroll med eget tomt pycache-prefiks passerte uten modellimport, GPU eller TEST. Ingen claim, genuin modellforward eller oppdatering; begge planer er fortsatt ikke autorisert.

29.10 — ALLE 60 KORRIGERTE MINUTTHODER: To eksisterende hashbundne aggregatrapporter stemmer eksakt på AUC og gyldige rader for alle 60 horisonter. I én gjenbrukt Q1-fold var punkt-AUC over 0,5 for 54 av 60 hoder, median 0,51285; for fem minutter 0,49698 og for 60 minutter 0,51895. Ingen av de 60 samtidige dag-/ukeintervallene hadde nedre grense over 0,5; for 60 minutter var nedre grensene 0,48962/0,49126. Utfallene og horisontene overlapper, radantallet varierer fra 16 512 til 17 253, og 54 hoder er ikke 54 uavhengige forsøk. Dette er ikke dokumentert retningsedge. Ingen ny bootstrap, modellforward, fit, prediction-payload-lesing eller TEST.

29.10 — KORRIGERT 5MIN-MODELL ER SKJEV MOT NED: Det eksisterende hashbundne OOF-resultatet viser at den sist trente 60-hodersmodellen predikerte NED på 13 942 av 17 192 modne, observerte ikke-null rader i gjenbrukt Q1 2025, mens 8 868 faktisk gikk OPP. 7 227 av de faktiske OPP-utfallene ble klassifisert NED. Alle prognosesannsynligheter lå mellom 0,4 og 0,6; de 13 942 under 0,5 hadde snitt 0,4862 OPP mens faktisk OPP-andel var 0,5184. AUC var 0,496976 og treff 48,604 %. På de samme 256 faste TRAIN-diagnoseradene klassifiserte startmodellen 255 NED og den ferdig trente 197 NED; de faktiske utfallene var 128 OPP og 128 NED. TRAIN-utvalgets logloss falt fra 0,784835 til 0,692311 og AUC steg fra 0,440979 til 0,545410, men utvalget er lite og innad i TRAIN. Den kildehashbundne 5min-utgangen ble initialisert med seedet torch.nn.Linear-standard, ikke fra TRAIN-prioren på 0,5091 OPP. Det er et mulig spor, ikke bevis for at initialiseringen forårsaket OOF-skjevheten eller at en annen initialisering vil hjelpe. Dette er observert skjevhet og svak rangering for én gjenbrukt utviklingsfold, ikke bevis for hvorfor den oppsto eller for en handelsregel. Ingen ny modellforward, fit, prediction-payload-lesing, TEST eller handel; kildekvittering står i NEXT_RUN_POLICY.

29.10 — BPS OG TRENINGSMENGDE AVKLART SOM SEPARATE TIDLIGERE FORSØK: På 17 197 samme modne, observerte ikke-null 5min-rader i gjenbrukt Q1 2025 ga signert bps-MSE i tillegg til original retnings-BCE AUC 0,496823 mot 0,503021 for retningsmodellen alene. En separat sammenligning med samme start og retningsmål ga AUC 0,503021 etter én TRAIN-gjennomgang mot 0,500950 etter tre. Dag-/ukebaserte parintervaller for begge AUC-forskjellene inkluderer null. Ingen av forsøkene dokumenterte retningsbedring. De brukte den eldre trehodersfasiten før desimalrettingen, ikke senere korrigert 60-hodersmål; én gjenbrukt fold er ingen generell dom. For den separate bps-sammenligningen ble TRAIN målt på 256 diagnoseforeldre; trepass mot én pass er derimot sammenlignet på hele TRAIN-prefikset som beskrevet over. Tidligere status som sa at OOF-parevaluering manglet er rettet mot eksisterende hashbundne resultatreferanser i NEXT_RUN_POLICY. Syv filer i den gamle kvalifiseringens kilde-/testliste har siden endret seg; dens gamle preflight kan derfor ikke kvalifisere dagens kilde. Ingen ny kjøring eller TEST.

29.10 — KAPASITETSMATCHET M1-KONTROLL FORBEREDT, IKKE AUTORISERT: Tidligere pris+M5 mot M1 hadde ulik kapasitet fordi M1 la til et trenbart ledd som også så modellens eksisterende representasjon. To private kandidater har nå samme ledd, ekte M1-input, masker, korrigerte mål, TRAIN-normalisering, seed og 62 kildeeiere. Kontrollvarianten bruker observert gjennomsnitt/sum inne i modellen; den andre beholder fem enkeltverdier. Forskjell omfatter minuttvariasjon og rekkefølge samlet, ikke maskenes effekt. En gammel konstruktørhash i overgangsbeviset ville stoppet ekte kjøring; ny kildeovergang er bundet og ikke et resultatbevis. 296 fokuserte syntetiske tester og 59 tester av siste parvalidator bestod. De to faktiske arkitekturene hadde identiske startparametere unntatt representasjonsbuffer og identiske 60 startlogits på én syntetisk rad. To private firestegskampanjer med 256 initialprediksjoner hver er foreslått i docs/DIRECTION_M1_STRUCTURE_PRIVATE_QUALIFICATION_20261029.md; begge er stengt og trenger uttrykkelig nytt vedtak. Ingen genuin forward, trening, TEST eller handelseffekt.

29.10 — M1-RAPPORTKOBLING RETTET: Den nye kandidatens evalueringsrapport avviste 16 input fordi den strengt krever 14; en syntetisk test gjenskapte stopp før første modellprediksjon. Rapporten får nå kun sine 14 felt mens modellen fortsatt får alle M1-verdier og masker uendret. Manglende M1-felt og uventede ekstrafelt avvises fortsatt. 98 fokuserte tester bestod. En ekstra ende-til-ende-kontroll på én syntetisk rad førte deretter M1 og kompakt momentum gjennom den faktiske modellarkitekturen, den strenge rapporten og alle 60 prediksjonshoder med identiske resultater med/uten gradienter. Dette gjelder den nye kandidaten, ikke tidligere 50/50-resultater. Ingen genuin kandidatforward eller trening.

29.10 — KORRIGERT 60-HODERS SAMMENLIGNING FULLFØRT OG KONSUMERT: 47 602 TRAIN-foreldre, 2 976 oppdateringer og 17 288 senere foreldre; signert vakt PASS, kilde uendret, fysisk restart før kjøring og tomgangsgrense gjenopprettet til 160 W. På 17 192 samme modne observerte 5min-utfall var AUC ny 0,496976 mot gammel 0,500175, treff 48,604 % mot 48,877 %, og alltid OPP 51,582 %. Parrede dag- og ukeintervaller for AUC-forskjell inkluderer null; ingen av de 60 samtidige dag-/ukeintervallene dokumenterte AUC over 0,5. Ny logloss 0,694379 er svakere enn egen TRAIN-prior 0,692738. Uavhengig rangregning gjenskapte alle 120 modell-/horisontpunkter; bootstrapintervallene er ikke uavhengig gjentatt. Q1 2025 er gjenbrukt utvikling, ikke urørt bekreftelse eller økonomi. Ingen retningseffekt av fasitreparasjonen er dokumentert. Trening, TEST og handel er stengt. Resultat- og kvitteringsreferanser står i NEXT_RUN_POLICY.

29.10 — M1/M5 OG KOMPAKT MOMENTUM: På alle 47 602 TRAIN-beslutninger stemte M1-summer for OFI, kjent kjøp/salg og ukjent volum med M5-kilden; 287 utvalgte faktiske modellkontekstrader stemte også. Den tidligere trente modellen fikk ikke fem separat avsluttede M1-OFI-verdier som egen inputrute. En forskningsrute for fem kausale M1-OFI-verdier med egen observeringsmaske er koblet til retningsmodellen og kilde-/klokke-/TRAIN-normalisering bundet i syntetiske tester; den faktiske M5-kontraktmasken og rullplanen er kontrollert på 64 890 faste foreldre som beskrevet under; reell treningsadmisjon gjenstår. Midnattsmasken sperrer 00:01 etter flagget 00:00; åtte råprøver hadde ren siste boktilstand og gjenskapt OFI, men ingen regel er endret. Filteret F_LAST utelot 8,3 % meldinger i kvalitetsgodkjente TRAIN-minutter; i 18 avgrensede råminutter var 648 utelatte meldinger handler uten endret BBO og OFI var lik. Utvalget beviser ikke alle meldingene. RSI, MACD og ret_1 er parkert i XAU-MTF-profilen; uavhengig prisberegning stemmer på et TRAIN-utvalg i alle fem tidsrammer. Den konkrete opt-in profilen for RSI14, MACD-linje/-histogram og ret_1 er koblet til forskningsmodellen, men ikke trent eller målt på faktiske markedsdata. Ingen målbar prediktiv eller økonomisk edge følger av disse inputkontrollene. Se auditfeltene i NEXT_RUN_POLICY.

29.10 — NY KANDIDATKVALIFISERING FORBEREDT, IKKE ADMITTERT: Intern dispatch peker nå begge matchede RSI/MACD-/M5- og RSI/MACD-/M5+M1-varianter til 60-hoderskvalifiseringen. Kandidaten må måle sin egen ferske 256-raders initialtilstand, kontrollere korrigert TRAIN-prior og gjenoppta eksakt; den krever ikke den tidligere modellens baseline som sin egen. Testene demonstrerer kontrollflyten med falsk modell/optimizer, fire fysiske syntetiske steg og egen resultatidentitet. 212 fokuserte tester bestod i den tidligere bølgen, da begge kampanjevalidatorer avviste kandidat-scope; privat-kvalifiseringsvalidatoren er senere avgrenset som beskrevet over, mens full-fit-validatoren fortsatt avviser scopet. Ingen genuin kandidatforward, GPU-kjøring, fit, TEST eller handel. Eksakt normaliseringsresultat er nå fullført som beskrevet under; ny kampanje/admisjon og reell initial-/gradient-/resume-kvalifisering mangler. Planene er senere oppdatert med gradientsperren beskrevet øverst.

29.10 — KONKRET M1-CHECKPOINTFEIL RETTET: I den nye, ennå ikke genuint trente M1-ruten ble endret center, scale eller normaliseringsidentitet tidligere først avvist etter innlasting; tre syntetiske negative tester gjenskapte at modellens tilstand da allerede var endret. De tre bufferne kontrolleres nå som immutable før noen restore-mutasjon. Etter rettelsen avvises alle tre før endring, korrekt restore fungerer, og to syntetiske M1-oppdateringer er identiske med avbrudd og gjenopptak inkludert optimizer/RNG. 116 fokuserte tester bestod. Feilen er ikke påvist i tidligere modeller uten M1 og forklarer ikke deres 50/50-resultat. Ingen ekte M1-fit, trening, TEST eller handel. Planene er senere oppdatert med gradientsperren beskrevet øverst.

29.10 — LÆRINGSKONTAKT PER INDIKATOR OG MINUTT: Den private gradientrapporten viser nå separate verdier for hver av fem M1-OFI-innganger og fem observeringsmasker. For RSI14, MACD-linje, MACD-histogram og ret_1 vises hver kolonne i den delte prisprojeksjonen og hver indikator-/tidsramme-gate på M5/M15/H1/H4/D1. Null rapporteres ærlig; dette er gradientkontakt, ikke bevist kausal effekt eller edge. Kontrollene sammenligner tallene mot faktiske gradienttensorer i en syntetisk instans av native-modellen etter én syntetisk oppdatering. 212 fokuserte tester bestod; ingen genuin kandidatforward, trening eller normaliseringsfit er kjørt. Planene er senere oppdatert med gradientsperren beskrevet øverst.

29.10 — M1 TRAIN-NORMALISERING FULLFØRT OG KONSUMERT: Én godkjent CPU-kjøring med 4 GB-grense gjennomførte én fit på 234 875 observerte M1-OFI-minutter fra 47 602 TRAIN-foreldre; 3 135 av 238 010 mulige minutter var manglende eller maskert. Fasitverdier ble ikke brukt. Modellforward og optimizersteg var begge null, og kilde/input var uendret. Normaliseringskontrakten er verifisert mot source-, klokke- og rullhasher. Kvittering, resultat og normaliseringsreferanse står i NEXT_RUN_POLICY; engangsadgangen er lukket og kan ikke relanseres. Fersk kvalifisering og separat godkjenning trengs før M1/M5-modellsammenligning. TEST og handel er stengt. Dette viser ikke prediktiv edge.

29.10 — KILDEBUNDET M1-INNLASTING KLARGJORT: Den interne kandidatklargjøringen kan nå laste en eksakt bundet M1-NPZ, rullplan og eksisterende normalisering etter installert lesevakt. Den kontrollerer SHA/stat før og etter lesing, skrivevern, source-/rull-/TRAIN-prefiksbinding, modellens M5-kontraktklokke og senere freeze; ukjente filer og endrede kilder avvises. En syntetisk filbasert test går helt gjennom _prepare til modellens featureview. 201 fokuserte tester bestod, inkludert gamle ruter og begge offentlige kampanjers avvisning av privat scope. Ingen reelle M1-input, normaliseringsfit, modellforward, trening eller TEST ble brukt i denne bølgen. Genuin source-admisjon, egen TRAIN-normalisering og fersk initial-/gradient-/resume-kvalifisering mangler; dette er ikke målt prediktiv edge. Den ventende betingede M1-auditens plan og 15 kildehasher er uendret.

29.10 — KANDIDATENS TRAIN-REKKEFØLGE RETTET FØR BRUK: Den nye private adapteren ville ellers alltid falt tilbake til fast blanding, selv om en fremtidig plan ba om kronologisk rekkefølge. Den krever nå et eksplisitt valg mellom kronologisk prefiks og den tidligere faste TRAIN-permutasjonen; ugyldig/manglende valg avvises. Begge sammenligningsarmene må binde samme valg. Fem målrettede syntetiske tester bestod; tidligere faktiske kjøringer eller resultater er ikke endret. Blanding innen TRAIN er i seg selv ikke påvist datalekkasje. Ingen faktisk fit eller modellforward.

29.10 — SYNTETISK M1/M5- OG MOMENTUMKLARGJØRING: Den interne _prepare-adapteren kan nå teste to matchede forskningsvarianter med korrigert fasit: kompakt RSI/MACD/ret_1 pluss eksisterende M5-ordreflyt, og samme oppsett med alle fem avsluttede M1-OFI-verdier/masker i tillegg. Den binder syntetisk M1-kilde, modellens M5-klokke, kontrakt, rullbytes og TRAIN-prefiksnormalisering. 184 fokuserte tester bestod, inkludert faktisk _prepare-ruting for begge varianter og avvisning i begge offentlige kampanjevalidatorer. Dette er kun syntetiske tester: kildesikret innlasting er nå også syntetisk kontrollert som nevnt over, mens genuin adgang, separat godkjent normalisering, faktisk initial-/gradient-/resume-kvalifisering og egen trening mangler. Ingen faktisk modellforward, fit eller effektmåling er gjennomført. Den forhåndsregistrerte betingede M1-auditens kildehasher er uendret.

29.10 — SYNLIGE KANDIDATGRADIENTER: Eksisterende private gradientrapport har en avgrenset utvidelse som rapporterer M1-inngang/-utgang og XAU-MTF-momentumprojeksjon separat for den nye forskningsprofilen. Rapporten finner faktisk antall ikke-null-gradienter; den krever eller antar ikke at første M1-inngangsgradient er ikke-null. Syntetisk test viser forventet null først ved nullinitialisert utgang og ikke-null etter en oppdatering; dette er ikke genuin læring. 75 fokuserte tester bestod. Admittert genuin lasting, initial-/gradient-/resume-kvalifisering og egen autorisert trening mangler fortsatt. Den ventende M1-informasjonsplanens kildehasher er uendret.

29.10 — FORBEREDT, IKKE AUTORISERT M1-MERINFORMASJONSAUDIT: En engangsplan er klar for å undersøke om siste lukkede M1-OFI rangerer OPP/NED innen rader med eksakt samme fysiske M5-OFI. Fast Q1 2025-utviklingsfold, 5min primær og alle 60 horisonter med samtidige dag-/ukeblokkbånd; 10 000 faste bootstrap-trekk per ordning. Bare observerte alle-fem-M1, modne og ikke-null mål inkluderes; manglende input og gruppestøtte rapporteres. Dette kontrollerer kun M5-OFI-summen, ikke alle modellens inputs, og gjenbrukt utvikling kan ikke bevise handelsedge. Planen gir ingen fit, model forward, optimizer, TEST eller handel. 14 syntetiske, fokuserte tester bestod. En syntetisk fullskala-timing på 17 288 foreldre, 60 horisonter, 10 000 trekk og 77 dag-/12 ukeblokker tok 27,63 sekunder uten profileringsvakt; dette er ikke genuin ETA eller markedsevidens. Ved forberedelsen ble målarkivets medlemmer lastet kun for form/dtype før planen ble låst; ingen målverdier eller statistikk ble vist/analysert. Kilde og plan er bundet, men execution_authorized=false. En capped forhåndskontroll på ren d22858b3 stoppet ved ONE_SHOT_AUTHORITY før payloadåpning; ingen outputkatalog eller claim ble opprettet. Nytt særskilt brukervedtak kreves før faktisk engangsmåling; ikke oppdater status til ADMITTED uten det. Se NEXT_RUN_POLICY for eksakt plan.

29.10 — MODELLENS FAKTISKE M5/M1-KLOKKE: En ny, separat avgrenset audit koblet 47 602 TRAIN- og 17 288 gjenbrukte Q1-utviklingsforeldre eksakt til native M5-cache og modellens hashverifiserte GC-sidecar. På alle 64 890 stemte rullindeks for hvert M1-minutt, modellens M5-kontraktflagg, OFI-maske/verdi, antall og dekningsandel; ingen avvik. Det faktiske GC-feltets metadata peker til trusted-restoration024 og samme rollhash som M1-reduksjonen. 024-kildens rå M5-tabell ble ikke åpnet/rehashert; dens kildebinding og stabile stat er kontrollert mot tidligere hashbundne kvitteringer. Den nye modellbundne clock-eieren er deretter kvalifisert på samme reelle inputrader: TRAIN 46 756 alle fem minutter, 376 delvis og 470 ingen; Q1 17 135 alle, 148 delvis og 5 ingen. Tre TRAIN- og to Q1-bøtter blandet kontrakter; 276 TRAIN-bøtter hadde ukjent kontrakt. Alle slike er maskert, uten observerte feilkontrakter. Eksakte rullplanbytes er bundet til clock/normaliseringskontrakt; feil M5-rad, kontraktmaske eller rulltilknytning feiler lukket. 45 fokuserte tester og deretter 23 etter ekstra negative bindingstester bestod. Første forsøk på den nye clock-auditen feilet før dataåpning med manglende importsti; 002 kvalifiserte forrige kodeversjon og 003 endelig kilde. Ingen ny normalisering, ekte modellforward, trening, TEST eller handel. Ingen admittert provider aktiverer den nye profilen eller M1-kilden på faktiske data; reell baseline/gradient/resume og separate autoriserte sammenligninger gjenstår. Se NEXT_RUN_POLICY for eksakte kilde-/resultathasher.

29.10 — FULL AKSE TIL MODELLBUNDET M1-RUTE: Den samme kilde-/modellklokke-eieren er kvalifisert på alle 118 249 foreldre som dagens featureview krever, til og med desember 2025. Kilde, rullplan og fysiske M5-inputbytes ble hashverifisert; ingen målverdier, modellevaluering, normalisering, trening eller TEST ble åpnet. 116 686 vinduer har alle fem M1-observasjoner, 942 er delvise og 621 mangler alle; åtte bøtter blander kontrakter og 276 har ukjent kontrakt. Ingen observasjon ble tilordnet feil kontrakt, og blandede/ukjente bøtter har null observerte M1-verdier. Deler av 2025 utenfor forrige Q1-sammenligning er kontrollert som input/klokke, ikke målt som prediksjon. Full-akse clock-sha c379c735c82427538aaf1b5cfa2aaae89622c470c177dd45c3f51be4eef395ac. Dette åpner ikke genuine fit: Genuin plan/admisjon må fortsatt opprettes og fersk initialfunksjon må kvalifiseres. Se NEXT_RUN_POLICY for resultat/script.

29.10 — FORSKNINGSRUTER OG KONTRAKTDEKNING: 108 fokuserte tester bestod, inkludert faktisk native-konstruktør, begge nye inputruter sammen i syntetisk modellforward, gradientkontakt, identiske gamle startlogits, null/missing, kausalt femminuttersvindu og blokkert produksjonsprofil. Fem ekstra målrettede tester bestod etter fjerning av ubrukte importer. Ingen ny ekte normalisering, modellforward, vektfit eller prediktiv effekt er målt. Endret ONLINE krever fersk initialbaseline. Et uavhengig gammelt M5-volumkontraktuttrekk har samme indeks som M1 på alle 29 960 TRAIN-/10 572 Q1-rader hvor begge finnes, men mangler 17 642 TRAIN-IDer inkludert 3 634 manglende klokker og 6 716 Q1-IDer inkludert 1 204 manglende klokker. Første audit stoppet på feil antakelse om full klokkedekning; den korrigerte audit rapporterer faktisk dekning. M1-reduksjonens og 006-kildens hashbundne rollplaner har samme 15-symbolrekkefølge og metadata og samme utvalgte symbol, basisdag og rollklokke på alle 791 dager. Dette bekrefter sammenlignet indeks-til-symbol for disse to kildene, men 006-kildens forbindelse til modellens M5-bokkilde er ikke selvstendig bevist. Manglende IDer kan ikke fylles fra 006 og dette er ikke en dokumentert modellfeil. Ingen autorisert fit-provider leverer ennå M1-kilden eller aktiverer momentum/M1-profilen i en genuin kjøring. Bevar alle vakter; separat kildetilknytning, ekte initialkvalifisering og ny eksplisitt autorisasjon for normalisering/trening trengs før et nytt forsøk. Se NEXT_RUN_POLICY for hasher og auditkvittering.

ET252-KOMPAKTINNGANG KILDEAVKLART: Rettelse til tidligere omtale: den ikke-lineære referansen brukte den navngitte kompaktprofilen, ikke et dokumentert bredere indikatorsett. Alle fem hashbundne snapshot-oppsett hadde 3 345 koordinater, ingen navngitt RSI eller MTF-ret_1/ret_20, og én lokal ret_1. Den tidligere relative AUC-forskjellen kan ikke tilskrives RSI; profil, normalisering, historikk og modell er ikke identiske med senere 60-hoder. Femfolds 5min-treff var 50,75 % mot alltid OPP 51,01 %, uten handelsedge. Test den brukerønskede kompakte indikatorutvidelsen etter fasitsammenligningen; den konsumert ET-planen må ikke gjentas. Kun eksisterende metadata/resultater lest, ingen vekter, nye data, fit, TEST eller handel.

PRISBASERT XAU-MTF-MOMENTUM KILDEAVKLART: den brukte 60-hoders kompaktprofilen velger 178 av 253 XAU-felt per tidsramme. Alle 29 momentumfeltene, blant annet ret_1 og RSI, er parkert i XAU-MTF-ruten på M5/M15/H1/H4/D1; dette følger profilvalget. Lokal XAU-ret_1 og GC-momentum beholdes. Eksisterende inventar viser at ret_1 og RSI hadde observert variasjon på alle fem tidsrammer. En tidligere uavhengig numerisk RSI-referanse bestod på samme cache på alle fem; prediktiv verdi og eventuell årsakssammenheng er ikke bevist. Modellens tidligere M5-sluttavkastning er ikke kvalifisert identisk med den utforskende M1-åpningsreferansen. Denne tidligere revisjonen gjorde ingen kildeendring, ny rådata-/featurepayloadlesing, modellforward, trening, TEST eller handel. Se rapporten `docs/INDEPENDENT_DIRECTION_FEATURE_AUDIT_20261028.md`. Den foreslåtte korrigerte ettpasstreningen tester fasiten, ikke denne separate momentumhypotesen.

KORRIGERT PRIVAT FASIT-/MODELLKVALIFISERING FULLFØRT OG KONSUMERT: etter fysisk boot 574 og 250 W kontrollerte uavhengig review alle 256 startforeldre/klokker og 60 logits byteidentisk mot opprinnelig baseline, alle 60 korrigerte TRAIN-priorer, gyldig target og gradient i 60 utganger på alle fire private steg, samt identisk resume. Alle fem GC-TF-projeksjons- og gategrupper hadde gradient; kartlokal projeksjon hadde null første og ikke-null andre steg. Dette beviser ikke per-feature kvalitet, retning eller edge. Signert vakt PASS, målt effekttopp 133,72 W, og idle-grensen ble gjenopprettet til 160 W med faktisk trekk 26,26 W. De seks loggede grenseprøvene var 250 W; den generiske vaktens grense er 300 W og enkeltgrensen for alle 130 prøver er ikke logget. Første stagingforsøk for Windows-config feilet før endring og ble rettet; bevis er bevart. Etter kvalifiseringen ble treningsflaggene lukket; den separat godkjente full-fit-ruten er nå klargjort over. TEST, handel og vektgjenbruk er stengt. Resultat: `docs/DIRECTION_DECIMAL_PRIVATE_QUALIFICATION_RESULT_20261028.json`; uavhengig review SHA-256 `f994b56dc5b5c76cb4edb1e4797a5b3e6556239a99c65c35fd8157fac5e5b81c`.

TIDLIGERE FORBEREDELSE AV INITIALPARITET (NÅ KJØRT OG VERIFISERT OVER): den nye private desimalqualifiseringen krever nå byteidentiske foreldre, bar-/beslutningsklokker og 60 startlogits på de samme 256 radene som tidligere dense-baseline, uten å lese vektene. Den lagrede prediksjonsfilen og dens 256 akser/masker stemmer korrigert fasit; to fortegnsceller skiller seg. Dette er fil-/kildekontroll og syntetiske tester, ikke en ny modellparitet. Selve target-grenen og desimaleieren har uendret AST siden faktisk datakontroll. Design: `/home/andre2/src/GX1_CURRENT/docs/DIRECTION_DECIMAL_INITIAL_PARITY_20261028.json`, SHA-256 `9c8e1499d3add7f2bab6af06df318096685efdd7e3d07a9c3ef23ce781335f8f`. Ingen modellforward, trening, GPU, TEST eller handel.

KORRIGERT PRIVAT TARGET-GREN KJØRT PÅ EKTE MÅLFILER, UTEN TRENING: native `_prepare` brukte den separat autentiserte desimalforbrukeren for 118 249 foreldre og alle 60 horisonter. Originale ankre ble kontrollert mot originalt mål; 976 falske ikke-null-fortegn ble fjernet, og alle 60 modne TRAIN-opp-/nedtellinger stemte uavhengig kildeaudit. Fem minutter: 24 069 OPP og 23 213 NED. Fysisk provider og featureview var stubbet og aksene hentet fra tidligere fysisk verifisert targetarkiv; genuine featuregather, modell, GPU, forward, optimizer og resume er ikke kvalifisert. Torch ble bare importert på CPU fordi den eksisterende måleieren eier horisontkonstanten; CUDA var skjult og ikke initialisert. Forsøk 001 og 002 feilet på auditens import-/scopevakter og er bevart og konsumert. 003-resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_DECIMAL_PRIVATE_INPUT_20261028_003/OUTPUT/RESULT.json`, SHA-256 `26f17302fcc5b8c1dee562de2b82f39ddafdc8a060a3f9c84922e98b59913399`; uavhengig datakontroll `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_DECIMAL_PRIVATE_INPUT_INDEPENDENT_CHECK_20261028_003.json`, SHA-256 `35912018b7982d4de582afa6967fd4f1e40003716118ec2470b9221d1a8d289a`. Ingen TEST eller handel, ingen målt retningsbedring.


KORRIGERT PRIVAT DATAKONTROLL 003 KILDEBUNDET: ny separat én-gangsplan etter konsumert importfeil 001 og overlappende verifikasjonsøkt 002. Den ytre dataverifikasjonen lukkes før den ekte måleieren åpner sin egen; syntetisk regresjonstest bestod. CPU-import, tom CUDA-synlighet og modellkallvakt beholdes. Fysisk provider og featureview stubbes; ekte målarkiver og alle 60 TRAIN-priorer kontrolleres. Ingen modellforward, optimizer, GPU, TEST eller handel. Plan: `/home/andre2/src/GX1_CURRENT/docs/DIRECTION_DECIMAL_PRIVATE_INPUT_20261028_003.json`, SHA-256 `b6251037173a697555e75898ddbf56201979fb838a24ed9de48efecf417361d2`. Kjør én gang, kontroller og lukk.


KORRIGERT PRIVAT DATAKONTROLL 002 FEILET OG ER KONSUMERT: kontrollens ytre filverifikasjonsøkt overlappet den ekte måleierens krav om en ny separat økt. Kildegjennomgang viser at native campaign-kontekstens økt er lukket før `_prepare`, men ekte privat kvalifisering er ikke kjørt. Ingen modellkonstruksjon, forward, optimizer, CUDA eller TEST. Ikke relanser 002. Feilkvittering `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_DECIMAL_PRIVATE_INPUT_20261028_002/FAILED_TERMINAL.json`, SHA-256 `705d1e15c2ab951ec35aed0eeb1f90bad4a6ba97baee87ec67c259ae10066f66`. Rett kun auditens scope og forbered ny separat 003.


KORRIGERT PRIVAT DATAKONTROLL 002 KILDEBUNDET: ny separat én-gangsplan etter konsumert 001-importfeil. Den eksakte måleieren lastes på CPU med CUDA_VISIBLE_DEVICES tom, deretter forbyr profilvakt modell- og CUDA-kall. Fysisk provider og featureview stubbes; ekte målarkiver og alle 60 TRAIN-priorer kontrolleres. Ingen modellkonstruksjon, forward, optimizer, GPU, TEST eller handel. Plan: `/home/andre2/src/GX1_CURRENT/docs/DIRECTION_DECIMAL_PRIVATE_INPUT_20261028_002.json`, SHA-256 `1b6431770e765b285620d58396a65f57d9dd0c87b3f1d8ce49bca7bb76d70ce3`. Kjør én gang, kontroller og lukk.


KORRIGERT PRIVAT DATAKONTROLL 001 FEILET OG ER KONSUMERT: den autentiske dense-måleieren importerer Torch via sitt horisontkonstant-eierskap, og auditens for strenge importvakt stoppet før target-grenen. Ingen modellkonstruksjon, forward, optimizer, CUDA eller TEST. Ikke relanser 001. Bevar feilkvitteringen `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_DECIMAL_PRIVATE_INPUT_20261028_001/FAILED_TERMINAL.json`, SHA-256 `6bc2b78e0e3f353ae4ef8ad7b9fde5c3d9680671720f9f6043a0a193bca30be8`. Forbered en separat 002 med CPU-bundet import av nøyaktig samme eier og profilvakt som forbyr modell- og CUDA-kall.

KORRIGERT PRIVATE MÅLINNGANG FORBEREDT: separat privat scope kobler eksisterende native target-forberedelse til den autentiserte desimalforbrukeren, beholder original ankerkontroll, fryser begge målarkivene og sammenligner alle 60 TRAIN-priorer mot kildeaudit. Eksisterende full-fit-validator avviser fortsatt dette private scopet. Syntetiske tester og en lesende kontroll av de faktiske 60 priorene bestod. Én capped datakontroll av faktisk target-gren er tillatt med stubbet fysisk provider og featureview; ingen modell, forward, GPU, optimizer, TEST eller handel. Plan: `/home/andre2/src/GX1_CURRENT/docs/DIRECTION_DECIMAL_PRIVATE_INPUT_20261028.json`, SHA-256 `cefc706fe588c2f646aec1c0b6b8ba8edcd957e33196f86b81472cd8417cf59f`. Kjør én gang, kontroller og lukk.

SAMME FEM-/SEKSTIMINUTTERSFORELDRE: på 16 482 felles modne, observerte ikke-null Q1-utfall var 5min-AUC 0,49959 mot 0,50017 på sine 17 192 egne utfall. 60min-AUC var 0,52100 på felles foreldre mot 0,52101 på sine 16 512 egne. Felles utvalg fjerner derfor ikke den svake femminuttersrangeringen i denne punktmålingen. Tallene er beskrivende, uten eget usikkerhetsintervall, og manglende framtidsutfall kan ikke vurderes kontrafaktisk. Utfallstilgjengelighet er aldri en beslutningsfilter. Capped terminalkontroll, sammendrag `docs/DIRECTION_SHARED_COHORT_20261028.json`, SHA-256 `e135f7f77fab73173dfca3fcadcb0de0000c8fdec7d07baa8871b8ac0db572cf`; ingen modellforward, trening, TEST eller handel.

MÅLMASKER OG M1-KILDEHULL MÅLT, FULLFØRT OG KONSUMERT: Alle 60 endepunktsmasker og kildepekere stemte med de faktiske M1-tidsstemplene i Q1 og fast Q2-utvalg, også ved en uavhengig ordbok-/mengdekontroll. Q1 har 63 manglende modne 5min-endepunkter og ett umodent; 60min har 756 manglende og 12 umodne. Av 17 276 foreldre der begge horisonter er modne, har 16 520 begge endepunkter, 63 mangler begge og 693 har 5min men mangler 60min. Kildehullene som påvirket 1–60min omfatter ett på 1 minutt og 63 på minst 64 minutter. Syv observerte 60min-endepunkter har likevel minst ett manglende mellomliggende M1-tidsstempel. Dette forklarer forskjellig observasjonspopulasjon, ikke AUC nær 0,50; full M1-dekning er ikke full tick-/path-sannhet. Ingen markedsårsaker antas. Retningsprioriteten består; dette åpner ingen ny FLAT-/risiko-/path-trening. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_OBSERVATION_GAPS_20261028_001/OUTPUT/RESULT.json`, SHA-256 `a8c21c9b93e59432e394af49213074d4d04612771f92addd8567476add4864e0`; uavhengig kontroll: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_OBSERVATION_GAPS_INDEPENDENT_CHECK_20261028_001.json`, SHA-256 `7a99c730859c5a5370b7de76b8eb588862d6c3c238d63baa5eb3ac2040000775`. Ingen modellforward, trening, TEST eller handel.

UTFALLSPOPULASJON OG M1-HULL KILDEBUNDET: én capped lesende kontroll av eksakte 1–60-minutters målmasker, umodne endepunkter, kildehull og manglende mellomliggende M1-tidsstempler er tillatt på Q1 og fast Q2-utvalg. Ingen årsak til markedshull antas. Ingen modell, trening, TEST eller handel. Plan: `/home/andre2/src/GX1_CURRENT/docs/DIRECTION_OBSERVATION_GAPS_20261028.json`, SHA-256 `743951ed3e97b36b648efeb8ea410f18988d4dad07faa60865eb3b5105807fc5`. Kjør én gang, kontroller og lukk.

CANDLE-SPREAD OG MÅLUTFALL MÅLT, FULLFØRT OG KONSUMERT: på Q1 var 9 972/17 192 ikke-null 5min-midtbevegelser (58,00 %) større enn summen av halv entry- og sluttspread ved native M1-candleåpninger; 95 var like og 7 125 mindre. Q2: 1 254/1 753 (71,53 %) større, 8 like og 491 mindre. Q1 for 1/15/60min var andelen 31,35/75,13/88,24 %. For 5min var Q1 median absolutt midtbevegelse 2,389 bp og median kombinert candle-spreadproxy 1,857 bp; 23,66 % av ikke-null-bevegelsene var under 1 bp. Dagens BCE gir alle kvalifiserte fortegn lik cellevekt, også små bevegelser. Dette viser mål-/kostnadsmismatch, men ikke årsaken til AUC ≈0,50. Candleåpninger er ikke verifiserte synkrone fyll; ingen gebyr/slippage, vei eller realisert økonomi. Uavhengig direkte kjøps-/salgs-tickkontroll stemte for alle 60 horisonter og begge perioder. Ingen modellforward, trening, TEST eller handel. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_CANDLE_SPREAD_20261028_001/OUTPUT/RESULT.json`, SHA-256 `ee4ee92e1e872e3af4121202afe8d4cc1510826955d885fd863bef55eaf099cb`; kontroll: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_CANDLE_SPREAD_INDEPENDENT_CHECK_20261028_001.json`, SHA-256 `dece4fbd7f3f560de5af60f8197ec9bf13123713115c5d88ae6f2ea00caee6cc`.

FASTIT OG OBSERVERT CANDLE-SPREAD KILDEBUNDET: én capped lesende måling av 1–60min-midtprisbevegelser relativt til native kjøps-/salgs-candleåpninger er tillatt på faste Q1-/Q2-foreldre. Eksakte kildepekere er kontrollert for alle horisonter. Candleåpningene er prisproxy, ikke dokumenterte synkrone fyll; ingen modell, trening, TEST eller handel. Plan: `/home/andre2/src/GX1_CURRENT/docs/DIRECTION_CANDLE_SPREAD_20261028.json`, SHA-256 `845bf95418620629e33e148d4aa38bc5381f49183381bedc26108b6539b8e9df`. Kjør én gang, kontroller og lukk.

5MIN TREFFPROSENT MOT ENKEL REFERANSE: samme lagrede 60-hodemodell traff 48,88 % mot alltid OPP 51,58 % på 17 192 korrigerte Q1-utfall; OPP ble valgt på 13,33 %. I fast Q2-utvalg var treff 49,29 % mot alltid OPP 50,20 %, med 4,16 % OPP-valg. Dette er punktmålinger på gjenbrukte data, uten separat parret treffprosentintervall. Q1 6min-hodet valgte OPP på 99,38 % av radene; dette betyr ikke 99 % sikkerhet. Capped terminalmål med hash-/aksekontroll, sammendrag `docs/DIRECTION_HEAD_ACCURACY_DIAGNOSIS_20261028.json`, SHA-256 `c35341662b2ec165b6006270776f7bc0275773bafae8dd85d111c179e097959e`. Ingen modellforward, trening, TEST eller handel.

NABOHODE-SAMMENLIGNING FULLFØRT OG KONSUMERT: fast gjennomsnitt av 4/5/6min-logits mot opprinnelig 5min på samme korrigerte fasit. Q1 17 192 kvalifiserte: AUC 0,50017 til 0,50278, parret dagbånd −0,00377 til +0,00884 og ukebånd −0,00222 til +0,00735. Fast Q2-utvalg 1 753 kvalifiserte: AUC 0,48969 til 0,48848, dagbånd −0,02376 til +0,02042 og ukebånd −0,02094 til +0,01942. Alle AUC-bånd inkluderer null. Q1 logloss/Brier ble bedre enn originalen med dag- og ukebånd under null, men dette er samlet sannsynlighetsscore, ikke separat kalibreringsbevis. Beskrivende punkter viser at en fast sannsynlighet fra TRAIN-andelen likevel hadde lavere logloss/Brier enn nabogjennomsnittet i begge periodene; forskjellen mot denne referansen har ikke eget intervall. Uavhengig kontroll av alle 60 horisonter og samtlige 10 000 trekninger stemte innen 2,23e−16. Gjenbrukte utviklingsdata, ingen modellforward, trening, TEST, handel eller kvalifisert edge. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_NEIGHBOR_POOLING_20261028_001/OUTPUT/RESULT.json`, SHA-256 `db39faf262d0cbc848dc7dfa26b3e52e5530846ac842afaa117e12db20388b27`; kontroll: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_NEIGHBOR_POOLING_INDEPENDENT_CHECK_20261028_001.json`, SHA-256 `eb382469e6f6d55d0c9011ac8373dffc1a16f6203213506b287191120474b9ff`.

NABOHODE-HYPOTESE KILDEBUNDET: én capped lesende sammenligning av faste 4/5/6min-logits mot det lagrede 5min-hodet er tillatt, med Q1 og fast Q2-utvalg og korrigert fasit. Øvrige 59 horisonter er kun beskrivende. Begge perioder er gjenbrukt utvikling; ingen ny modellforward, vekter, trening, TEST eller handel. Plan: `/home/andre2/src/GX1_CURRENT/docs/DIRECTION_NEIGHBOR_POOLING_20261028.json`, SHA-256 `179dc1fe9e1e7215adde766ccbe5ea1d48ad7cf8c52de2aba344fbbab7be0148`. Kjør én gang, kontroller og lukk.

KORRIGERT FAKTORSAMMENLIGNING FULLFØRT OG KONSUMERT: på samme 17 192 kvalifiserte 5min-rader er AUC for én pass/baseline 0,50298, bps 0,49676 og tre pass 0,50096. Parret bps minus baseline har dagbånd −0,01755 til +0,00605 og ukebånd −0,02127 til +0,00608; tre minus én pass har dagbånd −0,01002 til +0,00604 og ukebånd −0,00802 til +0,00430. Alle fire AUC-bånd inkluderer null. Tre pass hadde dårligere logloss/Brier med ukebånd over null, mens dagsbånd inkluderte null. Samme gamle modeller er bare omskåret med korrigert fasit; dette måler ikke korrigert retrening. Uavhengig kontroll av 1/5/15-punkter og alle 10 000 dag-/uketrekningers intervaller stemte innen 1,12e−16. Ingen modellforward, trening, TEST, handel eller kvalifisert edge. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_CORRECTED_FACTORS_20261028_001/OUTPUT/RESULT.json`, SHA-256 `48ba0989f59bc6edfc4feb2f5e5a7295d4c9280ee136eacd05a6f33f297e834e`; kontroll: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_CORRECTED_FACTORS_INDEPENDENT_CHECK_20261028_001.json`, SHA-256 `38c8f3246fe14182d4c9ed26e7b86fbd1fed64b80bed83fbcec99123efa5ecc3`.

KORRIGERT FAKTORSAMMENLIGNING KILDEBUNDET: én ny, capped lesende kontroll er tillatt for å sammenligne tidligere lagrede bps- og ekstra-passer-modeller mot samme baseline og korrigert desimalfasit, på faste 1/5/15-minutters utviklingsrader. Ingen modellforward, vekter, trening, TEST eller handel. Foreløpig kontroll bekreftet 17 288 felles foreldre og eksakte tidligere AUC-er, med 4/5/3 korrigerte falske fortegn. Plan: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_CORRECTED_FACTORS_PLAN_20261028_001.json`, SHA-256 `d027f658ad98f05ba94d97e8f408d22c6bce2aa7189999f7d4f63e84baea0ffc`. Kjør én gang og lukk autorisasjonen etter uavhengig kontroll.

FERSK GC-KONTEKST OG HISTORIKK MÅLT, FULLFØRT OG KONSUMERT: samme 1 754 Q2-foreldre med eksakt forrige M5-beslutning. Nåværende GC-kontekst ble erstattet med forrige, og nye GC-historikkrader skjult med sine masker: M5 1 754, M15 583, H1 143, H4 36, D1 0. Alle 110 originalbatcher gjenskapte lagrede logits byteidentisk. På 1 747 kvalifiserte 5min-utfall var AUC 0,49008 med fersk GC og 0,49939 med siste GC skjult; parret gammel minus fersk hadde dagbånd −0,01196 til +0,03160 og ukebånd −0,00583 til +0,02543. Begge inkluderer null. Tilleggsskjuling av GC-historikk versus tidligere kun-kontekst-inngrep endret ett 5min-fortegn og gjennomsnittlig sannsynlighet 0,00243 prosentpoeng. Dette er én gjenbrukt utviklingsperiode og muligens unaturlig input, ikke dokumentert edge. Uavhengig alle-60-/10 000-trekningers kontroll: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_FULL_STALE_INDEPENDENT_CHECK_20261028_001.json`, SHA-256 `719dd6f6555927312ce43755fc0f4bf21f0d43655cbc3c1145141ff36eb8f5ac`. Ingen trening, TEST eller handel.

KORRIGERT MÅLINNGANG KVALIFISERT PÅ GENUINE DATA: ny dataeier autentiserte original-/desimalfilene og alle koblede revisjonskvitteringer og returnerte korrigert 1–60-fasit for 118 249 foreldre. Fysiske foreldre-/barstart-hasher og gamle ankre stemte; 976 falske bevegelser ble fjernet uten motsatte reelle fortegn. Både syntetiske tester og capped lesende kontroll bestod. Dette er ikke integrert i et treningsscope og gir ingen ny treningstillatelse; fersk initialbaseline og eget nytt vedtak kreves. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_DECIMAL_BOUND_INPUT_ACTUAL_20261028_001.json`, SHA-256 `8415d195256ea7dd9bc9cd35661c1536fe57c1db330b0969d16a963a3381be19`. Ingen modellforward, trening, TEST eller handel.

KORRIGERT MÅL: faktisk treningsbinder godtar originalt 1–60-arkiv, men avviser det kildeverifiserte desimalarkivet med `ANCHOR_BYTES:mid_move`; den separate desimalforbrukeren godtar det strukturelt. De opprinnelige 1/5/15-fortegnene endres på 56/40/18 celler over 118 249 foreldre. Eksisterende treningsscope og privat kvalifisering er bundet til originalmålet; en ny separat kilde-/autorisasjonsbundet rute og fersk initialbaseline kreves, ikke svekket ankerkontroll. Sammendrag: `docs/DIRECTION_DECIMAL_TRAIN_BINDER_BLOCK_20261028.json`, SHA-256 `56a2773843ae6abfe212ec37e9bd669a7e3b57946d827906c34536eb5252f407`. Ingen trening, TEST eller handel.

FERSK GC-KONTEKST MOT FORRIGE M5, FULLFØRT OG KONSUMERT: 1 754 av de 1 760 faste Q2-foreldrene hadde eksakt forrige beslutning. Alle 110 originalbatcher gjenskapte lagrede logits byteidentisk. På 1 747 kvalifiserte 5min-utfall var AUC 0,49008 med nåværende GC-kontekst og 0,49944 med forrige; parret differanse gammel minus nåværende hadde dagbånd −0,01189 til 0,03165 og ukebånd −0,00563 til 0,02544. Siste GC-kontekstendring skiftet 5min-fortegn på 87 rader, gjennomsnittlig absolutt sannsynlighetsendring 0,515 prosentpoeng. Dette beviser ikke nytte: GC-MTF-historien forble aktuell, og utvalget er gjenbrukt utvikling. Uavhengig 60-horisont-/10 000-trekningers kontroll: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_STALE_CONTEXT_INDEPENDENT_CHECK_20261028_001.json`, SHA-256 `a0e997e5a0134d11cc0a09fc3cc91115ea729fd04cae233100efeed32d8adcb8`. Original RESULT `metrics.limitations` beskrev feilaktig historieablasjon; originalkvitteringen er bevart, korreksjonen står i kontrollen og teksten er rettet i kilde for nye kjøringer. Ingen trening, TEST eller handel.

REGISTERSCOPE OG FOLD: GC-feltregisteret og det eldre bevarte registeret er laget over mai 2024–desember 2025, som omfatter senere utviklingsfold; det er ikke et uavhengig fryst 2024-only register. Registeret oppgir at utfall ikke ble brukt. Alle 1 205 beholdte GC-verdier var likevel observert minst én gang på de første 47 602 TRAIN-foreldrene; alle registrets observasjonstall gjenskapte de 118 249 faktiske foreldrerekkene, og normaliseringen ble tilpasset første prefix. Det er ikke funnet direkte fasitlekkasje i denne kontrollen, men full prefix-only gjenoppbygging, eldre revisjoner og andre kilder er ikke dermed kvalifisert. Sammendrag fra capped terminalmåling: `docs/DIRECTION_GC_REGISTRY_FOLD_SCOPE_20261028.json`, SHA-256 `6eed8d1142c65fdcf5dd8b6819304e7d742c315ee97215b94409be21e3b1c67d`. Ingen modellforward, trening, TEST eller handel.

GC-VERDIER OFTE UTENFOR TRAIN-OMRÅDET: av 1 205 navngitte GC-verdi-/maskeposisjoner har 14 895/17 288 Q1-foreldre minst én observert verdi utenfor sitt normaliserte TRAIN-område, median 189 posisjoner; i fast Q2-utvalg gjelder det 1 760/1 760, median 204. Dette er beslektede koordinater, ikke antall uavhengige indikatorer. Alle 1 205 TRAIN-min/maks stemmer metadata, og en separat råverdimetode gjenskapte totaltallene eksakt. M5 GC-VWAPs observerte TRAIN-maks var 2 801,405; Q1- og Q2-median var ca. 2 907/3 324. For dette ene feltet er 5min-AUC i Q1 0,49445 innenfor og 0,50353 over TRAIN-maks: nivåskiftet alene er ikke dokumentert årsak til svak rangering. Utforskende, ingen kausal konklusjon. Sammendrag fra capped terminalmålinger: `docs/DIRECTION_GC_CONTEXT_RANGE_DIAGNOSIS_20261028.json`, SHA-256 `a32df96dc55669ef6efc4b5b47a28249adf8cd71f4f5b32362e197d090f27962`. Ingen modellforward, trening, TEST eller handel.

MODELLEN HAR LÆRT NOE SAMMENHENG, MEN IKKE DOKUMENTERT EDGE: på samme 256 lagrede TRAIN-foreldre økte mediankorrelasjonen mellom 59 nabohoder fra 0,0303 før til 0,7787 etter 2 976 oppdateringer; negative nabopar falt fra 29 til 7. For 5/6min var korrelasjonen −0,0413 før og 0,6100 etter på disse treningsradene, men 0,1698 i Q1 og −0,0454 i Q2 ved alle undersøkte foreldre. Det er et lite TRAIN-utvalg og ulike senere populasjoner, og korrelasjon er ikke prediktiv edge. Resultat: `docs/DIRECTION_HEAD_TRAIN_LEARNING_CHECK_20261028.json`, SHA-256 `f1befc6b5ce33de2019e5193311411646819390e7ce7b8e51a75285a43f901a2`. Ingen modellforward, trening, TEST eller handel.

PRISREFERANSE, UTFORSKENDE: bruk fortegnet på M1-midtprisendringen fra t−10 til t−5 minutter, kun når sluttcandle er lukket før beslutningen, og mål motsatt retning ved t til t+5. På 17 033 Q1-rader traff dette 50,75 % mot 51,61 % for alltid OPP; parrede dag-/ukeintervaller mot alltid OPP og modellens AUC inkluderer null. På 1 738 Q2-utvalgsrader traff det 52,47 % mot 50,17 % for alltid OPP; referansens AUC 0,52473 mot modellens 0,48931, med positive parrede dag-/ukebånd. Q2-valg og referansen er undersøkt i gjenbrukte data, og dagsintervallet mot alltid OPP inkluderer null. Ingen urørt bekreftelse, kostnader eller økonomisk edge. Resultat: `docs/DIRECTION_PAST_MOVE_REFERENCE_DIAGNOSIS_20261028.json`, SHA-256 `1d1273e9e500fcf32586fbc6ce24e1d663d44155b64a8f6e1b77d3cd8ba867e0`. Ingen nye modellkall, trening, TEST eller handel.

NY DIAGNOSE AV LAGREDE 5-/6-MINUTTERSHODER: på 17 157 felles kvalifiserte Q1-rader predikerte 5min OPP på 13,33 %, 6min på 99,38 %, mens fasit var 51,57 % og 50,83 %; AUC var 0,49987 og 0,49355. Q2-utvalget viser 4,17 % mot 99,49 % predikert OPP, og AUC 0,48967/0,46826. Fasitens og prognosenes korrelasjoner og alle 59 nabopar er dokumentert; modellen har separate parameteriserte hoder med felles representasjon. Dette er en konkret kalibrerings-/rangeringssvikt på gjenbrukte data, ikke bevist årsak eller validert rettelse. Korrigert full TRAIN-fasit er ca. 51 % OPP på begge horisonter. Resultat: `docs/DIRECTION_HEAD_COMPOSITION_DIAGNOSIS_20261028.json`, SHA-256 `2af5f7bd7ff5c2623318db972b15b0f5bf0efcc9cb523ce0d7b1683c1279717c`. Ingen nye modellkall, trening, TEST eller handel.

DIREKTE GC-PROJEKSJONSMÅLING FULLFØRT OG KONSUMERT: på 275 aktive Q1- og 80 aktive Q2-foreldre endret undertrykkelse av de fem TRAIN-usette normaliserte GC-verdiene ingen 5-minutters fortegn. Gjennomsnittlig endring i 5-minutters OPP-sannsynlighet var henholdsvis 0,00190 og 0,07463 prosentpoeng; alle opprinnelige batcher gjenskapte lagrede logits byteidentisk. På øvrige 1–60-minutts utganger skiftet 9/16 500 og 47/4 800 foreldre-horisontceller fortegn. Inngrepet er avgrenset til SMC-kontekstprojeksjonen, andre ruter er bevart; det beviser ikke at feltene er unyttige eller årsak til svak edge. Uavhengig NumPy-kontroll: `docs/DIRECTION_UNSEEN_GC_PROJECTION_CHECK_20261028.json`, SHA-256 `15620cc577cfd42ec66b2458f28e4ad87140f921433fd86497c5b30343b4689d`. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_UNSEEN_GC_PROJECTION_20261028_001/OUTPUT/RESULT.json`, SHA-256 `b4a42e6189a493acd0d45fe47b3972be7fad98f6a469da18e5898ea03ebf86bf`. Ingen trening, TEST eller handel.

BEKREFTET LÆRINGSMANGEL: de 33 navngitte GC-kontekstkoordinatene var normalisert null på alle 47 602 første-prefix TRAIN-foreldre og hadde uendrede kontekstprojeksjonskolonner. I Q1 ble bull-orderblock-alder ikke-null på 275/17 288 foreldre. I fast Q2-utvalg ble bearish orderblock-retouch, low, high og opphavsvolum ikke-null på 80/1 760 foreldre; low/high/opphavsvolum nådde normaliserte maksimum 7,12/7,22/12,27. Masker og opprinnelig normalisering ble brukt. Dette viser usett støtte i en bestemt projeksjonsrute, ikke bevist effekt på prognosen eller årsak til manglende edge. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_UNSEEN_GC_SUPPORT_20261028_001/OUTPUT/RESULT.json`, SHA-256 `48b28abd82380efa51ab4ff8912152e586b4535c615c5c7091eae68bc910d3c7`. Ingen modellkall, trening, TEST eller handel.

Separat kontroll av 33 GC-kontekstkoordinater uten variasjon i første TRAIN-prefix er FULLFØRT og KONSUMERT. Den skal bekrefte TRAIN-støtten og måle om koordinatene blir aktive i Q1 og fast Q2-utvalg, med de opprinnelige maskene og normaliseringen. Ingen modellkall, trening, TEST eller handel. Plan: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_UNSEEN_GC_SUPPORT_PLAN_20261028_001.json`, SHA-256 `2f134daf9d8ee14bef190a0ccdb6b828d4639fd6cc80bcccdd58d8135a2a6a9b`.

UAVHENGIG NUMERISK KONTROLL FULLFØRT: `close_position_range20` er gjenskapt med et separat NumPy-orakel og stemmer byteidentisk på M5/M15/H1/H4/D1, totalt 1 544 699 native rader før juli 2025. Alle undersøkte TRAIN-/Q1-/Q2-vinduer stemmer. Sammen med tidligere indeks er 735/890 aktive MTF-felt-/tidsrammeposisjoner dekket av uavhengig tallkontroll i denne avgrensede sammenstillingen; de resterende 155 er 31 trendlinjefelt × 5 tidsrammer med eksakt eierreplay, ikke uavhengig formelbevis. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_CLOSE_POSITION_NUMERIC_20261028_001/OUTPUT/RESULT.json`, SHA-256 `7f381bdf8d876471fd67ae0b2d8289737b3bdc42d7d40140bc2b43ba11f9598a`. Ingen modellkall, trening, TEST eller handel.

Separat uavhengig tallkontroll av `close_position_range20` på alle fem tidsrammer er FULLFØRT og KONSUMERT: NumPy-orakel fra faktisk 2009-kilde sammenlignes med cache og modellvinduer. De fem cachefilene hashkontrolleres i den kapasitetsvaktede kjøringen. Ingen modellkall, trening, TEST eller handel. Plan: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_CLOSE_POSITION_NUMERIC_PLAN_20261028_001.json`, SHA-256 `b4bc584240c559097f66c9820622ee32066464b6ab2ef98aa946000d24210e9d`.

ALLE FEM TIDSRAMMER: med faktisk native forhistorie fra 2009 ble 31 aktive trendlinjefelt gjenskapt byteidentisk mot cache på 1 070 080 M5-, 357 385 M15-, 90 075 H1-, 23 270 H4- og 3 889 D1-rader før juli 2025. Delte/hele tilstandskjøringer stemte, og undersøkte modellvinduer hadde null avvik. De tidligere H1/D1/H4-avvikene skyldtes min forkortede 2010-start, ikke en påvist cachefeil. Dette er same-owner-replay, ikke uavhengig matematisk formelbevis eller edge. Den siste tidligere ikke-dekkede MTF-posisjonen `close_position_range20` er nå uavhengig tallkontrollert; se øverst. H4-resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_H4_NATIVE_HISTORY_REPLAY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `debdc7287f3e8f87ffcad1bbe72469446230d0670b24f3f66c0639e5a55b33c0`. Ingen modellkall, trening, TEST eller handel.

D1 REPLAY FRA FAKTISK 2009-HISTORIKK ER FULLFØRT: alle 31 aktive trendlinjefelt stemte byteidentisk med cache på 3 889 D1-rader; delt/full state og alle undersøkte modellvinduer stemte. Tidligere D1-avvik skyldtes den forkortede replaykilden. H4 er nå også fullført. Eierreplay er ikke uavhengig formel-/edgebevis. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_D1_NATIVE_HISTORY_REPLAY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `52e8b24b0e0df6a4684847071bd15563c93f931e7cfb5cd5b7117b983db93fd7`. Ingen modellkall, trening, TEST eller handel.

M15 REPLAY FRA NATIVE 2009-HISTORIKK ER FULLFØRT: 31 aktive trendlinjefelt stemte byteidentisk med cache på 357 385 M15-rader, med eksakt delt/full tilstand og null avvik i undersøkte modellvinduer. Eierreplay er ikke uavhengig formel-/edgebevis. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_M15_NATIVE_HISTORY_REPLAY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `1f06f563c2258c9766c53963f8883a3c9f84d6c1dd8b3f1142f67da554442c38`. D1 og H4 er nå også fullført. Ingen modellkall, trening, TEST eller handel.

Separate native-historie-kontroller for M15, D1 og H4 er FULLFØRT og KONSUMERT sekvensielt. Hver sammenligner 31 aktive trendlinjefelt, tilstand og faktiske modellvinduer mot riktig 2009-kilde; resultatet inspiseres og lukkes før neste start. Ingen modellkall, trening, TEST eller handel. Planer: M15 `555a45affade691bd5d7b480e9902ed9fa4d4e4ce8d5215e782bc1fdc5f24988`, D1 `20a829c4d84bc00ee9f05a4239262aecc0c5d2e4d10422d0ec9704939aca15f1`, H4 `35b0ff2d587f3bbf98eb0f784c3e3acc0ee899ab84251752477a3425a280dca8`.

M5 REPLAY FRA NATIVE 2009-HISTORIKK ER FULLFØRT: alle 31 aktive trendlinjefelt stemte byteidentisk med cache på 1 070 080 M5-rader; delt/full beregning og 1 070 080 felles M5-prisrader stemte også. Ingen avvik i 47 602 TRAIN-, 17 288 Q1- eller 1 760 Q2-vinduer. Dette er eierreplay, ikke selvstendig formel- eller edgebevis. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_M5_NATIVE_HISTORY_REPLAY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `d1c39ec3f8ad7edbda23f620ea6ce5fec606eebe9ad763080cc1847c7dedd3ed`. Ingen modellkall, trening, TEST eller handel.

Kildebundet M5-kontroll fra native historikk tilbake til 2009 er FULLFØRT og KONSUMERT. Den skal gjenskape 31 aktive trendlinjefelt og kontrollere original cache samt faktiske 16-raders M5-vinduer. M5-cachehash kontrolleres inne i den kapasitetsvaktede kjøringen. Ingen modellkall, trening, TEST eller handel. Plan: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_M5_NATIVE_HISTORY_REPLAY_PLAN_20261028_001.json`, SHA-256 `98d65cfa6158fe20778f80a439a1b6818133f9bcb4257b531801455d8589adcc`.

H1 REPLAY MED FAKTISK 2009-FORHISTORIE ER FULLFØRT: 17 hashbundne native årsdeler ga eksakt pris-/klokkelikhet på 1 070 080 overlappende M5-rader. Alle 31 H1-trendlinjefelt stemte byteidentisk med cache på 90 075 H1-rader; null avvik i 47 602 TRAIN-, 17 288 Q1- og 1 760 Q2-vinduer. Deling av beregningen med state ga samme resultat. Det tidligere H1-avviket skyldtes at min kontroll startet i 2010 og mistet historisk state. Dette er eierreplay, ikke uavhengig formel-/edgebevis. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_H1_NATIVE_HISTORY_REPLAY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `94b524013ab9f3cf98e5ad679954b036ba15bd12286e8f87d0c08c7dcfbf4dae`. Ingen modellkall, trening, TEST eller handel.

Separat H1-kontroll fra faktiske native M5-årsdeler 2009–2025 er FULLFØRT og KONSUMERT. Hele overlappet med M5_ENRICHED kontrolleres før H1-historikk, cache og faktiske modellvinduer sammenlignes. Dette avklarer den tidligere forkortede kildetesten. Plan: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_H1_NATIVE_HISTORY_REPLAY_PLAN_20261028_001.json`, SHA-256 `48d7db83212b7810040bb3d559dbeb4d106f1e2e43ac29cef56a1349d45a4536`. Ingen modellkall, trening, TEST eller handel.

KORRIGERT PROVENIENS: den aktuelle cachen ble laget i `build_entry_exit_m1_enriched_frame_v1.py` fra `native_stage` i float64 med opprinnelig historikk fra 2009, deretter skåret til den bearbeidede M5-filen som starter i 2010. Våre D1/H4/H1-kontroller startet i 2010 og manglet dermed faktisk forhistorie. Standard-CLIens float32-cast er en annen byggevei, ikke bevist brukt her. Dens hypotetiske H1-replay avvek i 706 173 celler og er FULLFØRT/KONSUMERT, ikke bevis for cachefeil. H1-kontrollen fra hashbundne native årsdeler er nå fullført; se øverst. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_H1_BUILDER_PRECISION_REPLAY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `f83ea53ff0dea30a03545ab81329b2009ad917e27cc7051fdf69aa27b16418c1`. Ingen modellkall, trening, TEST eller handel.

Separat hypotetisk H1-kontroll med standard-CLIens float32-cast er FULLFØRT og KONSUMERT. Den skal måle samsvar med cache og størrelsen på prisavrunding uten modellkjøring. Byggefilsinnholdet matcher commit a7b88c7f. Plan: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_H1_BUILDER_PRECISION_REPLAY_PLAN_20261028_001.json`, SHA-256 `af9c10ea33d1b97fc57c578bcdf0983d28d49cbee0bf9a013bd1ec8fce4cc919`. Ingen trening, TEST eller handel.

TIDLIGERE, UFULLSTENDIG PRESISJONSHYPOTESE: de siste D1/H4/H1-gjenskapingene brukte originale double-priser, men cachebyggeren `gx1/scripts/prebuild_multi_tf_cache_v4.py` caster open/high/low/close/volume til float32 før MTF-beregning. Derfor er registrerte forskjeller presisjonsfølsomhet, ikke dokumentert cachefeil. H1 double mot cache skilte 10 native treningsrader i fire channel-felt, inne i 3 527/47 602 TRAIN-vinduer; fire Q1-rader inngikk i 3 449/17 288 vinduer. Q2-utvalget hadde null. Dokumenter nå en separat builder-tro float32-gjenskaping og størrelsen på eventuelt råpris-gridtap før en årsak eller rettelse konkluderes. H1-resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_H1_TRENDLINE_REPLAY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `2998b8d8ff3b74f9891c03f1640529f94b61654dfb00bee1d09202d764b14923`. Ingen nye modellkall, trening, TEST eller handel.

Én kildebundet CPU-kontroll av 31 aktive H1-trendlinjefelt er FULLFØRT: gjenskap fra originale M5-priser og full kausal H1-historikk, kontroller cache, tilstand og faktiske 96-raders modellvinduer. Ingen modellkall, trening, TEST eller handel. Plan: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_H1_TRENDLINE_REPLAY_PLAN_20261028_001.json`, SHA-256 `26fe4863bf65db30a609dd8eb6a013f57d6badbe83253384a7b3aa2d3813dd3b`.

H4-trendlinjekontroll med original double mot float32-basert cache er FULLFØRT og KONSUMERT: 31 aktive felt og tilstandsoverføring ble kontrollert på 23 270 historiske H4-rader. Det er 3 235 eldre cacheavvik, siste 2. februar 2023 kl. 14 UTC. Ingen ligger i de 47 602 TRAIN-, 17 288 Q1- eller 1 760 valgte Q2-vinduene; kanonisk window-eier bekrefter yttergrensene. Det gjelder eierreplay, ikke uavhengig formelbevis. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_H4_TRENDLINE_REPLAY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `e0162ea76483cc5e9c1013c3bf4e94711a57c3ceeb2447f471b256aac5ed4d44`. Window-eierkontroll: `docs/DIRECTION_H4_WINDOW_OWNER_CHECK_20261028.json`, SHA-256 `5edc05d74cb09cb8f69544c5353ecb786f5637a8913a8418a93b48d417511171`. Ingen modellkall, trening, TEST eller handel.

Én kildebundet CPU-kontroll av 31 aktive H4-trendlinjefelt er FULLFØRT: gjenskap fra originale M5-priser og full kausal H4-historikk før 1. juli 2025, kontroller cache, state og faktiske 96-raders modellvinduer. Ingen modellkall, trening, TEST eller handel. Plan: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_H4_TRENDLINE_REPLAY_PLAN_20261028_001.json`, SHA-256 `270033853babf3651e18aa5ad05f402209308951c45c521d891dcc42b72586b2`.

Forskjellen mellom double-gjenskaping og float32-basert D1-cache er lokalisert: 7 551 celler fordelt på 460 forskjeller i manglendehet og 7 091 ulike endelige float32-biter; siste avvikende D1-start er 18. april 2023 kl. 22 UTC. Ingen av radene inngår i de 47 602 TRAIN-vinduene, 17 288 Q1-vinduene eller 1 760 valgte Q2-vinduene. Tidligste faktisk brukte D1-rad er 9. mai 2023 kl. 22 UTC; den kanoniske window-eieren bekrefter yttergrensene for hver kronologisk gruppe. Årsaken til det eldre historiske avviket er ukjent; 31 D1-felt er eierreplay-kontrollert for disse vinduene, ikke uavhengig formelvalidert. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_D1_MISMATCH_LOCALIZATION_20261028_001/OUTPUT/RESULT.json`, SHA-256 `590ab9d42c34f56aeba5be50ac33ec140eb565b3a2bcaf25d28125633c9ee525`. Window-eierkontroll: `docs/DIRECTION_D1_WINDOW_OWNER_CHECK_20261028.json`, SHA-256 `72164316e48f1159b2c1f8f0474503ce9cfb7ca329cea80c5b20f5bbefbe544a`. Ingen modellkall, trening, TEST eller handel.

Separat CPU-lokalisering av det målte D1-avviket er FULLFØRT og KONSUMERT. Den skal identifisere tidspunkt, manglendehet kontra numeriske forskjeller og om avvikende D1-rader faktisk ligger i 252-raders TRAIN-/Q1-/Q2-vinduer. Forrige kjøring forblir fullført og relanseres ikke. Plan: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_D1_MISMATCH_LOCALIZATION_PLAN_20261028_001.json`, SHA-256 `f9f3277706dd654789768130c2409c077585b973e4db2c6e4375eff12f787989`. Ingen modellkall, trening, TEST eller handel.

D1-trendlinjereplay er FULLFØRT og KONSUMERT med AVVIK i eldre historikk: av 3 889 D1-rader × 31 aktive felt avviker 7 551 lagrede/nyberegnede celler. Hele kjøringen og todelt tilstandsoverføring er identiske. På 174 nåværende D1-rader i modellens TRAIN-periode og 127 senere rader er avviket null. Påvirkning på undersøkte 252-raders vinduer og forskjellstyper er nå avklart øverst; kildeårsak er ukjent. Ny separat avgrenset lokalisering er nødvendig. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_D1_TRENDLINE_REPLAY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `7e1b469bb33598527976b03ec7fd8f892b6f18e08801de2e5d115c40444970ef`. Ingen modellkall, trening, TEST eller handel.

Én kildebundet CPU-kontroll av 31 aktive D1-trendlinjefelt er FULLFØRT: gjenskap fra originale M5-priser og full kausal D1-historikk før 1. juli 2025, sammenlign eksakte cacheverdier og kontrollér overføring av tilstand ved prefixgrensen. Dette er eierreplay, ikke uavhengig formelbevis eller edge. Plan: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_D1_TRENDLINE_REPLAY_PLAN_20261028_001.json`, SHA-256 `d1491d086f11a36871721205ac8c3d759f797c4edb645ab05e179a6dd7661ba0`. Ingen modellkall, trening, TEST eller handel.

Kildebundet TRAIN-metadata viser at alle 32 lokale kartkoordinater og alle 39 kartkoordinater per M5/M15/H1/H4/D1 varierte i første prefix; det er ikke bevis for korrekt formel eller nytte. I M5-historieruten var tre posisjoner svært sjeldne: retest-fail opp 7, retest-fail ned 11 og channel-apex 3 av 47 798 unike native rader. Tilsvarende lokale posisjoner hadde 295, 290 og 5 106 ikke-null av 47 878 rader; populasjonene er ikke identiske, og rutene bruker ulike vinduer og bånd. Uavhengig tall-/kausalitetskontroll for 32 posisjoner per tidsramme gjenstår. Rapport: `docs/DIRECTION_CHART_TRAIN_SUPPORT_20261028.json`, SHA-256 `8a110c09d5e4332a06b11a0d9445e732d531dd764c1c4f609c0ee26508979959`. Ingen modellkall, trening, TEST eller handel.

Et beskrivende tillegg på samme M5-beslutninger fant at fem- og seksminuttshodene gir ulike binære tegn på 1 669/1 751 Q2-rader, mens de faktiske utfallene har ulike tegn på 224/1 751. Gjennomsnittlig opp-sannsynlighet er 48,13 % og 52,03 %; dette er derfor terskelfølsom usikkerhet, ikke i seg selv en bevist modellfeil. Alle 59 nabopar finnes i diagnosen. Ingen modellkall, trening, TEST eller handel.

Kombinert GC-kontekst-/historieintervensjon er uavhengig VERIFISERT på alle 60 minutter og begge armer; tidligere originalprediksjoner er byteidentiske. 5min-retningen har en konkret skjevhet: i Q2-utvalget predikerte modellen opp 73/1 753 ganger, mens faktisk opp var 880/1 753; accuracy 0,492869 og alltid-LONG 0,501997. På samme 256 TRAIN-rader predikerte den opp 1 gang ved start og 55 ganger etter 2 976 oppdateringer; 128 utfall var opp. Hele TRAIN-prefix hadde 24 081 opp og 23 227 ned i opprinnelige etiketter. Dette er observert kalibrerings-/signskjevhet, ikke bevist rotårsak eller gevinst ved en rettelse. Diagnose: `docs/DIRECTION_SAVED_CALIBRATION_DIAGNOSIS_20261028.json`, SHA-256 `3d33cc50dd960d99a47873b5dcba3b1092cf6968a135952299c8a17ea23a206a`. Uavhengig resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_CONTEXT_VERIFY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `00a84af7b6edda97e7574d8b464c615574940360150458577a289f1fd0d45e3f`. Ingen nye modellkall, trening, TEST eller handel.

CPU-etterregning av den fullførte GC-kontekst-/historiekontrollen er FULLFØRT og KONSUMERT. Alle 60 horisonter, begge armer, korrigerte mål, byteidentiske tidligere originaler og parrede 5/60min dag-/ukeintervaller kontrolleres uten modellkall. Plan: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_CONTEXT_VERIFY_PLAN_20261028_001.json`, SHA-256 `0a1936f10f82d08527af2ab61320bbaee28c7ca6265982cb9cfd4d1b77f2f076`. Ingen trening, TEST eller handel.

Kombinert GC-kontekst-/historieintervensjon 002 er FULLFØRT og KONSUMERT; alle 1 760 originaler ble gjenskapt byteidentisk. Middel absolutt logitendring over 60 utganger var 0,07858. Foreløpig 5min AUC: 0,489688 original mot 0,493635 ved intervensjon, 1 753 kvalifiserte; parrede dag-/ukeintervaller inkluderer null. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_CONTEXT_FOLLOWUP_20261028_002/OUTPUT/RESULT.json`, SHA-256 `3759c43b7eed35b40eb16817592a6a8fa4070bcd2354ee7e7be33b04028566ee`. Uavhengig skår- og målkontroll er fullført; se øverst. Gjenbrukt utvikling, unaturlig intervensjon og ingen påvist økonomi. Ingen trening, TEST eller handel.

Plan 001 ble avvist av launcher før workerstart: `-B` sto før påkrevd `-m`. Ingen data, checkpoint eller modell ble åpnet, og output/claim finnes ikke. Feilterminal: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_CONTEXT_PRELAUNCH_REJECTION_20261028_001.json`, SHA-256 `a6a045b577665b977aac09daa69c963a0409a7d54e9a2f06f6c1198a0dbbf7fb`. Plan 001 er lukket. Plan 002 er separat kildebundet med korrekt kanonisk kommando: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_CONTEXT_FOLLOWUP_PLAN_20261028_002.json`, SHA-256 `a5a64fe53f41a796612ecf82de7d25df24de4ce12a296cba94631278144fa402`. Samme klart avgrensede GC-kontekst-/historiekontroll, ingen trening, TEST eller handel.

Kombinert GC-kontekst-/historieintervensjon ble separat admittert og er nå konsumert som én avgrenset, kapasitetsvaktet modellmåling på de samme 1 760 forhåndsvalgte 2025 Q2-radene. Alle 2 410 aktive GC-kontekstkoordinater er 1 205 verdi-/maskepaer; koblingen til lokal sekvens er komplett og uten maskekryss til andre felter. Alle tidligere originalprediksjoner må gjenskapes byteidentisk før de sammenlignes med felles undertrykking av GC-kontekst, tidsaliaser og 25 historieprojeksjoner. Plan: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_CONTEXT_FOLLOWUP_PLAN_20261028_002.json`, SHA-256 `a5a64fe53f41a796612ecf82de7d25df24de4ce12a296cba94631278144fa402`. Design: `docs/DIRECTION_GC_CONTEXT_FOLLOWUP_20261028.json`. Ingen trening, TEST eller handel. Foreløpig utfall og manglende uavhengig kontroll står øverst.

Senere-periode GC-historieintervensjon er FULLFØRT og KONSUMERT på 1 760 forhåndsvalgte 2025 Q2-foreldre. Den frosne fold1-modellens referansebatch var byteidentisk; begge prediksjonsarmer og korrigerte mål er lagret. På 1 753 kvalifiserte 5min-rader er AUC 0,489688 original og 0,489699 uten GC-historieprojeksjon; dag-/ukeintervaller inkluderer null. Dette er én gjenbrukt utviklingsperiode, med annen GC-kontekst beholdt. Uavhengig kontroll er FULLFØRT: begge armer, alle 60 minutter og dag-/ukeintervaller ved 5/60 minutter ble etterregnet med separat rangsum og blokksøk; største avvik var under 1e-12. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_MTF_LATER_VERIFY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `f641aefdd33805cf467d1d639641d19cb59938fb33d9b2d9e401b9ffc2bb7ba0`. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_MTF_LATER_FOLLOWUP_20261028_001/OUTPUT/RESULT.json`, SHA-256 `02d8cf20527f9219ec82b974b0dd741522539cab4f7a6790358d1b4d65cdd5e6`. Ingen trening, TEST eller handel.

Publiseringsblokkeringen er REPARERT i kilde: den eksisterende lesesperren tillater nå eksakt egen outputmappe bare som read-only `O_DIRECTORY`, uten å åpne andre mapper eller skriveflagg. Kun én kodelinje endret; 26 fokuserte syntetiske tester bestod. Treningsreceipt binder fortsatt gammel kildehash, så fremtidig replay må binde kildeovergangen. Se `docs/DIRECTION_GC_OUTPUT_DIRECTORY_GUARD_FIX_20261028.json`. Ingen ny modellkjøring.

GC-historieintervensjon 001 ble avvist før workerstart fordi CUDA-vakten krever absolutt kanonisk Python-sti; ingen data eller modell ble åpnet. Kvittering: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_MTF_INTERVENTION_PRELAUNCH_REJECTION_20261028_001.json`, SHA-256 `e7a6f93e93609c3a41cb348c18283250b91a72fc8586560a5b65b60a9f95f609`. Korrigering av feiltastet source-head i kvitteringen: `3e224ee46146497591567c3b47084690403b086d`, dokumentert i `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_MTF_INTERVENTION_PRELAUNCH_CORRECTION_20261028_001.json`. Engangsplan 002 stoppet før modellforward: egen TEST-navnsperre avviste en hashbundet syntetisk testkvittering. Feilterminal: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_MTF_INTERVENTION_20261028_002/FAILED_TERMINAL.json`, SHA-256 `f02ab488558bac131f21061bae0a7a9cad23bf997fcb5e4f982d867a374058b1`. Plan 003 tillater bare de to eksakt hashbundne syntetiske testmetadatafilene; markeds-TEST er fortsatt sperret. GC-historieintervensjon 003 gjennomførte 218 modellkall og byteidentisk replay, men er FAILED fordi sluttsynkronisering av outputmappen ble sperret. Separat recovery er FULLFØRT: originale skårer for alle 60 minutter er uavhengig gjenskapt nøyaktig på 1 736 faste foreldre, og outputmappene er synkronisert/hashbundet. Foreløpig 5min-AUC på 1 726 kvalifiserte er 0,514611 original mot 0,514270 uten GC-historieprojeksjon; parrede dag-/ukeintervaller for differansen inkluderer null. Ablasjonslogits ble ikke lagret, så ablasjonsresultatet er ikke uavhengig etterregnet og opprinnelig kjøring forblir FAILED. Annen GC-kontekst var beholdt. Recovery: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_INTERVENTION_RECOVERY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `ce131292529d42cda9653c1fa263c84be46c7ea0077d8d557a627d26c0cd0359`. Ingen trening eller TEST. Forsøket gjaldt: hver tiende opprinnelige senere batch, først byteidentisk replay, deretter nulling av 25 GC-historieprojeksjoner og parret 1–60min-måling mot korrigert fasit. Annen GC-kontekst beholdes. Ingen trening, TEST eller handel.

Navngitt GC-kontroll er FULLFØRT og KONSUMERT: 39 uendrede koordinater i GC-historier og 33 i gjeldende kontekst har alle normalisert TRAIN-støtte 0–0; 31 av historikkoordinatene er også uendret i kontekst. Dette er felt/tidsramme-posisjoner, ikke antall uavhengige indikatorer. Flere gjelder kvalitetsklokker, men blant dem er D1 bearish orderblock-pris/volum og H4/D1 `signed_complete`; separate tilgjengelighetsmasker kan fortsatt variere. Delvis kjent GC-delta har derimot variasjon og oppdaterte parametere på alle fem tidsrammer. Ingen effekt på retning er bevist. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_PROJECTION_FIELDS_AUDIT_20261028_002/OUTPUT/RESULT.json`, SHA-256 `083f330c0a7166114bd2926fd948e4db378e36de99bc5b02f27bcb7344ada40b`. Ingen modellforward, trening, TEST eller handel.

Engangskontroll av 60-minuttersmodellens start- og sluttparametere er FULLFØRT og KONSUMERT: alle tre residualenes 16 384 vekter endret seg fra null; SMC-kontekstmatrisen har rang 128 av 128 etter standard FP32-rangtoleranse. I M5 SMC ble 40 320 av 40 448 GC-projeksjonsvekter og 315 av 316 gate-biaser endret. Dette viser parameteroppdatering, ikke at input gir nyttig retningsprognose. Finn navngitte koordinater som var uendret og kontroller prefixstøtten. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_PROJECTION_STATE_AUDIT_20261028_001/OUTPUT/RESULT.json`, SHA-256 `57268760afd73589f2ab5d721ac7eb85c8932e44bfc44278093f2f78d4e00159`. Ingen modellforward, trening, TEST eller handel.

Selvstendig kilde-/prefixkontroll av sammensettingen: SMC-konteksten har 1 094 aktive numeriske koordinater til én 128-dimensjonal projeksjon (616 binære, 540 `_observed`); M5 SMC projiserer 316 GC- og 76 XAU-koordinater til samme 128-dimensjonale familietoken. Alle 316 GC-navn finnes også i aktiv gjeldende kontekst. Dette er mulig informasjonspress, ikke bevist tap eller årsak til svak retning. 55 av 56 treningskildeeiere matcher fortsatt; den ene endringen er den særskilt dokumenterte read-only katalogvakten. Se `docs/DIRECTION_GC_PROJECTION_SOURCE_REVIEW_20261028.json`. Ingen modellkall.

Separat desimalmålforbruker er STRUKTURELT KVALIFISERT på faktiske kildebundne mål: alle 118 249 foreldre, 60 horisonter, klokker/masker og uavhengige fortegnsmatriser stemmer; 17 288 lagrede fold1-rader er koblet tilbake til originalmålene. Av 6 946 049 observerte bps-celler har 6 784 709 ulik flyttallsverdi, men største absolutte forskjell er 4,214e-12 bps. Originalanker gjelder fortsatt bare originalmålet. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_DECIMAL_CONSUMER_AUDIT_20261028_001/OUTPUT/RESULT.json`, SHA-256 `87b2240484f48b1c7341467947357e25cda28c8d47fcf91f43b87614fa664632`. Ingen modell eller trening. Ny autentisert treningsrute og fersk baseline kreves fortsatt.

Separat korrigert 1–60min-desimalmål er FULLFØRT og KONSUMERT: alle 118 249 foreldre, 60 horisonter, opprinnelige klokker/masker og revisjonens fortegnsmatriser er kontrollert. 976 falske bevegelser ble fjernet; ingen reelle OPP/NED ble snudd. Målarkiv: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_DECIMAL_TARGETS_20261028_001/OUTPUT/DENSE_DECIMAL_TARGETS.npz`, SHA-256 `cb31d4754d3db1282b1c9c5523aee661da4fbfc4938ea4bf803ca8716829bfeb`. Resultat SHA-256 `71ea36e8fd8048a32aa1136f342264bdbd28122ae674dc1452a3d17207979648`. Arkivet er ikke koblet til trening; separat målforbruker og fersk baseline kreves. Ingen modellforward, trening, TEST eller handel ble utført.

H1-slope-respons er FULLFØRT og KONSUMERT på samme gjenbrukte fold: faktisk 5min OPP er 50,53 % ved negativ og 52,08 % ved positiv slope; modellens gjennomsnittlige OPP-sannsynlighet er 48,76 % og 48,73 %. Både korrigerte dag-/ukeintervaller for faktisk gruppeforskjell omfatter null. Dette er utforskende, ikke bevis på frakobling eller årsak. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_SLOPE_RESPONSE_AUDIT_20261028_001/OUTPUT/RESULT.json`, SHA-256 `3f2bd5814f2f9d207b81d66ddc7b845023f8a07e3bf1ba7a55699f38ff9f87fe`. Ingen modellforward, trening, TEST eller handel.

Faktisk M5-tidskontroll er FULLFØRT og KONSUMERT: 41 124/118 249 vinduer (34,78 %) har minst ett hull; median vindutid er 8 timer, 90-persentil 9 timer og 99-persentil 57 timer. I senere fold1 har 6 008/17 288 vinduer hull; 5min-AUC er beskrivende 0,502787 uten og 0,496839 med hull, uten usikkerhet eller kausal intervensjon. Posisjonskoding teller rader; tids-/sesjonsfelt finnes. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_HISTORY_CLOCK_AUDIT_20261028_001/OUTPUT/RESULT.json`, SHA-256 `9eff85319123a5cee2ae6d0ed6e1b8a33025e77d3ad04684cd5cfa2c0379a374`. Ingen modellforward, trening, TEST eller handel.

Forhåndsdefinert 60-minutters usikkerhetsmåling er FULLFØRT og KONSUMERT: 10 000 dag-/ukeblokktrekninger, 77 dager og 14 uker. Ingen av 60 minutter har begge simultane nedre AUC-grenser over 0,5. Ved 60min er AUC 0,521054, simultant dagbånd 0,491561–0,550547 og ukebånd 0,493632–0,548476. Ved 5min er AUC 0,500183 mot gammel 0,503021 på identiske 17 197 kvalifiserte foreldre; parrede AUC-intervaller inkluderer null. Kandidaten har svakere logloss/Brier i begge blokkskjemaer. Modellen svarer NED på 86,67 % av 5min-radene, mens faktisk OPP er 51,58 %; gjennomsnittlig OPP-sannsynlighet er 48,74 %. Én gjenbrukt utviklingsfold, og geometri/antall utganger ble endret samtidig. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DENSE_ONEPASS_UNCERTAINTY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `47f20a687853e9ce8617aa96dd1cf0a8c47987b4c580c78f1805ceaeef60a687`. Ingen nye modellkall, trening, TEST eller handel.

Desimalfortegn-audit 002 er FULLFØRT og KONSUMERT: 591 682 kildeklokker, alle 118 249 foreldre og 60 mål før 2026. Av 6 946 049 observerte celler var 976 (0,0141 %) lagret som bevegelse ved eksakt uendret desimalmidtpris; ingen reelle OPP/NED fikk motsatt fortegn. Fold1 femminutt har 26 slike TRAIN-celler og 5 senere celler; senere AUC for faste prediksjoner endres fra 0,500183 til 0,500175. Ingen retreningseffekt er målt. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_DECIMAL_SIGN_AUDIT_20261028_002/OUTPUT/RESULT.json`, SHA-256 `9f3441e79017b4e2d00656e9843af3a96ca3c2bef143275b7b28d927aed5b096`. Forsøk 001 feilet på arkivomslag og er bevart. Ingen modellforward, ny trening, målendring, TEST eller handel ble åpnet. Korrigert desimalgrid-mål er nå materialisert med separat identitet; separat kvalifisert forbruker og ny baseline kreves før modellbruk. Videre: selvstendig feature-/samspillsrevisjon.

60-minutters énpass-trening er FULLFØRT og KONSUMERT: fysisk boot 573, 47 602 TRAIN-foreldre, 2 976 optimizersteg og 17 288 senere utviklingsforeldre; ressursvakt PASS, kilde uendret og TEST ubrukt. Oppgaven er deaktivert, opprinnelig effektgrense 160 W er tilbake, og faktisk idle-trekk var 23,31 W. Kvittering: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DENSE_ONEPASS_FIT_20261028_001/RUNTIME/receipts/invocation-0001.json` SHA-256 `5f6a242dae28ffa802ca235da64d8512c0b5437f8f6fa8f45082b8f20006f51c`. OOF-punkt-AUC 1/5/10/15/30/60min er 0,501055/0,500183/0,498516/0,505191/0,511837/0,521054; simultane bånd er nå målt som beskrevet over, uten kvalifisert edge. Ved 5min er treff 48,87 % mot fast TRAIN-prior 51,58 %; logloss er 0,694562 mot 0,692740. Egne foreldremasker varierer per horisont. Ingen ny trening eller relansering er åpen.

Selvstendig feature-/målrevisjon PÅGÅR: `docs/INDEPENDENT_DIRECTION_FEATURE_AUDIT_20261028.md`. Lik desimalmidtpris kan gi falskt OPP/NED i begge målbyggerne; den faktiske frekvensen og virkningen på faste skårer er nå målt som beskrevet over. Streng H1-aggressor er tilgjengelig ved bare 36 av 47 602 beslutninger, mens delvis kjent H1-delta er tilgjengelig ved 45 214; H4/D1-streng signering er aldri gyldig i de kontrollerte kildefilene. Tre aktive D1-pris-/volumfelt har konstant TRAIN-verdi og normaliseres til null, med separat tilgjengelighetsmaske. Samlet er 29 aktive GC-verdifelt i aktuell kontekst konstante når observert i første TRAIN-prefix; dette omfatter kvalitetsfelt, og aliaser er ikke nye indikatorer. Disse funnene beviser ikke årsaken til svak AUC. Korrigert målarkiv og forhåndsdefinert usikkerhetsanalyse er fullført; målforbrukeren er nå strukturelt kvalifisert; neste steg er konkrete undersøkelser av featurebruk og samspill samt separat autentisert treningsrute. Nye genuine datalesinger krever egen binding. Bevar originalmål og checkpoints.

Privat 60-minutters native mekanikkvalifisering er FULLFØRT og KONSUMERT: fysisk boot 572, 250 W-grense, 256 initialforeldre, 32 faste TRAIN-foreldre, 4 private optimizeroppdateringer og bitidentisk gjenopptak. Alle 60 hoder fikk gradient; kart-MTF hadde gradient i første steg, lokal kartgren først fra steg 2 fordi residualutgangen starter på null. Forseglet TEST var ubrukt. Signert vakt PASS; maks faktisk GPU-trekk 132,01 W. Privat oppgave er deaktivert, original idle-grense 160 W gjenopprettet og faktisk tomgang målt 22,81 W. Kvalifiseringsvektene må aldri brukes i full trening. Egen kontroll: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DENSE_NATIVE_QUALIFICATION_20261028_001/ACTUAL_ROOT_REVIEW_001.json` SHA-256 `7b5768f2310446716f9954ab2e5887831e5dc8e6bde143c7bc4fa72156b17b24`; aggregert `docs/DIRECTION_DENSE_PRIVATE_QUALIFICATION_20261028.json`. Ingen ny prediktiv edge er målt. Den etterfølgende separat admitterte énpass-treningen er nå fullført og konsumert, se aktuell status over.

Den nye kompakte profilen gjenoppretter 32 lokale og 39 MTF-kartgeometrifelt per tidsramme, uten å fjerne de gamle kompakte feltene. Ekte dense-target-arkiv er kontrollert mot eksisterende foreldre, klokker og 1/5/15-ankre, og alle 60 mål har strikt cutoff. Kildeovergang: `docs/DIRECTION_DENSE_SOURCE_TRANSITION_20261028.json`; design: `docs/DIRECTION_DENSE_MINUTE_MODEL_20261028.json`. Kombinert endring av mål og geometri kan ikke isolere årsaken til en eventuell resultatendring. Privat engangsplan skal aldri relanseres.

Kronologisk sign-only énpass-kjøring er FULLFØRT og KONSUMERT: 47 602 TRAIN-foreldre, 2 976 optimizersteg og 17 288 senere foreldre; signert ressursvakt PASS, fysisk boot 571, TEST ubrukt. Kjøringen er deaktivert og tomgangsgrensen er verifisert tilbake til 160 W. I samme utviklingsfold er femminutters logloss 0,695463 kronologisk mot 0,693514 blandet; dette er aggregatpunkter, ikke parret rangering eller årsaksbevis. Den parrede sammenligningen er FULLFØRT og KONSUMERT: identiske foreldre, klokker, utfall og masker; 5min AUC 0,498501 kronologisk mot 0,503021 blandet på 17 197 kvalifiserte rader. AUC-differansens korrigerte dag-/ukeintervaller omfatter null; kronologisk kandidat har dårligere logloss og Brier. Dette dokumenterer ingen AUC-bedring og ingen årsak. Resultat: /home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_ORDER_PAIRED_RESULT_20261028_001/RESULT.json SHA-256 f62b5afad12c784fe681f7ee2bdd5f07fba9d721a454860627e16b85d12db3d3. 13 delte kildefiler har endret hash siden referansen; tapsfunksjon og utvalgte batching-funksjoner er AST-identiske, men full post-update-paritet er ikke målt. Siste B2-gruppering er ulik. Originale kjøringer relanseres ikke. Bevis: /home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/ORDER_CHRONOLOGICAL_SIGN_ONE_PASS_20261028_001/ACTUAL_ROOT_REVIEW_001.json SHA-256 df1668188c0c968e78790482a20d3ce8c001bf0aae201f3be30b36f0c1fcdb65.

Brukeren ønsker 250 W som fysisk øvre grense under neste separat admitterte GPU-trening; alle eksisterende temperatur-, effekt- og minnevakter beholdes, og idle-grensen settes tilbake til 160 W etterpå. Forrige trening hadde 300 W-grense, men høyeste målte trekk var 136,12 W; økt effektgrense har derfor ikke bevist fart.

Brukerens prioritet er ekte retningsmåling for hvert minutt 1–60 etter M5-close. Ekte 1–60-mål er FULLFØRT og KONSUMERT på 118 249 TRAIN/development-foreldre; originale 1/5/15-mål er byteidentiske, og fire fold-spesifikke klokke-/maturitytellinger er dokumentert. Resultat: /home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_DENSE_MINUTE_TARGETS_20261028_001/RESULT.json SHA-256 1c832da918aee94ab2535898aab340e33e771c8ad9739e97ece21599be50128b. Fold1 har 17 215/17 197/17 080/16 890/16 513 gyldige 1/5/15/30/60-minuttutfall; populasjonene er ulike. Den senere énpass 60-output-modellen er nå fullført på fold1. Fast femminuttssignal mot alle 60 utfall i fire kronologiske utviklingsfolder er FULLFØRT og KONSUMERT: ingen minutter har begge simultane dag-/ukebånd over 0,5. AUC er 0,500368/0,504000/0,505630/0,504117/0,505545/0,508159 ved 1/5/10/15/30/60 minutter; hvert bånd omfatter 0,5. Foreldrepåslag er ulike og all utvikling er gjenbrukt. Dette er samme 5min-signal over 60 mål, ikke 60 trentede utganger. Resultat: /home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_DENSE_FIXED_SIGNAL_RESULT_20261028_001/RESULT.json SHA-256 359fa314187140da43ba0de499829930f67c1d2324e994d0b4cdc8d108278136. Egne minuttutganger er nå trent i én fold; virkningen av desimalfortegn på faste skårer er målt ovenfor, mens faktisk quote-alder og eventuell effekt av retrening med korrigerte mål er ukjent. Arbeid videre etter den aktuelle selvstendige revisjonen over og ny eksakt admisjon. Bevar kompakt input, alle vakter, separat null/missing og forseglet TEST. 480-minutters kurvedekoder beregnes også i retningsløpet selv om output kastes; tidsgevinst er ikke målt, og eventuell endring krever bevart vakt- og stateparitet.

Strengt avgrenset mekanikkvalifisering 003 er FULLFØRT og KONSUMERT etter fysisk omstart 570. Original starttilstand og alle 256 initialprediksjoner var identiske, og fire private optimizeroppdateringer ga bit-for-bit identisk gjenopptak. De 32 faste blandede TRAIN-foreldrene ble sortert; begge batcher hadde observerte og ikke-null GC-input i alle 25 familie/tidsramme-grupper, og alle aktuelle gradientgrupper var aktive. Utvalget er dekningstilpasset og beviser ikke prediksjonsevne eller edge. Kontrollkvittering: /home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/ORDER_NATIVE_B16_SUPPORTED_QUALIFICATION_20261028_003/ACTUAL_ROOT_REVIEW_001.json.

Tidligere gradientdiagnostikk feilet fordi første kronologiske TRAIN-batch (foreldre 0–15) hadde null GC-masker for H4 og D1, med forventet null gradient for disse grenene. Dette forklarer kontrollblokkeringen, ikke svak markedsretning generelt. Selve den planlagte fulltreningen skal beholde alle 47 602 TRAIN-foreldre, inkludert de første 32. Fulltreningen er nå fullført og konsumert, med alle foreldrene beholdt. Effektgrensen var 300 W under kontrollen, med målt topp 132,43 W; idle er kontrollert tilbake til 160 W. Ingen fartgevinst eller retningsedge er bevist.

Rekkefølge-audit002 er FULLFØRT og KONSUMERT på 47 602 TRAIN-foreldre. Begge rekkefølger gir 2 976 ikke-tomme batcher; 5min-andelen av loss-koeffisientmassen er 33,3973 % blandet mot 33,4010 % kronologisk. Siste B2 har ved 5min én OPP/én NED blandet, men bare én gyldig OPP kronologisk; siste kronologiske 15min har ingen gyldige utfall. Opprinnelig blandet koeffisientmasse er kontrollert mot tidligere uavhengig aggregat. Dette er loss-geometri, ikke gradient, modellresultat eller årsaksbevis. Se /home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/ORDER_BATCH_GEOMETRY_ROOT_REVIEW_20261028_002.json. Ingen modell, trening, CONTROL eller TEST ble åpnet.

Selvstendig kontroll av treningsrekkefølge: Original276 og trepass-kjøringen har samme gjenskapte MT19937-orderhash; hvert prisvindu beholder stigende kausale klokker. TRAIN/later-grensen er kronologisk. Samme B2-sluttrader gjentas per pass; effekt på retningsedge er ikke målt. I fire utviklingsfolder varierer 5min-OPP-andelen fra 13,65 % til 100 %, mens faktiske OPP-andeler er 50,68–51,58 %. På 70 294 gyldige 5min-rader er samlet treffrate 50,30 % mot 51,10 % for fast TRAIN-prior. Gjenbrukte utviklingsdata, ingen intervaller; terskelskjevhet er ikke årsaksbevis. Prioritert énfaktors forslag: docs/DIRECTION_ORDER_ABLATION_DESIGN_20261028.json. Rekkefølgekandidaten og den parrede senere sammenligningen er nå fullført og konsumert. Ny trening er stengt inntil en egen nøyaktig plan er admittert.

Presisering av én-/trepass-resultatet: TRAIN-foreldrene ble presentert i
en fast MT19937-blanding, ikke i markedsrekkefølge. De to ekstra passene
gjentok denne rekkefølgen fra samme ettpass-checkpoint med optimizer/RNG
bevart. Skillet til senere utviklingsobservasjoner var kronologisk; resultatet
måler ikke kronologisk glemsel eller løpende markedsadaptasjon.

Evalueringsprotokollen krever nå både vanlig TRAIN-prior og en separat
størrelsesvektet TRAIN-prior. Dette hindrer at endret vektet klasseandel
alene tolkes som lært markedsinformasjon. Primær sammenligningsgate er uendret.

Kilde-/metadata-kontroll viser at kompakt retningsbaseline velger bort
32 lokale chart-geometrifelt og 39 slike MTF-felt per tidsramme. Samtlige
hadde ikke-konstant transformert min/maks i fold1-prefiksets eksisterende
normaliseringsmetadata; faktisk dekning og retningsverdi er ikke fastslått.
Se `docs/DIRECTION_CHART_GEOMETRY_MODEL_CONSUMPTION_20261028.json`.

Den tidligere B16-kvalifiseringsplanen er avløst av de fullførte og
konsumerte kjøringene beskrevet over. Brukerens bestilling om fysisk omstart
før hver ny separat admittert trening og 250 W øvre effektgrense består.
Bevar gjeldende effekt-, temperatur-, minne- og capped-vakter.

Effektkontroll 28.10: idle-effektgrensen var 160 W, ikke et målt forbruk; samme baseline trakk
høyst 134,7 W ved en 160 W-grense, så høyere grense har ikke bevist fart.
Policy angir 300 W maks og reduksjon til 200 W ved 80 C, mens installert
keeper-kode bruker 220 W ved innstilling over 220 W. Avviket er inaktivt
ved 160 W. Rettet vakt er installert og bekreftet: 300 W-grensen ble brukt
under kontrollen med 132,79 W målt topp, 49 C GPU-kjerne og 54 C minne.
Etter at kontrollen stoppet ble idle-grensen kontrollert tilbake til 160 W.
Forskjellige arbeidslaster gjør tallene uegnet til å hevde tidsgevinst. Se signed-guard-loggen i NATIVE_M5_DIRECTION_PREFIX_FIT_FOLD1_20261025_276.

Fast evalueringsprotokoll for størrelsesvektet retning er skrevet før
noen genuin kvalifisering eller kandidat: samme femminuttersforeldre,
parret dag-/ukeblokk-AUC, logloss og Brier, med forhåndsbundet streng gate.

Kilde for ekte B16-kvalifisering bestod 24 syntetiske kontroller og er
fryst; genuine data/modelkall er ikke kjørt eller admittért. Et første
startforsøk ble avvist før test av wrapperen og er bevart som konsumert.

Første syntetiske kvalifiseringsforsøk ble avvist før tester fordi env ble
brukt som startkommando inne i capped-wrapperen. Forsøket er konsumert;
en ny engangskontroll bruker Python direkte og bevarer samme sikkerhetsvakter.

Kilde for avgrenset ekte B16-kvalifisering er fryst for én syntetisk
kontroll. Genuine modellkall og trening er fortsatt lukket. Designet binder
den tidligere startprediksjonen og stopper før oppdatering ved ulikhet.

Den vektede treningsadapteren har bestått 20 fokuserte syntetiske
kontroller, inkludert én forward per batch og eksakt syntetisk resume.
Genuin initial-/gradient-/resume-kvalifisering og treningsadmisjon gjenstår;
ingen ekte kandidat er trent, og ingen retningsbedring er dokumentert.

Modell-/state-binding for vektet retning har bestått 146 syntetiske
kontroller. Startlogits, trenbare parametere og RNG samsvarer med baseline;
objektiv og skalabinding er egne, og syntetisk resume er kontrollert. Ingen
treningsadapter, ekte modellkall eller trening er ennå åpnet.

En bounded, TRAIN-prefiksnormalisert størrelsesvektet retnings-BCE har
bestått 12 fokuserte syntetiske kontroller. Kun tapskontrakten finnes;
ingen modellrute, ekte data eller trening er åpnet. Dette beholder 1/5/15-
baselinen og er ikke en all-1..60-modell eller bevist bedring. Se
`docs/DIRECTION_MAGNITUDE_WEIGHTED_SIGN_DESIGN_20261028.json`.

ET252-sammenligningen er COMPLETE og konsumert på samme 1 042 førstetreff-
foreldre. AUC er 0,476606 mot native 0,426768; differansen er +0,049839,
men parret ukeintervall omfatter null. Referansen har andre inputs, og
sammenligningen er selektert og gjenbrukt utvikling; ingen stabil forskjell,
årsak eller edge er bevist. Se `docs/DIRECTION_FIRST_HURDLE_REFERENCE_RESULT_20261028.json`.

En naiv partial-set-loss har en bevist degenerert løsning ved 1min/10bps:
alle 17 288 mulige utfallssett inneholder «ingen treff», så en modell som
alltid svarer dette kan oppnå null tapsgrense uten å lære retning. Fasiten er
ikke koblet til trening. Se `docs/DIRECTION_HURDLE_LABEL_IDENTIFIABILITY_20261028.json`.

Matched 5min-sammenligning er COMPLETE og konsumert: Av de samme 1 042
kjente første-treff-retningene stemmer 1 017 med sluttfortegnet og 25 går mot.
Samme skår har AUC 0,426768 mot første treff og 0,431849 mot sluttfortegn på
akkurat disse radene. Dermed forklarer ikke forskjellen mellom de to fasitene
alene svak rangering i dette selekterte utvalget. Ingen ny modell, CONTROL
eller TEST. Se `docs/DIRECTION_FIRST_HURDLE_ENDPOINT_ALIGNMENT_RESULT_20261028.json`.

Første-treff-kontrollen er COMPLETE og konsumert på 17 288 gjenbrukte
fold1-utviklingsforeldre, alle minutter 1–60 og fast 10bps gross quote. Ved
5min nådde 1 042 kjent førstegangsside, 16 230 ingen side, 14 hadde ukjent
rekkefølge og 2 manglet bane. Eksisterende 5min-skår hadde AUC 0,426768 på
de 1 042 selekterte sidene; dag-/ukeintervallene ligger under 0,5. Dette er
ikke all-foreldre-ytelse eller grunnlag for å snu signalet, og modellen er ikke
trent på fasiten. Undersøk først om sluttfortegn og første treff faktisk er
enige på samme 1 042 rader. Ingen modell, CONTROL eller TEST ble åpnet. Se
`docs/DIRECTION_FIRST_HURDLE_OOF_RESULT_20261028.json`.

Endepunktsparitet på 118 249 TRAIN-foreldre var allerede verifisert i
`CANDLE_MIDPOINT_ROOT_REVIEW_001.json` under den uavhengige evidensroten.
En ny identisk kontroll ble unødvendig gjentatt og ga samme aggregater; den er
ikke ny evidens om edge. Duplikatkilde er fjernet fra arbeidsbreet og bevart i
Git. Høy/lav har et tidligere kildeidentitetsbevis, men første-treff-fasiten,
modellkoblingen og faktisk læring er ikke kvalifisert. Se
`docs/DIRECTION_TRAIN_YEAR_QUOTE_PARITY_RESULT_20261028.json`.

Fasiten bevarer også et konservativt sett av mulige utfall ved usikker
rekkefølge. 43 syntetiske kontroller bestod og engangsadgangen er konsumert.
Å maskere ukjente treff alene gir utfallsavhengig utvalg, særlig ved 1min.
Ingen tapsfunksjon, ekte første-treff-fasit eller modellkobling er kvalifisert.

Eksisterende M1-leser åpner hele kildeklokken og kan lese en hel radgruppe før
avskjæring ved TRAIN-grensen. Separate årsfiler 2024 og 2025 er funnet via
metadata, men er ikke åpnet eller verifisert. Kontroller eksakt kildeidentitet,
tids-/radbinding og 1/5/15-fasit før ny ekte måling. Se
`docs/DIRECTION_TRAIN_QUOTE_SOURCE_REVIEW_20261028.json`.

Forskningsfasiten for hvilken side som først passerer en fast bps-grense
er implementert i eksisterende første-gunstige-quote-eier. 42 syntetiske
kontroller bestod, inkludert opp/ned, reversering til start og 1–60min.
Ukjent M1-rekkefølge og hull blir ikke fasit; intet treff er ikke en FLAT-
handlingsregel. Engangsadgangen er konsumert. Ekte data, modellkobling,
trening og faktisk prediktiv verdi er ikke målt.

Parret 1-mot-3-pass TRAIN-sammenligning er fullført og konsumert på samme
47 602 foreldre. 5min TRAIN-AUC økte fra 0,515888 til 0,533534, og logloss
falt fra 0,693095 til 0,692069, mot fast TRAIN-prior 0,692984. På senere
fold1 var AUC 0,503021 mot 0,500950; begge korrigerte differanseintervaller
omfatter null og logloss-punktestimatet forverres. Dette er en målt forskjell
mellom TRAIN-fit og senere utvikling, ikke dokumentert handelsedge eller en
entydig årsak. Se `docs/DIRECTION_MATCHED_EXPOSURE_FULLTRAIN_RESULT_20261028.json`.
Alle engangsadganger og treningsporter er lukket; ingen tung jobb pågår.

Parret full-TRAIN-sammenligning er syntetisk kvalifisert med 23 beståtte
adapterkontroller. Den gjenbruker tre énpass-prefikspartisjoner og ett trepass-
uttrekk, binder foreldre/klokker/utfall/masker og måler 1/5/15min AUC, logloss
og Brier mot énpass og fast TRAIN-prior. Dette er beskrivende treningsfit.
Ingen ekte prediksjonspayload eller modell er åpnet gjennom kildeadgangen.
Den separate engangsadgangen er fullført og konsumert, som beskrevet over.

Full TRAIN-måling for 3-pass-terminalen er COMPLETE og konsumert: 47 602
foreldre, 2 976 B16-forwards, eksakt 256-raders paritet, null optimizer- og
baklengssteg og signert vakt PASS. Windows-oppgaven er deaktivert. Foreløpig
5min TRAIN-logloss er 0,692069 etter tre pass mot 0,693095 etter ett; eksakt
radkobling og parret TRAIN-AUC er fullført som beskrevet over. Resultatet gjelder
treningsfit og viser ikke senere generalisering eller handelsedge. Se
`F_FROZEN_EXTRACTION_threepass_terminal_prefix_CAMPAIGN_20261028_001/ACTUAL_ROOT_REVIEW_001.json`
under programroten.

Den eksisterende frosne måleruten er utvidet for 3-pass-terminalens hele
TRAIN-prefiks og syntetisk kvalifisert: 277 tester bestod under CPU-tak, og
57 kildeeiere og faktiske JSON-bindinger er kontrollert uten payloadlesning.
Kildekvalifiseringen åpnet ingen ekte checkpoint, input eller modellkall;
den etterfølgende, separat admittérte målingen er nå fullført som angitt over.

1-mot-3-gjennomgangene er nå parvis sammenlignet på 17 288 senere fold1-
foreldre. Ved 5min er AUC 0,503021 mot 0,500950; både dag- og ukeintervall
for differansen omfatter null. Tre gjennomganger ga høyere logloss/Brier i
punktestimat ved 1/5/15min. Den forhåndsbundne forbedringsgaten bestås ikke.
Dette er én gjenbrukt utviklingsfold, ikke handelsedge eller en dom over 30
passeringer. Parret TRAIN-vurdering på 47 602 foreldre er nå fullført. Engangsadgangen
for sammenligningen er konsumert og alle treningsporter er lukket. Se
`docs/DIRECTION_MATCHED_EXPOSURE_PAIRED_RESULT_20261028.json`.

Den parvise 1-mot-3-gjennomgangskontrollen er syntetisk kvalifisert, med
identiske observasjoner, parrede dag-/ukeblokker og fast 5min-hovedmål for
AUC, logloss og Brier. Fast antall trekk, seed og lineær kvantilmetode er
bundet før adgang til prediksjonene. 11 adapterkontroller bestod etter siste
kildeendring; ingen ekte payload, modellforward eller optimizer ble brukt.
Den etterfølgende CPU-sammenligningen er fullført og konsumert, som beskrevet
ovenfor. Parret TRAIN-sammenligning for 47 602 foreldre er nå fullført.

Fast 10min-krysshodeanalyse gjelder fire TRAIN-utviklingsfolder: native5/15-
skårer sammenlignes med ET252 på identiske modne 10min-fasiter. Den første
syntetiske engangskontrollen bestod 8 tilfeller og er konsumert. En senere
kildekontroll fant at ukeblokker på tvers av folder ble resamplet uavhengig,
og at kalenderen ble valgt etter kvalifiseringsmasken. Kilden er rettet til
felles kalendervekter før maskering. Rettelsen bestod 10 syntetiske tilfeller,
og engangsadgangen er konsumert. Første faktiske forsøk stoppet før claim og
payloadlesing da arbeidsområdet ble endret parallelt; forsøket er konsumert.
En ny engangsplan ble fullført på 70 043 sammenlignbare TRAIN-utviklingsutfall.
På 10min-fasit er AUC for 5min-skår native 0,505640 mot ET252 0,515228; begge
99%-intervaller for forskjellen er negative, med ukeintervallet nær null.
For 15min-skår omfatter intervallene null. Skårene er ikke trent for 10min,
og gjenbrukte utviklingsdata dokumenterer ikke handelsedge. Se
docs/DIRECTION_10_CROSS_HEAD_RESULT_20261027.json. Adgangen er konsumert.

Tidligere 5min-referansesammenligning brukte felles 1/5/15/30/60-maske og
utelot 3 182 av 70 294 gyldige 5min-rader. Eksisterende per-horisontresultat
viser native AUC 0,504000 mot ET252 0,512791 på hele 5min-utvalget;
fellesutvalget viser 0,503735 mot 0,513092. Punktforskjellen består på begge;
korrekt felles ukeusikkerhet er fortsatt åpen. Se
docs/DIRECTION_FIVE_MINUTE_COHORT_REVIEW_20261027.json.


10-minuttersutfall kan nå forberedes fra to lagrede femminuttersutfall med
delt prisrad og eksplisitt avrundingsusikkerhet; bps må sammensettes, ikke
adderes direkte. 27 syntetiske kontroller bestod, inkludert prisrad-,
tids-, 15min-paritets- og bpskontroll. Engangsadgangen er konsumert.
Ingen lagrede datapayloads, modellforwards eller trening ble åpnet.


Forskningskontrakt for første gunstige quote-bevegelse er implementert separat
fra eksisterende retning og global MFE. Den skiller same-side prisbevegelse
fra gross markout etter spread og beholder bps-størrelse, M1-rekkefølge-
usikkerhet og hullmasker. Én syntetisk engangskontroll ga 18 passmarkører og
returkode 0. Kontrollprogrammets terminal ble likevel FAILED fordi dobbel -q
fjernet den forventede tekstsummeringen; originallogg og avvik er bevart og
verifisert. Adgangen er konsumert. Ingen faktiske data, modellkobling, trening
eller dokumentert retningsforbedring følger av testen.


DIRECTION_BPS_PREFIX_FIT_FOLD1_20261027_001 er COMPLETE og konsumert.
Fersk signed-bps+BCE-kandidat brukte samme 47 602 TRAIN-foreldre som baseline276:
1/5/15 minutter, B16, seed20260911, samme rekkefølge/AdamW og én gjennomgang.
Originale worker-, progress-, controller- og vaktkvitteringer samsvarer:
2 976 optimizersteg, 4 090 forwards og alle 17 288 senere utviklingsforeldre.
Worker brukte 7 277,822 sekunder. Ett av maksimalt fire vinduer var nødvendig;
ubrukte vinduer2–4 er lukket. Ingen privat kvalifiseringsvekt ble gjenbrukt.
Kilde84086af8 og policy var fryst gjennom hele kjøringen på fysisk boot560.
Modell/eier/cgroup er borte; originallåsen device78/inode35 er ledig.
Windows-task er Disabled. Alle trenings- og launchautorisasjoner er lukket.
BPS-treningsadgangen og denne sammenligningsadgangen er lukket etter konsum.
Originale artefakter, prediksjoner og checkpoints bevares.

Se docs/DIRECTION_BPS_PREFIX_FIT_RESULT_20261027.json og originale
ACTUAL_ROOT_REVIEW.json, VERIFIED_TERMINAL.json og RUNTIME-kvitteringer.
Root har kun kontrollert kilde, JSON, vaktlogg og prosess-/låsmetadata.
Den forhåndsbundne parvise sammenligningen er fullført på 17 288 senere
utviklingsforeldre. Ved5min er kandidatens AUC 0,496823 mot 0,503021 for
baseline276; korrigerte dag-/ukeintervaller for differansen omfatter null.
BPS-MSE forbedres ikke dokumentert mot både prefiksmean og null. Ingen
retningsforbedring er dokumentert. Se docs/DIRECTION_BPS_SAVED_COMPARISON_RESULT_20261027.json.

Forutgående mekanikk er kvalifisert og konsumeres ikke på nytt:
81 syntetiske prefiks-/vaktkontroller, 54-bindingers metadata-preflight og
genuin002 kontroll av delt initialisering, bps-gradient og eksakt resume.
Egen prefiksskala er 2,763416 / 5,256305 / 9,117405 bps ved1/5/15 minutter.
Ingen normalisering ble refittet, og CONTROL/TEST var lukket.

## Neste minste steg

En avgrenset oppfølging av størrelsesskåren mot eksisterende kausal H1 ATR-z-
fortegn er syntetisk kvalifisert og målt på samme 70376 lagrede foreldre.
Native oppskår AUC for abs(5min-endepunkt)>10bps er 0,551582; H1 ATR-z-fortegn
alene 0,557799, parrede 99%-intervaller for forskjellen inkluderer null. Innen
hver faste H1-tilstand er native størrelses-AUC 0,548139. Fra eksisterende
sign-aggregater er native retnings-AUC innen samme H1-tilstander 0,503991.
Parpopulasjonene er ulike; ingen av differansene er et kausalt bidrag.
Gjenbrukte utviklingsdata og midtprisendepunkt beviser ikke tradingedge.
Se docs/DIRECTION_MAGNITUDE_H1_REGIME_RESULT_20261027.json.
Kildehistorikk i policy skiller nå 19-tilfellers forgjenger fra 22-tilfellers gjeldende kilde; begge faktiske engangsplaner er konsumert.
Se `direction_magnitude_h1_regime_source` i NEXT_RUN_POLICY.json.


En kontroll av eksisterende femminuttsskårer mot større endepunkt er forberedt.
Den bruker fast +/-10 bps og alle modne, observerte foreldre, også nuller.
19 syntetiske tilfeller og én avgrenset måling på seks lagrede TRAIN-payloads
er fullført og konsumert; ingen modell eller rådata ble åpnet. På 70376 modne,
observerte 5min-foreldre er native AUC 0,536890 for opp over 10 bps og 0,440893
for ned over 10 bps med negert oppskår. Dag-/ukeintervallene er nominelle og
gjelder gjenbrukte utviklingsdata. Fra samme aggregerte parsummer er oppskårens
AUC mot stor bevegelse i begge retninger 0,551582, ET252 0,517339; egne intervaller
for størrelsesmålet mangler. R1/ref259 i råresultatet er nullplassholdere, ikke
referansemodeller. Dette peker mot mulig størrelses-/regimeinformasjon, ikke
bevist årsak eller edge. H1 ATR-z fortegnets forskjell i gjennomsnittlig
oppsannsynlighet skifter fortegn mellom folder; dette er kun en grov, etterfølgende
kildekontroll og hverken bekrefter eller utelukker regimeforklaring.
Midtprisendepunkt er ikke første treff eller handel.
Se docs/DIRECTION_FIVE_MINUTE_HURDLE_RESULT_20261027.json. Trening er stengt.
Se `direction_five_minute_hurdle_rank_source` i NEXT_RUN_POLICY.json.


Undersøk årsaken til at det nye bps-tapet ikke forbedret senere5min-rangering,
uten å velge terskel, delgruppe eller horisont etterpå. Bruk først ferdige
aggregerte resultater og eksisterende kausale kontroller. Kandidaten spådde OPP
på 2 024 av 17 197 kvalifiserte5min-rader; logloss var 0,693465 mot 0,692740
for egen prior. Dette peker mot et mulig kalibreringsproblem, men forklarer ikke
svak rangering alene. Videre ny måling krever avgrenset plan og admisjon.
Ingen automatisk flerfoldutvidelse er autorisert. En beregningshorisont er ikke
en maksimal holdetid eller handelsregel.

## Hva som er endret og fortsatt åpent

Forskningsfunksjonen for første gunstige quote er utvidet med en eksplisitt,
fast bps-grense; null beholder opprinnelig fasit. Ugunstig bevegelse måles
fra entryreferansen, ikke fra grensen. 31 syntetiske tilfeller bestod under
4G-vakt; ekte data, modell og trening ble ikke brukt. Kontroll og hash finnes
under `direction_first_favorable_bps_hurdle_source` i NEXT_RUN_POLICY.json.
Grensen gjelder strengt større bevegelse; den er en måledefinisjon, ikke
tapsgrense eller maksimal holdetid. M1-rekkefølge og faktisk nytte er åpne.


Opt-in bevarer eksisterende fortegns-BCE og tilfører tre ubundne signed-bps-
middelverdier fra samme udetechede hidden. Loss=BCE+prefiks-RMS-skalert MSE,
fast relativvekt1. Observerte nuller deltar i bps-loss; de lærer ikke FLAT.
Default sign-only-rute beholdes.144syntetiske bps- og68routekontroller bestod.
Størrelse gir ulik supervisjon til+0,1og+50bps; det beviser ikke bedre edge.
Middelverdi gir ikke usikkerhet, første gunstige treff eller ugunstig forløp.
Prisbanens eksisterende pre-MFE-MAE gjelder før første forekomst av horisontens
største gunstige ekstrem, ikke nødvendigvis før første gunstige bevegelse.
Det er en annen kontrakt enn det rene retningsforsøket. Se
docs/DIRECTION_INPUT_AND_FIRST_FAVORABLE_REVIEW_20261027.json.
RMS garanterer ikke balanserte gradienter; store feil kan dominere.

Prisfasiteier støtter minutter1–60, inkludert10, syntetisk kvalifisert med36tester.
Genuin modell og eksisterende targets bruker fortsatt1/5/15.
Ingen ekte utvidet fasit eller10/30/60-/60-outputnative er målt.
Fasit R_h=10000*(mid(D+h)-mid(D))/mid(D), D=M5bar_start+5minutter.
60 marginale utganger er ikke en felles prisbane eller intraminuttrekkefølge.
Senere short krever en ny kausal beslutning. Kontroll mot native open viser
noen null/ikke-null-avvik, men ingen snudde LONG/SHORT-fortegn på målt TRAIN.

## Beholdt årsaksevidens

Småbevegelses-majoritet ble ikke støttet:0<absR<=1bp er39,17/20,20/11,84%
av TRAIN ved1/5/15;26,2694% av første prefiks frossen batchvektet BCE.
SisteB2celle får8ganger B16cellekoeffisienten; skade fra dette er ikke bevist.
Se docs/DIRECTION_BPS_ALLOCATION_RESULT_20261027.json.

Fast lineær slutt-readout forbedret TRAIN-AUC5 fra0,515888til0,539421,
men senere fra0,503021til0,503360.99%dag/ukeintervaller inkluderer null;
senere logloss forverret ved allehorisonter. Kandidathodet innføres ikke.
Se docs/DIRECTION_FROZEN_READOUT_RESULT_20261027.json.
FirefoldnativeAUC5=0,503735motET252=0,513092 på67112samme senereforeldre;
negative korrigerte parvise intervaller.118249TRAINlokale2622felt/masker
hadde eksakt paritet. Full numerisk MTF-paritet er fortsatt åpen.
GC-flow sterk samtidigGC,svak senereXAU i avgrenset analyse.
Endrede vekter, gradienter og lavereTRAINloss beviser ikke nyttig læring.
I C003 falt samlet varians mellom foreldre rett før native276-retningshodet
med ca.98% på tre faste TRAIN-paneler fra initial til slutt. Både faktisk
informasjonstap og årsak er ubevist; konteksten mates også direkte i joint-kilden.
Treningsprefikset var tidsavgrenset, men presentasjonsrekkefølgen var en fast
MT19937-permutasjon, ikke kronologisk. Se docs/DIRECTION_REPRESENTATION_GEOMETRY_REVIEW_20261027.json.
En offline-kontroll av lav-GELU-bidrag er fullført og konsumert. På de samme
256 TRAIN-foreldrene økte den identifiserbare delmengden med lav ideal
GELU-derivert fra 13,9% til 29,0% av koordinatene. På 17 197 senere 5min-rader
var AUC 0,503021 med hele skåren mot 0,502744 uten delmengdens bidrag;
parrede 99% dag-/ukeintervaller for differansen omfatter null. Dette viser
ikke at GELU bør endres og utelukker ikke informasjonstap. Se
docs/DIRECTION_GELU_CONTRIBUTION_RESULT_20261027.json.

M5 Entry: prisforløp minutter–timer, litenMAE før førsteMFE, begge retninger/FLAT.
Fullt prosjekt bevarer åtte spesialistfamilier og D1/makro. Den valgte kompakte
276/277/279/280-profilen aktiverer sju familier: chart_geometry er parkert,
mens 61 D1/makro-felt beholdes. Dette er scope, ikke en avvisning av familien.
TRAINmai2024–desember2025; CONTROLjan–mars2026lukket.
Juni2026gjenbrukt utviklings-VAL; TESTfrajuli2026forseglet for tilpasning/utvalg.
Ingen Exit,broker,live/paper,ordre,spending ellerpromotion.
Målet er aktivt og uferdig. Teknisk PASS er ikke dokumentert retningsforbedring.

## Selvstendig retningsmål — kompakt input og manglende edge

Egen undersøkelse uten kontakt med den andre chatten. Autoritet/status ligger i
`independent_direction_edge_20261027` i NEXT_RUN_POLICY.json. Brukeren har
bekreftet kompaktprofilen som første scope. Førsteanalyse med dokumenterte funn:
`docs/DIRECTION_INDEPENDENT_FIRST_ANALYSIS_20261027.json`. Bevisrot:
`/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63`.

Candle-fasiten for 118249 TRAIN-foreldre har et lite nativeopen/legacy-midpoint-
avvik uten positiv/negativ fortegnssnuing. Sammenligningene beholder original245.
Bps-hovedtesten gir AUC5 .503021→.496823 og forhåndsregel FALSE. I et separat
utforskende ni-cellers kart er returhodets AUC1/5/15 .496283/.496262/.496172;
alle 12 korrigerte relative intervaller inkluderer null. Nullterskel gir alltid
UP på 5 minutter og alltid DOWN på 15; treffrate alene er ikke skilleevne.
Opprinnelig RESULT er gjenskapt eksakt. Ingen økonomisk edge er kvalifisert.

F-kildens strenge metadata-optin er kvalifisert med 137 syntetiske tilfeller;
368 fokuserte tilfeller er bestått totalt på sine respektive komponenter.
001 ble bevart som FAILED på grunn av feil testantall og avvist lesing av egen
outputroot. Ny002 rettet disse to forholdene og besto; ingen payload/model/fit.
En fersk metadata-preflight av `NATIVE_0U_TIMING256_RECIPE_001.json` har bestått
mot eksisterende F-kontrakt og 57 faktiske kildehashes uten payload eller modell.
`NATIVE_0U_RECIPE_METADATA_REVIEW_001.json` binder resultatet. Preflighten var ikke i seg selv en
måling av 0U-paritet og ga ikke kjøreautoritet.

Egen 0U-timing256 er fullført på ny fysisk boot med bestått maskinvakt.
Uavhengig bytekontroll mot originale 256 TRAIN-foreldre besto på alle 16
opprinnelige felt; 16 fremoverkall, null optimizersteg og null reverse.
Kvittering og bevis finnes i `NATIVE_0U_TIMING256_001/ACTUAL_ROOT_REVIEW.json`
(SHA-256 1cd4148e6db49cdbfd792938225126133f2cc2d3c3dd8747c6b2d05808b15d01).
Dette bekrefter målekjeden, ikke læring eller edge. Oppstartsoppgaven er deaktivert.

Egen 0U-prefiksdel [31744,47602) er fullført med 15858 foreldre, 992
fremoverkall, null oppdateringer/reverse og bestått maskinvakt. Uavhengig
kontroll: `NATIVE_0U_PREFIX_PART3_001/ACTUAL_ROOT_REVIEW.json`
(SHA-256 6663b1a04d3e3335c47cab5cdd0ed7b0325a44349b72faf847ea85dbbfef29d8).
Oppstartsoppgaven er deaktivert; ingen edge følger av teknisk PASS.

Den faste analysen av lagrede utdata er uavhengig reprodusert uten nye
modellkall eller fit. TRAIN5min (47308 kvalifiserte) AUC start/slutt
.507550/.515888; sluttmodellens logloss .693095 er samtidig dårligere enn
fast TRAIN-prevalens .692984 på de samme radene. Alle tre horisonters samlet
eval-BCE er .693073 mot .692944 for samme TRAIN-referanse; dette er en
beskrivende aggregasjon, ikke egen signifikansprøve.

Senere gjenbrukt utvikling5min: 17197 kvalifiserte av 17288, AUC .503021
med korrigerte dag-/ukeintervaller som inkluderer .5. Logloss og Brier er
dårligere enn fast TRAIN-prevalens; 14849/17197 prediksjoner er DOWN ved
den forhåndsdefinerte nullterskelen. Dette er ikke TEST eller økonomisk edge.
Uavhengig review `WHOLE_PREFIX_SIGN_VERIFY_001/RESULT.json` SHA-256
c5df54350f2ef39c08014252f46914f3446e3f2704363f8691180738e966e236;
aggregert diagnose `WHOLE_PREFIX_AGGREGATE_DIAGNOSIS_001.json` SHA-256
0beb6cb9f234e46ed2e2e35d3b4313dbf645ee5b67a3450dc454fdc097361ff2.
På de samme 17197 senere femminuttsradene var gjennomsnittlig oppsannsynlighet
49,504% mot 51,579% observerte opp. Logloss .693514 ligger nær konstanten
.693509 med modellens eget kohortgjennomsnitt; fast TRAIN-referanse .692740.
Dette er en beskrivende etteranalyse, ikke årsaksbevis eller kalibreringsvalg.
En strengt stigende kalibrering endrer ikke skårens AUC på samme rader.
Bevis: `PROBABILITY_LEVEL_DIAGNOSIS_002.json` SHA-256
87f7853fc719bcb9bad987d36f01a155feaa8fad22b1b39550454d379e8e58c6.
Policyens referanse til førsteanalysen er samtidig rettet til aktuell filhash;
tidligere hash pekte på en eldre dokumentversjon, ikke på en modellfil.

Eksisterende GC-M5-bok-OFI var merket observert på 17192 av de samme
17197 kvalifiserte radene; fortegn alene ga AUC .498132. Fortegn av kjent
volumdelta og bok-midrespons ga .491888/.493144. Dette er utforskende
punktverdier uten intervaller: observasjonsmasken beviser ikke full dekning
eller siste-minutt-aktualitet, og måler ikke kombinasjoner eller årsaksbidrag.
Bevis `COMPACT_GC_SIGNS_DIAGNOSIS_001.json` SHA-256
edf74e390f25fde88c0c3de64389746c70c79cdcf29215b4010417ea819a565e.

GC-M5-kildeklokke på 17128 kvalifiserte fold1-foreldre: profil-snapshot-alder
var 0 på alle; spotbasis-alder var høyst ett minutt på 17111, eldre på 13
og manglet på fire. Bokdekning var full på 16977, delvis på 146 og manglet
på fem. Siste M5-OFI-fortegn stemte med lokal input på alle valgte rader.
Profilklokken er snapshottilgjengelighet, ikke siste quote; full bokstøtte
beviser heller ikke siste quote eller executionlatens. Ingen variasjon i
profilalder og bare 13 eldre basisrader betyr at tilført verdi av ferskere
informasjon fortsatt ikke er målt. 69 kvalifiserte rader ble ikke koblet;
grense-radgrupper ble hoppet over. Kun én gjenbrukt fold, ingen modell/fit.
Uavhengig aritmetikk-/kildekontroll, uten ny rådatareplay:
`GC_EXISTING_CONTEXT_CLOCK_REVIEW_001.json`, SHA-256
ae221fc07ce2da847d62907de4fb3501afa1bfd271da3546aba99f473387e4e3.
Engangskjøringen er konsumert; CONTROL, TEST og treningsporter er stengt.

Kompaktmodellens GC-M5-sidebane inneholder allerede et felt for avstand
til siste godkjente, fullt dekkede bokminutt. I lagret pre-CONTROL-kildeutvalg
på 118720 unike M5-rader var 117992 null, 79 positive og 649 ukjent. Dette er
ikke samme populasjon som fold1-prediksjonene, og feltet beskriver ikke siste
faktiske quote eller målt nytte. Bevis `M5_BOOK_TRAILING_SUPPORT_INVENTORY_002.json`
SHA-256 4c385bf3240ce53b8e0410ac9f43d0326aba849aeb04d9dfaede998dbd057b18.

M5-profil-snapshot-alder var nøyaktig null på alle 118249 observerte
2024–2025-TRAIN-beslutningsrader, også i den overlappende fold1-delen.
Dette feltet skiller ikke disse radene og er snapshotklokke, ikke siste quote.
Kompaktprofilen har 2622 aktive signalfelt, inkludert 2410 GC-verdi-/maskefelt;
aliaser skal ikke dobbeltelles som selvstendige indikatorer. Dimensjonen
beviser ikke støy eller årsak til manglende edge. Eget bevis:
`M5_PROFILE_SNAPSHOT_AGE_DIAGNOSIS_002.json` SHA-256
cf6de22b1fe5c4ecaa2864463f90c22d75e690d896fda08b560076652ff62c71;
`COMPACT_INPUT_PROFILE_INVENTORY_001.json` SHA-256
82ae3e43660d0754d85240f9e391653bf4b68202a6856f396c6f2500b16c6c3b.

Lagret 5min-retning er sammenlignet med eksisterende 95min MFE/før-MFE-MAE
Entry-mål i fold1. På 15113 like, observerte LONG/SHORT-foreldre stemte
faktisk framtidig 5min-fortegn med etterpå-bestemt path-side i 10163 tilfeller
(67,25%; majoritetsreferanse 51,82%). Framtidig fortegn er ikke tilgjengelig
for beslutningen. Modellens AUC mot path-side er .508985; korrigerte dag-/
ukeintervaller inkluderer .5. Av 15730 gyldige path-rader er 592 hindsight
FLAT etter eksisterende 0-nytte. Dette er én gjenbrukt fold, ikke handler,
realisert økonomi eller en 95min holding-/Exitregel. 392 ikke-koblede rader
og øvrig modnings-/måldekning er beholdt i kvitteringen. Uavhengig
tidsjoin og punkttall er reprodusert; seks blokkresamples er kontrollert
med fysisk dupliserte rader. Hele 2048-replikaters intervall er ikke
uavhengig kjørt på nytt. Bevis `SIGN_VS_PATH_VERIFIED_ROOT_REVIEW_001.json`
SHA-256 2603916cfdbcffde1d8ba17ff44ef1e8565fa28edff2640c46a2bf9d8db5a980.
Engangskjøringen er konsumert; CONTROL, TEST og trening er stengt.

Den separate 5min-/95min-kontrollen er fullført og konsumert uten modell,
trening, Exit, CONTROL, TEST eller nye indikatorer. Samme gjenbrukte fold
beviser ikke generalisering eller økonomisk edge.

Engangsmålingen av siste M1s kjente kjøps-/salgsregistreringer feilet ved
lagring: tidsvariabelen ble overskrevet med en NumPy-array. Ingen AUC eller
verdiresultat ble lagret. Den ufullstendige kvitteringen viser 2562 strengt
kvalifiserte rader med eksakt fem-M1/M5-paritet, hvorav 2530 også hadde
gyldig 5min-retning, mot 17197 opprinnelige kvalifiserte. Dette er ikke
generaliserbar dekning og ukjent aggressor er ikke gjettet. Original plan
er FEILET OG KONSUMERT; minste rettelse krever ny separat binding. Bevis:
`LAST_M1_TRADE_COUNT_FAILURE_REVIEW_001.json` SHA-256
de75d6834b048a922e7f15e1b7fa4105f293718209ec48608e172aed5bfe2e18.
Ingen modell/trening, nye modellindikatorer, CONTROL, TEST, Exit eller broker.

Separat, tidsvariabelrettet 002 er fullført: på 2530 strengt kvalifiserte
rader ga fast M5-tellingsfortegn AUC .517398 og med siste M1 som forhåndsvalgt
rangeringsforfining .514226; differanse -.003172. Korrigerte parrede dag-/
ukeintervaller inkluderer null. Ingen tilført verdi er dokumentert for denne
ene testen. 2530/17197 opprinnelige rader er bare om lag 15%; ingen general
dom over ukjente/utelatte, M1-OFI, volum eller modellen. Ingen nye
modellindikatorer, fit, CONTROL, TEST, Exit eller broker. 001 er fortsatt
FEILET/KONSUMERT, og 002 er FULLFØRT/KONSUMERT. Uavhengig rådatajoin,
punkttall og alle 4096 fysiske dag-/ukeblokkresamples samsvarer. Dette er
samme fold, ikke en ny framtidsperiode. `LAST_M1_TRADE_COUNT_VERIFIED_ROOT_REVIEW_001.json`
SHA-256 55b14e74936b740988ba6047647bf9d10a69a0ce796e178980d1273ca3cde405.

Den uavhengige M1-/M5-kontrollen er fullført og konsumert uten modell,
fit, nye indikatorer, CONTROL, TEST, Exit eller broker. Ingen bedre
rangering er dokumentert for denne strenge delpopulasjonen.

Utforskende volumgyldig M1-/M5-kontroll er fullført: 16959/17197
retningsegnede rader, AUC .508177 for M5-fortegn og .507971 med siste M1
som fast rangeringsforfining; differanse -.000206. Korrigerte parrede dag-/
ukeintervaller inkluderer null. 14429 rader har ukjente registreringer i M5,
7912 i siste M1, og åtte siste minutter har ingen kjent side. Ukjent er aldri
tilordnet kjøp/salg. Fem M1-tellinger matcher M5, og tidligere 2530-raders
strenge punktresultat er reprodusert. Etteranalyse på samme gjenbrukte fold,
ikke M1-OFI, volumdelta, modellablasjon, ny generalisering eller økonomi.
Uavhengig kontroll med alternativ tidskobling og fysisk gjentakelse av alle
4096 dag-/ukeresamples gjenskaper resultatene på samme data. Kvittering:
`LAST_M1_TRADE_COUNT_VOLUME_VERIFIED_ROOT_REVIEW_001.json` SHA-256
6ee0b7f6266481ad1dbe2834e5fbe678055965f63ccba1f1499de5e2d84ab5ff.
Engangsscope er konsumert; ingen fit, modellendring, nye modellindikatorer,
CONTROL, TEST, Exit eller broker.

Den uavhengige volumgyldige M1-/M5-kontrollen er fullført og konsumert på
samme utforskende fold og populasjon. Den er ikke en ny fremtidig test;
ingen fit, modell, nye indikatorer, CONTROL, TEST, Exit eller broker.

Den originale treningskvitteringen viser 47602 foreldre og 2976 fullførte
bakover-/optimizersteg. Første og siste batch hadde ikke-null gradienter i
alle tre retningshoder og GC-gater/projeksjoner på M5/M15/H1/H4/D1. Det
avklarer kontakt i disse batchene, ikke om treningsmengden var tilstrekkelig
eller om signalene gir retning. Original kvittering: window-0001/RESULT.json
SHA-256 889103d6a81cf725e7ee4e676c0ecf294774dd6eacd7af44b77f9ab871441912.

M1-bok-OFI: Fem kvalifiserte minutter i samme kontrakt gjenskaper M5-OFI
nøyaktig på alle 17066 kvalifiserte barer. På 16977/17197 retningsegnede
rader ga M5-fortegn AUC .498154 og fast siste-M1-forfining .494998;
differanse -.003156, korrigerte dag-/ukeintervaller inkluderer null. Ingen
merverdi er dokumentert for dette fortegnet. Samme gjenbrukte 2025-Q1-fold;
ikke kontinuerlig/ikke-lineær OFI, modellintervensjon eller økonomi.
Uavhengig tidskobling og fysisk gjentakelse av alle 4096 dag-/ukeresamples
stemmer på samme reduserte historiske data, ikke en ny fremtidsperiode.
Kvittering: `LAST_M1_OFI_VERIFIED_ROOT_REVIEW_001.json` SHA-256
421f3a3b91865d8743998844b8ab69690dd9aa79f0cdb667bbda9b2b458840f9.
Engangsscope konsumert; ingen modell, trening, nye indikatorer, CONTROL,
TEST, Exit eller broker.

Den uavhengige OFI-engangskontrollen er fullført og konsumert. Ingen modell,
trening, nye indikatorer, CONTROL, TEST, Exit eller broker ble brukt.

På samme senere utviklingsfold er AUC-punktene for 1/5/15 minutt henholdsvis
.500882/.503021/.507870. Ved opprinnelig nullterskel svarer modellen NED på
90.86%/86.35%/91.51%, mens faktisk OPP-andel er 49.69%/51.58%/52.39%.
Dette er en terskelskjevhet, ikke en påvist årsak. 1/15-minuttstallene har
ikke egne intervaller og folden er gjenbrukt; ingen horisontseleksjon.
`MULTIHORIZON_SIGN_DIAGNOSIS_002.json` SHA-256
fe243fe3fe95a238026f40c30ee572e2aefeea54ca37de66b94d45c0742443a0.

Kontinuerlig M5-OFI mot fast femdobbel vekt på siste M1 er målt på samme
16977 rader: AUC .495971 versus .489661, differanse -.006309. Korrigerte
parrede dag-/ukeintervaller inkluderer null. Vekten ble valgt etter tidligere
fortegnsresultat; utforskende gjenbrukt fold, ingen modellintervensjon eller
økonomi. Resultat: `LAST_M1_OFI_WEIGHT_RESULT_001.json` SHA-256
85364044ea0d70d2c415ef7a5e8b3958741c2ec8cf16e9793e2c3c8f629752b9.
Uavhengig tidskobling og alle 4096 fysisk dupliserte dag-/ukeresamples
stemmer på de samme reduserte historiske dataene. Kvittering:
`LAST_M1_OFI_WEIGHT_VERIFIED_ROOT_REVIEW_001.json` SHA-256
3a871185b5234b38e9f18bf7c3ab2b0ae7117cb859e6c0efff571e0a1e74bbb0.
Kontrollen er konsumert uten modell, trening, nye indikatorer, CONTROL,
TEST, Exit eller broker.

Startmodellens senere prediksjoner finnes ikke. Undersøk først om en nøyaktig
matchet eksponeringssammenligning allerede finnes; ellers forbered separat
plan med treningsportene stengt. Ingen nye indikatorer, treningspass, TEST/
CONTROL, Exit, broker, ordre, spending eller promotion er åpnet. Målet er aktivt.

Et konkret 1-mot-3-pass design står i
`docs/DIRECTION_MATCHED_EXPOSURE_DESIGN_20261027.json`. Det gjenbruker terminal276
og foreslår to pass med samme data, rekkefølge, AdamW og RNG. Eksisterende
énpass-state/cursor kan ikke representere dette; passbevisst tilstand og resume
må kvalifiseres uten å svekke gammel kontrakt. Designet åpner ikke kode,
checkpoint, modell eller trening.

En integrert 1-mot-3-pass-videreføring er implementert i eksisterende state-,
cursor- og treningsløype. Den gjenbruker samme foreldre, rekkefølge, delbatch,
AdamW-historikk og RNG. 353 fokuserte og 154 tilknyttede syntetiske kontroller
bestod på aktuell kilde; opprinnelig terminalstate har identisk tilstandssegl før og etter
baseline-evalueringen. Ingen ekte checkpoint-/data-/modelltilgang eller
trening ble åpnet av denne kilderevisjonen. En ny, separat avgrenset kontroll
må først bevise ekte restore- og prediksjonsparitet under ny kilde, før en
fersk bundet kampanje kan vurderes. 1-mot-3-sammenligningen er ikke utført,
og ingen retningsforbedring er dokumentert. Eksakt status og kildehasher står
i direction_matched_exposure_continuation_source_20261027 i NEXT_RUN_POLICY.json.

Eksisterende frozen-diagnose støtter nå en separat 256-raders terminal-restore
og original prediksjonsparitet med null optimizersteg. En oppdaget manglende
kobling til kvalifisert historisk inputkompatibilitet er rettet også i selve
videreføringen. 353 syntetiske kontroller og metadatakontroll av 57 kildeeiere
bestod. Ingen ekte checkpoint-/data-/modelltilgang eller kjøring er åpnet;
fersk eksakt engangsadmisjon og alle vakter kreves for faktisk paritetstest.
Se direction_continuation_zero_update_source_20261027 i NEXT_RUN_POLICY.json.

Den nye terminal-restorekontrollen er FULLFØRT og KONSUMERT: originalens
256 faste TRAIN-prediksjoner var bitidentiske etter migrering, og modell-,
buffer-, AdamW-, RNG- og modustilstand ble bevart. Den brukte 16 forwards,
null reverse-/optimizersteg og 200,22 sekunder; originale vakt-/terminal-
kvitteringer er verifisert, oppstartsoppgaven er deaktivert og originalene
bevart. Bevis: DIRECTION_TERMINAL_CONTINUATION_ZERO_UPDATE_PROOF_20261027_001.json
SHA-256 743de80de51229ffb8ace855d288e8a66ea115364b9e9070e00b4e9676242cd7.
Dette er mekanisk paritet, ikke retningsforbedring. Den forbrukte planen skal
ikke relanseres. En ny eksakt 1-mot-3-pass-kampanje og alle vakter kreves før
trening; TEST og CONTROL er stengt. Se dynamic_m5_direction_continuation_zero_update
og direction_matched_exposure_continuation_source_20261027 i NEXT_RUN_POLICY.json.

En ny 1-mot-3-oppskrift er forberedt som metadata, med originalt mål og
samme 47 602 foreldre per pass: 5 952 ekstra optimizersteg, totalt 8 928.
Den er ennå ikke fysisk forhåndskontrollert eller admittért for trening.
Recipe SHA-256 61f0911ae432813af809a03134f1e7c7ee9b0ff1f7a05c972b256ac1a034dd45.
Ingen checkpoint eller markedsdata ble åpnet ved denne forberedelsen.

Forhåndskontrollens filvakt bestod tre positive og sju negative syntetiske
kontroller. Den separate CPU-engangskontrollen av 1-mot-3-oppskriften er
FULLFØRT og KONSUMERT: originalens to navngitte checkpoints fikk eksakt hash,
ingen ble deserialisert, og ingen modell, markedsdata eller trening ble åpnet.
En fersk kampanje med kilde-, maskinvare- og tidsvakter kreves før trening.

En konkret kildekontroll fant at videreføringens filvakt ville avvise den
opprinnelige initialtilstanden under checkpoint-/resumeverifikasjon. Den tillater
nå kun lesing av begge eksakt bundne checkpoints og avviser skriving. 354
fokuserte og 154 tilknyttede syntetiske kontroller bestod, inkludert selve
cursor-verifikasjonen under filvakten. Siden kilde er endret må en ny separat
nullstegskontroll bekrefte paritet. Forrige oppskrift og forhåndskontroll er
historiske og skal ikke relanseres. Trening er stengt.

Runnerens faste FINAL_TRAIN-måling omfatter 256 foreldre. Designets fulle
47 602-raders TRAIN-vurdering krever en separat komplett, fast prefiksuttrekking
etter 3-pass-treningen; den kan ikke hevdes fra 256-raderspanelet.

Den planlagte fellesgaten for 1 mot 3 pass krever både AUC, logloss og Brier
på dag- og ukeblokker. Feil i én komponent betyr ikke at retningsrangeringen
ikke forbedret seg. Behold fellesgaten, men forhåndsbind og rapporter 5min AUC
og sannsynlighetsskårer separat; senere data må ikke brukes til kalibreringsfit.
Se docs/DIRECTION_EXPOSURE_METRIC_REVIEW_20261027.json.

## Bps-hodets separate retningsrangering

Begge tidligere lagrede bps-sammenligninger beregnet retnings-AUC fra
logit_up; mean_return_bps ble vurdert med regresjonsfeil, ikke separat
opp/ned-rangering. Det utelukker ikke at dette andre hodet rangerer retning
annerledes. En uforsøkt, eksplorativ kontroll av de to lagrede OOF-utgangene
er forhåndsspesifisert i
/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2/DIRECTION_BPS_HEAD_RANK_DESIGN_20261027_001.json.
5min er fast primær; 1/15min er sekundære. Samme observasjoner, parrede
dag-/ukeblokker og uendret baseline brukes. Mean-bps er ikke sannsynlighet.
Seks syntetiske kontroller bestod. Engangsanalysen av to lagrede OOF-filer er
fullført og konsumert: AUC5 for bps-hodet 0,496262 mot native276 0,503021
på 17 197 identiske kvalifiserte rader. Parrede 99% dag-/ukeintervaller for
forskjellen omfatter null. Ingen bedre rangering er dokumentert; dette er
gjenbrukt utvikling, uten ny modellkjøring, trening eller økonomisk vurdering.
Ingen ny modellkjøring, trening, terskeltilpasning eller payloadlesing er
autorisert av designet. Resultatene blir utviklingsdiagnostikk, ikke edge.

Ny separat nullstegskontroll 002 er admittert på reparert kilde: 256 faste
TRAIN-rader, 16 forwards og null optimizersteg. Den tidligere 001 er konsumert.
Trening er fortsatt stengt; resultat krever opprinnelige guard-/terminalkvitteringer.

Nullstegskontroll 002 er FULLFØRT og KONSUMERT på reparert kilde. Alle 256
TRAIN-prediksjoner var bitidentiske; modell, buffere, AdamW, RNG og moduser
ble bevart. 16 forwards, null optimizersteg, vakt PASS, oppstartsoppgave
deaktivert. Bevis DIRECTION_TERMINAL_CONTINUATION_ZERO_UPDATE_PROOF_20261027_002.json
SHA-256 4afa60e4f9cac17b6b69c672491f504538b7daf19a419aeb030da8e1eba018fe.
Dette er teknisk paritet, ikke retningsedge. Fersk 1-mot-3-oppskrift kreves.

Fersk 1-mot-3-oppskrift 002 er bundet til parity-beviset på reparert kilde.
Én CPU-begrenset metadataforhåndskontroll er admittert for de to eksakt
bundne checkpoint-hashene; ingen modell, markedsdata eller trening er åpnet.
Den tidligere beståtte syntetiske filvakten gjenbrukes etter bytekontroll.

Fersk 1-mot-3-oppskrift 002 har bestått én konsumert, CPU-begrenset
metadataforhåndskontroll: to eksakt bundne checkpoint-hasher, null
deserialiseringer, modellkall eller optimizersteg. Terminal SHA-256
651a20a8f3a3c127008dbfec2fef0de54522e4499bda833e71f9828fde09a5be.
Trening er stengt til separat kampanje og fersk avgrenset admisjon.

En konkret scopekontroll fant at kampanjekontrollørens metadataoperasjoner
hasher bundne checkpointfiler utenfor capped audit. En liten valgfri capped
kontrollvei kvalifiseres før videreføring; vanlig installasjonsvei og alle
øvrige vakter beholdes. Ingen modell eller trening er åpnet.

Den valgfrie capped kontrollveien er implementert. 18 Windows-controller-
kontraktstester, PowerShell-syntaks og et faktisk CPU-capped parserstopp
bestod uten modell eller payload. Opprinnelig installasjon er uendret uten
overstyring; den nye veien må bindes eksplisitt ved neste kampanje.

Den ferske 1-mot-3-pass-kampanjen 002 er avgrenset admittert: to ekstra
pass fra original276, maks 5952 nye optimizersteg, samme mål og data, fire
tidsbegrensede vinduer og full TRAIN-vurdering som separat gjenstående steg.
Kampanjen er ikke materialisert eller startet. TEST/CONTROL og all handel er
stengt. Opt-in capped kontrollør må være bundet før den første oppstarten.

1-mot-3-pass-kampanjen 002 er FULLFØRT og KONSUMERT i to vinduer.
5952 ekstra optimizersteg og 17288 senere foreldre er scoret; andre vindu
hadde null optimizersteg. Begge signerte vakter bestod, oppstartsoppgaven er
deaktivert, trening stengt og vindu 3–4 tilbakekalt. Aggregert logloss og Brier
er høyere etter tre pass ved alle tre horisonter, men AUC og parrede intervaller
er ennå ikke beregnet. Full TRAIN-vurdering på 47602 rader gjenstår.
ACTUAL_ROOT_REVIEW_001.json SHA-256 78a123219f19c70ecdc2a2b1a5b6f5110f0563edcb5484e307e77baad1defd1f. Ingen edge er dokumentert.

En ny kildeavgrensning for parret én-/trepass-evaluering er admittert. Den
skal kontrollere identiske foreldre, klokker, utfall og masker samt beregne
5min AUC og logloss/Brier med felles dag-/ukeblokker. Ingen genuine prediksjoner,
modellkall eller trening åpnes før separat faktisk engangsadmisjon.
