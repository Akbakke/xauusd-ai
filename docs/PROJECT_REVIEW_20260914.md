# GX1: målt tilstand og prioriterte forbedringer — 14. september 2026

**Anbefaling:** Bevar arkitekturen og resultatene. Avklar og rett økonomimål, selektivitet og posisjonsbruk først. Undersøk den målte konsentrasjonen i Entry før flere lag, modeller eller features vurderes. Neste epoch skal fortsatt ikke startes.

Brukerens presisering er at Entry skal velge gode LONG/SHORT-muligheter og avstå fra svake handler; Exit skal holde når videre gevinst forsvarer risikoen og lukke når utsiktene forverres. Holdetid skal kunne følge markedet. Ingen tallfestet risikovekt, tapsgrense, posisjonsgrense eller maksimal holdetid er godkjent gjennom denne presiseringen.

Gjennomgangen bruker én agent. Undersøkt kilde er ren `033d62d6fde120f42ab48ebfd971229e42edcc58` i `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`. Handover-kontroll 07:52:19 UTC viste ingen native treningsprosess og checkpoint 315 / 19 908 optimizersteg. Første epochs uforanderlige EMA, 19 588 steg, er eneste modellgrunnlag for juni-analysen. Epoch 2 er ikke brukt til å forklare juni-resultatet.

Nye målinger er samlet i [PROJECT_AUDIT_METRICS_20260914.json](../handover_snapshot/PROJECT_AUDIT_METRICS_20260914.json). Metoden er aritmetikk på eksisterende VAL-output og inspeksjon av SHA-verifisert EMA på CPU. Ingen nye modellforwards, bakoverpass, GPU-jobber, trening eller terskelsøk er kjørt. Ingen runtime-/modellkode er endret. TEST er ikke åpnet.

## 1. Hva som faktisk er bygd

Modellen har **9 617 497 lærbare parameterverdier** fordelt på 732 parametertensorer. Første epochs checkpoint bekrefter 238 lokale signalfelter, 71 kontinuerlige kontekstfelt og 176 MTF-felter per tidsramme. Det avtalte «200 features» er altså ikke en avkortet 200-kolonners matrise. Det eksisterende fulle settet er bevart.

- Entry bruker lokal M5-historikk og M15/H1/H4/D1. Exit bruker lokal M1-historikk og M5/M15/H1/H4/D1, samt Entry-representasjon, prissti og livstidssammendrag.
- De åtte familiene er struktur/sving, SMC/likviditet, trend/EMA, volatilitet/kompresjon, momentum/flyt, sesjon/regime, chartgeometri og candle/prisaksjon.
- Koden har faktisk attention mellom familier og mellom tidsrammer. EMA og momentum kan derfor inngå i lærte samspill. Det er ikke bare åtte isolerte avstemninger.
- Lokal representasjon og global kontekst når også Entry-Q direkte. En liten vekt i siste MTF-pooling er derfor ikke bevis på at et input ikke påvirker modellen.
- Entry-Q avgjør LONG/SHORT/FLAT. Prognose-, timing-, MAE-, kvantilrisiko- og størrelseshoder er hjelpeoppgaver; de er ikke ferdig kalibrerte handelsfiltre.

Aktuelle kodeeiere: `entry_v10_ctx_hybrid_transformer.py:1124–1133,2149–2266,3658–3759`, `entry_specialist_feature_groups_v1.py:46–107` og `entry_exit_feature_base_v1.py:58–59`. Kodebaner og metadata fra [dataaudit](audit_20260912/DATA_CAUSALITY.md) er gjenbrukt; aktuelle feature-, indeks-, adapter-, sampler- og Bellman-kontrakter har ingen Git-differanse fra den auditerte kilden. Dette er ikke en ny rekonstruksjon av alle historiske featureverdier.

## 2. Nye, målte funn

### A. Entry bruker de siste vektingstrinnene svært ensidig

| Første epochs juni-EMA | Målt verdi | Hva dette betyr |
|---|---:|---|
| Dominerende lokal familievekt | Sesjon/regime: **99,7495 %** i gjennomsnitt; størst på 5 508/5 508 rader | Lokal sluttvekting er nesten ensidig |
| Dominerende MTF-familierute | **D1 × SMC/likviditet: 99,9977 %**; størst på 5 508/5 508 rader | Samarbeidspoolingen er nesten én rute |
| Effektivt antall MTF-familieruter | **1,00018 av 32** | Kapasitet til bred vekting brukes svært lite her |
| Separat Entry-tidsrammevekt | D1: **98,3582 %** i gjennomsnitt | Også dette poolingtrinnet er D1-dominert |
| Entry-featureporter ved øvre grense | **24 587 av 3 877 632**, altså 0,6341 %; 51 av 704 koordinater berørt | Oppfyller ikke eksisterende krav om åpent gateintervall |
| Exit-samarbeid | **32,6315 effektive ruter av 40** | Exit viser vesentlig bredere sluttvekting |
| Exit-featureporter ved grensene | **0** | Entry-problemet skal ikke feilaktig generaliseres til Exit |

Alle Entry-featurekoordinater har målt standardavvik over eksisterende minimum. Saturering betyr derfor ikke at disse koordinatene er konstante på alle rader. Delte porter kan dessuten få Exit-gradienter. Vi har ikke bevist permanent døde parametere.

Første epoch har fortsatt nesten ensidig Entry-ruting; dette var også et problem i seedanalysen. Attention før pooling kan blande inn andre familier og tidsrammer. Nyttig bidrag må derfor måles ved kontrollerte endringer av gyldige inputs og ved tap-/resultatendring, ikke utledes av gatevekt alene. Begrensningen ved å bruke attention som forklaring er også undersøkt empirisk i andre domener. [Jain og Wallace](https://arxiv.org/abs/1902.10186).

Eksisterende input-influence-kontrakter kan gjenbrukes, men deres åtte prober og svært små responsgrenser viser primært at input når frem. De beviser ikke positiv økonomisk nytte eller EMA×momentum-synergi. Én-familie- og parvise inngrep må sammenlignes på samme kausale tilstander; fysisk sammenhengende features må endres gjennom eieren sin. Vilkårlig nulling/permutering kan skape ugyldige markedsbilder. Begrens videre måling til de konsentrerte rutene og de relevante samspillene først; ikke start en stor kombinasjonsjakt.

### B. FLAT finnes, men modellens Q gir den ingen sjanse

Entry valgte 4 180 LONG, 1 328 SHORT og 0 FLAT. Begge handels-Q er positive på **alle 5 508 rader**:

| Rå Q i Bps | Minimum | Median | Maksimum |
|---|---:|---:|---:|
| LONG | 0,7770 | 1,2408 | 2,1059 |
| SHORT | 0,2906 | 1,0233 | 2,9455 |
| FLAT | −0,0751 | −0,0050 | 0,0525 |
| Valgt side | 0,7770 | 1,2463 | 2,9455 |

FLAT-target er eksakt null, mens FLAT-output er lært. En diagnostisk erstatning av bare FLAT-output med null gir **nøyaktig samme 5 508 handelsvalg**. Å rette denne lille FLAT-avvikelsen er derfor ikke løsningen på overhandlingen.

Kodeeier er `entry_fitted_q_v1.py:252–316,345–368`: LONG/SHORT-targets kommer fra det frosne Exit-nettets første tilstandsverdier; FLAT avslutter muligheten med null. Entry lærer dermed verdien av den eksisterende Exit-økonomien, og kan arve dens HOLD-problem. Ingen kalibrert usikkerhet eller særskilt «jeg stoler på denne handelen»-mekanisme inngår i argmax-regelen.

Q-margin er forventningsforskjell i Bps, ikke sannsynlighet. Det allerede utførte topp-10-prosent-filteret ga fortsatt negativ hypotetisk månedssluttberegning. Gjenbruk dette negative resultatet; ikke søk nye juni-terskler til en positiv variant dukker opp. Et nytt selektivitetsmål bør måle økonomisk resultat og nedsiderisiko mot andelen aksepterte muligheter. Metoder for læring med mulighet til å avstå finnes, men deres generelle resultater dokumenterer ikke GX1-lønnsomhet. [SelectiveNet](https://arxiv.org/abs/1901.09192).

### C. Dagens handelsmuligheter er ikke en gjennomførbar kontostrategi

`unified_exit_entry_policy_evaluation_v1.py:65–137` summerer like store, uavhengige Entry-muligheter. Den har ingen samlet kapitalbegrensning eller posisjonsstatus som blokkerer en ny inngang.

Ny aritmetikk over de allerede lagrede handelstidene gir **3 287 samtidige valgte forløp** på topp, første gang 30. juni 14:05 UTC. Beregningen bruker fill fem minutter etter M5-start, faktisk Exit-tid og åpen posisjon frem til observert månedsslutt ved avkorting. Dette er overlapp i hypotetiske handler, ikke registrert kontobelåning.

En enkel kronologisk adgangsregel med høyst én posisjon, brukt på de samme lagrede forløpene, ville bare akseptert første LONG og avvist de neste 5 507 inngangene: første handel holdes måneden ut. Dette er en diagnostisk avspilling uten ny modellrespons, ikke en validert ny strategi. Funnet viser hvorfor færre innganger alene ikke reparerer dårlig Exit.

**Anbefaling:** Begynn med én samlet XAUUSD-posisjon, uten pyramidering, som forslag til første realistiske utførelsesregel. Ikke aktiver dette uten at det er del av avtalt mål. Læring/evaluering må samsvare med regelen. Verdien av å vente på en senere mulighet må behandles dersom kapitalen bindes i én handel; dagens uavhengige FLAT=0 representerer ikke denne fremtidige muligheten. Dette kan uttrykkes med eksisterende Entry/Exit-hoder og riktig tilstand/targets, men er mer enn et ettermontert terskelfilter.

### D. Aktivt hode betyr ikke at prediksjonen er god

| Lagret juni-metrikk | Resultat | Tolkning |
|---|---:|---|
| Entry-Q MSE mot rapportert konstant gjennomsnitt | **2,66 % svakere** | Teknisk aktiv Q er ikke dokumentert god verdiprediksjon |
| Forecast MSE mot samme type baseline | **0,85 % svakere**; Pearson **0,0173**, rank-korrelasjon **−0,0369** | Svakt aggregert signal i de fire forecast-outputene |
| Timing MSE | **20,29 % lavere** | Noe hjelpeprediksjon ser nyttig ut; dette er ikke handelsgevinst |
| Volatilitetsprognose MSE | **23,05 % lavere** | Volatilitetsinformasjon bør undersøkes før flere features legges til |
| Trendline-event | AUC **0,8194**, ECE **0,0125** | Et hendelseshode har nyttig separasjon på sin etikett, ikke bevist side-/gevinstsannsynlighet |

De rapporterte regresjonsmålene kan blande sider og horisonter. Konstant gjennomsnitt er en beskrivende VAL-referanse, ikke en separat trent sammenligningsmodell. Forecast lærer L1, så MSE alene avgjør heller ikke om treningsoppgaven er løst. De svake korrelasjonene styrker behovet for en målrettet vurdering av retning/timing.

**Unngå en konkret feiltolkning:** Tail-risk lærer 90-prosentkvantil med pinball-tap. Dets høye MSE mot gjennomsnittet beviser ikke dårlig kvantilkalibrering. Riktig videre måling er kvantiltreff, pinball-tap mot TRAIN-frosset kvantilbaseline og resultater per side/horisont. Slike mål kan ikke gjenopprettes fra bare lagrede aggregater. Heller ikke dip-kvantiler skal bedømmes som forventningsprognoser.

Entry-MAE gjelder 95 minutter og er ikke en tapsgrense for flerdagers handler. «Tillitsfull handel» må skilles fra både denne prognosen og lærte oppgavevekter.

### E. Entry må måles mot markedet uavhengig av Exit

Brukeren har etter gjennomgangen uttrykkelig bestilt videre konkret arbeid og
presisert at Exit ikke skal måtte reparere systematisk dårlige innganger.
En ny beregning gjenbruker de allerede lagrede LONG/SHORT-prisstiene fra første
epochs EMA. Ingen ny inferens, terskeltilpasning eller trening er gjort.
Bevis: [ENTRY_INDEPENDENT_QUALITY_20260914.json](../handover_snapshot/ENTRY_INDEPENDENT_QUALITY_20260914.json).

| Fast diagnosehorisont | Felles gyldige rader | Valgt sides gjennomsnittlige netto Bps | Valgt side best av LONG/SHORT |
|---|---:|---:|---:|
| 15 minutter | 5 508 | −6,3459 | 49,0378 % |
| 60 minutter | 5 508 | −8,3884 | 47,9484 % |
| 240 minutter | 5 171 | −18,4727 | 42,2162 % |

Dette måler utfallet av de faktiske Entry-valgene ved forhåndsvalgte horisonter,
uten å bruke modellens Exit-tid. Det er ikke samlet policy-PnL eller bevis på at
den beste horisonten kan handles. Andelen beste side er en sammenligning av to
observerte nettoforløp, ikke en kalibrert sannsynlighet eller et universelt krav
om 50 % treff. Avkastningens størrelse, kostnader og markedsfordeling teller også.
Gjennomsnittet for valgte innganger er negativt i alle de lagrede kalenderdelene
ved alle tre horisonter. Dette er én utviklings-VAL, ikke flere uavhengige tester.

For 60 minutter gir alltid LONG −8,5966 Bps, alltid SHORT −2,5746 og FLAT 0.
Entry er marginalt bedre enn alltid LONG, men svakere enn de to andre faste
referansene. Ved 240 minutter er alltid SHORT positiv på det felles utvalget;
det gir ikke grunnlag for å velge en SHORT-regel ut fra juni i ettertid.

Den aktuelle koden ble kontrollert på ren `2b6b7b51`:

- `entry_fitted_q_v1.py:252–316` setter LONG/SHORT-target lik detached første
  Exit-tilstandsverdier og FLAT-target lik null. Dette er Exit-avledet
  handelsverdi, ikke en uavhengig markedsfasit.
- `unified_exit_random_access_training_v1.py:534–583` bygger broen fra det
  frosne Exit-nettet. `entry_v10_ctx_train_v3.py:9201–9206` fører dessuten
  Exit-gradienter tilbake til Entry-representasjonen. Detached targets betyr
  derfor ikke at hele Entry er gradientmessig isolert fra Exit.
- `build_entry_v10_ctx_training_dataset_v3.py:521–533` lager faktiske fremtidige
  close-til-close-returer for K=1/5/12/24 M5-bars (5/25/60/120 minutter).
  `entry_v10_ctx_train_v3.py:476–489` trener eksisterende forecast-hode med L1
  mot disse etikettene. Dette er prisprognoser uten full handelskostnad og har
  ingen selvstendig myndighet til å velge LONG/SHORT/FLAT.

**Anbefaling:** Behold økonomisk samarbeid, men krev selvstendig målbar
Entry-kvalitet. Avdekk først svikten i eksisterende prisprognose og representasjon;
ikke opprett et duplisert forecast-hode eller gjør frakobling av Exit-gradienter
til en uprøvd standardløsning. Rå Q kan verken bevise retningssignal eller trygg
inngang når fasiten selv bygger på den feilende Exit-økonomien. Separat
markedssupervisjon finnes allerede, men nytten er foreløpig svak. Kontroller
horisontvis signal og relevant gradient-/inputpåvirkning med eksisterende eiere
når tillatt måleprofil er klar. Ingen årsakssammenheng mellom delingen og svakt
signal er ennå målt; ingen nye treningsmål er aktivert.

## 3. Økonomimålet må korrigeres konsistent

Det eksisterende funnet består: SHORT-HOLD gir null løpende belønning, tapsrealisering gir negativ belønning, og ingen økonomisk slutt tvinger tapet inn. Mer trening alene dokumenterer ingen løsning. [Ferdig Entry/Exit-analyse](ENTRY_EXIT_REVIEW_20260914.md).

Min anbefalte målretning er **endring i kostnadsjustert posisjonsverdi, med uttrykkelig vurdering av nedsiderisiko**. Åpne gevinster og tap skal telle økonomisk. Exit skal sammenligne forventet videre verdi med lukking nå. Det gir rom for en god trend og økonomisk grunn til å forlate en forventet ugunstig utvikling.

Dette må innebære følgende før implementering:

1. Definer én konsistent verdi ved gjennomførbar bid/ask, fra faktisk fill. Spread, slippage og finansiering skal belastes én gang. Når gevinsten allerede er ført løpende, kan Exit ikke få hele gevinsten en gang til.
2. Vis regnskapsidentiteten på uendrede prisstier: summen av verdiendringer, kostnader og nødvendig sluttverdi skal stemme med økonomisk resultat. Test både sider, umiddelbar Exit, flere HOLD, markedsgap og naturlig avkorting.
3. Samordne med dagens tidsdiskontering. En naiv prisdifferanse under diskontering er ikke automatisk lik dagens terminale kontantstrøm.
4. Avkorting betyr manglende fremtidig observasjon. Markedsverdien ved siste observasjon kan måles separat uten å kalle den modellstyrt Exit. Ingen kunstig 512-, minibatch- eller månedssluttregel skal snikes inn.
5. Bestem om resultatvalg skal bruke realisert pluss åpen markedsverdi i en kronologisk strategi. Dagens full-policy-score er utilgjengelig ved én valgt åpen handel; med tålmodig Exit kan patience 5 ellers forbrukes uten et valgbart checkpoint. Dette krever en eksplisitt ny metrikkavtale, ikke ommerking av gammel juni-score.

En presisering til samtalens forslag: Å bare flytte belønning tidligere kan være *reward shaping* som bevarer samme optimale policy. Da fjerner det ikke nødvendigvis målkonflikten. En endring som faktisk gjør åpent tap økonomisk tellende må avklare målet, sluttverdien og grensene, ikke bare gi tettere feedback. Potensialbasert shaping og policyinvarians er behandlet av [Ng, Harada og Russell](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf). Anvendelsen på GX1 her er vår analyse, ikke et publisert GX1-resultat.

Jeg anbefaler ikke en vilkårlig maksimal holdetid som hovedløsning. Vi trenger fortsatt et valg om hvor mye forventet nedside vi aksepterer for oppside. En eventuell absolutt tapsgrense er en separat sikkerhetsregel, med utførelse ved tilgjengelig pris og samme behandling i trening/evaluering. Tall skal fastsettes før ny evaluering, med TRAIN-grunnlag og brukerens risikoramme; juni skal ikke brukes til å velge en heldig grense.

Den tidligere observerte første-minutt-feilen i LONG-finansiering er omtrent **0,001026694 Bps per handel**. Den er liten og forklarer ikke tapene. Når økonomieieren uansett må endres, bør dette intervallet behandles riktig fra fill, samtidig i Entry-target og evaluering. Ingen gammel resultatfil skal overskrives.

### Implementert kandidat etter nytt arbeidsmål

Den observerte økonomifeilen er nå rettet i en eksplisitt, **inaktiv** kandidat
i eksisterende kode. Regnskapsvalget heter
`liquidation_value_increments_v1` og er bundet til økonomikontrakt v3 og
økonomisteg v2. Byggeren krever et eksplisitt valg. Eksisterende v2-kontrakter,
standardadferd, frosne resultater og checkpoints er ikke ommerket.

La L_t være gjennomførbar netto lukkeverdi fra fill, inklusive tur-retur-
kostnad og første minutts finansiering, men eksklusive finansiering allerede
ført ved tidligere HOLD. La f_t være neste intervalls finansieringskostnad,
r_t det eksisterende risikofradraget, og V_neste beste lovlige Q ved successor.
Q-feltene beholder verdiskalaen; ingen hoder eller features er erstattet:

```text
Q_exit(t) = L_t
Q_hold(t) = -f_t - r_t + (1-gamma_t)*L_neste + gamma_t*V_neste

Q_hold(t) - L_t
  = (L_neste - L_t) - f_t - r_t + gamma_t*(V_neste - L_neste)
```

Dermed sammenlignes HOLD med forventet videre prisendring, kostnad og risiko,
uten at realisering av et gammelt tap i seg selv gjør HOLD bedre. For en
avsluttet sti er den diskonterte målsummen eksakt L_0 pluss diskonterte
verdiendringer minus finansiering/risiko. Kontantresultatet summeres separat
uten denne ikke-kontante korreksjonen. Ingen gevinst eller kostnad telles
to ganger. Den tidligere manglende første-minutt-finansieringen er inkludert
én gang i v3, i både Exit-target, videre Entry-verdi og evalueringsregnskap.

En konstant posisjonsverdi på -100 Bps gir fortsatt -100 ved ubestemt kostnadsfri
HOLD, mens det gamle målet gir null for denne policyen. Ved flat pris og null
kostnad er HOLD og EXIT likeverdige; dette er ingen ny maksimal holdetid.
Et videre prisfall gjør HOLD dårligere dersom fortsettelsesverdien gjenspeiler
fallet. Modellen må fortsatt lære dette riktig. Ved eksisterende årlige
diskonteringsrate er korreksjonen per M1 liten; vi har ikke bevist raskere
kredittildeling eller løst Entrys svake markedssignal.

42 målrettede tilfeller besto under eksisterende 4 GiB audit-vakt, CPU 0–7,
én numerisk tråd. Først 39 relevante tilfeller, deretter bare tre nye
integrasjonstilfeller. Verifikasjon omfatter regnskapsidentitet med lukkeintervall,
begge sider, riktig én-gangs-finansiering, byte-identiske FP32-belønninger i
scalar/vector-banen, Entry-avhengig cache, bevart naturlig censurering,
Bellman→Entry-bro, VAL-regnskap og v3-byggerens TRAIN-/VAL-binding.
[Maskinbevis og kilde-/logghasher](../handover_snapshot/MTM_OBJECTIVE_VERIFICATION_20260914.json).

**Gjenstår før bruk:** Full-policy-evalueringen må fortsatt få eksplisitt åpen
markedsverdi og kronologisk posisjons-/resultatdefinisjon. Å legge sluttmarken til
måleresultatet skal ikke late som modellen valgte EXIT, og naturlig avkorting
skal ikke gjøres til terminal i treningsfasiten. Risikoavklaring og korrekt
overføring til nytt optimaliseringsmål gjenstår, sammen med GPU256-paritet,
samlet fart og resume-bevis. NEXT_RUN_POLICY.json er uendret; ingen trening,
ny inferens eller TEST-tilgang er brukt til denne rettelsen.

### Native evaluering og cache — neste verifiserte trinn

Den inaktive økonomivarianten har nå native resultat v3 med en separat
valuation for hver posisjon. Ved split-slutt er nettoverdien bokført cash pluss
gjennomførbar gjenværende lukkeverdi. For utility legges den samme sluttverdien
til med akkumulert diskontering. EXIT-markørene og cash-ledgeren endres ikke:
en siste prisobservasjon er ikke et modellvalgt salg. Ukjente datagap og
ufullstendig beregning kan ikke oppgraderes til komplett månedsscore.

Det rapporteres både uavhengige muligheters markerte verdi og en kronologisk
kontroll med én fast posisjon om gangen. Kontrollen behandler kjent EXIT før
Entry ved samme klokke, uten pyramidering eller rentesrente, og teller modellens
FLAT separat fra muligheter som ble blokkert av en opptatt posisjon. Den er
uttrykkelig ikke innført i treningsfasiten eller early stopping. Endelig avtale
for posisjonsbruk, kapitalens alternativverdi og resultatvalg gjenstår.

En nødvendig følgefeil i cache ble også rettet: inngangsavhengig mark kunne gi
én ferdigberegnet HOLD-cacheoppføring per inngang og intervall. Nå gjenbrukes
bare felles intervallkostnader, og marken beregnes på nytt. En regresjon med
400 forespørsler, 100 innganger, to sider og to intervaller ga to oppføringer
med eksakt likhet mot scalar-banen i begge rekkefølger. Faktisk minnebruk og
samlet gjennomstrømning på full VAL er fortsatt ikke målt.

57 målrettede CPU-tester besto, inklusive syntetisk native batch256 pause/resume
med åpne tap, ukjent gap og compute-avkorting. Et kontrolltilfelle med to lukkede
gevinster og ett åpent tap ender på -13,5 Bps; en overlappende +1000 Bps-handel
får ikke plass og kan ikke forbedre resultatet. Dette er en syntetisk test av
korrekt beregning, ikke GX1-avkastning. Resultatvalidatoren avviser endrede
summer selv når ytre JSON-hash er beregnet på nytt.
[Kilde-/loggbundet bevis](../handover_snapshot/MTM_VAL_VERIFICATION_20260914.json).

Ingen treningsaktivering, ny analyse av det faktiske GX1-checkpointet eller
TEST-tilgang er gjort. V2-resultater og gammel juni-score beholder sin opprinnelige
betydning. Risikoavklaring og de øvrige portene gjelder fortsatt.

## 4. Læringsforløp og alternative metoder

En full epoch har 313 399 Entry-par og fire samplede Exit-overganger per par, begge sider, pluss første-tilstandsanker. Dette er 1 253 596 samplede overgangstilstander; ikke alle mulige minutter av alle handler. Sampleren har allerede aldersgrupper og ingen påvist uniform-minutt-feil.

Bellman-target bruker ett observert successor-steg. Target-nettet oppdateres ved ny epoch, etter 19 588 optimizersteg; `entry_v10_ctx_train_v3.py:13135–13149`. Dette er en mulig flaskehals for å føre langsiktige konsekvenser tilbake til Entry. Nettverksgeneralisering og seed påvirker dette, så det er **ikke** en hard 30-minutters horisont ved 30 epocher. Mål residualer og verdi mot faktisk fremtid før frekvens eller targetmetode endres.

Første epochs EMA viser at tapsvektene faktisk er lært: effektiv faktor Entry-Q 8,189, Exit-Q 0,234 og forecast 0,101. Oppgavene har ulike enheter og tapstyper; faktorforholdet er ikke et mål på gradientdominans eller handelskonfidens. Vektingen er allerede av typen lært usikkerhetsvekting. [Kendall, Gal og Cipolla](https://arxiv.org/abs/1705.07115).

Dagens EMA-koeffisient etter én full epoch er omtrent 0,36787 på initial EMA-tilstand i den lineære rekursjonen. Det beskriver treghet i vektene, ikke at 36,8 % av beslutningene skyldes seed. Ikke bytt EMA, seed eller gjenstart trening på dette grunnlaget alene.

| Prioritet | Mulighet | Minste begrunnede neste steg / beslutningsregel |
|---|---|---|
| P0 | Konsistent økonomimål, åpen verdi og økonomisk resultatvalg | Avtal verdi-/risikodefinisjon og lag smal regnskaps-/targetverifikasjon i eksisterende eiere |
| P0 | Realistisk posisjonsbruk og selektiv Entry | Avtal første posisjonsregel; mål faktisk kronologisk resultat, eksponering og avståelsesandel med samme regel |
| P1 | Gatekonsentrasjon og saturering | Gjenbruk input-influence, mål gyldig familie/TF-påvirkning og gradient ved de berørte portene; rett først påvist skala-/læringsproblem |
| P1 | Bedre læring fra lang prissti | Sammenlign targetresidualer før eventuell hyppigere targetoppdatering eller begrensede flertrinnstargets; ingen ny full treningsarkitektur |
| P1 | Bedre Entry-tillit | Evaluer avståelse mot nettoresultat og nedsiderisiko, per side og regime. Kalibrering må være tidsmessig utenfor modellens fit. Rå Q eller softmax er ikke nok |
| P1 | Riktig måling av risikohoder | Kvantiltreff/pinball og underestimerte tap per side/horisont; koble ikke 95-minutters hjelpehode direkte til ubundet livstid |
| P1 | Robusthet gjennom tid | Bruk forhåndsbestemte kronologiske fit-/kalibreringsvinduer innen TRAIN for nye forsøk. Juni er brukt utviklings-VAL; TEST forblir forseglet |
| P1 | Kostnads- og gapsensitivitet | Gjenbruk bundet bid/ask og tillatte slippage-scenarier. Sammenlign eldre TRAIN med mange gap mot nyere sammenhengende perioder; ikke syntetiser ukjente priser |
| P2 | Enkle sammenligningsmodeller | En forhåndsdefinert EMA/momentum-regel, regularisert lineær modell eller gradientboostede trær på samme kausale data kan teste om kompleksiteten gir merverdi. De erstatter ikke alle features i hovedmodellen |
| P0 | Selvstendig Entry-kvalitet mot faktiske priser | Fast-horisontkontroll er utført i del 2E. Undersøk eksisterende forecast-signal før nye hoder, og skill prisfasit fra Exit-bootstrap; ingen hindsight-optimal Exit som urealistisk fasit |
| P2 | Fordeling av fremtidig verdi | Kvantil-/distributional Q kan gi informasjon om nedside utover et gjennomsnitt. Vurder først om eksisterende risikohoder kan brukes riktig; nytt hode krever ny evidens |
| P2 | Hysterese, hendelsesbaserte innganger eller re-entry-regel | Vurder ved målt re-entry-churn etter økonomirettelsen. Færre bars eller fast cooldown velges ikke fordi juni ser bedre ut |
| P2 | Tap-/gradientbalansering og mindre negativ deling | Mål oppgavevise gradienter før GradNorm, endret loss eller delvis separasjon av representasjoner. Ikke anta at alle hjelpeoppgaver hjelper Entry |
| P3 | Ensemble, ny backbone, større modell, PPO/SAC eller separat modell per regime | Bare dersom billigere kontroller viser en bestemt begrensning. Mer kapasitet, flere modeller og mer GPU reparerer ikke et galt mål |

Flertrinnsreturer, fordeling av Q, selektiv prediksjon og gradientbalansering er dokumenterte generelle metoder, ikke dokumenterte forbedringer for GX1. Relevante primærkilder: [Fedus mfl.](https://proceedings.mlr.press/v119/fedus20a/fedus20a.pdf), [Dabney mfl.](https://arxiv.org/abs/1710.10044), [Geifman og El-Yaniv](https://arxiv.org/abs/1901.09192), [Chen mfl.](https://proceedings.mlr.press/v80/chen18a/chen18a.pdf). Studier av tabulære datasett begrunner en rimelig trebaseline, men beviser ikke at trær slår GX1s tidsseriemodell: [Grinsztajn mfl.](https://arxiv.org/abs/2207.08815).

Forsøk må begrenses og telles. Med mange prøvde regler kan beste backtest være et seleksjonsresultat; dette er særlig relevant når juni allerede er brukt flere ganger. [Bailey mfl.](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf). Et historisk TRAIN-vindu som dagens modell allerede har sett, kan ikke kalles out-of-sample bare fordi analysen kjøres senere. Ekte kronologisk sammenligning krever at fit stoppet før vurderingsvinduet.

## 5. Datagrunnlag, drift og gjenopptakelse

Tidligere verifisert kilde-/metadataaudit støtter kausale lukkede klokker, eksakt parent/child-mapping, full inputdekning og TRAIN-eid normalisering. Native feature- og kontrakteiere er fortsatt de samme på de undersøkte punktene. Den gamle operative teksten om å fortsette trening er historikk og gjelder ikke nå.

Vesentlige begrensninger består: 3 136 ukjente kildegap i TRAIN (1 985 fra 2021), ingen slike gap i juni; bevart v7 base-normalisering, ikke den medfølgende v8-transformen; prospektivt kostnadsscenario, ikke bevist historisk broker-PnL. Ingen ny full feature-rekonstruksjon var nødvendig for denne gjennomgangen. Tidligere trening/resume-bevis og beståtte paritetstester er gjenbrukt, ikke kjørt om igjen.

| Krav før videre trening | Nåværende bevis |
|---|---|
| Økonomi-/risikomål | Prinsipp og anbefaling konkretisert; tallfestet risiko, posisjonsregel og ny resultatmetrikk ikke fastsatt |
| GPU-paritet med VAL 256 | Mangler; CPU-like tilstandsbytes beviser ikke like GPU-Q/handlinger |
| Samlet fart med åtte CPU-arbeidere og VAL 256 | Mangler; eksisterende CPU-trinn er målt +24,65 %, ikke hele pipeline |
| Korrekt gjenopptakelse | Ordinær fremgang fra checkpoint er tidligere observert; eksakt numerisk likhet mot uavbrutt løp for neste oppsett mangler |
| Gatehelse og økonomisk gyldig kandidat | Første epoch har både valgt naturlig avkorting og Entry-saturering; ingen godkjent lønnsom modell |

Bare native campaign gjennom `gx1_capped_run.sh` er tillatt videre. Profilen er fortsatt VAL 256 / åtte CPU-arbeidere / 10 800 s VAL, native 12 000 s, ytre vakt 13 800 s, FP32 uten TF32 og eksisterende caches, klokkeprofil og maskinvarevakter. Timesvis modellobservasjon ved lange kjøringer. Ingen alternative gamle VAL-/smoke-/treningsveier er brukt.

**Endret økonomimål er ikke eksakt gjenopptakelse av samme optimaliseringsproblem.** Vi må skille videreføring med uendret mål og verifisert resume fra overføring av bevarte vekter til et nytt, eksplisitt bundet mål. Gamle targets, optimizer-/EMA-tilstander og kontrakt-SHA-er skal ikke ommerkes som om intet var endret. Velg minste dokumenterte overgang når målet er avklart; bevar checkpoint 315 og epoch-1-EMA uforandret.

## 6. Foreslått rekkefølge og avgrensning

1. Fastsett økonomisk verdi, risikohåndtering, posisjonsregel og hvilken resultatmetrikk som tillater åpne posisjoner. Holdetid kan fortsatt følge modellens markedsvurdering.
2. Gjør bare nødvendig endring i eksisterende økonomi-/target-/evalueringskode. Verifiser regnskap, grenser og Entry–Exit-samsvar med små deterministiske tilfeller før modellkjøring.
3. Mål de konkret påviste Entry-problemene: konsentrerte porter, svakt retursignal, avståelse og relevant risikokalibrering. Gjenbruk eksisterende verktøy og velg én rettelse om gangen når årsaken er målt.
4. Dokumenter GPU-paritet, samlet fart og resume for den eneste tillatte profilen. Disse er fortsatt porter, ikke antatt bestått.
5. Start først deretter avtalt full femårstrening, opptil 30 epocher, VAL per epoch og patience 5 under avtalt resultatmål.

Denne gjennomgangen gir et kontrollert kart over de viktigste observerte svakhetene og relevante alternativer. Den beviser ikke fravær av alle feil eller at en uprøvd forbedring vil gi gevinst. De åpne målingene er eksplisitt navngitt. Det er ikke grunnlag for å starte en stor omskriving eller å love positiv Bps.
