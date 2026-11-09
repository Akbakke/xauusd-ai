# CURRENT_HANDOVER — GX1

Arbeidssted: `/home/andre2/src/GX1_CURRENT`, gren `work/gx1-current`. Les
[GX1_RULES.md](GX1_RULES.md) før arbeid og kjør `bash scripts/gx1_handover.sh --check`.
[NEXT_RUN_POLICY.json](NEXT_RUN_POLICY.json) er eneste maskinlesbare arbeidsstatus;
terminalkvitteringer og faktiske prosesser overstyrer tekst her. Eldre arbeid,
feil og kvitteringer finnes i Git og på sine opprinnelige artefaktstier. Åpne dem
bare ved et konkret avvik eller en ny nødvendig gatekontroll.

## Mål og gjeldende port

Lær løpende framtidig OPP/NED/FLATT markedstilstand **før** Entry-plassering.
M5 er prognosetakten; kausalt lukkede M15/H1/H4/D1 og eksisterende genuine
indikatorhistorikk er input. Alle fem TF har egne framtidige mål. Nåværende
forskningsfokus er SMC og teknisk analyse etter brukerens vedtak 08.11; GC-planen
er utsatt. Entry, Exit, ordre, broker, handel, spending og forseglet TEST er stengt.

Policystatus: TA_M5_EQUAL_OBSERVATION_SAVED_LOSS_AUDIT_ADMITTED.

Én ny CPU-kontroll er nå innrømmet av PLAN.json i samme loss-rot:
alle1849lagrede FIT-batcher,3698rålogit-autogradkall,0modell/optimizersteg.
S/F-normalisering, full uavhengig taps-/gradientformel og uendrede øvrige
koordinater kontrolleres.4GiB/512MiB/én tråd/900s. Ingen senere arrays.

M5-sidevektingen er nå implementert som eksplisitt opt-in med fast fold1
TRAIN-binding. Hver kjent M5-observasjon får q/N før eksisterende TF-divisor;
samlet tapmasse beholdes. Aktivitet, øvrige TF, input, mål, scorer og gamle
standardkontrakter beholdes. Siste batch på seks rader behandles likt.
42 fokuserte syntetiske tester består, inkludert gradienter, akser før
forward, gamle/nye state-rundturer og avvist kryss-restore. Syntaks består.
[Kildekontroll](/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_EQUAL_OBSERVATION_LOSS_20261109_001/SOURCE_VALIDATION.json).

Dette er ennå ikke prøvd gjennom alle lagrede FIT-logiter eller native fit.
Neste nødvendige kontroll er én avgrenset CPU-kontroll på lagrede FIT-logiter,
uten modell/optimizer eller senere data. Ny varmstart må eksplisitt binde
det verifiserte importerte checkpointet til den nye tapskontrakten; eksisterende
streng gammel restore må ikke omgås. Ingen fit er innrømmet; kun den avgrensede lagrede-logit-kontrollen ovenfor. AdamW er uendret; Newton med koblet L2 fra readout-forsøket er
ikke gjenskapt av denne vektingsrettelsen.


Den nye [nullstegsimporten](/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_EQUAL_WEIGHT_IMPORT_20261109_001/REVIEW.json)
har native PASS og et uavhengig verifisert checkpoint. Kampanjekontrolleren
feilet separat med exit1 og ga ingen campaign-receipt. Planen er konsumert,
oppstartsoppgaven deaktivert; ingen kjøring eller trening er nå innrømmet.
Original RESULT/COMPLETE-terminal/progress/checkpoint/ACTIVE_INVOCATION er
bevart. Ingen gjenkjøring eller omskriving til kampanje-PASS.

Native kontroll tok128,227s: streng original FINAL CUDA-restore, eksisterende
dedicated-migrering, bare fire importerte FP32 M5-tensorer, seks fulle forwards
på de faste32TRAIN-radene og5918native head-kall på alle29574/17753cacherader
for begge mål.0backwards/optimizersteg. Modellkropp/cache-representasjon og
øvrige outputs var bitlike; checkpoint-rundtur og resume-outputs var bitlike.
Ny modeldigest ae772fd755cbfa07b1b7bcca8bccd2c8a8889806989a289ddb8267972c281b37.
Nytt checkpoint616a22e467ebf35bded61705e00f99531d28420db90c30b8d755d447c612679f.
Full fersk encoderbaseline dekker32TRAIN-rader; full head-baseline gjenbruker
kvalifisert originalrepresentasjon. Ingen ny full encoderpopulasjon påstås.

Uavhengig CPU-lesekontroll sammenlignet lagret state med kvalifisert dedicated
INITIAL: eksakt fire endrede tensorer, bitlike koeffisienter, øvrig modellstate,
buffers, tom optimizer, RNG, modes og parentorder.12scoreceller samsvarer med
verifisert likevektskontroll, maksimalt4,78e-11. Ingen nye forwards eller fit.
GPU↔CPU rålogitavvik var under7,83e-8.250W beholdt, maks47°C/minne52°C;
signed guard avsluttet0. Vellykket total kampanjetid/speedup er ikke målt
fordi kontrollerens sluttrapportering feilet.

M5 Brier etter faktisk import, med fryst prior:
TRAIN S0,248473364 mot0,249651008; F0,249327011 mot0,249876211.
Senere S0,249652488 mot0,249827988; F0,249855791 mot0,249833416.
Begge slår TRAIN-prior. S slår senere prior, F taper fortsatt svakt.
Senere TRAIN er gjenbrukt utvikling; robust generalisering/økonomi er ubevist.

Den målte rapporteringsdefekten er fysisk gjenskapt: Windows-observatøren
avviste total_units0 når trenerprosessen fortsatt eksisterte. Smal kildefiks
tillater bare eksakt importrecipe, korrekt fase, terminalstatus og fire
heltallstellere0.7faktiske PowerShell-tester og syntaks består; legacy positiv
progresjon består, feil recipe/optimizer/parent/terminal/bool avvises.
Controller og øvrige vakter er urørt. Ingen UNC-fallback er bygget: lesing
bestod ved repro, så en slik ny feilårsak er ikke etablert.
36tidligere fokuserte tester, fysisk metadata-/transportpreflight og
commitvakter gjenbrukes. Native kilde338f8b3f var uendret under kjøring.
Alle gamle capture/restore/migration-metoder forble byteidentiske.
Original Windows-kilde er bevart i den installerte speilingen; rettet observer
er testet separat. En framtidig eksakt plan må binde og installere riktige bytes.

**Direkte neste steg:** Korrigert M5-vektingsloss er implementert og syntetisk
kontrollert som beskrevet øverst. Verifiser på lagrede FIT-logiter før eventuell
native varmstart/fit. Ingen ny fit er innrømmet.
Det verifiserte importerte checkpointet og alle ferdige bevis skal gjenbrukes.
GC/TEST/Entry/Exit/handel forblir stengt; F-priorgrensen gjenstår.

Den [parvise likevektskontrollen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_EQUAL_WEIGHT_20261109_001/REVIEW.json)
er COMPLETE og konsumert. Kun vekten mellom kjente observasjoner ble endret:
original128D-representasjon, alpha0,01, startprior, masker, solver og total
M5-sidevekt ble holdt fast. To nye konvergerte hoder; vektet kontroll gjenbrukt.
Hele kontrollen tok4,091s, ingen nativeforward eller nativevektendring.

M5 FP32 Brier, vektet→lik observasjonsvekt, med fryst prior:
TRAIN S0,248546831→0,248473364, prior0,249651008.
TRAIN F0,249391063→0,249327011, prior0,249876211.
Senere S0,250021975→0,249652488, prior0,249827988.
Senere F0,250045801→0,249855791, prior0,249833416.
Alle fire Brier/log-målinger bedres. Begge slår TRAIN-prior; S slår nå også
senere prior, mens F fortsatt taper svakt. Senere AUC S0,521943/F0,508323.
Dette isolerer en konkret skade fra tapsvektingen ved frossen representasjon;
hele native treningshistorien eller robust generalisering/edge er ikke bevist.

Alle24 scoreceller er uavhengig kontrollert (maks2,23e-16), begge mål
(maks2,78e-17), begge optimumgradienter (maks8,13e-10), vektmasse og
FP32 B16-prediksjoner (bitlike). Kontrollprediksjoner ble gjenbrukt bitlikt.
Source6164a2f0 var uendret. Begge CPU-planer er lukket og konsumert.
Worker ble avgrenset delegert, lest av root og kvalifisert med fokuserte
vektkontroller; en overstreng float-literal-test ble rettet, uten workerendring.

Alpha0,01 kom fra [kronologisk indre TRAIN-valg](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_INNER_L2_20261109_001/REVIEW.json):
fem forhåndsvalgte styrker, tre splitter,32fits/5,537s. Valget ble låst før
senere cache åpnet. Én hel B16 ble purget, streng M5-modning kontrollert.
Encoder/normalisering hadde sett hele TRAIN: dette er betinget head-valg,
ikke encoder-OOF. Senere TRAIN er gjenbrukt utvikling, ikke forseglet TEST.

[Skår- og vektdiagnosen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_INNER_L2_20261109_001/CALIBRATION_DECOMPOSITION.json)
viste inntil14× ulik observasjonsvekt. Effektiv OPP-andel var S0,526853
mot faktisk0,518681, F0,515518 mot0,511126. Lik vekting fjerner akkurat dette
avviket med bevart tapmasse S0,194591671/F0,199459167.
Aktiv native rute er native_F_objective→side_normalized_loss_v1; den eldre
loss_v2 er ikke siste side-normalisering. Native kode/vekter er fortsatt uendret.

Likevektskoeffisientene og originalcache er grunnlaget for den nye
nullstegsimporten øverst. Den siste native FINAL har en annen kropp og brukes
ikke. F-priorgrensen er fortsatt uoppfylt.

Den nye [M5-fit-kjøringen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DEDICATED_M5_FOLD1_FIT_20261109_001/TERMINAL.json)
er COMPLETE og konsumert: original FINAL varmstartet til egne M5-sidehoder,
tom AdamW, fersk full INITIAL, én passering med1849updates/0skips,
samme29574 FINAL-rader og separat17753 senere TRAIN-rader.
Kilde8e1277f1 var uendret; trener, guard og observer avsluttet med0.
Windows-oppgaven er deaktivert; ingen ny nativefit er innrømmet.

M5 conditional-side Brier, lavere er bedre:
TRAIN S0,268542461→0,262282190, prior0,249651008 (N10010).
TRAIN F0,302929987→0,286292874, prior0,249876211 (N11909).
Senere S0,270574780→0,268627655, prior0,249827988 (N6096).
Senere F0,316566859→0,300100645, prior0,249833416 (N7165).
Log-loss bedres også, men begge M5-mål taper fortsatt mot prior på begge
populasjoner. Dette viser forbedring etter fortsettelsen, ingen kvalifisert edge.
Eget M5-hode og ekstra trening er ikke isolert fra hverandre i denne native
sammenligningen; tidligere frossen readout-kontrast isolerer kun head-begrensningen.

Før/etter og gammel/ny senere måling har identiske observasjonsbindinger,
alle60 score-antall og uendrede priorer. Separat leser har kontrollregnet alle
gjennomsnitt fra summer/antall; ingen nye forwards eller target-replay.
Fersk full INITIAL gjenskaper originale FINAL-aggregater med største
middelavvik9,71e-10; dette er ikke en ny rå per-rad-prediksjonskontroll.
[Resultat og tidsmåling](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DEDICATED_M5_FOLD1_FIT_20261109_001/PERFORMANCE_AND_LEARNING_REVIEW.json).

Hele kontrollerkjøringen tok7171,361s (1t59m31s), mot5051,843s for tidligere
recovery. Ny kjøring inkluderer full INITIAL (1691,751s) og256flere faktiske
updates. Ingen målt total speedup. Fit2623,855s/1849updates=1,4191s/update,
mot1,4067 tidligere (omtrent0,9prosent langsommere). FINAL1708,333s og
senereTRAIN1025,917s.250W ble beholdt; maksGPU56°C/minne64°C.

Den [faktiske M5-mekanikken](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DEDICATED_M5_MECHANICAL_20261109_002/RESULT.json)
er gjenbrukt: strictoriginalrestore, bitlike uendrede utganger, begge M5-hoder
med gradienter og bitlik2mot1+resume1.69 fokuserte fittester og commitvakter bestod.
Bare fem rute-/bindingsfiler endret; øvrige80 kildeeiere bevarte kvalifiserte bytes.


Den fulle [koblingskontrollen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_SHARED_OBJECTIVE_20261108_001/RESULT.json)
er COMPLETE og konsumert. Seks konvergerte readout-fits på samme frosne
FINAL-representasjon, samme pass0 sampled-S/native-F-fasit og eksakte
batch16-/TF-sidevekter; 7,992 s, null nye native forwards eller vektendringer.
Felles hode kontra eget M5-hode, med felles hode for de fire andre TF:
TRAIN M5 S Brier0,255734 mot0,246126; F0,252135 mot0,247188.
Felles hode taper altså mot prior0,249651/0,249876 på TRAIN selv ved
konvergens; eget M5-hode slår prior. Dette isolerer en konkret begrensning
i dagens readout ved den frosne representasjonen, ikke hele treningshistorien.
Senere S/F bedres fra0,258874/0,259311 til0,254622/0,253853 ved å skille M5,
men taper fortsatt mot prior0,249828/0,249833. Ingen kvalifisert edge.
Alle poeng/tellinger er kontrollert med0avvik; separat batchvis rekalkulert
treningsobjektiv avviker maksimalt1,05e-15. Alle seks løsninger konvergerte.

Minste native rettelse er implementert som TA-only opt-in
dedicated_m5_side=True: to egne 128→1 M5-sidehoder, 258 nye parametre.
Aktivitet, øvrige TF, genuine input/familier, mål og tapsvekter beholdes.
Standardkontrakter/state-nøkler/RNG og gamle checkpoints er bevart.
89 unike fokuserte tester består: gradientruting, vektmigrering, nye
optimizerparametre og bitlik syntetisk gjenopptakelse. Én testforventning
brukte ulik Q-form; den er rettet og kun egen testfil kjørt om, 26/26 PASS.
Faktisk metadatabinding bevarer opprinnelige felter, vinduer, ruter,
normalisering og priortall.
[Kildekontroll](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DEDICATED_M5_SOURCE_20261109_001/RESULT.json).

Varmstart kopierer alle gamle vekter og kloner de to sideradene; ny
modellidentitet beholdes. Dagens statekontrakt tillater ikke gammel AdamW-
historikk ved ny cursor0. Ny tom AdamW er derfor eksplisitt valgt; originalt
checkpoint bevares. Faktisk CUDA-migrering/RNG/gjenopptakelse er kvalifisert ovenfor; fersk
full initialbaseline og den innrømmede engangsfit-planen er nå fullført ovenfor.
Horisontspørsmålet står åpent; ingen target endres.

Kontrollen på [opprinnelige indikatorinput](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_INPUT_CONTROL_20261108_001/RESULT.json)
er COMPLETE og konsumert. 1 713 kolonner med alle genuine aktuelle felt og
siste lukkede M5/M15/H1/H4/D1-rad, samme frosne TRAIN-normalisering og samme
29 574 TRAIN-/17 753 senere TRAIN-rader. To faste LightGBM-modeller à256
grunne trær, ingen søk/early stopping, null nye native forwards/modeller.
Total 634,710 s; inputklargjøring 603,492 s, begge fit/score samlet25,868 s.
Inputcache og private modeller er bevart. Ingen hovedmodellendring.

M5 S: TRAIN/senere AUC 0,788409/0,510227; senere Brier0,254091 mot prior0,249828.
M5 F: TRAIN/senere AUC 0,769324/0,509349; senere Brier0,253952 mot prior0,249833.
Alle lagrede tellinger, Brier/log, gjennomsnitt og AUC er rekalkulert med
eksakt null avvik. De største TRAIN-gain-bidragene omfatter prisposisjon,
ATR-normalisert candle-geometri, EMA, trendlinje-/BOS-aldre og volatilitet.
De fleste av disse holder seg innen tidligere min/max senere i tid.
Dette er deskriptivt; verken feature-gain eller range-drift etablerer årsak.
Kontrollen svekker en ren representasjonsforklaring, men utelukker ikke verdi
i lengre historikk, andre tidsperioder eller modeller. Økonomisk edge er ukjent.

Brukeren er spurt om hovedmålet er neste 5-minutters bevegelse, en lengre
markedsreaksjon vurdert hvert 5.minutt, eller separate mål for begge.
Det skiller prognosetakt fra forventet reaksjonshorisont. Ingen target- eller
hovedmodellendring gjøres på grunnlag av en antagelse; gjeldende M5-fasit
og alle øvrige TF-mål er bevart. Retningsmålet er aktivt; ingen ny fit er innrømmet.

Full frossen FINAL-utlesning er COMPLETE og konsumert: 29 574 TRAIN-rader
og 17 753 senere TRAIN-rader, 2 959 enkodinger, null backward/optimizer.
Alle S/F-side-tellinger og Brier/log-gjennomsnitt på fem TF ble gjenskapt med
eksakt null avvik. Vekter og kilde var uendret. Trener og kontroller avsluttet
med 0; total kontrollertid 2 584,765 s (43 min 5 s), topp-RSS 5,85 GB.
250 W ble beholdt som brukeren krevde; ingen effektinnstilling ble endret.
[Full cache](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_FULL_CACHE_20261108_001/RESULT.json)
bevares med originale kvitteringer og to NPZ-partisjoner.

Den konvekse [M5-kontrollen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_CONVEX_M5_20261108_001/RESULT.json)
er COMPLETE og konsumert: fire konvergerte private readout-tilpasninger på
hele sidekjente TRAIN, 4,615 s, null nye native forwards. Senere TRAIN ble
kun scoret. Originale M5 S/F-prognoser hadde gjennomsnittlig OPP 0,630727/
0,741344 mot faktisk 0,513944/0,513050. S-rangeringen var svakt reversert:
AUC TRAIN/senere 0,476902/0,484692. TRAIN-fit kalibrering lærte S-slope
-0,249721 og F-slope 0,097952; dette er ikke bevis for en target-signfeil.

Senere Brier original/kalibrert/full128D/prior:
S 0,270575 / 0,249662 / 0,253804 / 0,249828 (N=6096).
F 0,316567 / 0,250001 / 0,252964 / 0,249833 (N=7165).
Full128D-readout bedret TRAIN til S/F 0,244985/0,246605, men senere AUC
var bare 0,505592/0,505114. Kalibrering fjerner mye feil sannsynlighetsnivå;
kun S har en liten senere punktgevinst mot prior, uten kvalifisert
statistisk/generell fordel. Fleksibelt readout løste ikke generaliseringen.
Hovedmodellen er uendret. Alle tellinger, Brier/log og vektrekonstruksjoner
er kontrollert separat mot cache, eksakt null måleavvik, ingen ny fit.

Brukeren ba om en annen smart angrepsvinkel. Neste direkte hypotese er en
liten kontrollmodell på de opprinnelige kausale indikatorinputene, med samme
M5-mål og kronologiske perioder. Den kan skille informasjonsmangel i input
fra informasjon som går tapt i den lærte representasjonen. Kontrollen er nå fullført som beskrevet ovenfor. Ingen promotering er tillatt.

Aktivt brukermål08.11: jobb bare direkte med retning. Første avgrensede
kontroll er fullført på eksisterende TRAIN-fasit på 1,24 s, uten modellkall.
M5 global/batchvektet OPP-andel er S 0,518681/0,526853 og F 0,511126/0,515518.
Denne statiske priorforskjellen alene forklarer ikke det målte Brier-fallet.
Checkpoint-prøven er COMPLETE: 246 TRAIN-rader, 32 CPU-enkodinger,
64 dekodergradienter, ingen optimizer, 153,21 s. Sluttmodellens TF-sidekorrelasjon
er minst 0,998620 for S og 0,999533 for F. På 5/18 rader med kjent motsatt
M5/D1-fasit ga S/F aldri motsatt side; totalt var det likevel 5/3 motsatte
prognoser av 246 rader. Total dekodergradient motarbeidet M5 S i 7/16 og F i
3/16 batcher; aggregert pekte den gunstig. Dette beviser svak TF-differensiering
og lokal konflikt i utvalget, ikke årsaken til hele treningsresultatet.
Den kontrollerte readout-sammenligningen er også COMPLETE og konsumert:
samme frosne backbone/projeksjon, identiske startprognoser, 480 private
oppdateringer per arm, 246 TRAIN- og 249 senere TRAIN-rader, 198,07 s.
På senere M5 S er Brier delt/TF-spesifikk/prior 0,270946/0,256937/0,249602
(N=75); F 0,265582/0,255479/0,250228 (N=107). Begge M5-mål bedres også
på TRAIN, men prior slår begge armer. Øvrige TF har blandede resultater.
Dette støtter delt readout som begrensning i denne kontrasten, ikke som
hele årsaken til tidligere native fit. Hovedmodellen er uendret.
Lagrede prognoser er rekalkulert mot cachede masker/counts; null nye forwards.
[Kontrollert sammenligning](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_READOUT_COMPARISON_20261108_001/RESULT.json)
og frosne dekoderinputs bevares for gjenbruk. Retningsmålet er fortsatt aktivt.

Korrigert TA S/F fold1 er nå **COMPLETE og konsumert**. 256 oppdateringer
ble gjenbrukt, 1 593 nye fullførte samme pass, totalt 1 849 uten skip.
Fersk initialparitet og full model/AdamW/RNG/order-restore bestod.
INITIAL/FINAL har identisk observasjonskjede, alle nevnere og frosne priorer.
På M5 TRAIN ble S-side-Brier dårligere fra 0,249789 til 0,268542 og
F-side-Brier fra 0,255374 til 0,302930. På senere TRAIN ble S/F henholdsvis
0,270575/0,316567 mot prior 0,249828/0,249833. Lavere er bedre.
Dette er sidekjente observasjoner, ikke treffprosent eller valgte handler.
Samlet TRAIN-tap falt fra 2,8812 til 2,4015 hovedsakelig via tilstand/aktivitet;
fallet dokumenterer ikke bedre retning. Bare D1 S slo senere prior på begge
sidemål; ingen statistisk kvalifisert gevinst eller årsaksforklaring følger.
[Resultat og tidskontroll](/home/andre2/GX1_RUNS/MARKET_STATE_TA_NATIVE_FOLD1_RECOVERY_20261108_001/PERFORMANCE_AND_LEARNING_REVIEW.json)
binder målinger, nevnere og filhasher. Trener/observatør/kontroller avsluttet
med 0, kilden var uendret, ingen TEST ble brukt. Windows-oppgaven er deaktivert,
ingen treningsprosess er aktiv, og resterende tre vinduer er ikke autorisert.

[Fullføringen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_NATIVE_INPUT_COMPLETION_20261108_001/RESULT.json)
er COMPLETE: 600 ekstra historikkrader × 94 felt = 56 400 sammenligninger,
null avvik. M5/M15/H1/H4/D1-gapene var 94/63/95/96/252 rader. Alle ni
modellflater har null ikke-endelige verdier på unike konsumerte rader.
315 lokale felt og 132 kontekstfelt er ikke konstante i dette utvalget;
noen tilstedeværelses-/sesjonsfelt er konstante på lengre TF.
Frosne S/F-priorer bruker eksakt 29 574 modne fit-rader, null score-fit.
385,86 s og 5,74 GB topp-RSS; null nye modeller, gather, normalisering,
lokal31-replay, backward eller optimizer. Én tidligere forward gjenbrukes
eksplisitt som evidens. Fullføringsplanen er konsumert.

Den tidligere lokal31-diagnosen er ACTUAL_SOURCE_MATCH: 47 697 rader ×
31 felt, 1 478 607 sammenligninger, null oracle/cache-/producer-avvik.
Den gamle auditreferansen startet i 2009, mens faktisk produsent startet
i juni 2010. Ingen produksjonsfeil ble demonstrert; ulike state-historikker
og absolutte koordinatstarter ble ikke isolert fra hverandre.

Ny [TA S/F-mekanikk](/home/andre2/GX1_RUNS/MARKET_STATE_TA_SF_MECHANICAL_20261108_001/RESULT.json)
er PASS: 32 modne fit-rader, seks forwards, fire backwards/vektoppdateringer,
fersk baseline og bitlik gjenopptakelse. 154,96 s, 5,75 GB topp-RSS og
1,79 GB toppallokert GPU-minne. Trener/observatør/kontroller avsluttet med 0.
Oppgaven er deaktivert og planen konsumert. Dette er ikke markedspresisjon.

Brukeren ba om å rette den målte ytelsesflaskehalsen straks. Ettpass-fit
ble derfor avbrutt før kildeendringen; planen er ABORTED_CONSUMED.
Full INITIAL på 29 574 rader og checkpoint ved 4 096 rader / 256 updates
er bevart. Checkpointets hele state- og komponenthasher er kontrollert.
Oppdateringer etter siste checkpoint i den avbrutte jobben er UNKNOWN;
dens FINAL og senere TRAIN-score ble ikke fullført. Startoppsettet ignorerte SIGINT; SIGTERM ga guard-exit143
og Windows-task-exit1. Ingen vellykket trenerterminal eller resume-cursor
ble publisert. Oppgaven er deaktivert, GPU/prosess/lås er ledige.
[Operatørterminalen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_NATIVE_FOLD1_FIT_20261108_001/OPERATOR_INTERRUPTION_TERMINAL.json)
bevarer dette som avbrudd, ikke PASS. Ingen gammel plan relanseres.

Den minste ytelsesrettelsen erstatter 2 700 gjentatte tuple.count-skanninger
med Counter. I 25 like metadata-målinger sank feltvalideringen fra
66,994 til 0,834 ms (80,34 ganger); hele TA-profilbyggingen fra
117,099 til 50,161 ms (2,33 ganger). Gyldige profiler, kontrakter,
feltrekkefølge og seks kontrollerte normal-/feiltilfeller er identiske.
Alle validatorer kjøres fortsatt; modell, features og regning er uendret.
Tre fokuserte TA-profil-/normaliserings-/konsumtester består. Tre nye
duplikatfeiltester følger den eksisterende obligatoriske pre-commit-gaten.
[Resultatet](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DUPLICATE_CHECK_PERF_20261108_001/RESULT.json)
binder før/etter-målingene. Fasemåling og faktisk recovery-varighet står nedenfor;
ingen gammel fullført total finnes for en målt helkjøringssammenligning.

Den separate [inputfasekvitteringen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_NATIVE_INPUT_20261108_003/INPUT_TENSOR_PHASE.json)
er komplett innen sitt scope: 47 327 beslutningsrader og alle overlappende
inputvinduer, 7 726 606 020 råverdier bitlike, null normaliseringsavvik og
én faktisk forward med åtte familier/fem TF.
Disse bevisene er gjenbrukt i fullføringskvitteringen ovenfor.
003 stoppet i M5-lokalreplay; de øvrige MTF-historikkgapene er nå kvalifisert.
Alle originale `CLAIM/FAILED/TERMINAL` og logger for input001–003 og
diagnosen er bevart. Ingen konsumert plan relanseres.

Brukeren presiserte læring av berøring→brudd→retest på samme trendlinje.
Dagens representasjon har to nærmeste-linjeplasser uten linje-ID, ankere
eller byttemarkør; hendelsesfelt kan gjelde andre registerlinjer.
Tapet har S/F-framtidsmål, men ingen separat fasit/evaluering av samme-linje-
reaksjoner. Dette er en konkret representasjons-/målegrense; effekten på
prediksjonsevne er ikke målt. Ingen ny modellarkitektur er innført.
GC-registerplanen er utsatt og uklamet. Juni 2026 er brukt utviklings-VAL;
forseglet periode følger bare forward-journalregelen i GX1_RULES.md.

Den nye [recovery-planen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_NATIVE_FOLD1_RECOVERY_20261108_001/CAMPAIGN/CAMPAIGN_PLAN.json)
fullførte i første vindu etter fersk Windows-boot624. Den bevarte modellen,
targets, normalisering, feltrekkefølge og optimizer; 85 kildeeiere var bundet.
Bare Counter og seks nødvendige recovery-eiere avvek fra originalen.
Input-/mekanikkjobbene og full INITIAL ble gjenbrukt; originalplanen relanseres ikke.

Hele controller-vinduet tok 5 051,84 s (**84 min 12 s**); runnerens monotone
måling til før terminalpublisering var 5 033,16 s. Resterende fit tok
2 240,82 s, full FINAL 1 681,83 s og senere TRAIN-score 1 013,11 s.
Første nye 256-blokk tok 363,60 s mot gamle 565,66 s: 35,7 % kortere tid.
Full evaluering av samme 29 574 rader tok 48,4 % kortere tid enn gammel
INITIAL på 3 260,23 s. Gamle grenser er filtid-intervaller; nye er monotone.
Fit bruker senere rader, INITIAL/FINAL ulike vekter. Dette er målt
operasjonell fasegjennomstrømning, ikke et identisk A/B-forsøk.
Gammel fullført total og gammel senere-score-tid finnes ikke; ingen målt
prosentgevinst for hele løpet påstås. [Tidsgrunnlaget](/home/andre2/GX1_RUNS/MARKET_STATE_TA_NATIVE_FOLD1_RECOVERY_20261108_001/HISTORICAL_TIMING_BASELINE.json)
bevares. Ingen ekstra epoke eller lengre historikk er åpnet.

## Det som faktisk er målt

- **Eksplisitt TA-kandidat og korrigert læringsmåling:** Profilen
  `full_xau_ta_mtf_entry_research_v1` beholder 315 signal-, 132 kontekst- og
  253 MTF-felt, alle fem TF, åtte familier og 61 makrofelt. GC er fjernet
  før modellinput; modne fold1-statistikker projiseres uten refit.
  Faktisk FeatureView/checkpoint-identitet er kontrollert lesende:
  `3e2761afaf1d29e6537843e9e59b592a7c149a9cd99c97f866a2cd1e88611906`.
  Ny rute bruker korrigert S/F og hele samme 29 574-raders TRAIN-populasjon
  før/etter fit, med globale aktivitets-/sidemasker og frosne priorer.
  Senere 17 753 TRAIN-rader scores separat. Gammel initialkvittering
  med null forwards er ikke en læringsmåling. Inputkvalifikasjonen er komplett innen deklarert scope;
  ny fireoppdateringsmekanikk består. Den avbrutte fitens 256 oppdateringer
  er gjenbrukt, og den nye recovery-planen har fullført hele læringsmålingen.
  Retningen ble dårligere på M5/M15/H1/H4 for begge hoder allerede i TRAIN;
  bare D1 S/F forbedret begge sidemål fra initial og slo prior på TRAIN.

- **Funksjonsgjennomgang:** EMA bruker SMA-seed og alfa `2/(p+1)`;
  ATR bruker klassisk TR og Wilder14. SMC-pivot blir synlig etter bekreftelse,
  BOS bruker strict close-brudd, og alle inputbarene skal være lukket.
  Ingen ny produksjonsfeil er påvist i disse formlene ved kildekontroll.
  Modellen får indikatoravstander/helninger og candle-geometri.
  Prisoppdateringsantall er ikke omsatt volum.
  En reell makrodatofeil rundt høstens DST ble reprodusert og rettet:
  tilgjengelighet følger nå kalenderen D+2 ved NY-midnatt, ikke 48 elapsed timer.
  Alle 9 000 bundne kildedatoer fram til 2024-12-31 har identisk tilgjengelighet
  før/etter rettelsen; kandidaten trenger ingen ny materialisering for denne
  rettelsen. Kalender-/driverkontroll: 11 beståtte tester.

- **Test- og formelbevisgrense:** Ny TA-gather mistet C-kontiguitet og
  signaturrefleksjon mistet de fem gamle GC-argumentnavnene. Begge er rettet;
  sju berørte tester består.
  [Kilde-/testkvittering](/home/andre2/GX1_RUNS/MARKET_STATE_TA_SOURCE_QUALIFICATION_20261108_001/REVIEW_20261108T131458787986Z.json) SHA 68fcf96b270bf5c944fae4b3f6724ce25ad92c2ed25aaa3e1cdc329b55288a6d. Øvrige korrigerte taps-/batch-/state- og
  initial/final-tester bestod første fokuserte kjøring.
  Én eldre GC-relative-kildetest feiler på en uendret, foreldet
  `gc_order_flow_v1.py`-SHA; ingen GC-reparasjon åpnes i denne bølgen.
  186 lokale felt har eksakt M5-cache-alias. De 31 lokale trendlinjefeltene
  bruker egne 96-barsparametere og kontrolleres separat. Auditen inkluderer
  også 37 HTF/session-felt med eksplisitte asof-/avrundings-/klokkekontroller;
  den nye auditeierens kontroller er utvidet til 16 beståtte tilfeller.
  Fysisk tensorfase bestod; lokal31 er nå kvalifisert mot faktisk kilde som beskrevet over.
  61 makrofelt har eldre samme-eier-paritet/asof-bevis, ikke et uavhengig
  formelorakel; dette er en uttrykkelig bevisgrense.
   COT i 2024 bruker as-of +7 kalenderdager som tilgjengelighetsproxy.
   Faktiske publiseringstidspunkt og historisk datavintage er ikke kvalifisert
   per rapport. Ingen konkret 2024-lekkasje er påvist; full historisk
   tilgjengelighet er UNKNOWN og blir ikke bevist av teknisk input-PASS.
  Tidligere 253-feltsunion 158+31+1+63 og asof/postnorm-bevis gjenbrukes
  på deklarerte rader; de eldre faktisk konsumerte historikkgapene er nå også målt.
  [Tidligere kvitteringskontroll](/home/andre2/GX1_RUNS/MARKET_STATE_TA_INPUT_LOSS_REUSE_REVIEW_20261108_001/REVIEW_20261108T120132104946Z.json).

- **Input og mål:** 118 249 modellkoblede TRAIN-rader, 26 måneder GC MBP-1 og
  kausale fem-TF-flater er bevart. Foldprefikser, masker og normalisering er
  kvalifisert i sine deklarerte vinduer, ikke som bevis for prediktiv verdi.
  Hver genuine indikator trenger fortsatt kontroll helt til faktisk nytte;
  numerisk paritet og gradient på én rad er ikke edge. Lokal framtidstilstand
  `S_s(v)` bruker valgt aktivitet `R≥0,5 ATR`; i 0,5–1,0 ATR er side uavklart
  `{NED,OPP}`. Beslutningsforankret prisforskjell `F_s(t,v)` er et **separat**
  retningsmål med egen maske, ikke en ommerking av S-gråsonen. I senere TRAIN
  hadde 52 931 M5-rader S-side uavklart og gyldig F-endepunkt; bare 13 947
  hadde `|F|≥0,5 ATR(t)` og dermed meningsfull F-side.
  [F-dekning](/home/andre2/GX1_RUNS/MARKET_STATE_F_SUPERVISION_COVERAGE_20261108_001/RESULT.json)
  SHA `61b1e21a2dac1b86c8263e60fdcec5cd12b652135fcb991e231607493c942312`.

- **SMC/teknisk grunnlag:** Eksisterende 26-måneders M5-kilde og den eksakte
  modellklokken har 118 249 felles TRAIN-rader. Kausalt lukket M5 BOS og
  EMA20-helning gir 6 737 sidejusterte hendelser totalt; fem senere TRAIN-folds
  har henholdsvis 1 006/953/1 017/1 038/1 032 modne kandidater før noe
  framtidsutfall ble lest. [Kildedekning](/home/andre2/GX1_RUNS/MARKET_STATE_SMC_EMA_BOS_20261108_001/PREFLIGHT.json)
  SHA `f87a834f985cc53f1e5517f76753f6b00b4ef66d07735cc1d51bcc56594681b7`.
  Eldre SMC-familiefit kvalifiserte ikke retningsverdi på sitt tidligere mål;
  enkel EMA50/200-referanse ga ikke stabil sidegevinst på nytt S-mål. Ingen
  av dem testet den forhåndsbundne BOS+EMA20-sammenhengen ved 60 minutter.

- **Fast SMC+EMA20-forsøk, negativ retningsport:** Engangsplanen
  [MARKET_STATE_SMC_EMA_BOS_20261108_001](configs/research/MARKET_STATE_SMC_EMA_BOS_20261108_001.json)
  er konsumert. På 4 876 modne TRAIN-hendelser var ekstra signert
  60-minutters F-forskyvning mot EMA-kontroll i samme uke/side `−0,006385`
  H1-ATR, 95 % ukeblokkintervall `[−0,022610,+0,010314]`; bare 1/5 fold
  var positiv. Halv-ATR-kandidatens event-hit-rate var `0,488753`; det er
  ikke en låst markedstilstand eller handelsmåling. Uavhengig pandas-gruppering
  ga samme 4 876 rader og samme fem foldeffekter. Første auditforsøk tok med
  tre rader over foldcutoff; dette var auditens egen indeksfeil, rettet før
  kvittering, og påvirket ikke engangskjøringen. [Resultat](/home/andre2/GX1_RUNS/MARKET_STATE_SMC_EMA_BOS_20261108_001/RESULT.json)
  SHA `79fd9bccd08c78af6f4fb2de59f0416258e73546b0a302969ce152fd4822e40b`;
  [uavhengig kontroll](/home/andre2/GX1_RUNS/MARKET_STATE_SMC_EMA_BOS_20261108_001/INDEPENDENT_CHECK.json) SHA
  `d7a036c2065a29ea4f1a98c1c87cf4f62f2abc3ca51ae652243551ce5c36729d`.

- **Native tilstand, gammel v1:** Én fold-1-fit brukte 29 574 modne TRAIN-rader
  og 1 849 AdamW-steg. På 6 096 senere sertifiserte M5-siderader var
  side-Brier `0,277631` mot fryst prior `0,249828`; ingen retningsgevinst.
  Kilde/algebra viste at v1-tapet ga direkte skjev sidegradient når fasiten
  bare sa AKTIV; OPP-skjevheten er målt, men årsaken er ikke isolert.
  [Fit-resultat](/home/andre2/GX1_RUNS/MARKET_STATE_TIME_QUERY_FOLD1_FIT_20261107_002/RUNTIME/prefix-fit/window-0001/RESULT.json)
  SHA `08172fc2c17f8eaf4f0457e38bdd3aa750bfbbf9ae719cb587aabd4d93d21254`.
  Ingen ny v1-fit.

- **Ny S/F-rute:** Faktorisert v2-S og separat F-hode deler én native enkoding.
  På to ekte TRAIN-rader besto ett fysisk forward/backward: begge hoder og
  backbone fikk gradient, mens ukjent S-side og liten F-bevegelse fikk null
  direkte sidegradient. Vektene var bitlike før/etter; null optimizer/TEST.
  [S/F-nullsteg](/home/andre2/GX1_RUNS/MARKET_STATE_NATIVE_F_CPU_ZERO_STEP_20261108_001/RESULT.json)
  SHA `a2bb6cb8f586ea85f18ca4b313ffe28117dee67659216d00641d0a25cb0ccc7f`.
  Dette er teknisk kobling, ikke fit eller prognoseevne.

- **Enkle kronologiske referanser:** Femfolds S-LightGBM ga M5 side-Brier
  `0,249819` mot prior `0,249440`, men bedre aktivitet. En separat femfolds
  F-M5-LightGBM med alle 3 991 tabellfelt versus samme oppskrift uten de
  2 410 GC-navngitte feltene fikk **FAILED-terminal etter alle 20 fits** på
  grunn av skygget kilde-HEAD i siste bokendekontroll. Originalplan og
  terminal er konsumert; ingen RESULT eller omkjøring. Lesende hash-/mål-/
  prognosekontroll av de fem foldartefaktene ga på 34 760 meningsfulle
  M5-siderader Brier prior/full/uten GC `0,249755/0,250705/0,250821`.
  Prior slo full i alle fem folder; GC-tillegget var `+0,000115` Brier,
  3/5 positive folder og 95 % UTC-ukeintervall
  `[−0,000183,+0,000425]`. Den låste retningsporten feiler. På 88 397
  aktivitetsrader var prior/full/uten GC `0,238637/0,230894/0,231863`;
  aktivitet og GC-tillegget var positive i 5/5 folder. Dette er diagnostikk
  fra en feilet engangsjobb, ikke et fullført kampanjeresultat eller dom over
  native tidssekvenser.
  [Feilterminal](/home/andre2/GX1_RUNS/MARKET_STATE_F_M5_GC_ABLATION_20261108_001/TERMINAL.json)
  SHA `ce1b1862d1808fc67d63720b20e5c4075bf82b478ec5afbd2391b71d06f4e4a3`;
  eksakte foldreferanser, skår og begrensning står i policyens gate
  `market_state_F_m5_gc_ablation_20261108_001`. Den konkrete variabelskyggen
  i beholdt kilde er rettet etter kjøringen.

- **GC-tolkning:** Kjent aggressivt kjøps-/salgsvolum, sidevis gjennomsnittlig
  handelsstørrelse, OFI og GC-prisrespons når inputruten. Største registrerte
  handel har ikke egen sideetikett; dagens felt identifiserer ikke presist ett
  stort kjøp i aktiv modellvisning. De nye sidevise maksimumsfeltene er fysisk
  bygget, men ikke registrert eller konsumert av modellen. En enkel positiv
  GC-kjøpsubalanse ga ingen stabil framtidig XAU-retningsverdi; det tester ikke full native sekvensmodell eller absorpsjon
  ved struktur. [GC-flytreferanse](/home/andre2/GX1_RUNS/MARKET_STATE_GC_FLOW_REFERENCE_20261107_001/RESULT.json)
  SHA `256710c697cba9324cb4635aa323e99f4319ddbd1c418fa54ec549f04fde4946`.

- **GC-side-dekning på faktisk modellakse:** Av 118 249 M5 TRAIN-rader har
  117 626 gyldig GC-observasjon og global største handelsstørrelse; 117 541
  har både kjente aggressive kjøps- og salgsregistreringer. Sidevis
  størrelsesmaske finnes, men ingen sidevis største handel i den fysiske
  tabellen. Dette er presist informasjonstap for store handler, ikke mangel
  på GC-side generelt og ikke målt prediktiv verdi. 97 917 rader har også
  minst én handel med ukjent aggressorside (`N`); det kan sameksistere med
  kjente kjøp og salg. Kun eksakt M5-modellakse er kontrollert her.
  [Dekningskvittering](/home/andre2/GX1_RUNS/MARKET_STATE_GC_SIDE_LARGE_TRADE_COVERAGE_20261108_001/RESULT.json)
  SHA `7fdf2adb1138e0d777f85181c59d4272a50c6e832c6391d40ee9519ac684d184`.

- **Sidevis storhandel, kilde og fysisk data:** Eksisterende GC-samler bevarer nå
  største betrodde `T`-kjøp (`B`) og -salg (`A`) separat per M1 og reduserer
  dem kausalt til M5/M15/H1/H4/D1. Ti nye kandidater har samme sidevise
  observasjonsmasker og flow-eier; `N` gjøres ikke om til kjøp/salg. 77
  fokuserte kilde-/syntetiske tester består. Gamle fysiske GC-tabeller og
  nåværende native register/visning mangler fortsatt de nye feltene; nye
  fysiske tabeller er kvalifisert nedenfor. Prediktiv verdi og Entry er ubevist. Policyens
  `market_state_gc_side_max_source_20261108_001` binder dette kildesteget.
  Ny engangsplan [sidevis GC-bygg](configs/research/NATIVE_GC_SIDE_MAX_RESTORE_20261108_001.json)
  SHA `57c076ffe2797297fa502a6fc32a0f0d38f4f347eaa71c9517e0d9c3b350bd47`
  er kjørt én gang og konsumert. Den gjenbrukte kvalifiserte råkilder og
  samme byggeeier: 66 filer/892 100 712 poster, 1 603 s under 1 800 s/4 GiB-
  kapsel. [Terminal](/home/andre2/GX1_RUNS/NATIVE_GC_SIDE_MAX_RESTORE_20261108_001/RESULTS/TERMINAL.json)
  SHA `664d7b9aca3d969cfd7a72c34f1b5a186d2c37fd1f3a079a0bcacd5446abebd6`.
  Alle gamle 270 TF-felt er numerisk like på fem TF. Nye sidevise maksverdier
  har eksakte masker og likhet med maksimum over godkjente M1-minutter;
  [fysisk kontroll](/home/andre2/GX1_RUNS/NATIVE_GC_SIDE_MAX_RESTORE_20261108_001/PHYSICAL_SIDE_MAX_AUDIT.json)
  SHA `9cefe83b23b7c02d848308ce756deeb8575205edfb89551478ad740db28c4dcc`,
  [M1–TF-paritet](/home/andre2/GX1_RUNS/NATIVE_GC_SIDE_MAX_RESTORE_20261108_001/PHYSICAL_SIDE_MAX_M1_TF_PARITY.json)
  SHA `9bbcaf4bec20cd1e4bc642c8e60b8c4ee0cfd41ccfa2b803d51247a36942d7a0`.
  På de eksakte 118 249 M5-modellklokkene har 117 541 begge nye felt;
  [klokkedekning](/home/andre2/GX1_RUNS/NATIVE_GC_SIDE_MAX_RESTORE_20261108_001/MODEL_M5_SIDE_MAX_CLOCK_COVERAGE.json)
  SHA `3f8fcaa460b5f8e7f90d93cd84c504df914d1717bc1422bb8eb67f676d19289b`.
  Native register/layout/visning har ennå ikke konsumert de nye feltene.

- **Ny retningsvekt, teknikk og ett målt pass:** På de fem senere TRAIN-foldenes faste
  endepunkter hadde S entydig side i 33–44 % og F i 37–41 % av informative
  rader. Det gamle v2-tapet delte sidefeilen på alle informative rader. En
  separat side-normalisert tapsmodul deler nå på faktisk sidekjente rader;
  S/F-ruten bruker den for begge hoder. Den gamle v2-filen er byteidentisk
  med kvalifisert featurevisning.
  Den målte kandidatens F-vekt er bundet til 1,0 mot S; ingen scorevalgt vektsøk.
  11 fokuserte tester består. Ny fysisk nullstegskjøring på to ekte
  TRAIN-rader besto: S-/F-hode og backbone fikk gradient; ukjent S-side og
  FLATT F-side fikk null direkte sidegradient. Vektene var bitlike før/etter;
  én forward/backward, null optimizer, null TEST. 87,6 s og 5,35 GiB topp-RSS
  under 12 GiB kapsel. [Ny nullstegs-PASS](/home/andre2/GX1_RUNS/MARKET_STATE_NATIVE_F_CPU_ZERO_STEP_20261108_002/RESULT.json)
  SHA `7302d760741ae0329bd936a3bc0461dfedc2ada3a337fb23700bb2be7207b84b`.
  Dette beviser teknisk kobling, ikke læring eller retning. Policyens
  `market_state_native_F_side_normalized_design_20261108_001` binder designet.

- **Optimizer/gjenopptakelse, faktisk teknisk PASS:** Én guarded native S/F-prøve
  på de tidligere valgte TRAIN-radene 2–33 fullførte med seks forwards,
  fire backwards, fire AdamW-oppdateringer og tre modellkonstruksjoner.
  Avbrutt/gjenopptatt forløp var bitlikt referanseforløpet; trener- og
  observatørutgang var 0. 132,8 s, 5,76 GB topp-RSS og 1,98 GB toppallokert
  GPU-minne under 20 GiB/250 W-vakter. Planen er konsumert, og ingen
  prediktor-fit eller retningsverdi følger av dette.
  [Resultat](/home/andre2/GX1_RUNS/MARKET_STATE_NATIVE_F_PRIVATE_4STEP_20261108_001/RUNTIME/private-research/RESULT.json)
  SHA `55936ccfb06bddcebebfb10a6a591de29143fae3f64036594463f04985ccd97e`;
  [kampanjekvittering](/home/andre2/GX1_RUNS/MARKET_STATE_NATIVE_F_PRIVATE_4STEP_20261108_001/RUNTIME/receipts/invocation-0001.json)
  SHA `d274e2f0ea3d27842645105199b92761562965562cb2752d85169cd6fbc08f13`.
  Policyens `market_state_native_F_private_4step_20261108_001` binder
  engangsutfallet.

## Ikke gjort, hindring og neste steg

Native S/F har nå ett fullt korrigert fold1-pass med senere TRAIN-score,
men ingen ny femfolds OOF-kvalifikasjon. GC-merverdi i den native
tidssekvensmodellen, samme-linje-reaksjonsverdi, Entry-verdi og økonomisk edge
er ubevist. 61 makrofelt mangler uavhengig formelorakel; COTs faktiske
historiske publisering og vintage er fortsatt UNKNOWN.

**Neste steg:** Kun den målte native M5-vektingsfeilen og kontrollen beskrevet øverst.
Importert checkpoint er teknisk verifisert; rapporteringsfeilen skal ikke
utløse omkjøring. Ingen fit er innrømmet. S har slått senere prior i den
fryste head-kontrollen, F taper fortsatt svakt. Målet er aktivt.
Brukerens spørsmål om 30 epoker og fem år er fortsatt åpne hypoteser.
Denne fitten dekker omtrent mai–september2024, senere score oktober–desember2024.
Ett pass og én fold er ingen generell dom over lengre læring eller historikk.

Den tidligere [GC-registerplanen](configs/research/MARKET_STATE_GC_SIDE_MAX_REGISTRY_20261108_001.json)
forblir bevart, men er utsatt og uklamet etter brukerens nye prioritering.
`training_enabled=false` stenger ny trening; koblingskontrollen og alle tidligere
retningsdiagnoser, input-kontrollen, den konvekse M5-planen, cacheplanen
og native fitplanene er konsumert.

Kilde/metode for denne fasen: [docs/NATIVE_LEARNING.md](docs/NATIVE_LEARNING.md).
Detaljert historikk og eldre beslutninger: Git, ikke nye arkivmapper her.
