# CURRENT HANDOVER — XAUUSD M5 Entry, minutter til timer

Autoritativt arbeidssted: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les GX1_RULES.md. NEXT_RUN_POLICY.json eier aktuell status, scopes og eksakte
bevisbindinger. Flere avgrensede agenter, én integrator og én capped tung jobb.

## Mål og arbeidsrekkefølge

Operatørens siste18.10-presisering og aktive mål: bare Entry i denne bølgen.
M5 er beslutningsklokken. Lær om prisen går riktig vei fra ett minutt til mange
timer, med signert MFE og minst mulig konservativ MAE før første MFE. Dager er
støtte. Verken95min,480min eller en dagsberegning er tvungen exit/holdetid.
Ingen Exit-vurdering, Exit-target eller Exit-gradienter nå.

Alle genuine indikatorer, åtte spesialister, niende D1/makrokontekst og lukket
M5/M15/H1/H4/D1 skal følges til faktisk modellkonsum. M30 krever tilført verdi.
Bruk de26rike MBP-månedene først: TRAIN [2024-05-01,2026-01-01), VAL
[2026-01-01,2026-04-01), senere utviklingskontroll [2026-04-01,2026-07-01).
Alle26m er gjenbrukt utvikling. TEST fra juli er forseglet. Ingen broker,
live/paper, ordre, spending eller promotion.

Den bestilte ene fulle nativeepoken står ved lag, etter faktisk input-/mål-/
normaliserings-/fersk-baseline-/paritets-/optimizer-/resume-/maskinvareadmission.
Trening er stengt. Fellesmodellen med åtte temporale spesialister er hovedspor;
SMC-/MBP-modeller eller ensemble krever kronologisk OOF-evidens for merverdi.

## Gjort og målt

- Rå001:66filer,892100712records,9173893prisnivårader og30ufiltrerte
  kontraktreduksjoner bevart.019eventpass og020observerte SMC-rettelse er konsumert.
-024trusted-T-restaurering er COMPLETE,1202.084s, uendret kilde og TEST=false.
  Identiske Tposter var feilaktig avvist. Ingen T dedupliseres;17overlappende
  PTPminutter/18bok-eventbøtter sensureres. Netto78305M1 og25059620kontrakter
  gjenopprettet. Alle opprinnelige reduksjoner/prisnivårader og gammel gyldig
  eventstatistikk er kontrollert eksakt. Non-T-actionflow er ikke kvalifisert.
-025registry er COMPLETE,29.254s:1205genuine verdier+1205masker,995nye+210gamle,
 30observerte konstanter,0uobserverte/0formelaliaser. Den bruker gammel fysisk
 109507TRAINklokke og er ikke admission for en ny dynamisk M5-populasjon.
-253primitive felt perTF er implementert med ekte BID/ASK og217eksplisitte
  sammeformelaliaser. Alle fem lukkede TF og M5Entryruten er bevart.
-022M5/cache/model-source er COMPLETE,2554.675s, uendret kilde og TEST=false.
 1140982M5rader,2576enrichedkolonner,2725signal/2542ctx-cont/1ctx-cat.
-026M1 er COMPLETE,6092.253s, exit0, uendret HEADb669d8f0 og TEST=false,
  avsluttet18.10 15:42:11UTC. Publisert1111058rader/2576kolonner; footer-/
  klokkeprøver viserM1 fra2023-05-09T22:03Z til2026-06-30T23:59Z.
  ParquetSHA7928886648820c6e2ddb5057628c980f75264efbab77d94623732f7a88e1efcd;
  sidecarSHAf4e3ab32955cbf7b16ba3e279d070f5b4b39d764f122dd63c0b7415e6810136d.
  TerminalSHA0294e5c02093c991e008e24c77fe5cb3e5393f89f1af63ded263379bac05ee3e.
  Dette er featurepublisering, ingen native dataset-/modell-/læringsadmission.
-558unike restaurerings-/kildekontroller,75forbrukercaser og496M5-emisjons-/
  sekvenslesercaser bestod før026. Bevisene dekker daværende funksjon; ny
  dynamisk modell/target er ikke kvalifisert av disse tallene.
- Sekvenslesere bruker eksisterende diskbaserte M5loader i stedet for hele
  1140982×2725RAMmatrisen. Binær normalisering bevarer samme identitet som
  runtime. Alle originale feilrapporter/checkpoints består.

## Konkret ny designgrense

Et algebraisk moteksempel viser at MFE minus pre-MFE-MAE kan ha positiv
forventning for både long og short uten retningsinformasjon. Derfor skal den
ikke alene styre ny trade/FLAT-fasit. Modellfeedback må skille endpointretning,
MFE og konservativ pre-MFE-MAE med egne aktive tap og observasjonsmasker.

Kjent p/T-quote kan bevise terminal prisendring selv om mellomliggende bane
mangler. Ukjente hull sensurerer MFE/MAE; kortere observerte prefikser beholdes.
En terminal-open-inkluderende bane bruker hele M1bar førT og bare eksakt open
vedT, aldri Tbarenshigh/low. Closed-bar-only har egen kjent querytype, aktive MFE-/MAE-mål og maske.
Samme encoder og samme pathheads brukes med eksplisitt kjent endpointkode;
closedfasit erstatter aldri inclusivefasit. Dette beholder faktisk observerte
1minutfall også når neste open mangler ved kildeslutt eller splitgrense.
GC-futuresquotes beviser ikke spotrekkefølge eller historisk OANDAkalender.

Et delt M1-intervallindeks/pointers og liten horisontbetinget dekoder bruker
samme encoderforward for1..480min. Ingen stor modell/head perminutt, ingen
lagret tett entry×480framtidsbane. Dagersmål er separat maskert støtte.
En ratio av predikert netto retning og pre-MFE-MAE er kun en navngitt
kvalitetskandidat, med kjent følsomhet nær0; ingen bps-/cash-/probabilitetsclaim
eller native autoritet før faktisk kontrakt og kvalifisering. Det varige
lav-MAE-målet er allerede vedtatt og skal ikke spørres om igjen.

Design-/algebrabevis er samlet i
`/home/andre2/GX1_RUNS/DYNAMIC_M5_ENTRY_DESIGN_20261018_029`.
Gammelt027utkast med fast95minprimærmål skal IKKE kjøres.028registryutkast
må bindes til faktisk ny fysisk klokke, ikke den gamle027planen.

## Kvalifisert komponentkode, fortsatt åpen native kjede

030 avsluttet med650beståtte og29feil, fryst kilde uendret. Tre lokale
feilårsaker ble rettet uten å svekke vakter. Faktisk pointer-bindingmodus ble
også samlet hos én eier.031 kjørte de fem berørte testfilene: 233beståtte,
exit0 og uendret fryst kilde. Dette dekker syntetiske quote-/maskesemantikker,
faktiske fixturefilhash-/klokke-/pekerkoblinger, samme encoder med begge
endpointtyper, aktive gradienter/tap, Entry-only optimizer og gammel
kontraktoppførsel. Det er konsistensbevis, ingen markedsfit eller edge.

Native initialmåling/koordinator er fortsatt en konkret åpen kobling:
fresh-prefix-konstruktøren sender ikke dynamisk fase/querykwargs, gammel
prefixbinding krever Exit/referencepolicy, og observatørene forventer Q/token
og Exit-forward. Disse må få en eksplisitt Entry-kurvegren før baseline og
native epoke. Ingen bps-alias for ratioen. Ekte evaluerings-/avståelses-/
dekningsspor, metadata-successor og fysisk TRAINfit er fortsatt påkrevd.

## Neste fysiske bygg032

Nytt engangsbygg `NATIVE_GC_DYNAMIC_DATASET_BUILD_20261018_032` er bundet som
`m5_dynamic_observed`, før eksakt ren-kilde-binding/claim. Det bruker ferdige
022/026features og friske032outputstier; gammelt027utkast startes aldri.
History-start er målt faktisk M5-kilderad etter026s første M1:2023-05-09T22:05Z.
En capped4G-kontroll hashbandt råquotes/026 og sjekket hele alignmentklokken:
1110839rader etter eksisterende219rader warmup,0manglende kildepunkter, samme
siste tidspunkt2026-06-30T23:59Z. To importfeil i eksternt kontrollskript ble
rettet; originale feillogger/skript består, ingen data eller repo-kilde ble endret.

032 kjører eksisterende sekvensielle capped eiere: sourcecascade, TRAIN-ranker,
signalmanifest, M1/M5-featurebaseprojeksjon, pretest fysisk datasett og uttømmende
sekvensrekonstruksjon. Maks64800s, én invokasjon, ingen budsjettutvidelse.
025 er kun produksjonsprovenans; COMPLETE betyr fysisk materiale uten ny
inputadmission. Faktisk ny TRAIN-klokke, registryfit, metadatasuccessor,
normalisering og native Entry-koordinator/baseline må fortsatt kvalifiseres.
Ingen native trening eller Exit-vurdering starter ved032.

## Neste steg

1. Bevar kvalifiserte komponenter og lag en eksplisitt Entry-kurvegren i
   native konstruktør, prefixbinding og initial-/gradient-/paritetsmåling.
   Legacy-M1-definisjoner er AST-identiske med foregående HEAD.030/031-bevisene
   er beskrevet over; ingen ny fullsuite uten konkret berørt endring.
   Kilde er fryst under neste capped jobb. Ingen Exit-forwards/trening eller
   gammel Q-bps-/cash-semantikk kan brukes som dynamisk kurvemåling.
2. Bind komplette022/026artefakter og bygg nytt fysisk26mM5datasett med ny
   targetkontrakt, perkomponentmasker og ekte M1quoteproveniens. Ingen gamle
   legacy37-/largest-horizon-/dags-uniongater eller falske nullobservasjoner.
3. Mål faktisk fysisk TRAINklokke, gjør fersk wholeTRAINregistryfit og bind
   gjeldende immutable signal-/datasetmetadata.025s109507klokke gjenbrukes
   ikke som nye bevis. Gjenbruk featurebytes bare ved eksakt kilde-/verdi-/
   rolle-/maske-/evaluator-ekvivalens. Unngå gammel011wideM1read.
4. Normaliser hele fysisk TRAIN én gang. Mål fersk ONLINE=TARGET=EMA0,
   full fysisk feature-/MTF-konsum, paritet, Entry-only optimizer/resume og
   ekte LONG/SHORT/FLAT-/kvalitets-/deknings-/forklaringsfunksjon.
5. Bevis maskinvid idle, trygg fersk fysisk Windowsboot og aktuelle vakter.
   Bind unik TRAINpopulasjon/N/ordre og ceil(N/16), behold siste delbatch,
   tren nøyaktig én epoke med varige checkpoints. Stopp før VAL/lærerrefresh/
   nyepoke. TRAIN256/CONTROL256 er måleprober, ikke treningsbudsjett.

## Ærlige grenser

022/026featureflater er publisert. Dynamisk fysisk datasett, ny TRAINfit,
normalisering, aktuell Transformerforward/-læring, generalisering og
nettoøkonomi er ikke fullført. AlltidFLAT er ingen påvist selektivitet.
Beslutningsspor må vise faktiske inputs/masker/prognoser og gyldige
familie-/TF-kontrafaktiske bidrag; gatevekter alene er ingen årsaksforklaring.

MBP-1 viser BBO, ikke dypere bok/MBOordreidentitet. Hele prisgrid beholdes;
modellen bruker deklarerte sammendrag, ikke alle råbaneinteraksjoner.
Makroens fullstendige historiske originalvintager er ikke bevist. Ingen
Pythonfil er bevist slettbar etter referanse/import/CLI/test/eierkontroll.
DATA/RUNS ryddes bare av retention-eieren. Originalmålinger bevares;
korte gamle tester brukes ikke som generell negativ prosjektdom.

Metode: docs/NATIVE_LEARNING.md. Formel-/TF-inventar: docs/FEATURE_AUDIT.md.
Aktuell tilstand og alle eksakte kjøre-/bevisbindinger: NEXT_RUN_POLICY.json.
