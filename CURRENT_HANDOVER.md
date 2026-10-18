# CURRENT HANDOVER — XAUUSD M5 Entry, minutter til timer

Autoritativt arbeidssted: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les GX1_RULES.md. NEXT_RUN_POLICY.json eier arbeidsstatus og eksakte bindinger.
Flere avgrensede agenter, én integrator og én capped tung jobb. Ingen endring
av kilde/policy/controller under fryst kjøring. Faktiske prosesser og terminaler
overstyrer prose; fullførte/avbrutte engangsplaner startes aldri igjen.

## Mål og avgrensning

Bare Entry i denne bølgen. M5 er beslutningsklokken. Lær separat netto
endepunktsretning, signert MFE og konservativ MAE før første MFE fra ett minutt
til mange timer; dager er støtte. Verken95min,480min eller dagshorisont er
maksimal holdetid eller tvungen exit. Ingen Exit-vurdering/-target/-gradienter.

Alle genuine indikatorer, åtte spesialister og niende D1/makrokontekst skal
følges gjennom lukket M5/M15/H1/H4/D1 til faktisk modellkonsum. M30 krever
målt merverdi. Samme modell med eksisterende spesialister er hovedspor;
SMC-/MBP-modeller eller stacking krever identisk kronologisk OOF-bevis.

Bruk de26rike MBP-månedene først: TRAIN [2024-05-01,2026-01-01), VAL
[2026-01-01,2026-04-01), senere utviklingskontroll [2026-04-01,2026-07-01).
Alle26m er gjenbrukt utvikling. TEST fra juli er forseglet. Ingen broker,
live/paper, ordre, spending eller promotion. Den bestilte ene fulle nativeepoken
står ved lag etter alle eksisterende admissionporter. Trening er fortsatt stengt.

## Gjort og målt

-024restaurerte faktisk78305M1/25059620GCkontrakter fra identiske Tposter som
  var feilaktig avvist.66råfiler/892100712records og9173893prisnivårader er
  bevart; ingen T dedupliseres.17PTPminutter/18bok-eventbøtter sensureres.
-025registry har1205genuine verdier+1205masker,30observerte konstanter,
  0uobserverte/0formelaliaser. Den bruker gammel109507TRAINklokke og er kun
  produksjonsprovenans for ny fysisk populasjon; ny wholeTRAINfit gjenstår.
-253primitive felt perTF og217eksplisitte sammeformelaliaser finnes. Alle fem
  lukkede TF og genuine BID/ASK-ruter beholdes. Formel-/eierinventar:
  docs/FEATURE_AUDIT.md. Ikke slett ekte features for å omgå en kontroll.
-022M5/cache/model-source og026M1 er COMPLETE, uendret kilde og TEST=false.
  Event003 har1140982M5enrichedrader og1111058M1enrichedrader,2576kolonner,
  2725signal/2542ctx-cont/1ctx-cat.022/026startes aldri igjen.
- M1alignment er uttømmende kontrollert:1110839rader etter219rader eksisterende
  warmup,0manglende punkter, siste2026-06-30T23:59Z. History-start for fysisk
  M5 er faktisk kilderad2023-05-09T22:05Z. Uendrede bevis gjenbrukes.
-034gjenbrukte032s byteidentiske rankermatrixcheckpoint og kildebevis.
  Ranker229.073s, signalmanifest66.440s, M1featurebase1277.268s og
  M5featurebase730.810s fullførte. M1featurebase:1110839rader,
  SHA c2bfdf4634150c38614ccf2923fee48f6b4e15dcca282a05bc7e0c0771b88a4b.
  M5featurebase:1140763rader,
  SHA baa4a2ee4820caa2cabf48018c670d1b7ac0b5fd827d08b632ebeb4b0ecb79b9.
  Dette er publiserte komponenter, fortsatt ingen fysisk TRAIN/VAL- eller
  modellinputadmission.

## Aktuell hindring:044ubrukt bred GC-arbeidskopiering

044 er konsumert:21:48:08UTC,600.466s, uendret kilde, builderexit137 etter
461.102s. Ingen044prosess lever. 043s første kildefikser virket:188031rader
passerte session-, signal-, tapejoin- og swingkontroll. Siste logg er så
TAPE_CONTEXT_RECOMPUTE; første ubekreftede operasjon er bred df.drop(15),
deretter merge og warmupreset. Kernel viser ny10G/512MOOM, men direkte live
044cgroupbinding og eksakt drept instruksjon er ikke bevist.

Full kildegjennomgang viser at2410råGCkolonner bæres i df og aldri brukes
av df_ctx, source_extra, mål eller faktisk modellinput. Senere bygges enda en
fullGCmatrise inn i merged3 bare for feltpresence; alle2725signal/2542ctx+1cat
emitteres fra den eksakte M5-memmapen. Rettelse: owner-avledet22kolonners
arbeidslesing, behold komplett schema/hash/full-source smal prisvakt, hold
mellomtabellene smale og bevis GCverdi+observasjonsmaske feltvis mot faktisk
M5input. SammeGCverdifunksjon kjøres én gang per felt, ingen chunkvis gjentatt
helkildekonvertering. Ingen ekte modellkanal eller øvrig guard fjernes.

045ga64PASS/1fixturefeil. Testspy fanget også legitim int64klokkelikhet;
bare spyavgrensningen er rettet, produksjonskoden er byteidentisk.046består:44PASS i berørt testfil+eksaktctx-only, øvrige21writerkontroller
gjenbrukt fra045. Produksjonskode byteidentisk045→046. Målingen viser2410
float32GCkolonner og188031historierader,1812618840bytes(1.688GiB) per
hel GCkopi. På4096ekte rader kopierte både drop og merge alle2410kolonner;
0delte. Bare footer/timekolonne/én4096batch ble lest. Full fysisk peakRSS
er fortsatt ikke kvalifisert. Deretter bare ny unik047 fysisk dataset-only. Bevar044terminal/delartefakter; aldri
044relansering eller caputvidelse.039inputs/strictv5PASS/040klokke/ranking
og uendrede kontroller gjenbrukes. Fysisk TRAIN/VAL er ennå ikke publisert.

## Entry-kurver og native kjede

Delte originale M1quotes/pointers og en liten horisontdekoder bruker samme
encoderforward for1..480min. Terminalmarkout og banemål har uavhengige
observasjonsmasker. Ukjente gap sensurerer MFE/MAE; kortere gyldige prefikser
beholdes. `closed_m1_prefix` og `terminal_open_inclusive` har egne aktive
MFE-/MAE-mål med samme heads; ingen fasit erstatter en annen. Fremtidige
masker/censor/peak/realisert elapsed brukes aldri som inputs.

Ratioen R/(abs(R)+MAE) kunne ikke velge FLAT på grunn av høy MAE når en
R-prognose var positiv. Nå er kandidaten eksplisitt kvalitets-bps:
predikert netto endepunktsmarkout minus predikert pre-MFE-MAE upper.
Koeffisient1 er valgt lik-bps kvalitetssemantikk; ingen cash/optimalitet eller
ubetinget felles forventningsclaim ved ulike masker. Bare predikerte
horisonter sammenlignes, aldri fremtidens beste fasit. Output-APIv12.
Native LONG/SHORT/FLAT-autoritet krever fortsatt målte kvalitets-/deknings-/
avståelsesbevis, fersk baseline og etterprøvbart beslutningsspor.

030/031komponentbevis består.033hadde249beståtte/30feil; de berørte
availability-/fixturefeilene ble rettet.035kjørte bare de tre berørte filene:
54beståtte,0feil, uendret kilde. Modell-/targetkontroller gjenbrukes.
Dette er kilde-/konsistensbevis, ingen markedsfit, generalisering eller edge.

Native konstruktør er rettet for eksplisitt fase/querybinding og frysing av
inaktive parametre før optimizer. Prefixbinding/epoke0-ordre bruker ekte
dynamisk wholeTRAIN-vitne, klokke, pointere og rollemasker uten Exitadapter.
Dynamiske koordinater har egen v2-identitet og stopper før Exit-målinger;
måle-/launchadmission er fortsatt eksplisitt stengt. Disse rettelsene
er kvalifisert:036 hadde424PASS/10fixturefeil;038 kjørte bare de to
berørte fixturefilene og fikk55PASS/0feil. Produksjonskoden er byteidentisk
mellom036/038. De opprinnelige feilkvitteringene beholdes.
Full_train-komponentkjeden og initial-/gradient-/paritetsobservatører krever fortsatt en Entry-kurvegren uten legacy Q/cash/
token/Exit-forward. Komponentrettelser alene er ingen nativekjøreautoritet.

037 stoppet før claim/tyngre trinn fordi M1-klokkebevisets hele kildefilhash
var eldre enn SMC-rettelsen. Ingen komponenter ble bygget.037 er konsumert,
aldri relansering.040 kjørte den eksisterende eksakte klokkekontrollen på nytt
med aktuell fil:5.732s, exit0, uendret kilde,1110839beholdte rader,0mangler
og samme rå-/beriket klokkehash. Ny039 bruker dette beviset og de samme
kvalifiserte produksjonsfunksjonene; ingen ekstra modell-/rankerfit.

## Fysisk PC-omstart er gjennomført

Brukerens18.10-forespørsel er oppfylt ved første trygge terminalgrense.
Shutdown exit0; faktisk ny Windowsboot2026-10-18T18:44:29.5000000Z.
WindowsOpenSSH og Ubuntu-22.04 er autentisert etterpå. Ren HEAD4634fc65 og
12lagrede artefakthasher er kontrollert eksakt etter omstart. Gamle
kampanje-/bootstrapoppgaver ble ikke reaktivert. `REBOOT_CONFIRMED.json`
i034runroten overstyrer det eldre PENDING-notatet. Samme forespørsel skal
ikke utløse enda en omstart. Faktisk maskinvare-/telemetriport er separat
fra boot og må kontrolleres før neste relevante kjøring.

## Neste steg

1. 046har kvalifisert22kolonners arbeidslesing/GCfeltparitet og faktisk
   begrenset kopimåling. Bevar alle modellkanaler og beståtte kontroller.
2. Ny unik047 er klar etter046. Bind ren commit/policy/plan/controller/
   validator og start én gang: smal arbeidsramme og faktisk feltvis GCparitet
   mot alle M5modellkanaler. Dataset-only, samme capper og039inputs/v5PASS.
   Ingen044relansering eller nativeadmission.
3. Mål faktisk fysisk TRAINklokke, gjør fersk wholeTRAINregistryfit og bevis
   effektiv verdi-/rolle-/maske-/evaluator-ekvivalens før nye immutable
   kontrakter. Ingen blind SHA-utskifting eller svekket klokkegard.
4. Normaliser hele fysisk TRAIN én gang. Bevis fersk ONLINE=TARGET=EMA0,
   hele feature-/MTF-konsumet, paritet, Entry-only optimizer/resume og ekte
   valgte/avståtte LONG/SHORT/FLAT-funksjoner med kvalitets- og dekningsspor.
5. Bind faktisk N/ordre og ceil(N/16), behold siste delbatch og tren nøyaktig
   én epoke med varige checkpoints etter gjeldende admission. Stopp før
   VAL/lærerrefresh/ny epoke.

Fysisk TRAIN/VAL, ny registryfit, normalisering, aktuell Transformerlæring,
generalisering og nettoøkonomi er fortsatt ufullført. AlltidFLAT beviser ikke
selektivitet. Gatevekter alene er ingen årsaksforklaring. MBP-1 viser BBO,
ikke L2/MBOordreidentitet; rå prisgrid bevares, modellen bruker deklarerte
sammendrag. Makroens komplette historiske originalvintager er ikke bevist.
DATA/RUNS ryddes bare av retention-eieren. Ingen Pythonfil er bevist slettbar
etter full referanse-/import-/CLI-/test-/eierkontroll. Historikken finnes i Git.

Metode: docs/NATIVE_LEARNING.md. Eksakte status-/bevisbindinger: NEXT_RUN_POLICY.json.
