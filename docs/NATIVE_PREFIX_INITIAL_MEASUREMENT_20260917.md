# Native førmåling ved steg null — 2026-09-17

Førmålingen trenger samme native måleeiere og bevart fersk starttilstand som
det fryste forsøket. Den eksisterende budsjettkontrollen avviste nullsteg.
Denne avgrensede utvidelsen tillater ett uttrykkelig bundet nullstegsvindu.
Andre oppsett krever fortsatt positivt steg- eller epochbudsjett.

Lagret INITIAL_STATE.pt gjenbrukes med opprinnelig modell, AdamW, EMA,
scheduler og CPU/Python/NumPy-tilstand. CUDA starter fra samme deklarerte seed.
Eksisterende koordinator lagrer et ordinært initialcheckpoint og stopper før
første TRAIN-batch. Eksisterende måleeier beregner deretter Entry, Exit-state0
og de1024 fryste sampletilstandene for hver av TRAIN256 og CONTROL256.
Begge bruker uendret initialmodell som frossen lærer, med hver sin datogrense.
Målingen bevarer modellmodus, tilfeldig tilstand og aktiv checkpointpeker.
Ingen læreroppdatering, fit, økonomisk rollout eller TEST er åpnet.

12 fokuserte syntetiske CPU-tester består under4GiB-vakt. Direkte trening og
nullstegspause med gjenopptak gir eksakt samme vekter, optimizer, EMA,
scheduler, tilfeldig tilstand og treningsrekkefølge. Tilstand bevares også
ved feil mellom TRAIN- og kontrollmåling. Campaign stopper for gjennomgang
etter ett nullstegsvindu; gamle oppsett kan ikke bruke unntaket.
Faktiske uforanderlige inputbindinger er kontrollert gjennom scope-eieren.

NEXT_RUN_POLICY åpner nå bare dette ene native førmålingsvinduet, med
training_enabled=false og uendrede maskinvarevakter. Kilde skal først være
committet og bundet i recipe/campaign. Faktisk måling er ikke startet.
Dette er teknisk bevis, ingen ny læring eller generalisering.
Rå måleutvalg, observasjoner og vekter forblir i GX1_DATA.

## Faktisk native resultat — ferdig

Én måling er fullført på commit0d50bbd9, fysisk boot452. Kontrolleren startet
2026-09-17T07:30:23.646868Z og terminalkvitteringen ble ferdig07:44:33.772085Z.
GuardPASS, trainer0, observer0, null optimizersteg. Guardens observerte topper
var50C kjerne,54C minne,150,3W og2754MiB GPU-minne. Dette er målt drift,
ikke nye sikkerhetsgrenser. Windows-tasken er deaktivert etter avslutning.

Begge roller har256 Entry-/Exit-ankerobservasjoner og1024 samplede
Exit-observasjoner. Originale Entry-ID-er, samples og rekkefølge er eksakt
identiske med de ferdige kohortene. Targets er numerisk endelige og stemmer med
observerte og bootstrap-komponenter. De beskriver den faste referansepolicyen,
ikke samlet profitt fra modellens egne handelsvalg. Modell, target, tom AdamW, EMA steg0,
scheduler og alle lagrede CPU/Python/NumPy-RNG-felt er eksakt bevart i det
durable native startcheckpointet. CUDA-RNG er lagret fra deklarert seed20260911.

TRAIN Exit velger HOLD1013/1022 ganger av1024 for LONG/SHORT; kontrollen
1001/1019. Første måling viser dermed nesten konstant HOLD før trening.
Entry velger127LONG/90SHORT/39FLAT på TRAIN,27/162/67 på kontrollen.
Dette er atferd fra utrente vekter, ingen læring eller handelsfordel.

Samplet Exit-MSE før trening er864,79/859,50 på TRAIN og1143,47/1140,48 på
senere kontroll. TRAIN-konstanten gir864,71/859,42 og1143,52/1140,48.
Rapporten inneholder også Entry-/ankerfeil, sentrert feil, nivå og spredning.
Alle konstanter er beregnet fra TRAIN; kontrollen brukes ikke til tilpasning.
Før/etter-forskjeller og usikkerhet skal først beregnes mot fast ONLINE ved256.

Den første recipe-forberedelsen ble avvist før modellarbeid fordi den inneholdt
det overflødige feltet exit_backup_steps=1 sammen med den fryste referanse-
policyen. Bare dette metadatafeltet ble fjernet; den avviste oppskriften,
korrigeringen og begge forberedelseslogger er bevart. Modellkoden ble ikke endret.

Førmålingsunntaket er stengt. Neste arbeid er den ene fryste256-stegs native
recipe-bindingen og fast sluttmåling mot disse lagrede observasjonene.
Ingen lærings-/generaliserings-/økonomiport er bestått, og ingen mer omfattende
trening eller TEST er åpnet. Bevar råobservasjoner og vekter i GX1_DATA.
