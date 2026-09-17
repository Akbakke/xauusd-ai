# Ferdige normaliseringer koblet til native referanser — 2026-09-17

Indeksens referansefiler pekte fortsatt på den gamle normaliseringen fra hele
TRAIN. Nye immutable referanser bruker nå den ferdige normaliseringen som
bare ble fittet før 1. mars 2026. Ingen normalisering er beregnet på nytt.

Eksisterende build_bundle-eier bygget nye referanser og kontrollerte klokker
og utførbare inngangspriser. Begge bridge-vitnene er eksakt like originalene
utenom normaliseringsbindingen og vitnets kontrollsum. Alle episode-ID-er,
fill-ID-er, sekvensbindinger og sampler-kontrakter er uendret.

De opprinnelige indeksfilene gjenbrukes byte for byte: 313 399 fysiske TRAIN-
rader og 5 508 historiske VAL-rader. Ingen rad, pris, tidsstempel, parent/child-
koordinat, successor-count, sensurering eller terminal endres. Eksisterende
indeks- og kildevalidatorer er kjørt mot de nye manifestene. Det frosne
CONTROL256-utvalget er bundet til samme opprinnelige fysiske TRAIN-indeks.
Historisk VAL får kun en konsistent transformreferanse; ingen VAL-modell
eller økonomivurdering er kjørt. TEST er urørt.

Normaliseringens eksisterende base-, summary- og composite-artefakter brukes
direkte. Gamle artefakter og checkpoints er bevart. Den historiske state-view-
kildefilen for de uendrede fysiske episode-ID-ene er fortsatt proveniens,
ikke en alternativ oppstartsvei. Fersk modelloppstart bruker eksplisitt den
nye composite-kontrakten; eldre bundle-metadata er kun konstruksjonsmal.

Produksjonskoden er uendret. Forberedelsen kjørte én gang med eksisterende
CPU-/minnevakt, uten modellkonstruksjon, forward, optimizer eller refit.
Det ble ikke kjørt en ny syntetisk testsuite for denne referanseoppdateringen.

Faktisk fersk oppstart og frosne TRAIN-/kontrollmålinger må nå bindes, inkludert
dispatch for fast ONLINE-sluttpunkt. Det frosne 256-stegs læringsforsøket er
ikke kjørt eller åpnet av denne forberedelsen. Ingen læringsport er bestått.

Bevis og eksakte artefaktbindinger:
handover_snapshot/PREFIX_IMMUTABLE_BINDINGS_20260917.json.
Forberedelsesplan, script, logg og resultat er bevart i GX1_DATA.
