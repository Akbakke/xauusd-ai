# Prefix-normalisering — én avgrenset klargjøring

Blokkeringen er at de gamle normaliseringene brukte hele TRAIN, inklusive
mars–mai-kontrollen. Tidlige Entry-ID-er alene fjernet ikke senere Exit-states.
Eksisterende eiere er allerede rettet og testet; ingen ny produksjonskode nå.

Én CPU-jobb bruker frosne47814 Entry-ID-er og cutoff2026-03-01. Den lager faktisk
ny base/context/MTF- og lifetime-normalisering gjennom de eksisterende eierne.
Eksisterende fullstendig sekvensbevis gjenbrukes. Normaliseringsvisningen bytter
bare populasjonsbindingen og dens hash, uten å kopiere gamle fit-statistikker.

Alle313399 parent-/child-klokker er kontrollert like; Entry-ID-er trenger ingen
ny mapping eller utvelging. Hele de opprinnelige successor-counts beholdes og
sammenlignes eksakt etter fit. Alle valgte tilstandsklokker og MTF-vinduer må
ligge før cutoff. Sammensatt normalisering bygges/valideres av eksisterende eier.

Guard: capped producer,16GiB RAM,512MiB swap,én numerisk tråd,3600s tidsgrense.
Faktisk kontroll viste31GiB tilgjengelig før planlegging. Større minne enn de
syntetiske testene er nødvendig fordi eksisterende fit-eier leser de bundne
M1-featurematrisene; ingen sikkerhetsgrense endres.

Ingen modell-forward, optimizer, policyfit, label-refresh, featureberegning,
kontrollutvalgsbytte eller TEST. Dette er forberedelse, ikke læringsbevis.
Native fersk oppstart og eksplisitt binding av transformene/fasitene gjenstår.
Plan, operator og bevis bevares under GX1_DATA/.../PREFIX_NORMALIZATION_20260917.

## Første forsøk og minste rettelse

Populasjonen er ferdig:47814 Entries,53007 unike lokale M5-rader og263998
Exit-current-rader. Base-fit stoppet på ctx_cont.d1_ema_stack_aligned_v2:
median1,skala0. Ingen base-statistikk ble publisert; summary-fit var ikke startet.
Første terminal, logg, plan og operator beholdes. Den lokale overføringshjelperen
hadde først en navnekollisjon med standardbibliotekets operator; rettet før
første remote forberedelse, uten å påvirke kilde eller fit.

Kildeeieren definerer EMA-tilstanden som-1,0,+1. Bare når et slikt eksakt felt
er konstant, brukes dets kildebestemte enhetsskala. TRAIN-median og eksisterende
invertible asinh beholdes; ingen binær-/embeddingkonvertering. Alle senere tegn
kan representeres, men dette betyr ikke at modellen har lært et usett regime.
Ukjente konstante kontinuerlige felter avvises fortsatt. Ingen generell epsilon.

33 relevante tester består på første forsøk:16 nye cases og17 eksisterende.
Alle tre konstante tegn på signal/context/MTF bevarer de tre senere tegnene,
med eksakt domene-/metadatakontroll og inversjon.17 øvrige kontraktfunksjoner,
inkludert apply/invert, er AST-uendret. Ikke-konstant fit er uendret.

RETRY_PLAN binder kun uferdig base-/summary-fit og sammensatt binding. Ferdig
populasjon gjenbrukes med hash; ingen datoer, rader, features eller kontrollmål
endres. Samme capped producer16GiB/512MiB og3600s. Ingen modellkjøring.

## Generell feil avdekket; første unntak erstattet

Det andre forsøket passerte EMA-feltet, men stoppet på D1 bull_divergence_strength
med median0/skala0. Den feltspesifikke rettelsen var for snever og er fjernet.
Begge mislykkede kjøringer er bevart; ingen base-fit ble publisert.

Den eksisterende fit-funksjonen har nå én eksplisitt prefix-policy:
allow_constant_train_fields. Bare prefix-eieren aktiverer den. Endelig konstante
felter beholder TRAIN-median og rå enhetsskala gjennom eksisterende invertible
asinh; scale_source=constant_train_unit_scale registrerer ingen observert
variasjon. Standardmodus og alle ikke-konstante statistikker er uendret.
Ingen feltliste, epsilon, binærdomene, clipping eller nyere observasjoner brukes.

41 relevante tester består på første forsøk:33 normaliseringskontrakt og8
prefix-integrasjon, inklusive invarians ved endring av framtidige features.
Senere verdier beholder distinkte, endelige, inverterbare representasjoner.
Dette gir ingen læring av den manglende variasjonen; konstante felter oppgis
åpent i det faktiske resultatet. RESUME_PLAN binder kun de uferdige fasene.
