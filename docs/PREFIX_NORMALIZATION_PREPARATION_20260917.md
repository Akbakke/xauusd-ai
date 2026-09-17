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
