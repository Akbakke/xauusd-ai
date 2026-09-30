# Avgrenset sweep-/AVWAP-/aktivitetstest — 30.09.2026

Brukeren godkjente den foreslåtte datakontrollen og tekniske hypotesen med «ja gjør dette».
Full B forblir blokkert på egne kildekrav. Denne armen erstatter ikke B.

Eksisterende Dukascopy-cache kontrolleres først for innhold, format, kronologi innen fil,
bid/ask-spread, duplikater og kvotert volum. Fire udaterte filer er tomme; hovedcachen
har 264 filer, hvorav én i 2025-mappe. Originale hentekvitteringer er ikke funnet.
Dekoderkontrollen kan derfor ikke alene kvalifisere datoer eller sammenhengende dekning.
Ingen ny nedlasting, kontotilgang eller tickbasert lønnsomhetstest er bestilt av manifestet.

Den økonomiske testen gjenbruker den bundne OANDA M5-historikken og eksisterende
SMC-, volum-, risikostyrings-, kontantregnskaps- og inferenseiere. Bare forskningseieren
endres. Én fast hypotese: en ensidig bekreftet sweep fades ved avslutningen av fem
nye sammenhengende M5-barer, dersom prisen da er på fadesidens side av ankret VWAP
og aktiviteten vol_ratio_5_20 er positiv. En ny sweep i mellomtiden ugyldiggjør kandidaten.
AVWAP vekter close med antall prisoppdateringer fra sweepbaren til bekreftelsesbaren.
Det er seks observerte barer og ingen tilbakedatering av ankeret.

Sammenlign på samme reserverte populasjon: sweep alene, rullerende VWAP20 med
samme aktivitet, den foreslåtte kombinasjonen og LONG; FLAT er null på samme rader.
Avviste filtre beholdes som nullresultater og frigjør ikke alternative handler.
Utførelse bruker første observerte bid/ask ved eller etter beslutningen; målehorisont
er 12 M5-barers veggklokketid. Dette innfører ingen maksimal native holdetid.

Les 2010 som oppvarming, rapporter 2011–2025 og vurder primært 2021–2025.
Hele historikken er gjenbrukt utvikling; ingen del kalles urørt holdout.
Ingen parametere tilpasses, og TEST åpnes ikke.

Fire slippage-nivåer per utførelse (0/0,5/1/2 bps), faktisk bid/ask og to
finansieringsscenarioer beholdes fra tidligere instrumenter. 96 endepunkter korrigeres
samlet med paret stasjonær bootstrap/max-t (1999 trekk, 20 kalenderdagers forventet blokk).
GO krever alle åtte primære s1-sammenligninger over 1 bps med simultan nedre grense,
og positiv solvent økonomi i den senere perioden. Ellers NO_GO ved minst én relevant
øvre grense under 1 bps, eller INKONKLUSIV. GO er bare grunnlag for ny bekreftelse.

Kjørbare autoriteter, committet før måling:
- configs/research/TA_SWEEP_DUKASCOPY_AUDIT_20260930.json
- configs/research/TA_SWEEP_PREREG_20260930.json

Status: implementert og fokusert mekanikk kontrollert; ekte målinger ikke kjørt ennå.
