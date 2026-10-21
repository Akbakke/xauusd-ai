# Originalprosjektmålet — brukerens første bestilling

Denne bestillingen bevares ordrett som prosjektets faste retning. Den skal ikke
omskrives til det som tilfeldigvis er bygget eller har bestått en test.
Senere uttrykkelige brukerpresiseringer gjelder foran eldre formuleringer;
de operative reglene står i [GX1_RULES.md](../GX1_RULES.md). Nåtilstand,
bevis og neste steg står bare i [CURRENT_HANDOVER.md](../CURRENT_HANDOVER.md)
og `NEXT_RUN_POLICY.json`. Originalteksten er ingen ny kjøre- eller handelsadgang.

## Ordrett originalbestilling

<!-- original-request:start -->
Mål: lage en fullautomatisk tradingbot mot xauusd. Vi skal handle på M5 timeframe, med Multi timeframe til å få «aksept» i det «lange løp» at man ser opp mot hverandre (m15,h1,h4,d1). Vi kan jo også vurdere 30 min om dette gir oss en fordel. Vi skal ha gode entries: short i bearish, long i bullish. Korte eller lange trades spiller ingen rolle så lenge det er minst mulig MFA før MFE. Kvalitet på tradesa før kvantitet. Vi har 8(9) «familier» som overvåker hver sin side av markedet, og til slutt samarbeider om trade. Jeg har også lastet ned mbp-1 GC data for å lage indikatorer/features som matcher dette (order book, volume profile, columative volum data, BBAI, spread expansion shock, average order size filter, order flow footprint, volume point of control +++) og andre smart money concept som order blocks, sweeps og slikt. Vi har tidligere slitt veldig med å skaffe edge uten mbp-1 datasett (kun chart indikatorer 8 familier), nå må vi prøve med. Jeg har 10 år med ohlc data og 26 måneder med mbp 1. vi konsentrerer oss om disse 26 månedene med full mbp datasett før eventuelt trening på hele datasettet til slutt. Oppgave: du er en data analytiker med 160 i iq og skal gjennomgå ABSOLUTT HELE PROSJELTET, gå dypt inn i hver eneste indikator (ca 150 stk, og ALLE skal gå igjen på samtlige multi timeframe), at målene og koblingene med alt henger riktig koblet til hverandre, alle har en riktig funksjon, ingen død og/eller unødvendig kode eller ubrukelige filer er i prosjektet. Fullfør også mbp datasettet og sørg for at dette blir utnyttet TIL DET FULLE med all info vi kan få tak i. Oppdater også  .md filen handover script med dette målet så jeg slipper å forklare det samme gang på gang. Vi skal etterstrebe å gjøre det som trengs for å lykkes i dette prosjektet. All tidligere historikk med at dette ikke fungerer, sletter du, fordi vi tror ikke på at dette ikke fungerer (i hvert fall ikke en kort 10 minutter test der vi ikke har trent transformeren en gang). Dette holder ikke. Optimaliser kode til en hver tid, men ikke heng deg opp og kode deg fast i ubetydelig koding. Husk dette som regler. Når du kjører gpu/cpu utnytter du maskinen til maks, med restarter en gang i blant så maskinen ikke henger seg opp. Ser du noe vi mangler av indikatorer eller ubrukelige indikatorer så IKKE HOLD IGJEN på noen ting, da justerer vi. ALDRI antyd, da dobbeltsjekker du heller og får sannheten frem. Er du usikker så spør du. Oppdater rules og bruksanvisningen og kjør fler agenter. Effektivt og konkret og rett på problemene.
<!-- original-request:end -->

## Senere presiseringer som må leses sammen med originalen

- Først offline M5 Entry-edge: LONG, SHORT og FLAT, kvalitet før antall og liten
  MAE før første MFE. Lukket M15/H1/H4/D1 gir kontekst; M30 krever målt merverdi.
- M1 er etterfølgende prisfasit fra ett minutt til mange timer. Entry-kadensen
  er M5. Ingen Exit-vurdering eller Exit-trening nå; dager er støtte.
- Første runder bruker den navngitte kompaktprofilen MBP-1/SMC/EMA/ATR og
  nødvendig klokke/makro. Åtte familier og niende D1/makro beholdes, og resten
  av genuine indikatorbiblioteket bevares for senere kvalifisering/tillegg.
- Fellesmodellen er hovedsporet. Separate modeller/stacking krever identisk
  kronologisk OOF-merverdi. Mål skal følge kausale inputs og egne framtidsmasker.
- Ingen generell negativ prosjektdom fra korte tester. Faktiske målinger,
  originalkvitteringer og checkpoints bevares; dårlige resultater omskrives ikke.
- Bruk verifiserte data på nytt, én capped tung jobb, fryst kilde under kjøring,
  trygge omstarter og kort oppfølging hver time ved stabil drift. Ingen TEST-fit,
  broker/live/paper/ordre/spending/promotion i denne offlinefasen.

Fullstendige og gjeldende detaljer eies av GX1_RULES. Målt læring, generalisering
og økonomisk edge rapporteres hver for seg; teknisk PASS erstatter dem ikke.
