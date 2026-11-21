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

- Aktiv fase er native M5/MTF-markedsprognose før Entry. M5 er hovedklokke;
  kausalt lukket M15/H1/H4/D1 og alle åtte familier samt D1/makro skal støtte
  nyttig framtidig prognose. Scalp/intradag/swing er senere anvendelser av
  lærte muligheter; beslutningstakt er ingen fast holdetidsgrense.
- Hver TF skal bedømme neste helt framtidige bar/tilstand med de samme genuine
  familiene. M30 vurderes som en ekstra strøm ved siden av M15; dagens kjørte
  profil har fortsatt bare M5/M15/H1/H4/D1. Ingen numeriske taskvekter er valgt.
- M1 brukes bare i senere hands-on-analyse/oppfølging av en allerede åpen handel.
  Retningsinputs, fasit, akser, masker, prior, normalisering og utvalg bruker
  native M5 eller høyere, aldri M1 eller gamle M1-avledede artefakter.
- Den aktive profilen er full ikke-GC TA med genuine OHLC-/candle-/chart-/
  struktur-/nivå-/EMA-/ATR-felt på alle fem TF og D1/makro. GC/MBP er beholdt
  for en senere separat kvalifisert sammenligning; kompakt MBP-førsteprofil
  er et tidligere vedtak, ikke den nåværende kjøringsidentiteten.
- Fellesmodellen er hovedsporet. Samarbeid skal vurderes på prediktiv nytte,
  ikke bare tilkobling, attentionvekter eller like retningsstemmer. Framtidige
  mål, output, tap, prior og score skal ha samme sanne kausale semantikk.
- Bevar OHLC-form, spikes, wicks, range, støtte/motstand og mulige forløp.
  Netto-nøytralitet er ikke alene bevis for et sideveis/rolig marked.
- Ingen generell negativ prosjektdom fra korte tester. Faktiske målinger,
  originalkvitteringer og checkpoints bevares; dårlige resultater omskrives ikke.
- Gjenbruk verifiserte data og beståtte kontroller, én capped tung CURRENT-jobb,
  fryst kilde under kjøring og trygge omstarter. Stabil langkjøring følges opp
  hvert30.minutt etter siste operatørvalg. Ingen nativeTEST, broker/live/paper/
  ordre/spending/promotion er åpnet av denne offlinefasen.

Fullstendige og gjeldende detaljer eies av GX1_RULES. Målt læring, generalisering
og økonomisk edge rapporteres hver for seg; teknisk PASS erstatter dem ikke.
