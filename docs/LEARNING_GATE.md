# Læringsport for GX1

Det uavhengige Entry-målet fra09.10 har ennå ingen native initial-/læringsmåling.
Den fullførte nullmålingen med gammel Exit-basert Entry-fasit er historisk bevis. Samlet læring,
generalisering og positiv kostnadsjustert økonomi er ubevist.
Nåstatus og launchomfang eies av NEXT_RUN_POLICY.json, ikke historiske resultater.

## Før mer omfattende trening

1. Samme ordnede features, tidsrammer, normalisering, mål, masker, successors,
   reference-policy, cutoff og utvalg bindes før sammenligningen.
   En endret ONLINE-funksjon krever egen fersk nullstegsinitialisering;
   lik vekthash er ikke funksjonsparitet. Historiske lærere er ikke ny baseline.
2. Entry LONG/SHORT/FLAT og Exit HOLD/EXIT_NOW vurderes separat på begge sider,
   mot samme ferske initialisering og kausalt tilpassede TRAIN-konstanter.
   Rapporter MSE, sentrert feil, korrelasjon, verdi-/målfordeling og LONG−SHORT-
   kontrast. Ren biasflytting er ikke tilstandsavhengig læring.
3. Behold handlingenes fordeling, referanseverdi/regret og tidsblokk-usikkerhet.
   Alltid FLAT er ikke dokumentert selektivitet; alltid HOLD er ikke dokumentert
   tålmodighet. Entry alene består ikke samlet Entry/Exit-port.
4. Skill gjenbrukt TRAIN-fit fra senere CONTROL/VAL-generaliseringsmåling.
   Reference-policyverdier og lærerestimater er ikke realisert strategiprofitt.
   Rapportér alle deklarerte perioder i faktisk gjeldende populasjon, ikke en
   tidligere recipes faste månedstall.
5. Bruk forhåndsbundne kriterier fra gjeldende design/kontrakteier.
   Ingen terskel velges etter resultatet. Effekt, baselines, variasjon og
   usikkerhet avgjør; uklart utfall er ikke PASS og åpner ingen utvidelse.

Operatørens designskifte09.10 bruker en eksplisitt Entry-only fase. Dens
fasit er observert BID/ASK-markout netto under bundet kostpolicy ved den
opprinnelige TRAIN-eide horisonten; ingen Exit-modell inngår i Entry-targetet.
Eksisterende multihorisont- og risiko-hjelpemål består. I denne fasen gjelder
de parvise læringskravene Entry LONG/SHORT og deres retningskontrast.
Exit-målinger er rapporterte diagnoser, ikke et krav om eller bevis for
Exit-læring uten Exit-optimizersteg. Større Entry-trening krever at Entry-
læring, stabilitet, usikkerhet og driftskontroller består. Dette åpner ikke
samlet Entry/Exit-port: den krever separat Exit-trening som bevarer godkjent
Entry-funksjon, og etterfølgende komplett økonomivurdering.

For v38 gjelder fryst TRAIN256 og CONTROL256, høyst4096 Entries/256 optimizersteg,
før eventuell separat, endelig budsjettregistrering. Den nye målbaselinen og
smoke-fasen er ennå ikke utført. Ingen full epoch/full VAL eller automatisk utvidelse er åpnet.

## Generalisering og økonomi

Juni 2026 er gjenbrukt utviklings-VAL, ikke urørt holdout. TEST er forseglet.
Økonomi krever gjennomførbare bid/ask-priser, alle kostnader/finansiering,
samlet kapitalregnskap og alle valgte handler inklusive åpne posisjoner.
Lukkede vinnere alene eller en prognosefeil alene viser ikke lønnsomhet.
Langhorisontreferansen er forhåndsvalgt alltid-LONG/kjøp-og-hold.

Svakt resultat skal diagnostiseres fra bevarte mål-, gradient- og outputbevis.
Ingen blind ekstra trening, brede søk, featurekassering, fast tapsgrense eller
maksimal holdetid. Native paritets-/resume-/ytelses-/maskinvareporter består.
Syntetiske tester beviser kontrakter, aldri kvalitet på genuine markedsdata.
