# GX1 arbeidsmål

Få modellen til å lære tilstandsavhengige Entry- og Exit-verdier og ta bedre
beslutninger enn relevante enkle baselines. Deretter kreves kronologisk
beslutningskvalitet og positiv kostnadsjustert økonomi. Målet er fortsatt aktivt.

Nå:512-prøven er fullført. Exit tar bedre tilstandsavhengige valg på gjenbrukt
TRAIN, men Entry er fortsatt FLAT256/256. Samlet læringsport er ikke bestått.
512-scope er stengt; den separate nullstegs TRAIN-evalueringen er ferdig med negativ netto; ingen ny kjøring er bundet. Se VEIEN_VIDERE.md og
CONVERGENCE512_REVIEW_20260919.md for det gjenværende Entry-spørsmålet.

Bevar alle 200 features, åtte familier, tidsrammer og kausale inputs. Ingen fast
tapsgrense eller maksimal holdetid. Beregningshorisont og bootstrap er ikke
handelsregler om holdetid. Bevar gjennomførbar BID/ASK-økonomi og kostnader.

Rapporter TRAIN-fit, senere generalisering og samlet økonomi hver for seg.
Konstant bias, all-FLAT/all-HOLD og bedre hjelpeprognoser alene er utilstrekkelig.
Samlet økonomi inkluderer alle valgte handler og åpne posisjoner. Ingen TEST,
live/paper eller spending. Ingen modell loves å være lønnsom «evig».

Én agent og én tung jobb. Bare native campaign og eksisterende vakter, TRAIN16,
VAL256/8CPU/3t når særskilt tillatt, FP32/TF32 av. Ingen full epoch/full VAL
før læringsport og oppdatert bundet policy. Ingen blind trening, søk eller
forebyggende refaktorering. Rett bare konkret påviste blokkeringer.

Langsiktig større trening er et mulig senere steg, aldri gjeldende starttillatelse.
Den aktuelle launch-policyen og eksisterende aktive jobb har prioritet over
alle historiske planer. Stående publiseringsautorisasjon gjelder ferdig kode,
dokumentasjon og aggregater; rådata, vekter og hemmeligheter bevares privat.
