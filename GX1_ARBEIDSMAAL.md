# GX1 arbeidsmål

Få modellen til å lære tilstandsavhengige Entry- og Exit-verdier og ta bedre
beslutninger enn relevante enkle baselines. Deretter kreves kronologisk
beslutningskvalitet og positiv kostnadsjustert økonomi. Målet er fortsatt aktivt.

Nå: Ny ONLINE-startbaseline er målt: frosne Entry-/Exit-targets, vekter og RNG er eksakt bevart. Én separat native256 TRAIN-prøve er bundet. Læringsporten er fortsatt ikke bestått.
Neste: utfør én bundet256-prøve og vurder tilstandsavhengig Entry OG Exit.
Ingen ny læring er målt med rettelsen. Se VEIEN_VIDERE.md.

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
