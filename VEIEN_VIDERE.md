# Neste steg: policy-konsistent selector lukket etter negativ måling — 24.09.2026

Gjeldende resultat og stoppbeslutning står i
[ENTRY_SELECTOR_CACHE_FIT_20260924.md](docs/ENTRY_SELECTOR_CACHE_FIT_20260924.md).
Den native representasjonsuttakingen og den ene forhåndsbestemte 127/129
Ledoit–Wolf-fitten er fullført. Verdiprediksjonene tapte mot original512 og
fit-konstantene; check-netto var negativ. Selector-hypotesen er lukket.

Ikke kjør extraction, fit, terskel-/lambda-/feature-/split-søk på denne
hypotesen. Ingen native trening, full epoch, full VAL, CONTROL, TEST,
live eller papirhandel er åpnet. training_enabled=false; ingen ny jobb er
bundet. Historisk design nedenfor er bevart for revisjon, ikke som startordre.

---

## Historisk plan før uttrekking og fit

Designavklaringen og tre brukerbestilte agentgjennomganger er ferdige.
Gjenbruk handover_snapshot/POLICY_CONSISTENT_ENTRY_REVIEW_20260919.json.
Ingen ny fit, forward eller optimizeroppdatering er utført. Kun uttrekking er bundet.
Selve modell-/treningsmatematikken er uendret. Native uttrekking er implementert:
en midlertidig observer-hook leser eksisterende Entry-head-input uten å erstatte
input/output. Den krever16 kall,256 rader, eksakt gammel Q og identisk full
Exit-kontrakt. Cachen bevarer hidden/Q/tokens; originalmodell/cursor kontrolleres.
Planen tillater0 fits og0 Exit-rollout-kall. Native oppstart gjenstår.

Syntaks og diff er kontrollert;19 målrettede tester bestod under eksisterende
beregningsvakt. Låseieren avsluttet, og den ene kølagte testprosessen fullførte.
Ikke gjenta disse testene uten nye relevante endringer. TEST_REVIEW ligger i
NATIVE_ENTRY_POLICY_REPRESENTATIONS_20260919 og handover_snapshot.

Én uttrekking er nå bundet i frozen_entry_selector_probe i NEXT_RUN_POLICY.json.
Fullfør obligatoriske commit-kontroller/push, materialiser én native campaign
med eksisterende klargjøringsrutine og følg controllerens vanlige maskinvare- og
oppstartskrav. Planen i entry_representation_preparation er kildeavhengighet.
Native paritet er fortsatt umålt; ingen fit er tillatt av uttrekkingsplanen.
Den fullførte fullpolicy-planen må ikke relanseres. Etter paritets-PASS bindes
én separat, kort cache-fit under auditvakten med reglene nedenfor.

Netto for hele frosne Exit512 på TRAIN256 er LONG−4,1080 / SHORT−5,1893 Bps.
Øvre rangerte halvdel gir−3,3433 Bps; faktisk Entry er FLAT256/256.
Exit slår umiddelbar lukking, men ingen profitabel Entry/Exit-strategi er påvist.

Entry lærer Q_mu, mens forløpene følger pi512. Lagrede fullpolicy-utfall er nå
bundet som støyende signerte labels for samme frosne pi512: begge sider for
alle256 rader, alle409 negative labels, opprinnelige kostnader og FLAT0.
Ingen framtidig beste side, gevinstutvalg, ekstra likvidasjon eller bootstrap.
Fill-/exitmetadata er labelproveniens og skal aldri brukes som Entry-input.

Entry-Q og entry_q_joint_hidden inngår begge i Exit-tokenet. Derfor bevarer
kandidaten hele original512-forwarden, originale Q-verdier og Exit-kontekst.
Et separat eksemplar av eksisterende lineære readout kan bare levere valg-Q
etter at originaltokenet er laget. Frysing av Exit-vekter alene er utilstrekkelig.

Neste konkrete leveranse er ett separat bundet native omfang som henter
original hidden for de samme256 radene og kontrollerer opprinnelige Q-verdier
og Exit-token. Sistnevnte er allerede hashbundet i lagret rollout-kontrakt;
gjenoppbygg nøyaktig samme kontrakt med original factory/cohort/modell/budsjett.
Ved identisk kontrakt kan fullpolicy-utfallene gjenbrukes uten1083 nye
Exit-forwards. Pariteten er foreløpig ikke målt. Gjenbruk eksisterende native
campaign, vakter, checkpoint-eier, inputs, readoutmatematikk og posisjonsregnskap.
Ingen separat runner, ny modellarkitektur eller ny targetsimulering.

Én forhåndslåst analytisk fit kan deretter undersøke hypotesen: fit127 fra
juni–september2025 og check129 fra oktober2025–februar2026. Alle fit-handler
avsluttes før checkperioden. Dette er likevel brukt TRAIN; originalmodellen
har allerede vært trent, og kontrollutfallene er kjent utviklingsbevis.
127 fitrader mot128 koordinater pluss intercept gir høy overtilpasningsrisiko.
Bruk den eksisterende Ledoit–Wolf-regelen fra fit-inputs alene, samme lambda
for alle tre handlinger og upenalisert intercept. Ikke gjenbruk gammel lambda.
Frys koeffisientene før check vurderes. Ingen parameter-, feature- eller terskelsøk.
Den historiske operatøren er matematikkreferanse, aldri alternativ oppstartsvei.

Forhåndsbestemt stopp: Brutt paritet/ugyldige tall avviser beregningen. På
check129 må både LONG og SHORT forbedre MSE og sentrert feil mot original512
og fit-konstanter. Uendret argmax med FLAT0 må velge handler og gi positiv netto
både over alle129 muligheter og i eksisterende én-posisjonsregnskap. Rapporter
måneder, antall handler og gevinstkonsentrasjon. Uklare eller konsentrerte
resultater gir ikke automatisk GO. Ingen handler eller svak økonomi gir STOP.
Ved STOP lukkes denne selector-hypotesen uten ny lambda, terskel, split eller
mer uendret trening. En teknisk feil kan bare få sin minste begrunnede rettelse.

Selv tydelig positivt utfall kan bare begrunne en separat bundet kronologisk
utviklingsmåling; det beviser ikke varig handelsfordel. Mars–mai og juni2026
er allerede utviklingsdata. TEST forblir forseglet. Ingen full epoch/full VAL,
live/paper/spending eller kostnadsendring. training_enabled er fortsatt false.

Ikke gjenta fullpolicy-rollout, rangeringstest, forkastet FLAT-bias/forecast-
hypotese eller beståtte fullsuiter. De tre agentene har levert; én tung jobb
om gangen gjelder fortsatt. Målet er aktivt og ikke oppnådd.
