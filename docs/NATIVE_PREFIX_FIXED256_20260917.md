# Ett fryst native256-stegsforsøk — 2026-09-17

Førmålingen er ferdig og skal gjenbrukes. Den konkrete mangelen var en bindende
kobling fra dens lagrede starttilstand til native trening og fast ONLINE-sluttmåling.
To eksisterende native eiere er utvidet. Modell, tapsfunksjoner, features,
normalisering, policies, sampler og selve treningskoordinatoren er uendret.

Recipe binder INITIAL_MEASUREMENT_AUDIT.json og dens opprinnelige resultater.
Native oppstart gjenbruker INITIAL_STATE.pt med eksakt modell, optimizer, EMA,
scheduler og tilfeldig tilstand. Ved gjenopptak overtar eksisterende koordinator
med sin ordinære durable tilstand. Den opprinnelige native rekkefølgen, TRAIN16,
47814 eligible TRAIN-rader og maksimum4096 trente Entries/256 steg beholdes.
Ingen læreroppdatering eller full epoch/fullVAL. Tre native vinduer er kun
mulighet til å gjenoppta mot samme samlede256-stegsgrense, aldri mer trening.

Ved nøyaktig steg256 brukes eksisterende måleeier på samme TRAIN256/CONTROL256
og1024 Exit-samples per rolle. Det er den siste ONLINE-modellen som måles;
EMA eller beste checkpoint velges ikke. Læreren må fortsatt være eksakt initial.
Alle regenererte target-/bootstrapverdier og masks kreves eksakt like den
lagrede førmålingen. Rapporterte fasiter gjenbrukes fra denne; bare prediksjoner
endres. Denne paritetskontrollen er ingen target-refresh. Ingen økonomisk rollout.
Målingen må holde seg innen gjenværende native-vindu og tre timers målegrense.
Checkpoint, modellmodus og tilfeldig tilstand bevares også ved målefeil.

14 fokuserte syntetiske CPU-kontroller består totalt: bundne førbevis, endret target/
kohort avvises, mellomliggende pause måles ikke som slutt, faktisk koordinator
ved256 brukes, og full trenings-/RNG-tilstand bevares. De14 inkluderer de to
berørte førmålingskontrollene. Dette er teknisk bevis, ikke faktisk læring.

NEXT_RUN_POLICY åpner kun dette forhåndsbestemte forsøket. training_enabled
forblir false utenfor unntaket. Etterpå kreves paret Entry/Exit-sammenligning
mot initialmodell og TRAIN-konstanter, fordelt på sider/måneder og ukeusikkerhet
etter det frosne designet. Et uklart eller negativt utfall gir ikke utvidelse.
Ingen tuning på kontrollen, TEST, live/paper eller spending er åpnet.
