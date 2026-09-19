# Neste: forbered og kjør den bundne frosne TRAIN-policyen

Kode, egen ONLINE512-checkpointbinding, observasjonsgrense og utvalg er ferdige.
24 målrettede tilfeller er bestått i native-binding/evaluator/handover; gjenbruk
også de tidligere13 cutoff-testene. Ekte originalcursor, checkpointbinding,
TRAIN256-kohort og uendrede modellfunksjonskilder er verifisert. Ingen nye
optimizersteg eller modellforwards er kjørt.

NEXT_RUN_POLICY.json.frozen_train_policy_evaluation binder null optimizersteg,
original512, faste256 TRAIN-rader, begge sider og én native invokasjon.
EVALUATION_PLAN.json og NATIVE_BINDING_REVIEW.json ligger i
/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/FROZEN_EXIT_TRAIN_POLICY_20260919.
Snapshotene ligger under handover_snapshot. training_enabled er fortsatt false.

Neste handling: commit/push ferdig kilde og bind eksisterende native campaign
fra ren kilde. Gjenbruk forrige PREPARATION_OPERATOR-mønster, current boot,
controller, eksklusive låser og maskinvarevakter. Kontroller det konkrete
native window før oppstart. Ikke bruk en separat runner. PREPARATION_RESULT.json
og faktiske prosesser/receipts avgjør om planen allerede er startet/fullført.
Ikke relanser en brukt plan. Frossen kilde må bevares under kjøringen.

Evalueringen bruker eksisterende TRAIN-state factory, cache og full evaluator.
ONLINE512 skal gjengi lagrede Entry-prediksjoner før Exit-rollout. Den regner
også umiddelbar EXIT med identiske priser/kostnader uten ekstra forwards.
Etterpå kontrolleres uendret modell og originalcursor. Ingen lærerrefresh,
endring av targets, modell, normalisering, tap, slippage eller Entry-terskel.

Grense:2026-03-01T00:00:00Z; tre forløp trenger cutoff. Alle åpne posisjoner
medregnes med siste utførbare likvidasjonsverdi; grensen konstruerer ingen EXIT
eller maksimal holdetid. All-HOLD øvre omfang er3375234 delte tilstander og
46573 policyforwards ved batch256; åtte CPU-arbeidere og10800s evalueringsvindu.
Dette er ikke målt kjøretid. Ressursstopp gir ufullstendig vurdering.

Vurder alle måneder/begge sider mot umiddelbar EXIT og FLAT0, og modellens
faktiske Entry-valg/ett-posisjonsregnskap separat. Bind og behold rangeringstesten
fra ENTRY_RANKING_AUDIT: én øvre/nedre halvdel innen måned, ingen terskelsøk.
Foreløpig hybridutfall er−0,3998/−5,7213 Bps; høyere rangering hjelper6/9 måneder,
men beste halvdel er negativ og alle130 velger LONG. Dette er kun en diagnostisk
TRAIN-gruppering, ingen online strategi eller gevinstbevis.

FLAT-Q-konstruksjon og eksisterende lang forecast som retningsgiver er avkreftet
med lagrede512-outputs; ikke bruk flere runder på disse uten ny konkret evidens.
Et positivt kontrafaktisk resultat beviser ikke kausal Entry eller generalisering.
Negative sidegjennomsnitt alene avkrefter heller ikke betinget edge.
Ingen blind trening, full epoch/full VAL, CONTROL/TEST, live/paper eller spending.
Målet er aktivt. Oppdater handover og lukk brukt scope etter terminal review.
