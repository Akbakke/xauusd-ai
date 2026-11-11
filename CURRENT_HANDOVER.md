# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og grenser

TA_DIRECTION_EDGE_CAUSE_PAIRED_JOINT_BOUNDARY_ADMITTED.
Målet 11.11 er aktivt: mål årsaker til svak retningsverdi i indikatorer,
sammensetting og læring. [Samlet status](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json).
Korte runder er feilsøking, ikke kapasitetsdom. Full kvalifisert TRAIN-historikk
og 30–40 epoker er senere skalering, ikke et krav ved hver rettelse.
Alle tidligere modellplaner er konsumert. Én ny passiv måling av begge
endepunkter er innrømmet etter fokuserte tester og faktisk preflight; ikke startet. Training/native_training/native_launch/full_epoch
og full_val er false. Begge gamle c171 Windowsoppgaver er Disabled. GC er på
pause; TEST, Entry/Exit, broker, handel, spending og promotion er stengt.

## Indikatorer og faktisk læring

Åtte TA-familier og M5/M15/H1/H4/D1 når faktisk modellkonsum. Numerisk dekning
og ikke-konstante TRAIN-støtter er bekreftet; det beviser ikke prediktiv nytte.
Alle 11 temporal-/akse-encodere og 228 undersøkte tensorer hadde vektendring og
Adam-andremomenter. En generell frakoblet læringsvei er avkreftet.
Mean+last kansellerer ikke systematisk. Variasjonsfall i blandingen er ikke
påvist prediktiv skade; tidligere fast MTF-intervensjon ga ingen bedre rangering.
Nærmeste trendlinjeslots mangler entydig samme-linje-identitet/ankere/endringsmarkør;
prediktiv kostnad er UNKNOWN. Alle genuine features og tidsrammer beholdes.
Låst M5-sluttlag og sannsynlighetsjustering ga tidligere liten Brier/log-forbedring,
med svak rangering. [Native evidens](docs/TA_M5_LOCKED_SHRINK_NATIVE_REVIEW.json).

Native short002 er COMPLETE/konsumert: 256 updates over 4096 kronologiske
TRAIN-rader, 0 skips, fersk tom AdamW og samme c171-initial/originaltap.
RESULT 1b25f095; kilde 9de7e20b uendret. Samme 16 B16-grupper før/etter;
3 grupper/48 rader fikk nye updates, 13/208 er andre historisk brukte TRAIN-rader,
ikke heldout. Kjent side S91/F104; radene er avhengige.
Totaltap 1,875017→1,892264; S/M5 Brier 0,246301→0,243797; F 0,251730→0,254568.
Begge M5-sidetap ble dårligere i de 3 nyeksponerte målte gruppene; dette er ikke
full evaluering av de 4096 fit-radene. Pooled AUC 0,53575→0,54121 / 0,46939→0,47681.
Gjennomsnittlig opp-sannsynlighet på kjent støtte steg 2,50553/2,19309 prosentpoeng.
[Kontroll og begrensninger](docs/TA_M5_C171_SHORT_LEARNING_REVIEW.json).

## Nivåskiftet er avgrenset, ikke ferdig forklart

[Sluttlag og representasjon](docs/TA_C171_SHORT_HEAD_ATTRIBUTION_REVIEW.json).
Beholdte gradienter og alle 256 AdamW-steg i M5-sluttlaget lar seg gjenskape;
direkte biasendringer er negative. Ingen generell optimizerfeil er påvist.
På gjenskapt initialinput gir sluttlagets parameterendring alene opp-sannsynlighet
−0,03676/+0,22419 prosentpoeng, mot faktisk +2,50553/+2,19309. Det peker mot
representasjonsbanen, men identifiserer ingen bestemt encoderårsak.

Observer001s FP32-avvik er bevart, konsumert og ikke relansert. Observer002
matcher kjent originalprofil: 8 numeriske tråder og opprinnelige gradflagg.
Alle initialfelt og tap er bitlike; gjenskapt skjult vektor er likevel ikke et
historisk registrert skjult input. Denne begrensningen gjelder videre analyser.

## Fullført observasjon av felles query-projeksjon

[Resultat, tall og grenser](docs/TA_C171_QUERY_PROJECTION_GRADIENT_REVIEW.json).
ROOT: /home/andre2/GX1_RUNS/MARKET_STATE_TA_C171_QUERY_PROJECTION_GRADIENT_20261111_001.
Kilde 091c5985 uendret; RESULT 26886d10, TERMINAL b4afccfa, controller exit 0.
16 forwards, 256 rader, 176 autograd-kall; 0 optimizer/updates/checkpointskriv.
Alle 14 initialfelt og 16 tap er bitlike; modell/buffers/RNG/modus/hooks og
trainability er uendret, alle parameter.grad er None. 68 fokustester bestod.
Produservakter: 12 GiB/512 MiB, CPU 0–7/pids64, 250 W limit/170 W draw/65 C/80 C.

Åtte originale tapsbidrag er målt i query_projection[0], weight[128,134]+bias[128].
H4/D1 kan lokalt flytte M5 via delte hidden-vekter, bias og query-fraksjon.
TF-onehot-kolonnene er eksakt disjunkte mot M5: 560 negative kontroller.
F/D1 er aktiv i 6/16 grupper: lokalt oppdriv i F/M5-middellogit i 5, samtidig
økning av F/M5-tap i 2 (batchindices 0 og 1108). Fire aktive grupper hjelper tapet.
Gjennomsnittlig batch-dot for samlet gradient er positiv mot begge M5-tap i
alle 16, første 3 og øvrige 13 grupper. Dot av gjennomsnittsvektorer kan ha motsatt
fortegn. Dette begrunner ingen generell D1-fjerning eller ny tapsvekt.
Uavhengig kontroll e0bd316c: 4805 tallfelt/2487 tellefelt; maksavvik 8,88e-16,
maks relativ closure 4,22e-8. Alle gradientvektorer er bevart utenfor repoet.

Faktisk initial→final parameterendring i projeksjonen er også kontrollert fra
de to beholdte checkpointene, uten modell/forward. Dot mot initialderivatene
gir lokal M5-middellogitendring +0,012190 S/+0,009118 F, positiv i alle 16 grupper,
hovedsakelig via hidden-vektene. Faktisk endring med alle parametere er
+0,103761/+0,089288. Dette er ikke en eksakt finite-change-dekomponering;
differansen er ikke en ren upstream-årsak. Originaltapets lokale S/M5-endring
er gunstig i snitt, F/M5 ugunstig; begge er ugunstige i de 3 nyeksponerte gruppene.
ACTUAL_PROJECTION_DELTA_REVIEW f1da121c; uavhengig c7d83c9f kontrollerte 1104
tallfelt/516 tellefelt, maksavvik 7,11e-15 og begge hele modellhasher.

## Fullført gjenbruk av beholdte inputs

CPU-beregningene FINITE_QUERY_REPLAY_REVIEW 39b1acd1 og FINITE_JOINT_REPLAY_REVIEW
f355212c er komplette; kilde 10749744 uendret. Ingen fullmodell/backbone/gradient/
optimizer/trening. Fire forhåndsvalgte query-/sluttlagkombinasjoner og åtte
joint-/query-/sluttlagkombinasjoner bruker faktiske initial/final-vekter på fast
beholdt input. Originale B16-former og S-query-fraksjoner er bevart.

Eldre c171-cache er ikke bitlik matched002: headinput-avvik maks 1,84774e-5;
query-replay M5-råavvik maks 1,78814e-6 S/1,16229e-6 F. Ingen toleranse ble innført
for å skjule dette. Selv likt headinput ville ikke bevist historisk inputidentitet.

Ved fast beholdt 128D-hidden gir bare final query-projeksjon råmiddelendring
+0,012374 S/+0,009247 F; query+sluttlag gir +0,011449/+0,022075.
Ved fast beholdt 512D-source gir final joint-LayerNorm/linear + query + sluttlag
+0,042509/+0,045919, mot faktisk endring med alle parametere +0,103761/+0,089288.
Det ordnede joint-leddet etter final query/sluttlag er +0,031060/+0,023843.
Dette er betingede finite effekter, ikke historiske årsaksandeler. Differansen
mot faktisk endring er ikke en ren upstream-rest. Brier er blandet; begge M5-tap
blir dårligere i de 3 nyeksponerte gruppene i den hypotetiske all-tail-varianten.
Ingen kandidat er valgt, og ingen generell frysing/fjerning er begrunnet.

Uavhengige kontroller f35eb8ce og 70883132 bekrefter henholdsvis 290/543 tallfelt
og 188/259 eksakte felt; største numeriske avvik 2,66e-15/1,78e-15. Begge bruker
4 GiB/512 MiB og CPU 0–7. Første query-kontroll feilet etter regningen ved lesing
av en manglende cpuset-kvitteringsfil; originalfeilen er bevart, minste rettelse
til os.sched_getaffinity(0) bestod. Ingen modellkjøring inngikk i noen forsøk.
Alle fullrapporter og kildebindinger er lenket fra samlet årsaksrapport.

## Eksakt innrømmet observasjon

[Plan og grenser](docs/TA_C171_PAIRED_JOINT_BOUNDARY_REVIEW.json).
Ny ROOT: /home/andre2/GX1_RUNS/MARKET_STATE_TA_C171_SHORT_JOINT_BOUNDARIES_20261111_001.
PLAN c3ab604a, preflight c1a88f7e, faktisk cap-gate afa3d6e0. 29 hooktester og
63 worker-/rutetester består; uavhengig kildereview fant ingen konkret blokkering.
Én modell, 32 forwards over samme 16 B16-grupper ved initial og faktisk final;
512 evalueringsrader/256 unike TRAIN-rader. Én eksplisitt last av finalvektene,
ingen optimizer/RNG-restore, gradient, trening, checkpoint- eller kandidatskriving.
Originale 12 felt og 16 tap kreves bitlike per fase, initialt også begge headinputs.
Alle interne tilstander/identiteter/hooks skal være passive innen hver fase.
Faktisk 512D-source og 128D-hidden kopieres med to midlertidige pre-hooks; ingen
rekonstruksjon i hookene. Initial/final lagres som fase0/1 i samme TRAIN_FUSION.npz,
maks 2 MiB. 900 s worker/1200 s guard, samme 12 GiB/512 MiB, CPU 0–7/pids64,
250 W limit/170 W draw/65 C/80 C og 8 numeriske tråder som kvalifisert tidligere.

Første PLAN-publisering manglet foreldremappen og feilet før PLAN/CLAIM. Det er
bevart i PRE_PLAN_PUBLICATION_FAILURE; bare den nye mappen ble opprettet. Ingen
modellplan var da opprettet eller kjørt. Faktisk preflight og cap-valg bestod etterpå.
Etter terminal: kontroller paritet/passivitet og gjenbruk den parede fangsten for
å skille de fire kildeblokkene før joint fra joint/query/sluttlagets vektendringer.
Ingen modell-/lossendring eller full trening følger automatisk av observasjonen.
Unik historisk årsak, familiers prediktive nytte, senere generalisering og kapasitet
ved tilstrekkelig trening er fortsatt uavklart. TEST og handelsøkonomi er stengt.
