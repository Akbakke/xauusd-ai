# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og grenser

TA_DIRECTION_EDGE_CAUSE_QUERY_TRANSFER_MEASURED_CAUSE_UNRESOLVED.
Målet 11.11 er aktivt: mål årsaker til svak retningsverdi i indikatorer,
sammensetting og læring. [Samlet status](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json).
Korte runder er feilsøking, ikke kapasitetsdom. Full kvalifisert TRAIN-historikk
og 30–40 epoker er senere skalering, ikke et krav ved hver rettelse.
Alle modellplaner er konsumert. Siste gradientobservasjon er COMPLETE; ingen
ny modellkjøring er innrømmet. Training/native_training/native_launch/full_epoch
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

## Neste konkrete avklaring

Gjenbruk c171 TRAIN-fusjonscachens capture_entry_q_joint_hidden og de faktiske
initial/final-checkpointene. Kontroller først rad-/query-samsvar og rekonstruksjon
mot short002s initialpanel: den eldre fusjonscachen brukte en annen numerisk
profil. Kvalifisert input kan deretter avgrense endelig projeksjonsendring ved
fast initial hidden-input. Ved mismatch beholdes avviket og minste manglende
capture spesifiseres. Ingen ny native fit eller modell-/lossendring er innrømmet.
Unik historisk årsak, familiers prediktive nytte, fullprefix-fit, senere
generalisering og kapasitet ved tilstrekkelig trening er fortsatt uavklart.
Ingen ny edge er påvist; ingen kontroll bruker TEST eller etablerer handelsøkonomi.
