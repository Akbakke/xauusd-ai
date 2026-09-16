# Én kandidat: lær en eksplisitt referanseverdi før policyforbedring

Status: referansemålets matematikk, kompakte dataflyt og eksplisitte native
trainer-/recipe-/checkpointbinding er implementert og CPU-kontrollert.
Faktiske frosne sammenligningsmål og bundet læringsplan gjenstår. Ingen ny
native kjøring eller teacher-refresh er åpnet; dagens standard er uendret.

Begrunnelse: 120-minuttersprognosen har positiv kostnadsjustert TRAIN-verdi
på de komplette forløpene, mens Entry velger FLAT og får nesten null videreverdi.
Den eksisterende lærerens grådige valg kutter LONG-target etter ett steg i
1880/2048 overganger, men lar SHORT følge fem i2047/2048. Sterkere gradienter
løste ikke dette. Se FORECAST120_ECONOMIC_SIGNAL_20260916.md og
TARGET_COMPONENT_CAUSE_20260916.md; målingene skal ikke gjentas.

## Presist estimand

Bruk én stasjonær referansepolicy mu for videre evaluering: HOLD med
p=119/120, EXIT med1/120 ved hver gyldige beslutning. Samme p for begge sider.
Dette er en analytisk blanding i læringsmålet, ikke tilfeldige ordre i modellen.
Referansen har forventet120 beslutninger til EXIT og ingen maksimal holdetid.
Valget er én forhåndsbestemt hypotese knyttet til den eksisterende lange
prognosen, ikke en parameter som skal søkes over. Ved tvungen økonomisk terminal
velges gyldig EXIT. Ukjent successor skal aldri gjøres til terminal.

La r_i være eksisterende likvidasjonsrelative reward, gamma_i eksisterende
veggeklokkediskontering og Q_mu forventet handlingsverdi under mu etter første
handling. EXIT-advantage er fortsatt nøyaktig0. Da gjelder:

    Q_mu(s, HOLD) = E[r_0 + gamma_0 * p * Q_mu(s_1, HOLD) | s, HOLD]

Et observert n-stegs mål, n<=120, er:

    sum(i=0..n-1, p^i * product(j<i,gamma_j) * r_i)
    + p^n * product(j<n,gamma_j) * frozen_Q_mu(s_n,HOLD)

Bootstrap settes bare til0 ved en faktisk terminal/ugyldig HOLD, ikke fordi
beregningsvinduet eller legacy-dataene slutter. Ved tilgjengelighetsgrense brukes
siste gyldige boundary-estimat og rett maskering. Ingen framtid inngår i inputs;
framtidige observerte rewards brukes bare som supervisjon.

Dette er Q_mu, IKKE Q_stjerne. Det kan ikke gis eksisterende optimalitetsnavn
uten en uttrykkelig kontraktsendring. Modellen beholder sine handlingsvalg;
en grådig policyforbedring må vurderes mot den lærte referanseverdien og
observerte utfall. At den eksakte Q_mu gir en policyforbedringsmulighet beviser
ikke at en approximert kritiker gjør det. Det må måles.

## Minste implementering og stoppkriterium

Neste arbeid er en opt-in mål-eier med eksplisitt mu-identitet, separat fra
dagens grådige femstegsmål. Bevar eksisterende standard, checkpoints, alle200
features, kostnader, normalisering og optimizer. Regn blandingen analytisk;
ikke generer tilfeldige forløp eller nye modellhoder. Bare boundary-tilstanden
trenger lærerforward for dette målet, ikke alle120 mellomtilstander.

Før native kjøring: bevis rekurrensen mot eksplisitt forventning, terminal- og
sensursemantikk, side-likhet og uendret default. Bruk eksisterende finite-
absorption-kontrakt der den passer; forventet antall beslutninger er ikke et
bevis på maksimal eller forventet veggklokketid under markedspauser.

Deretter skal én avgrenset native kritiker-kandidat spesifiseres med frosne,
felles mu-mål på faktisk trent og separat TRAIN, tids-/sidesplitt og konstant-
baselines. Ingen replay eller broad sweep. Først ved bedre tilstandsavhengig
kritikerlæring kan én separat, bundet teacher-refresh/Entry-vurdering vurderes.
Ingen slik kjøring eller refresh er autorisert av dette notatet. Feil mot
forskjellige mål må aldri presenteres som læringsgevinst. Full epoch/VAL og
TEST forblir stengt.

## Fullført CPU-prototype

Ny eier: gx1/contracts/unified_exit_reference_policy_v1.py. Fast policyidentitet
61c8aaaaa362d820165ce69e38b46f8e17a65cf82283d9af1410515fc5c90547.
Resultatet har eksplisitt Q_mu-semantikk, observed/bootstrap-komponenter og
grensevekt; det kan ikke stille erstatte et eksisterende optimalt Q-mål.
Ved prototypekontrollen var ingen native import eller recipe koblet til funksjonen.

62 tester bestod samlet: den nye referansekontrakten, eksisterende frossen
femstegskontrakt og eksisterende økonomikontrakt. Forventningen er kontrollert
mot eksplisitt summering over alle mulige første EXIT-tidspunkter ved1,2,5 og
120 steg. Negativ bootstrap, ufullstendige forløp, terminal på bare én side,
sidebytte, fravær av gradient/RNG-drag og ugyldige bevis er kontrollert.

Absorpsjonseieren bekrefter120 forventede beslutninger i den abstrakte
referansekjeden. Det er ikke en økonomisk kjøreautorisasjon eller en grense
på veggklokketid. Ved gamma=1 har bootstrap fortsatt vekt0,366341 etter120
observerte steg. Beregningsgrensen blir altså ikke en tvungen EXIT.

Kvittering: handover_snapshot/REFERENCE_POLICY_CPU_PROTOTYPE_20260916.json,
SHA2569f9e64fa2b10044f3bf287dc3c80d1f026930414930049099bbb1cb3a923ee6f.
Logg og eksakte kildefiler er bevart under BASE/NATIVE_EXIT_PRIVATE_CLIP_20260916_REFERENCE/
REFERENCE_POLICY_CPU_PROTOTYPE_20260916. Audit/4GiB/512MiB-swap, ingen modeller,
forward, optimizer, GPU eller TEST. Ingen eksisterende produksjonsfil var
endret ved kontrollen. Ikke gjenta de beståtte testene uten relevant endring.

## Fullført datatilkobling

Eksplisitt reference_policy-argument går via eksisterende factory/adapter til
state-view og collate. Nytt v4-skjema binder nøyaktig samme policyhash som over.
Ingen silent aktivering fra batchmetadata: collate krever policyen eksplisitt.
Gamle ett-/femstegsmål og Entry-ankeret beholder standardatferden.

Hver mellomovergang bruker eksisterende økonomi- og veggklokkeeier, men bærer
bare rewards, masker, tidspunkt og økonomihasher. Bare current, første successor
og siste boundary får modellinputs. En120-stegsfixture materialiserer derfor
3 tilstander; collate har1 online-rad og2 target-rader inklusiv Entry-anker.
Ingen mellomliggende MTF-/path-/context-materialisering eller lærerforward.

26 nye integrasjonstilfeller består: reward-paritet inklusiv helgegap, eksplisitt
policybinding, uendret default/anker, reell terminal på én side, ulik tilgjengelighet,
negativ bootstrap ved sensurering, falske bevis og blokkering før forward.47
eksisterende state-/training-/trace-/relative-eiertester bestod før siste snevre
finite-evidenssjekk, og gjenbrukes. Endelig data/adapter/runner-kontroll ga37 PASS
og2 feil i uendrede gamle tekstasserts mot gx1_capped_run.sh. Begge filene er
verifisert byteidentiske med baseline90cbfe93. Ingen kjørevern er svekket og
ingen fullsuite er kjørt eller erklært grønn.

Kvittering: ../handover_snapshot/REFERENCE_POLICY_DATA_FLOW_20260916.json.
SHA256: f9846fdc8b217568f062de18584eb11dd8faaff79145c781510a0173f142e213.
Eksakte kilder, første feil, rettet regresjonslogg og sluttlogg er bevart under
BASE/NATIVE_EXIT_PRIVATE_CLIP_20260916_REFERENCE/REFERENCE_POLICY_DATA_FLOW_20260916.
Kun CPU-fixtures under audit/4GiB/512MiB-swap; ingen produksjonscheckpoint,
ny native trening, GPU eller TEST. En observert collator-navnekollisjon ble
rettet før de grønne kontrollene. Teknisk datatilkobling er ikke læringsbevis.

## Fullført native binding

Referansemålet går nå gjennom eksisterende full-train factory, adapter,
collator og trainer. Exit bruker Q_mu-returnen direkte; ingen optimalitets-max
på mellomsteg eller ved boundary. Én targetforward, én onlineforward og ett
backward; målkomponenter og policyidentitet følger resultatet.

Første avgrensede kandidat beholder Entry-broen mot den uendrede opprinnelige
læreren. Dette isolerer korreksjonen av Exit-målet. Det hevdes ikke at gamle
lærervekter allerede er Q_mu: de er initialisering ved første referanseiterasjon.
En senere teacher-refresh/Entry-endring krever fortsatt eget læringsbevis.

Original95/global5777 er eneste tillatte origin for referansen. Eksisterende
continuation-kvittering navngir endringen; vekter, lærer, Adam, EMA, scheduler,
RNG, epochrekkefølge og progresjon bevares. Native scope tillater bare én32-
stegs TRAIN-kandidat til global5809, uten VAL eller full epoch. Den åpnes bare
med en hashbundet reference_learning_plan og faktiske frosne mål for512 trente
og128 separate TRAIN-Entries. Source closure, lærer, policy og sammenligning
før/etter mot konstant-baselines per side/måned må stemme. Planen finnes ennå
ikke, og gjeldende NEXT_RUN_POLICY gir ingen starttillatelse for referansen.

175 CPU-tester består, derav31 nye. Nye kontroller dekker faktisk backward i
fixture, uendret Entry-bro, source-/scopeavvisning, bytebevaring ved overgang og
serialisert resume, og at EMA-historikken ikke kan miste referansesemantikken.
En utilsiktet endring i eldre økonomiovergang ble rettet før den grønne suiten.
Ekte source closure mot original95 er også kontrollert:156→157 bindinger,
bare unified_exit_reference_policy_v1.py legges til, ingen eier fjernes og alle
endrede eksisterende kilder ligger innen den snevre overgangslisten.

Kvittering: ../handover_snapshot/REFERENCE_POLICY_NATIVE_BINDING_20260916.json,
SHA256 027ab79dcf1893b0d2900592794d5c8ddcb0469031c2d140429d5b1de245e064.
Kilde, logger, JUnit og faktisk source-closurebevis er bevart under
BASE/NATIVE_EXIT_PRIVATE_CLIP_20260916_REFERENCE/REFERENCE_POLICY_NATIVE_BINDING_20260916.
Ingen produksjonscheckpoint er lastet, GPU eller native trening startet, eller
TEST brukt i denne kontrollen. Tidligere to uendrede runner-tekstfeil er fortsatt
separat dokumentert; ingen fullsuite-PASS eller modellæring påstås.

Neste arbeid: frys faktiske mu-mål for hele native offset1696:1728 og128
uavhengig valgte, disjunkte TRAIN-Entries fra eksisterende cache. Gjenbruk
nåinputs og BEFORE95-prediksjoner. Beregn bare manglende rewards og boundary-
verdier med bevart lærer, i én avgrenset CPU-forberedelse uten optimizer eller
backward. Deretter bind bevisene i eksisterende policy før en kandidat åpnes.
