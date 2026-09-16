# Én kandidat: lær en eksplisitt referanseverdi før policyforbedring

Status: isolert CPU-mål-eier er implementert og kontrollert. Den er ikke
koblet til native trening; dagens mål, modell, recipe og checkpoints er uendret.

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
Ingen native import eller recipe er koblet til funksjonen ennå.

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

Neste konkrete arbeid: koble en opt-in reward-trace og bare den siste
boundary-tilstanden til eksisterende native state-view, collate og recipe-/
trainer-binding. Bevar økonomi-eieren, source binding og dagens standard.
Gjenbruk allerede lagrede inputs/utfall der de dekker samme tilstand; unngå
modellinputs for alle120 mellomsteg. Lag frosne sammenlignbare mu-mål før én
avgrenset native læringsplan eventuelt åpnes. Teknisk PASS er ikke lærings-PASS.
