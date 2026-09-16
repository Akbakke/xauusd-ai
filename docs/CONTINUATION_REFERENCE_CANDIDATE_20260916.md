# Én kandidat: lær en eksplisitt referanseverdi før policyforbedring

Status: valgt hypotese for CPU-prototype og kontraktkontroll. Ingen target-,
modell- eller treningsendring er implementert eller åpnet for native kjøring.

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
