# Eksisterende verdilag kan hente signal — Entry LONG er fortsatt uavklart

Dette er diagnose på TRAIN, ikke en ny godkjent treningsmodell. Ingen original
checkpoint, optimizer, EMA eller frossen lærer er endret. Ingen GPU, VAL eller
TEST ble brukt. Referanse32 og alle ferdige målinger skal ikke gjentas.

## Avgrenset funn

Etter referanse32 var læringsraten 9,890738e-5, Adam hadde faktisk tatt 32 steg,
og Exit-hodet var oppdatert. Verken null læringsrate eller et frosset siste lag
forklarer svikten. Exit-tapsvekten var omtrent 0,112, ikke null.

Vi tilpasset ett delt, eksisterende lineært Exit-verdilag til de lagrede
representasjonene og de samme frosne Q_mu-målene. Én direkte minste-kvadraters
løsning med numerisk toleranse fra float32; ingen søk over modeller/tapsvekter.
Deretter ble eksisterende Entry-verdilag tilpasset den nye, frosne Exit-læreren
ved de korrekte første ankrene. Alle øvrige modellvekter og 200 features beholdes.

Entry-verdiene mates videre til Exit. Derfor ble begge lag også kontrollert
sammen i den faktiske modellen, med endringer bare i minnet. Exit-gevinsten består.
Entry-resultatet reproduseres med maksimal forskjell 0,00001145 Bps.

| Mål / separate 128 TRAIN-Entries | Gjeldende96 | Tilpassede lag sammen |
|---|---:|---:|
| Exit LONG MSE | 1070,5441 | 1006,6943 |
| Exit SHORT MSE | 1069,4183 | 1053,1474 |
| Entry samlet MSE | 37,5651 | 26,6493 |
| Entry LONG MSE | 32,2571 | 51,7595 |
| Entry SHORT MSE | 80,4376 | 28,1883 |

Entry-sammenligningen bruker samme nye lærer på begge sider av tabellen; den
skal ikke sammenlignes direkte med tidligere MSE mot den gamle læreren.
Tilpasset Entry velger 37 LONG / 38 SHORT / 53 FLAT, mot tidligere 128 FLAT.
Exit velger HOLD i 53,13% LONG og 48,83% SHORT. Begge Exit-sider slår null og
treningsutvalgets konstantbaseline på separat TRAIN. Entry samlet slår også
konstantbaseline 29,2092, men LONG-feilen alene blir verre. Samlet gevinst er
derfor ikke et PASS for begge retninger.

512 trente og 128 separate Entries har ingen overlapp. Begge dekker de samme
tolv TRAIN-månedene; separate TRAIN er ikke kronologisk holdout. Exit-MSE med
det første frosne Entry-laget bedres bare i 6/12 måneder for hver side.
Månedstall for alle kontroller står i maskinrapporten. Exit-koeffisientnormen er
stor, omtrent 620; numerisk følsomhet og overtilpasning er ikke avkreftet.

## Avkreftet og korrigert

Hypotesen om sterk kansellering mellom LONG- og SHORT-gradienter ble ikke støttet:
cosinus +0,0804, sentrert +0,4058. Separate sidehoder forbedret SHORT, men forverret
LONG mot det delte laget. Dette begrunner ingen arkitekturdeling.

Første Entry-diagnose er ugyldig og bevart med INVALIDATED.json. Operatøren tok
de siste N radene som ankere, men femstegscachen legger ekstra successors etter
ankerblokken. Korrekt blokk er transition_count:transition_count+selected_entry_count.
V2 kontrollerer eierskap og samsvar med tidligere verifiserte tilstand0-features.
Feilen var i diagnoseoperatøren; native trening og Exit-målingen er ikke berørt.
Bare ENTRY_READOUT_BRIDGE_PROBE_20260916_V2 skal brukes videre.

## Markedsutfall og grense for konklusjonen

Et gjenbrukt ankerutvalg på 275 trente og 67 separate Entries gir tilpasset Entry
henholdsvis +5,2506 og +0,8288 Bps gjennomsnittlig referansenytte over ALLE Entries.
På separat utvalg bidrar observerte komponenter +0,8343 og frossen bootstrap -0,0056.
Dette er likvidasjon pluss diskonterte utfall under referansepolicyen Q_mu,
ikke realisert kontant-PnL under modellens egen Exit-policy. Utvalget er lite,
delvis dekket og gjenbrukt. Profitt og kronologisk generalisering er ubevist.

Neste konkrete blokkering er separat Entry LONG-feil. Bruk V2-cachene til å
skille konstant kalibreringsfeil fra feil som varierer med tilstanden før én
begrunnet minste rettelse. Ingen nye forwards, arkitektursøk eller native-steg
følger automatisk av disse målingene. NEXT_RUN_POLICY.json forblir stengt.

Eksakte kilde-, checkpoint- og resultatbindinger samt månedstall finnes i
[maskinrapporten](../handover_snapshot/READOUT_LEARNING_REVIEW_20260916.json).
Alle operatorer, koeffisienter, cached representasjoner og outputs er bevart
under NATIVE_REFERENCE_POLICY_20260916_REFERENCE i GX1_DATA.
