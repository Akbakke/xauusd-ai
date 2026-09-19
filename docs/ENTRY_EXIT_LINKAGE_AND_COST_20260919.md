# Entry/Exit-kobling og kostnadsgrunnlag etter 512

Kode og kildebindinger bekrefter at Entry-target fortsatt er første utførbare
likvidasjon pluss 119/120 av Q_mu(HOLD) ved ankeret. ONLINE 512 velger ikke
handlingen i denne targetberegningen. Lærerrefresh endrer bare bootstrap-
verdier; den erstatter ikke de faste referansesannsynlighetene. Dette er den
avtalte kausale referanseberegningen, ikke en påvist implementeringsfeil.

Lagrede TRAIN-ankre tillater én avgrenset beregning: velg HOLD/EXIT etter
modellens prediksjon ved første utførbare Exit-state, deretter referansepolicy.
Handlingen velges ikke fra etterfølgende observert gevinst. Ankerinformasjonen
kan ikke brukes som input til den tidligere Entry-beslutningen.

|Beregnet Entry-utfall|LONG Bps|SHORT Bps|
|---|---:|---:|
|Opprinnelig referansepolicy|−3,7500|−7,9601|
|Ett 256-valg, deretter referanse|−3,7315|−5,7794|
|Ett 512-valg, deretter referanse|−2,9894|−4,6324|

512-forbedringen gir +0,7421/+1,1470 Bps mot 256 i denne hybridberegningen.
Begge sidegjennomsnitt er fortsatt negative. Dette er gjenbrukt TRAIN og én
avvikende beslutning, ikke full evaluering av 512-policyen eller strategiprofitt.
Hele policyen krever native forløpsvurdering; eksisterende measurement_only-
cohort avviser slik rollout. Denne sperren er beholdt, og ingen kjøring er bundet.

Kostnadskjeden forklarer også en viktig forutsetning: kommisjon er 0, mens
slippage er fast 2 Bps per utførelse, altså 4 Bps for inngang og utgang, i tillegg
til BID/ASK-prisene. Slippage er eksplisitt valgt konservativt før vurdering,
ikke estimert fra utførelsene. Policyen merker selv historisk kostnadssannhet
som ukvalifisert. PASS betyr at det eksplisitte parametersettet er komplett.

De 258 bevarte OANDA-fillene har kommisjon 0 og fillpris lik fullVWAP, men
mangler beslutningsquote koblet til fillklokken. Null differanse mot fullVWAP
måler ikke latenstidsslippage. Den rekonstruerte gjennomsnittlige første-
likvidasjonsfriksjonen er 5,8641 Bps; den faste slippage-forutsetningen står for
omtrent 68,2 prosent. Resten er ikke isolert som ren spread i denne beregningen.

Dette beviser ikke at 2 Bps er feil eller kan senkes. Brukeren er spurt etter
filbane til eksisterende quote/ordre/fill-logger. Ingen rå broker-rader er
publisert i disse nye aggregatene; eksisterende kildeartefakt er bare hashbundet.
Ingen nye handler, targets, fits, forwards eller optimizersteg er utført.
Bevar kostnader og modeller mens målegrunnlaget avklares. Ingen terskelsøk eller
billigere kostnader for å få et ønsket resultat. TEST forblir forseglet.

Tre avgrensede agentgjennomganger bekrefter at ingen Entry-kodeendring er
begrunnet av disse funnene. Negative sidegjennomsnitt avkrefter ikke mulig
betinget fordel. Manglende slippagemåling er en kostnadsusikkerhet, ikke et
bevis på manglende signal eller en nødvendig stopp for alle undersøkelser.
Neste konkrete spørsmål er hele den frosne Exit-policyens utfall under
uendrede kostnader, gjennom separat bundet TRAIN-omfang i native evaluator.
Dette må skilles fra profitabilitet for modellens faktiske Entry-valg.

Den direkte beviskjeden ble også kontrollert: broker-materializeren hentet
historiske API-transaksjoner og bevarte sanitiserte fills uten ordre-ID,
beslutningstid eller beslutningsquote. Råresponsene ble ikke bevart. Den
bundne quote-kilden er M1 BID/ASK OHLC uten ordre-/fill-ID; den gjenoppretter
ikke en slik kobling. Ingen av de direkte refererte kildene peker på en
sammenkoblet utførelsesjournal. Dette utelukker ikke separate logger andre
steder. Ingen ny API-forespørsel eller handel ble gjort.

NEXT_RUN_POLICY.json hadde fortsatt foreldet toppnivåstatus og neste handling
fra før den fullførte 512-prøven. Disse er rettet til fullført og ingen bundet
kjøring; training_enabled er fortsatt false. Originale kjøringsbevis er bevart.

Gjennomførbarheten er avklart: en separat TRAIN-rollout krever en avgrenset
kontraktsutvidelse, ikke bare ny konfigurasjon. Den eksisterende native
adapteren må håndheve observasjonsgrensen 2026-03-01T00:00:00Z før videre
states og økonomiske overganger. Grensen må rapporteres som egen sensurering,
med åpen siste utførbar likvidasjonsverdi og model_exit_executed=false.
Å forkorte available_state_count eller kalle grensen split_end er ikke korrekt.
Dagens measurement_only-cohort må fortsatt avvise rollout.

Eksisterende TRAIN-factory, begge sider, kausale inputs, native Exit-rollout,
resume/checkpointbinding og kostnads-/posisjonsregnskap kan gjenbrukes.
Et eventuelt separat omfang må binde rad-ID-er, modell, kalendergrense,
forwards/state-budsjett og én native invokasjon med eksisterende vakter.
Den historiske brede VAL-grensen er ikke et målt behov for denne prøven.
Ressursavbrudd gir ufullstendig resultat, ikke et økonomisk avslag.
Ingen slik kontraktsutvidelse eller kjøring er utført i denne gjennomgangen.
