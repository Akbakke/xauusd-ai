<!-- GX1_DOCUMENT_CLASS: CANONICAL | short current status -->
# GX1 status nå

Status 2026-09-11 etter autoritativ post-reboot snapshot.

**Trening er fortsatt blokkert, men kaldstarten er forklart.** BootId 359 startet
`10:39:11.5Z`. Engangsproben passerte `10:40:24.760Z`: første `/bin/true` tok
14,070 ms og neste eksakte `wslpath` tok 71 ms; begge returnerte 0. WSLService,
vmcompute og hns er friske.

Den gamle femårs epoch 1 er ferdig og bevart som historisk bevis. Den viste at
alle featurefamilier og gradientruter var koblet, men Entry/Exit var økonomisk
negative. Den gamle Exit-læringen tvang terminal ved 512 og skal ikke fortsettes
som epoch 2 i det reparerte problemet.

Lifecycle v2 er designet for ubegrenset tradelevetid: 480 lokal-M1-rader ved
første Exit-beslutning, opptil 512 post-entry-rader som rullerende detaljhale,
livstidsoppsummeringer, eksakt t/t+1 og ingen capacity-forced EXIT. TRAIN bruker
et fast, begrenset random-access-utvalg; full VAL følger begge sider og lar Exit
bestemme bare for åpne trades.

V16/pre-run-auditen fant en konkret launchfeil: Task Scheduler hadde
`RestartCount=3`. En kaldoppstartsfeil kunne derfor gjentas tre ganger på samme
defekte WSL-boot. Den innstillingen finnes fortsatt på den deaktiverte
kampanjeoppgaven og må fjernes.

Den sannsynlige enkle rotårsaken er at campaignens 8–10 sekunders kaldkallgrense
var kortere enn den målte friske kaldstarten på 14.07 sekunder. Rettelsen i kildekoden er én 30-sekunders grense for første WSL-kall, uten retry, reset, terminate
eller shutdown.

Kampanjen, gammel WSL-bootstrap og engangsproben er Disabled; proben har
resultat 0 og retries 0. Heartbeat er pauset. Post-reboot-baseline var clean på
`feature/unbounded-exit-lifecycle-v2-20260910` ved `fb4f060d60d7017bf4188684cf5c0b5f05e110b4`;
den nye source/docs/test-committen må bindes eksplisitt av collector før launch authority.
Det finnes ingen ACTIVE-, guard-, trainer- eller CUDA-evidens. Den senere signaturverifiserte GPU-avlesningen (registrert 14:34:18Z) viser
37 C kjerne, 44 C minne, 22.59 W forbruk, 160 W grense og 78 MiB brukt.
Målingen må bindes sammen med data/model/checkpoint/plan, og sanntidsvaktene
gjelder fortsatt ved hver oppstart.
Detaljene og eksakte neste gate står i `CURRENT_HANDOVER.md`.

Kildekontrollen er nå grønn: 4666 tester og 14 deltester bestått i den
samlede, ressursbegrensede kjøringen. Python-kompilering, shell-syntaks og
Git-diffkontroll passerte også. Lifecycle-v2-bindingene, kampanjemiljøet og
skillet mellom lovlig HOLD og manglende Bellman-mål er rettet. Gamle
ufullstendige oppskrifter er fortsatt blokkert. Neste port er ren commit,
kildebundet kampanjeplan og verifisert Windows-installasjon. Den gamle
oppgaven er arkivert; arbeidet fortsetter med bare én tung jobb om gangen.

Den reelle PowerShell-parameteren godtar nå kaldkallet på 30 sekunder.
Windows-regresjonen og den nye samlede kontrollen passerer.
