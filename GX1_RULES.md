# GX1-regler — bindende for alle agenter (mål presisert 18.10.2026)

Originalbestillingen er bevart ordrett i
[docs/ORIGINAL_PROJECT_GOAL.md](docs/ORIGINAL_PROJECT_GOAL.md).
Den faste retningen og senere uttrykkelige brukerpresiseringer skal bevares;
status eller beståtte tester skal aldri omdefinere målet. `NEXT_RUN_POLICY.json`
hashbinder originaldokumentet; handover kontrollerer bindingen ved overtakelse.

Dette er den ene regelkilden. Claude leser den via `CLAUDE.md`, Codex via `AGENTS.md`.
Status, gjort/ikke gjort og neste steg står bare i `CURRENT_HANDOVER.md` med
maskinlesbare bindinger i `NEXT_RUN_POLICY.json`, aldri som daglig historikk her.
Den eksekverbare statuseieren `bash scripts/gx1_handover.sh --check` overstyrer all prosa;
er dokument og kode uenige: stopp, og reparer dokument og kode sammen.

Reglene gjelder beholdt kilde og alle nye planer. Tidligere versjoner finnes i Git;
arkivfiler på disk er aldri kjøreautoritet.

## Varig prosjektmål og arbeidsmåte

Operatørbestilling 18.10.2026 er gjeldende prosjektretning: bygg en
fullautomatisk XAUUSD-bot med M5 Entry og kausal, lukket M15/H1/H4/D1-kontekst.
LONG skal kunne lære bullish tilstander og SHORT bearish tilstander. Prioriter
handelskvalitet og liten MAE før gunstig MFE fremfor antall handler; verken kort
eller lang holdetid er et mål i seg selv. M30 vurderes bare med en eksplisitt
sammenligning av tilført verdi. MFE/MAE er etterfølgende utfall, aldri inputs fra
framtiden. Ingen fast tapsgrense eller maksimal holdetid innføres av dette.
Operatøren presiserte18.10 eksplisitt: M5-prisforløp og MAE før MFE er hovedmål;
1/5/20dagers og øvrige lengre horisonter er støtte. Siste presisering18.10:
Entry skal lære faktisk prisretning fra ett minutt til mange timer, sammen
med signert MFE og liten konservativ MAE før første MFE. Dager er støtte;
fast95min er ikke hovedfasit. Exit vurderes eller trenes ikke i denne bølgen.
Dette avløser tidligere primær-h20-/fast95-vedtak, ikke lang kontekst eller
historiske kildekvitteringer.

Presisering21.10: Entry tar beslutninger på lukkede M5-barer. M1 brukes i
denne bølgen til etterfølgende prisfasit og MAE/MFE-oppløsning, ikke M1-Entry.
En prognose-/observasjonshorisont på ett minutt er ikke Entry-tidsrammen,
Exit-logikk eller en fast holdetid. M1-Exit hører bare til neste bølge etter
Entry-edge. Behold separate navn for beslutningstidsramme, framtidig
målehorisont og senere Exit-tidsramme i dokumentasjon og rapporter. LONG, SHORT og FLAT skal kontrolleres, også
kvaliteten på muligheter modellen avstår fra. Alltid-FLAT er ingen suksess.

Retningsmarkout og forløpskvalitet får egne observasjonsmasker. Eksakte
start-/sluttquotes kan bevise terminal prisendring selv når mellomliggende
MAE/MFE er ukjent. Ingen slik markout beviser realisert handelsøkonomi.
Fremtidige quotes, masker, peak og faktisk framtidig forløp er aldri inputs.
Ingen skjult max/snitt av framtidens fasit over horisonter kan bli Entry-Q.
En ny kvalitetsscore er en kandidat inntil enheten, eneste native autoritet,
kontrakt, fersk baseline og faktisk beslutningskvalifisering er bundet.

Revider hver genuine indikator: formel, enhet, kilde, familie, warmup, maske,
kausal tilgjengelighet, samtlige tidsrammer og faktisk vei inn i modellen.
Omtrent150 er brukerens anslag, ikke et godkjent antall. Tell eierne og skill
indikatorformler, signalverdier, aliaser og observasjonsmasker. Åtte
spesialistfamilier og den niende D1/makro-konteksten beholdes; forslag til nye
eller ubrukelige indikatorer avgjøres på dokumentert kilde og måling.

Prioriter de eksisterende26månedene med GC MBP-1 før utvidelse til hele
OHLC-historikken. Bruk all genuint observerbar handels-, BBO-, volum-,
profil-, footprint- og strukturinformasjon. MBP-1 viser BBO, ingen skjulte ordre,
ordreidentitet eller dypere bok. Ukjent aggressor forblir ukjent. GC er en
futuresinput til XAUUSD; kontraktskifte, faktisk dekning, kildeklokke og basis
skal følge featureverdien. Komplett og delvis observert bøtte er ulike domener.
Bevar price-grid og redegjør for informasjonstap i modellens sammendrag.

En kort CPU-, regel- eller smoketest kan aldri konkludere med at prosjektet
eller en ny MBP-Transformer ikke kan fungere. Fjern slike generelle dommer fra
aktive instrukser; bevar originale målinger og kvitteringer med eksakt omfang.
Ingen måling omskrives til positivt resultat. Inputkonsistens, kapasitet til
TRAIN-fit, generalisering og nettoøkonomi kontrolleres hver for seg.

Bruk eksisterende fungerende eiere og beståtte kontroller. Rett målte
blokkeringer og vedtatte designgap fortløpende med minst nødvendig endring;
ikke bruk tid på kosmetisk ombygging. Dobbeltsjekk utført arbeid. Usikre
målvalg spørres operatøren om mens uavhengig arbeid fortsetter.

## Første kompakte treningsprofil, operatørpresisering19.10.2026

Brukeren ønsker først MBP-1, SMC, EMA, ATR og minste nødvendige kontekst,
med candlestick-/chart-pattern- og overlappende felt parkert for raskere
læringsrunder. Hele rådata-/featurebiblioteket og fullmodellens åtte
spesialistfamilier bevares. Kravet om alle genuine felt i hver første
treningsprofil er avløst av denne avgrensede profilplanleggingen; feltvalg,
familie-/encoderstier og MTF-vinduer må være eksplisitt navngitt og hashbundet
før trening. Reell skip/reduksjon i beregning skilles fra bare nullmaskering.
Ingen felt erklæres støy uten måling. Eksakte aliaser skilles fra korrelerte
signaler; MBP-informasjon droppes aldri ved å gjette fra familienavnet.
Aktuell latency/minne måles på reelle data;30dagers trening er ingen målt
forutsetning. Kausalitet, kostnader, TRAIN-fit, fersk profilbundet baseline,
konsum/paritet/Entry-only/resume/maskinvare og alle sikkerhetsgrenser består.
Ingen ny epoke/HPO, Exit eller TEST-adgang følger av profilvalget.
Faktisk dynamisk paritet og Entry-only optimizer/resume må kvalifiseres før
epoken. Eventuelle tekniske updates krever egen eksakt hashbundet policy,
recipe/campaign og training_enabled=true, mens full_epoch_training_allowed
forblir false. Nullstegs initialmåling beholder training_enabled=false.
Kvalifiseringsvekter brukes ikke til hovedepoken eller modellutvalg.

Første navngitte profil er `compact_mbp_smc_ema_atr_clock_macro_entry_v1`,
kompilert av eksisterende featuregruppe-eier og bundet som eksakt
`entry_execution_profile={name,contract_sha256}` i metadata/recipe/modell/state.
None beholder fullmodellen. Full fysisk055validering kommer før intern gather;
aktive rapporteringsgater må følge samme profil. Syntetiske rute-/gradienttester
er ikke genuin konsum-/latency-/læringsmåling.

Kvalifisert filhashgjenbruk er avgrenset til én eksplisitt valideringsoperasjon:
første faktiske SHA bindes til stabil canonicalpath/dev/inode/mode/size/mtime_ns/
ctime_ns, kontrollert før/etter lesing. Hvert kall kontrollerer fortsatt filtype,
symlink, aktuell identitet og forventetSHA samt alle semantiske porter. Ny
operasjon eller endret identitet krever fysisk hash på nytt. Ingen varig cache
eller treningsadgang følger av dette; reell metadata-kjøretid skal måles.

Brukervedtak19.10: organiser datalesing og trening for minst mulig unødvendig
arbeid. Gjenbruk ferdige features, targets, normalisering og beståtte kontroller
innen deres eksakte uendrede omfang. Skill engangsvalidering fra batcharbeid;
ingen gjentatte fullscan/hashinger av samme input inne i batchløkka. Innen én
metadataoperasjon brukes eksisterende verifisert hasheier én gang per stabil
filidentitet. Ingen cache føres gjennom modell-/state-/checkpointarbeid.
Mål oppstart/validering, modellkonstruksjon, batchlesing, GPU-steg og checkpoint
separat før videre ytelsesendringer. Skill logiske lesevolumer/rchar fra
faktiske disklesinger/read_bytes; ingen TB-/hastighets-/ETA-påstand uten
korrekt målerolle. Rett målte flaskehalser med minste endring og bevar kausalitet,
masker, features/MTF og alle sikkerhetsvakter.

For checkpointpolicy kan en eksplisitt, runtime-only Session-kontekst binde én
full fysisk og semantisk inputkontroll til eksakt sessionkontrakt/prefix/monitor/
policy. Alle faktisk verifiserte immutable input- og kostavhengigheter fanges,
og filidentitet revalideres samlet før beviset forsegles. Hver gjenbruk krever
uendret canonicalpath/type/symlink/dev/inode/mode/size/mtime_ns/ctime_ns og
eksakte kontrakt-/prefix-/monitor-/policybindinger; endring feiler lukket uten
automatisk ny binding. En ny Session, også resume/read-only, gjør fersk fysisk
validering. Beviset serialiseres aldri og mates aldri tilbake til filhasheren.
Det erstatter ikke mutable checkpointfilers eksisterende SHA/load/save-kontroll.
Selve filhash-memoet er fortsatt avgrenset til den korte metadataoperasjonen.

Loss-only enheter kan tilpasses fra den eksakte wholeTRAIN-targetpopulasjonen
med separate koordinat-/horisontmasker og counts, etter erklært kontrakt og
egen capped fit-binding. Rå mål, prognoser og beslutningsverdier beholdes i bps.
Inputnormalisering refittes ikke av dette. TRAIN-RMS innebærer ingen sentrering,
epsilon, klipping eller gjettet relativ taskvekt; en uobservert eller nullskala
for en lærbar koordinat stopper før optimizer. Analytisk FLAT0 beholdes som
kjent null med eksisterende denominator. VAL-targetstatistikk er diagnostikk,
aldri fitgrunnlag, og CONTROL-resultater velger ingen skala. Rå prognosefeil og
dimensjonsløst optimeringstap rapporteres separat. En loss-endring eller fit
åpner ingen ny epoke, og erstatter ingen ny kilde-/baseline-/paritetskontroll.

## Én kodebase, avgrensede agentoppgaver

- Eneste kodebase: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`. Andre
  worktrees og grener er git-lagring, aldri arbeidssteder.
- Operatørbestilling 18.10 åpner flere agenter med eksplisitt oppgave- og
  fileierskap. Én integrator eier policy, lansering og samlet verifikasjon.
  Ingen agent starter en egen tung jobb eller endrer en annens filer uten
  koordinering. Alle arbeider i CURRENT og bygger på eksisterende arbeid.
- Én tung CURRENT-jobb om gangen. Ingen kilde- eller policyendringer mens den
  bundne jobben kjører. Leseagenter utfører bare lette kilde-/metadatakontroller.


Operatørvedtak 28.09.2026: separate prosjekter kan arbeide parallelt. CURRENT
bruker en egen prosjektlås; én tung jobb om gangen gjelder fortsatt innen CURRENT.
Den tidligere maskinfelles låsen slettes ikke. Cgroup-tak, krav til ledig minne,
CPU-/trådgrenser og GPU-/temperaturvakter består. Dette åpner ikke trening.

## Omfang

Kun offline XAUUSD: Entry på native M5 (eller en høyere lukket tidsramme når det er
vedtatt), Exit på native M1, de samme åtte kausale feature-eierne, og offline
train/OOS/replay-evidens. Live, paper, demo, broker, daemon, collector, publisher,
promotion, drift og online-adaptasjon er forbudt og kan ikke gjenåpnes av historiske
moduler på disk.

## Ikke-forhandlbare regler

1. **Kun XAUUSD-eksponering.** Operatørvedtak 29.09.2026 tillater kryss-aktiva- og
   makrodata som inputs i navngitte offline forskningsarmer. Kilde, serie, hentemetode,
   tilgjengelighet og tillatt omfang bindes i et immutabelt manifest før henting;
   mottatte bytes og historiske dataversjoner hash-bindes før bruk. Faktisk
   publikasjonstid og det vedtatte publikasjonslaget skal respekteres.
   Operatørvedtak 11.10.2026 åpner navngitte andre instrumenter som manifestbundne
   forskningsinputs og som forskningsmål i offline backtest (pooled estimering,
   porteføljespor); se eget avsnitt nederst. Native Entry-/Exit-kontrakter og
   serving forblir XAUUSD; innføring der krever egen operatørbeslutning på
   deklarert evidens. Kun manifestbundet forskningshenting innen det godkjente omfanget
   åpnes. Broker, collector, live/paper, ordre og øvrige nedlastingsruter forblir stengt.
2. **Ingen fallback, gjettet default, mutable `latest`, foreldet artefakt, syntetisk
   beslutningsinput eller myk gjennomslipp** — heller ikke i evidensen for beslutninger om
   koden:
   a. Hvert beslutningspåvirkende tall har nøyaktig én opprinnelse: navngitt konstant i en
      kontrakteier, statistikk tilpasset på ekte deklarerte data, eller eksplisitt
      CLI/recipe-input. Kan opprinnelsen ikke sies i én setning, er det en gjettet default.
   b. Aldri finn på en størrelse for å få noe til å virke; følg konvensjonen koden allerede
      bruker og si hvilken. Å fjerne et unntak er lov; å finne på et tall er ikke.
   c. Syntetiske, tilfeldige, placeholder- eller leketøysdata beviser bare at koden kjører.
      Konklusjoner krever ekte deklarerte bytes i ekte dimensjoner, eller bevis fra
      kilde/algebra som holder uavhengig av data.
   d. Oppgi evidensklasse for hver påstand: bevist fra kilde, målt på ekte data, målt på
      syntetiske data, eller ubevist. Nedgrader eller trekk tilbake straks klassen viser seg
      svakere, og dokumenter tilbaketrekningen.
   e. Et diagnoseinstrument rapporterer bare det som er gyldig der det kjører; utelat feltet
      heller enn å skrive null eller placeholder.
   f. En terskel mot en samplet statistikk må være minst statistikkens samplingfeil ved den
      faktiske utvalgsstørrelsen, eller sammenligningen tas på hele populasjonen. Oppgi n og
      grensen når en terskel innføres eller flyttes.
   g. Mål der beslutningen tas: på radene modellen trener/serverer på, når størrelsen er
      live, etter deklarert warmup, gjennom samme indeksmapping. Bevis at en gate så på
      riktige rader før du tror den om systemet.
   h. Verifiser mot sjekkens egen kode, aldri mot din modell av den: repliker gatens
      sammenligning felt for felt (inkludert sti-oppløsning, dtype og avrunding) før noe dyrt
      startes. Byte-like kopier på ulike stier er ikke utskiftbare når gaten binder stien.
3. **Én beslutningsautoritet.** Entry-retning kommer bare fra den aksepterte modellens unike
   argmax over sine handlingsverdier i bps (`entry_action_q_bps`, eller en eksplisitt vedtatt
   etterfølger), Exit bare fra samme bundle og delte encoders `unified_exit_action`
   (HOLD/EXIT_NOW). Dette er forventede verdier, ikke kalibrerte sannsynligheter. Ingen
   post-modell trend-, sesjons-, konfidens-, nytte-, terskel- eller lukkeregel kan veto-,
   snu- eller skape en handling. En separat Exit-modell er forbudt. Eksakte ties og
   manglende evidens feiler lukket.
4. **Alle genuine featurefamilier blir i den lærte stien.** Å pensjonere en regel fjerner
   aldri markedsevidensen den bygde på; håndskrevne stemmer erstattes av primitivene de ble
   bygd av. Sammensetningen eies av tuplene i `gx1/contracts/entry_model_native_signal_v1.py`
   — les tall ved å eksekvere eieren, aldri fra dokumenter.
5. **Mål og tap eies av kontraktene** (`gx1/contracts/entry_model_native_training_objective_v1.py`,
   `entry_model_native_train_recipe_v1.py`, `entry_fitted_q_v1.py`,
   `unified_exit_fitted_q_v1.py`, `unified_exit_reference_policy_v1.py`); les skjema, nøkler og
   flagg ved eksekvering. Sizing har aldri retningsautoritet og kan aldri skape en ordre når
   retningen er FLAT eller ugyldig.
6. **Train er lik serve**: eksakte ordnede felt, dimensjoner, normalisering,
   tidsrammekonstruksjon, hasher og beslutningssemantikk. Ubevist inntil samme bundne bundle
   har emittert og bestått en ekte paritetshendelse.
7. **Nyeste gyldige terminale evidens for samme funksjon/data/omfang vinner.**
   Nyere relevant rødt blokkerer eldre grønt; manglende eller misformet evidens
   er rødt for den aktuelle gaten. En gammel regeltest sperrer ikke et annet
   erklært mål eller en ny korrekt generasjon. Et GRØNT datasett slipper bare de bytene til neste gate —
   ikke en modell, retning, bundle eller launch.
8. **Run-ID er lineage, ikke godkjenning.** Hver rebuild har én immutabel build-`--run-id`;
   hver trening sin egen output-`--run-id` pluss en `dataset_run_id` som matcher build og alle
   split-manifester. Aldri auto-promoter en artefakt.
9. **Aldri slett under `/home/andre2/GX1_DATA` eller en aktiv run-sti** uten verifisert
   opprydningsvedtak, rekkeviddebevis som dekker manifest-til-manifest-referanser, og
   aktiv-prosess-sjekk — heller ikke bytes du selv laget. Eneste rute er retention-eieren
   `gx1.scripts.cleanup_gx1_evidence_v1` (plan → godkjenn → utfør, hash-bundet inventar;
   `resume` fullfører en avbrutt STAGED-transaksjon), kjørt i bakgrunnen. Opprydding er en
   stående plikt i samme bølge som foreldet artefaktene; en artefakt som er navngitt som
   baseline eller autoritet i et gjeldende dokument ryddes først etter dokumentendring og
   operatørvedtak, med oppgitt størrelse. En eiernektelse kan aldri overstyres av et
   håndbygd bevis.
10. **Fjern frakoblet kode og foreldede dokumenter** når call-site-scan, tester og
    evidenseierskap viser at de er unødvendige.
11. **Aldri eksponer hemmeligheter, force-push, hard-reset delt arbeid eller overskriv
    ikke-relaterte endringer.** Andres arbeid bygges på, aldri reverseres.
12. **Avslutt hver endring** med fokuserte tester, syntaks-sjekk, stale-path-scan,
    `git diff --check` og en ærlig liste over hva som er ubevist.
13. **Kildekoblingsrevisjoner må bevise import og kjørende bruk av eksakt kontrakteier.** En
    gjentatt modus-, dimensjons- eller feltliteral er ikke eierskap.
14. **Trening får beslutningspåvirkende verdier bare fra den kanoniske recipe-eieren.**
    Ambient miljø, wrapper-defaults og håndskrevet recipe-evidens er forbudt.
15. **Datasett-build- og treningsoutput-identitet er separate roller**; recipe, wrapper,
    trener, bundle og handover binder begge. Manglende, kollapset eller split-brain lineage
    feiler lukket.
16. **Framtidsutfall-måldomener er eksakte.** Spread-aware MFE og path-quality er signerte
    gjennom validering og tap; MAE er en ikke-negativ størrelse. Klipping, absoluttverdier
    eller parkerte nuller er forbudt målomskriving.
17. **Head-liveness krever de eksakte batch-nøklene** Dataset-mappingen emitter; aliaser,
    defaults eller dupliserte mål for å tilfredsstille en sjekk er forbudt.
18. **Input-normalisering tilpasses én gang på hele den fysiske TRAIN-populasjonen** før
    sampling; statistikk, kategoriske domener og hasher er immutabel bundle-tilstand.
    VAL, TEST og serve refitter aldri.
19. **Hvert kontekstfelt har nøyaktig én spesialisteier**; aliaser oppdages fra det faktiske
    signalmanifestet, må være bit-identiske og kan ikke skape en ny normaliseringseier.
20. **Kun atomisk publisering**: ingen bundle eller immutabel hendelse under endelig navn før
    alle bytes, hasher, strict-load og `fsync` er ferdige; atomisk no-replace rename med
    eksakt inventar.
21. **Bygg på eksisterende eier.** Ingen ny versjon eller parallell fil for en liten endring,
    alias eller workaround; en ny fil krever en genuint ny avgrenset autoritet.
22. **Avansert nok til å handle, aldri mer.** Mål eksisterende system → endre én
    recipe-verdi → utvid eksisterende eier → først da noe nytt. Diagnostiser før du bygger;
    av to like fail-closed-design velges det minste.
23. **Overtakelsesautoritet beskriver den aktive grensen.** Kildeimplementasjon, faktisk
    kjøring og aksept er tre tilstander; slå dem aldri sammen til «klar».
24. **En endret-sti-telling er ikke kildeidentitet.** Ignorerte filer (`.git/info/exclude`,
    `.claude/`) er usynlige for arbeidstre-fingeravtrykket; les `git worktree list
    --porcelain` og diffen før fortsettelse autoriseres.
25. **Gate-grønt er ikke kvalitets-grønt.** Svar i tre klasser: *målt* (tall, populasjon,
    dato), *bevist konsistent* (sier ingenting om kvalitet) og *ikke undersøkt* (oppgis
    uoppfordret). Kjør dyprevisjonen før du påstår noe, sveip en funnet defektklasse over alle
    eiere i samme bølge, og dokumenter hva som ble og ikke ble verifisert.

## Operatørvedtak09.10.2026: uavhengig Entry-læring

Entry skal lære stabile retningsmuligheter fra observerte markedsutfall,
uavhengig av et Exit-modellestimat. Dette åpner en eksplisitt bundet etterfølger
til dagens Entry-target, samt trinnvis trening der Exit-tap ikke får endre
Entry. Entry kontrolleres først; senere Exit-trening må bevare den godkjente
Entry-funksjonen. Samme bundle, alle genuine features og kausale tidsrammer
består. Ingen ny serving-autoritet finnes før kontrakt, native recipe og
faktisk målt evidens er oppdatert. Targets med flere observasjonshorisonter
innfører ingen maksimal holdetid. Konfidens må måles; den antas aldri.

## Beholdte forskningsgrenser

Operatørvedtakene 29.09–01.10.2026 åpnet bare navngitt, forhåndsregistrert offline
forskning med manifestbundne kilder, as-of/versjonshistorikk, populasjon, targets,
kostnader, baselines, inferens og beslutningsregel før utførelse.
CPU-fits må også gjennom capped audit/producer; de åpner ikke native trening.
Sju D1-felt i forskningsarmen erstatter ingen native featurefamilie.
Forskningsbaselines kan ha egen deklarert enkel logikk uten å endre native
modell-/head-/handlingsautoritet. Brede indikator-/modell-/terskel-/tapsvektsøk
er stengt. Fullførte engangsplaner skal aldri relanseres.

Den tidligere forskningsarmen B (FRED-serier, høyst 15 felt) og MACRO_CORE er
avløst av swing-programmets manifestbundne primærkilder (CFTC, OANDA, Treasury,
Cboe, SPDR). Gjeldende flate, kilder og tilgjengelighetsregler står i
NEXT_RUN_POLICY.json. Manglende kilder lukkes eksplisitt, ikke ved stille
fjerning av målet. Native innføring krever egen vedtatt kontrakt/evidens.

Kausal sweep-ankret pris-/aktivitetsrepresentasjon ble bestilt i eksisterende
SMC-eier og M5/M1-featurekjede. Negativ håndskrevet regeltest beviser ikke
manglende lært featureverdi. Ingen fast bekreftelses-, taps- eller holdetidsregel
følger med. Prisoppdateringsantall er ikke ekte utført volum/order flow.
Ingen ny modellarkitektur eller featurekassering utover det vedtaket
11.10.2026 (kveld) binder nedenfor.

## Kapasitet og vakter

Hver tung produsent, datasett-build, audit, trening eller replay går gjennom
scripts/gx1_capped_run.sh, eneste kapasitetsautoritet: én tung CURRENT-jobb,
hard cgroup-grense, begrenset swap, CPU-affinitet og tråder. Verdiene eies av
skriptet og bundne vakter, aldri av prosa eller ambient miljø.
Native GPU-trening går bare via eksisterende native campaign og etablerte
maskinvarevakter, innen NEXT_RUN_POLICY.json og eksakt kildebundet recipe.
Manglende host-tilstand/cgroup, låskonflikt eller overskredet grense er hard feil.
Aldri omgå, svekk, dupliser eller ubevoktet start en tung jobb.
Delvis output etter stopp/krasj/reboot er ugyldig uten completion-manifest/hasher.

Utnytt målt CPU-/GPU-kapasitet effektivt innen eksisterende verifiserte
kapasitets-, temperatur-, effekt- og minnevakter. «Maks» åpner ingen omgåelse.
Stabile kjøringer kontrolleres omtrent hver15.–30.minutt, gjerne sjeldnere;
aldri minuttvise modellrunder eller arbeid bare for å fylle ventetid.
Sporadisk PC-omstart gjøres ved trygg grense mellom fullførte kjøringer.
Bevar terminalkvitteringer; alle prosjektjobber, GPU-beregninger og prosjektlåser
på maskinen må være ledige. Ingen aktiv jobb avbrytes for periodisk omstart.
Etterpå bekreftes ny boottid, WSL/tilkobling, uendret kilde og intakte artefakter
gjennom eksisterende vakter. Native profiler kan kreve fersk fysisk Windows-boot.

## Brukte engangstillatelser

COST_TERMS_REVALIDATION_001 brukte det særskilte lesekall-unntaket 02.10.2026:
maksimalt ett GET for practice-kontovilkår og ett for XAUUSD-vilkår, ingen retry,
redirect, transaksjonsoppslag, ordre eller spending. Det er konsumert og kan
aldri gjenbrukes som broker-adgang. Nye kall krever ny eksplisitt autorisasjon.
Claimede benchmark-/auditplaner har samme ikke-relaunch-grense.

Operatørunntak19.10.2026: «Tillat kun de to leseoppslagene» åpner én ny,
eksakt bundet engangsplan med ett GET for konfigurert practice-kontosammendrag
og ett GET for XAU_USD-instrumentvilkår. Ingen retry, redirect, kontolisting,
nye transaksjonsoppslag, ordre eller kontoendring. Frosne tidligere
observasjoner gjenbrukes; aktuelle vilkår beviser ikke historisk kostsannhet.
Ingen øvrig broker-/live-/paperadgang følger av dette. Faktisk konsum står
bare i NEXT_RUN_POLICY og immutable request-/terminalkvitteringer.


Repo-opprydding omfatter verifiserte repo-filer og regenererbare cacher;
DATA/RUNS følger alltid regel 9. Brukerens særskilte godkjenning 06.10.2026
tillater retting/synkronisering av konkrete CURRENT-referanser i installerte
Claude-vakter og tilhørende hook-kommandoer. Andre globale innstillinger er urørt.

## Konsumerte vedtak 10.10.2026

De fire avgrensede Entry-edge-stegene og den endelige Entry-læringsundersøkelsen
(ENTRY_LEARNING_CURVE_20261010_001) er gjennomført og lukket; resultat og
bindinger står i NEXT_RUN_POLICY.json. De autoriserer ingen ny kjøring.
Operatørvedtaket 09.10 om uavhengig Entry-læring (observert markedsmål uten
Exit-estimat, trinnvis trening) består som grunnlag for etterfølgeren nedenfor.

## Operatørvedtak 11.10.2026: regel 1 åpnes for navngitte instrumenter

Brukeren vedtok: «Åpne regel 1 med andre instrumenter (det var initielt for å
ikke blande EURUSD men dette er parkert).» Bundet omfang: navngitte instrumenter
kan brukes som manifestbundne forskningsinputs og som forskningsmål i offline
backtest, herunder pooled/hierarkisk estimering av delte signaler over et panel
og et porteføljespor der et svakt signal måles på flere instrumenter. Hvert
instrument, hver kilde og hver periode forhåndsregistreres med kilde, serie,
hentemetode, tilgjengelighet, publikasjonslag og hashbundne bytes før bruk.
Native Entry-/Exit-kontrakter, serving og eventuell handel forblir XAUUSD til
et eget vedtak på målt evidens. Broker, collector, live/paper, ordre og spending
forblir stengt; ingen ny nedlastingsrute åpnes uten manifest.

Samme dag vedtok brukeren: (a) swing på D1 med 5 og 20 dagers horisont var
primærspor (18.10 setter M5/MTF/MBP som prosjektretning; gjeldende targetvalg
må bindes eksplisitt); (b) én native GPU-jobb og én CPU-forskningsjobb kan gå samtidig
innen CURRENT, hver under egen capped cgroup, minne- og trådgrense, aldri to av
samme slag; (c) OANDA-API brukes som manifestbundet lesetilgang til historiske
candles for navngitte instrumenter, aldri ordre eller kontoendring; (d)
candlestick-/mønsterfelt pauses ut av D1-forskningens inputsett uten sletting,
native beholder alle åtte familier. Programmet står i
docs/ENTRY_DIRECTION_SWING_PROGRAM_20261011.md.

## Operatørvedtak 11.10.2026 (kveld): forward-journal og native D1-kontekst

Brukeren vedtok «Ja, gjør begge» på de to forslagene etter stage C:

(1) **Forward-journal.** Den frosne 68-feltsflaten (A-pristilstand, COT 088691,
kryss-panel over 11 OANDA-instrumenter, Treasury/VIX/GLD-drivere) med
swing_ridge h20/h5 på XAU D1 (H1-rute) scores én kalendermåned om gangen av
eieren `run-swing-forward-journal`: fit bare på rader hvis 20-dagersutfall
lukket før måneden, hver måned scores nøyaktig én gang, immutable hashlenkede
månedsfiler, utfall løses i egne filer, og et fullført journalår evalueres med
den forhåndsregistrerte innen-år-regelen. Journalen leser forseglet periode
etter 2026-07-01 bare for de navngitte manifestbundne kildene og bare til
`scored_end_utc` i policyblokken `forward_journal_20261011`; hver forlengelse
er en eksplisitt policyendring. Native TEST-split leses aldri. Ingen refit på
scorede rader, ingen flateendring, ingen prediktor innrømmes av journalen.
Dette er den eneste bekreftelsesruten for D1-signalet.

(2) **Native D1-kontekst og horisont.** Transformerens neste steg er ikke flere
epoker på 95-minuttersmålet. Vedtatt: observert markedsmål på 1/5/20
handelsdager (eksplisitt bundet etterfølger etter vedtaket 09.10) og en ny
kontekstfamilie med D1-posisjonering, kryss-panel og makrodrivere med samme
as-of-tilgjengelighet som forskningsflaten. Dette krever egen kontrakt,
signalmanifest, normaliseringseier, datasett-rebuild og treningsbudsjett;
ingenting av dette er lansert av vedtaket. Åtte eksisterende familier,
kausalitet, vakter og TEST-forsegling for tilpasning består. Designet står i
docs/NATIVE_D1_CONTEXT_HORIZON_DECISION_20261011.md.

## Operatørvedtak 12.10.2026: minimum native læring på D1-horisontmålet

Brukeren vedtok «ja gjør dette … Bruk minimum nå i starten for å se at alt
fungerer som det skal, så kan vi vurdere å legge på fler ting senere».
Bundet omfang (docs/NATIVE_D1_MINIMUM_LEARNING_PLAN_20261012.md): Entry-only
læring (Exit-tap og Exit-gradienter inaktive, samme bundle) på det observerte
markedsmålet ved 20 handelsdager som eneste mål i tapet, med 1 og 5 dager
observert; full native v10-funksjon på NATIVE_D1-generasjonens inputs; ett
seed; avgrensede faser smoke 256 → fit 1 024 → kurve 4 096 steg med
CONTROL4096-observasjoner; forhåndsregistrert gjennomgangsregel. Brukerens
forslag om å sløyfe M5-sekvens, candlestick-/mønsterfelt og M1-Exit er ikke
åpnet: regel 4, Multi-TF og Exit på M1 består. Ingen full epoke, full VAL,
seleksjon på CONTROL, hyperparametersøk, TEST-lesing eller launch utenfor
den bundne campaignen. Hver fase krever egen policybinding og budsjett.

## Operatørvedtak 14.10.2026: gjennomgang før epoke, retention-autoritet og mindre vindu

Brukeren bestilte en grundig gjennomgang før noen epokekjøring («lag deg fler
agenter og let aktivt etter feil/mangler/mismatches/opprydd og slett alt vi
ikke trenger»), vedtok «ja, dette må fikses på. Så klart skal vi kunne
godkjenne, og så klart skal vi slette opptil ca 256 GB om vi ikke trenger
dette», og presiserte «vi trenger ikke kjøre alle 5 årene med train/val og
test, vi kan ta et mindre utgangspunkt på f eks et år for å se om pipelinen er
good og justeringer trengs, før "stor trening"».

Bundet: (a) Retention-eieren er den eneste sletteruten (regel 9) og skal kunne
akseptere en plan: launch-kontrakten beskytter eksplisitt deklarerte leaves
(`retention_protected_leaves`), bare deklarerte nøkler er referanser, og et mål
som et gjeldende dokument navngir med absolutt sti nektes til dokumentet er
endret. Opprydding av ureferert, foreldet evidens inntil omtrent 256 GB er
vedtatt gjennom plan → godkjenn → utfør i bakgrunnen; målene og størrelsene
bokføres per bølge i NEXT_RUN_POLICY.json. (b) Neste læringsfase bruker et
mindre kronologisk vindu (størrelsesorden ett år av TRAIN) før noen full
epoke; full epoke krever et læringssignal i det mindre vinduet og eget vedtak.
Ingen seleksjon på CONTROL, ingen TEST-lesing, ingen featurekassering,
training_enabled=false består. Agentene i gjennomgangen var leserettede;
Dette tidligere en-agent-kravet er avløst av18.10-vedtaket om koordinerte
agenter; én tung jobb og fryst kilde består.

## Operatørvedtak 17.10.2026: separat Transformer-/TCN-undersøkelse

Brukeren bestilte teknisk kartlegging, kvalifisering av korrigerte h20-mål og
en separat avgrenset GC/M5-sekvensstudie med liten Transformer og TCN.
Dette åpner kode og capped datakvalifisering i den nye bundne forskningsplanen,
ikke relansering av P0 eller automatisk trening. Native Transformer beholdes;
D1/h20 og GC/M5 skilles. Modellfits krever egen nødvendig NEXT_RUN_POLICY-binding
etter kvalifisering. All26GCmåneder er utvikling; TEST, native baseline/Exit,
handel, broker, spending og omfattende GPU-trening forblir lukket.

Samme17.10 godkjente brukeren avgrenset train/smoke på26månedersmaterialet etter
kontroll av informative features. Dette binder027/028: alle254 kvalifiserte
native markedsfelt og eksisterende F/L/V/basis/sesjon, fem kronologiske fold,
to faste lineære fits og128 steg hver for liten Transformer/TCN per fold,
maksimalt20fits/10nevrale fits,4G CPU-audit og1800s totalt. Ingen HPO eller
utvidelse. Makro med uavklart publisering og manglende signert flow fylles ikke.
Native h20/GPU/TEST og produksjon forblir lukket.

## Operatørpresisering 17.–18.10.2026: nytt rådatasett og én native epoke

Brukeren presiserte at CPU-ministudien ikke oppfylte oppgaven: nytt datasett fra
starten på26måneder med MBP-1 og alle genuine features i den faktiske native
Transformer-stien, deretter én full epoke (forventet12timer+). Dette er nytt
vedtak for en ny datasett-/treningsidentitet, ikke relansering av gamle planer.
H20 var beholdt i dette vedtaket;18.10 er M5-prisforløp hovedmål og H20 støtte.
Siste fordelingsønske er omtrent20måneder TRAIN og siste måneder
VAL/TEST; eksakte datoer bindes før normalisering/mål/trening. Alle26måneder
er tidligere undersøkt utvikling, så retrospektiv TEST er ikke urørt bekreftelse.
Brukeren ba også kontrollere full MBP-1-utnyttelse, herunder footprint og kausale
volumstøttede order-block-proxyer. Ukjent aggressor, manglende bokhistorikk og
ikke-observert dybde/ordreidentitet skal aldri konstrueres.

Én epoke er nå eksplisitt bestilt; den tidligere begrensede CPU-oppskriften og
kravet om enda et eget epokevedtak brukes ikke til å avvise denne bestillingen.
Før native launch består de faktiske datakvalitets-, mål-, kausalitets-,
normaliserings-, fersk-baseline-, paritets-, maskinvare- og campaign-gatene.
training_enabled=false er korrekt inntil en eksakt ny campaign er kvalifisert
og bundet. Ingen automatisk ekstra epoke/HPO, TEST fra juli, broker eller handel.
