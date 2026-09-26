# GX1-regler — bindende for alle agenter (konsolidert 26.09.2026)

Dette er den ene regelkilden. Claude leser den via `CLAUDE.md`, Codex via `AGENTS.md`.
Status og neste steg står bare i `CURRENT_HANDOVER.md` og `VEIEN_VIDERE.md`, aldri her.
Den eksekverbare statuseieren `bash scripts/gx1_handover.sh --check` overstyrer all prosa;
er dokument og kode uenige: stopp, og reparer dokument og kode sammen.

Reglene 1–25 er overført fra den arkiverte grenens konstitusjon
(`archive/gx1-engine-audit-v9-20260926:CLAUDE.md`), uten svekkelse; historiske anekdoter er
forkortet og ligger i git.

## Én kodebase, én agent

- Eneste kodebase: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`. Andre
  worktrees og grener (inkludert den arkiverte `audit/v9-premiere-20260905` i
  `/home/andre2/src/GX1_ENGINE`) er historikk og git-lagring, aldri arbeidssteder.
- Én agent (Claude eller Codex) og én tung jobb om gangen. Bygg på den andre agentens
  commits; aldri et parallelt spor. Er noe i strid med dette, stopp og si fra.

## Omfang

Kun offline XAUUSD: Entry på native M5 (eller en høyere lukket tidsramme når det er
vedtatt), Exit på native M1, de samme åtte kausale feature-eierne, og offline
train/OOS/replay-evidens. Live, paper, demo, broker, daemon, collector, publisher,
promotion, drift og online-adaptasjon er forbudt og kan ikke gjenåpnes av historiske
moduler på disk.

## Ikke-forhandlbare regler

1. **Kun XAUUSD.** Entry-kontrakter kan ikke avhenge av andre instrumenters markedsdata
   eller eksponere andre instrumenter. Kryss-aktiva-data er bare tillatt i eksplisitte
   forskningsarmer som gir evidens, ikke inputs, og krever navngitt kilde og immutabel
   manifest før henting. Broker-, collector- og nedlastingsruter er stengt uten eksplisitt
   operatørvedtak. Endring krever eksplisitt regelendring på et VAL-bekreftet resultat.
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
7. **Nyeste gyldige terminale evidens vinner.** Nyere rødt blokkerer eldre grønt; manglende
   eller misformet evidens er rødt. Et GRØNT datasett slipper bare de bytene til neste gate —
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

## Kapasitet og vakter

Hver tung produsent, datasett-build, audit, trening eller replay går gjennom
`scripts/gx1_capped_run.sh`, som er eneste kapasitetsautoritet: én tung jobb om gangen,
hard cgroup-grense, begrenset swap, CPU-affinitet og tråder. Grenseverdiene eies av skriptet
og de bundne vaktene; les dem der, ikke fra prosa. Native GPU-trening går bare via eksisterende
native campaign og etablerte maskinvarevakter, innenfor scope i `NEXT_RUN_POLICY.json`. En
forespørsel over grensene, manglende host-tilstand, låskonflikt eller manglende cgroup er
hard feil. Aldri omgå, svekk, bakgrunnsstart eller dupliser en tung jobb. Delvis output
etter cap-kill, krasj eller reboot er ugyldig inntil completion-manifest og hasher består.
