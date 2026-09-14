# Grunnregel fra brukeren — 2026-09-12

Prioriter å få modellen trent og måle resultatet. Gjør aldri unødvendig omfattende oppdateringer.

- Endre kode bare når en konkret, observert blokkering hindrer det avtalte treningsløpet. Navngi blokkeringen og gjør den minste nødvendige rettelsen i eksisterende kode.
- Ingen forebyggende refaktorering, nye rammeverk, ekstra rapportløp eller utvidelse av oppgaven uten brukerens uttrykkelige ønske. Allerede fungerende og verifisert arbeid skal gjenbrukes.
- Kjør bare målrettet verifikasjon som er nødvendig for den faktiske endringen. Ikke gjenta beståtte smoker, trening eller fullsuiter uten ny relevant feil.
- Én agent og én tung jobb om gangen. Kontroller kjøringer som varer i flere timer én gang i timen, gjerne sjeldnere når tilstanden er stabil (brukerpresisering 2026-09-13). Aldri minuttvis polling eller statusprat uten en konkret ny feil eller et nært forventet sluttpunkt. Lokal automatisk sikkerhetsvakt håndterer hyppige temperatur- og prosessmålinger uten modellbruk.
- Bruk tokens konservativt: vent når jobben må beregne, ikke fyll ventetiden med nye analyser, dokumentasjon eller sideoppgaver. Ikke del venting opp i minuttvise modellrunder. Gjenbruk kjent kontekst og beståtte resultater; les bare nødvendige filer og korte loggutdrag. Unngå hele manifest-, kildekode- og loggdumper.
- Før en kodeendring skal den konkrete blokkeringen og minste nødvendige rettelsen kunne forklares kort. Hvis arbeidet ikke bringer avtalt trening eller resultatmåling videre, skal det utgå. Bruk eksisterende løsning fremfor nye lag, rammeverk og generell opprydding. Når nødvendig verifikasjon består, fortsett treningen; ikke utvid testen eller endringen uten ny relevant evidens.
- Gi korte statusmeldinger ved vesentlig fremgang, feil eller resultat. Ikke gjenta en uendret status bare fordi en målfortsettelse eller timer aktiveres.
- Stående autorisasjon gjelder nødvendige handlinger innen avtalt oppgave. Ikke be om samme godkjenning på nytt.
- Bevar fullførte treningsresultater og aktiv kjøring. Ikke endre en frosset kilde eller starte kjøringen på nytt for dokumentasjon, opprydding eller spekulative forbedringer.
- Skill lokale driftsgrenser fra maskinvareprodusentens spesifikasjoner. En overskredet lokal temperaturgrense er ikke alene bevis på overoppheting eller utilstrekkelig maskinvare.

Gjeldende treningsmål og status står i `GX1_ARBEIDSMAAL.md`. Denne brukerregelen gjelder også arbeid i prosjektets eksterne Linux-repositorier.

## Current operator stop — 2026-09-14

Full June VAL after first five-year TRAIN epoch completed on 2026-09-14.
USER-REQUESTED STOP: GX1RandomAccessCampaignV2 is disabled/stopped and native
PID 723 was terminated. No next epoch before the Entry/Exit, MAE/MFE and
month-end HOLD findings are resolved. Epoch 2 had already started before the
request; final durable pointer is checkpoint 315, epoch_index 1, 19,908 total
optimizer steps, batch offset 320. Do not call invocation 7 an outer PASS.

The first-epoch immutable EMA (19,588 steps) and full VAL are preserved.
VAL completed 57,845,748 state views / 467,371 forwards, with 7,472 model exits
and 3,544 month-end censored sides out of 11,016 hypothetical side paths.
Entry chose 4,180 LONG / 1,328 SHORT / 0 FLAT. Of those 5,508 selected trades,
only 2,227 exited; 3,281 remained HOLD. Official full-policy net Bps is unavailable.
See docs/ENTRY_EXIT_REVIEW_20260914.md and handover_snapshot/ENTRY_EXIT_SUMMARY_20260914.json.
Do not present closed-winner statistics as complete-policy profitability.

Frozen executed source remains 03592fe6f1113736d0499c35ef98a3d9267e558c in
/home/andre2/src/GX1_VAL_PAUSE_ENVELOPE_V40. CURRENT_NATIVE_RUN.json retains
the exact source, data, recipe and session bindings. No executable source was
changed in place. V41 capacity preparation is separate, not active.

One agent, one heavy job; hourly model observations only when a long run is
active. Existing signed local guards retain their frequent safety measurements.
300 W cap, 85 C core, 80 C memory, 12 GiB VRAM, 20 GiB RAM and 512 MiB swap.
TEST remains sealed; no live/paper trading, promotion or external spend.

## Historical instructions and observations

Current source: 03592fe6f1113736d0499c35ef98a3d9267e558c at /home/andre2/src/GX1_VAL_PAUSE_ENVELOPE_V40.
Preserve completed TRAIN and 18,353,548 saved VAL views.
Use CURRENT_NATIVE_RUN.json. Older source references below are historical.
One heavy job, sparse checks, existing guards, no speculative changes.

# GX1 takeover instructions

Read `GX1_ARBEIDSMAAL.md`, `CURRENT_HANDOVER.md`, then `SYSTEM_MAP.md`.
Run `bash scripts/gx1_handover.sh --check` and `bash scripts/gx1_handover.sh`
from this documentation checkout on the training host. They verify the small
immutable bindings and observe the separate frozen training source/runtime.
A cloned repo on another host needs the named artifacts restored first;
missing files are missing evidence, not permission to start a new run.

## Current operator scope — 2026-09-13

The current instructions are in `GX1_ARBEIDSMAAL.md` and `CURRENT_HANDOVER.md`.
The operator authorized the full local lifecycle-v2 campaign: full one-year
TRAIN smoke plus full June VAL are complete; full five-year TRAIN now runs
for up to 30 epochs, with full June VAL after each epoch and patience 5.
TEST remains sealed; no paper/live, broker activity or external spending.
Standing authorization covers ordinary necessary in-scope work; do not ask
again for the same approval. The Windows controller owns automatic guarded
pause/reboot/resume. The frozen active source is identified by
`CURRENT_NATIVE_RUN.json`; never edit or rebind it for documentation.

Conserve tokens and time: one agent and one heavy job by default, approximately
15-minute observations, silence on routine healthy progress, no minute polling.
Fix only a concrete observed blocker with the smallest necessary change.
No speculative refactors, repeated passed smokes/full suites, opportunistic
benchmarks or extra cleanup. Targeted verification for a real change suffices.
The completed September 12 three-agent audit was explicitly requested and is
finished; it does not authorize continuing parallel agents.

Current plan/recipe values supersede the old operational limits below: 300 W
physical cap, 310 W actual-draw stop, 85 C core, 80 C memory junction, 12 GiB
VRAM; the keeper reduces power to 200 W at 80 C core. These are local operating
limits. Do not restore the obsolete 160/200 W policy or disable the campaign
because an old paragraph says training is blocked. Earlier dated hold, retry,
power, sampler and launch-state instructions below are historical for this
campaign. Technical contracts still apply; source and runtime evidence outrank
stale prose. `scripts/gx1_handover.sh` observes the explicit native binding;
it neither authorizes nor starts training.

The historical architecture and evidence rules remain in `GX1_RULES.md`;
its latest operator scope overrides earlier dated launch holds. Source owners
own dimensions, feature order, economics and model decisions. No handwritten
threshold or maximum holding period has been added. Never replace the learned
policy while a bound evaluation is running.

For documentation/observer work, inspect the diff, run shell syntax and focused
observer tests. Preserve installed Git hooks. Avoid a second heavy job while
the trainer owns its lock; commit-time capped checks need a natural idle slot.
Do not use a frozen training checkout for edits or commit a trained checkpoint,
raw data, runtime logs, credentials or private configuration to Git.
