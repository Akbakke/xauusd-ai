# Gjeldende GX1-systemkart — 17. september 2026

Kode: /home/andre2/src/GX1_CURRENT, branch work/gx1-current. GX1_ENGINE/.git er
felles Git-lagring, ikke alternativ oppstartsvei. Data: /home/andre2/GX1_DATA.

## Modell og læringskjede

Kausale native features og TRAIN-eid normalisering
→ Entry: lokal M5-historikk + M15/H1/H4/D1, LONG/SHORT/FLAT
→ Exit: lokal M1-historikk + M5/M15/H1/H4/D1, HOLD/EXIT_NOW
→ samme V4 BID/ASK-, kostnads- og økonomiberegning i trening og evaluering.

Alle 200 features, åtte familier og tidsrammer bevares. Hver timeframe bruker
sin tilgjengelige lukkede klokke. Antall features er ikke antall uavhengige
signaler. Sammenkobling alene dokumenterer ingen prediksjons- eller handelsfordel.

Entry-Q kan nå trene sin upstream representasjon: den blokkerende detach før
Entry-Q-mikseren er fjernet. Exit-tokenets eksisterende detach beholdes.
Dette innebærer ikke full isolasjon av alle delte parametere. Exit har egne
exit_episode_family_tf-rutere; null Exit-gradient på Entry-ruteren er ikke
bevis for at Exit-ruteren er frakoblet. Forecast er hjelpeoppgaver, ikke direkte
handlingsfasit. Exit har også kausal prissti og livstidssammendrag.

## Gjeldende targets i det avsluttede forsøket

- Exit-HOLD-target er et observert, diskontert Q_mu-returutfall under fast kausal
  referansepolicy: HOLD119/120, EXIT1/120 etter første handling. Maksimalt120
  observerte beregningssteg og gyldig frossen boundary-bootstrap bevares.
- Entry-target er første gjennomførbare likvidasjonsverdi + V_mu(state0),
  der V_mu=(119/120)*Q_mu(HOLD); ugyldig/terminal HOLD gir EXIT=0. FLAT=0.
- Positive og negative observerte HOLD-utfall teller. Tidligere
  max(observert HOLD-utfall,0) brukte framtidig informasjon til første
  handlingsvalg og er rettet. Maks over critic-estimater i legacy-grenen
  er en annen beregning og er bevart.

Dette er referanse-policyverdier, ikke optimal verdi eller faktisk profitt
under en lært greedy-policy. Framtidige utfall er targets, aldri online-input.
Ingen fast holdetid eller tapsgrense er innført. Bootstrap, successors og
kostnader er bevart. Femstegsoppsettet er historisk sammenligningsgrunnlag.

## Aktuell forsøksstatus og drift

Den korrigerte native256-prøven er fullført på955abf19. Fersk lagret
initialisering, normalisering/labels fittet på prefix-TRAIN, samme4096 Entries,
frossen lærer og slutt-ONLINE. Bare TRAIN256/Exit-anker/samplede states er målt.
Korrekt avledet Entry-baseline gjenbruker originale prediksjoner; Exit-målene
beholdes. Paret analyse er ferdig og læringsporten ikke bestått: Entry all-FLAT,
Exit alltid HOLD for LONG / EXIT for SHORT. Signaldiagnosen er fullført: kraftig
vekst i nesten felles representasjoner og rundt ti ganger mindre variasjon i
Entry-hidden. Inputnormaliseringen er uendret. Normalisering før tre residualprojeksjoner er implementert og testet;
én native256-prøve er særskilt bundet. Se VEIEN_VIDERE.md. Effekten på læring
er ennå ikke målt. Entry-hidden/Q inngår i Exit-tokenet, så endringer må også kontrollere
bevaring av lærerens Entry-/Exit-outputs og frosne mål.

Én kjørevei: eksplisitt NEXT_RUN_POLICY → bundet native campaign → etablert
Windows-launcher/controller → gx1_capped_run.sh → native kandidatvindu.
TRAIN16, VAL256/8CPU/3t når særskilt tillatt, FP32/TF32 av og etablerte vakter.
Det brukte unntaket er stengt. Ingen ny trening/full VAL/TEST er åpnet.

Handover er kun lesing. `current_work` gjelder dagens jobb; øvrige felt fra
COMPLETED_RUN.json er historikk. Historiske smoker/kildekopier er avhengigheter
og bevis, aldri alternative oppstartsveier. Se CURRENT_HANDOVER.md,
VEIEN_VIDERE.md og docs/LEARNING_GATE_20260916.md.
