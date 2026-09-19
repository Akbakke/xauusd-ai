# GX1 — overlevering19. september2026

Paret analyse av NATIVE_MAIN_ENCODER_NORMALIZED_FIXED256_20260918 er ferdig.
Entry har mer tilstandsavhengig TRAIN-signal; samlet Entry/Exit-port er ikke
bestått. Entry er fortsatt FLAT256/256; Exit HOLD for alle LONG / EXIT for alle
SHORT. Ingen beslutningsforbedring mot residual256. Se
[analysen](docs/MAIN_ENCODER_FIXED256_REVIEW_20260919.md) og VEIEN_VIDERE.md.

Eneste kodevei: /home/andre2/src/GX1_CURRENT, work/gx1-current.
Data: /home/andre2/GX1_DATA. GX1_ENGINE/.git er felles Git-lagring, ikke startvei.
Mac er overlevering: ./handover.sh --check. Linux: bash scripts/gx1_handover.sh --check.
current_work er nåstatus; eldre toppnivåfelt fra COMPLETED_RUN er historikk.

Treningskilde6b44c23d; analyse påda56102f. Ingen modell-/treningskode endret
siden a1c4b443. Native prøve avsluttet18Sep01:29:46UTC /03:29:46Oslo med
guard PASS,256 steg, checkpoint5/slot0/offset256. Task Disabled; ingen native
prosess. Brukt scope stengt; RESUMABLE/complete=false gir ingen fortsettelsesrett.
Ny initialbaseline og final ONLINE har eksakt samme kausale targets; state,
frossen lærer, optimizer/EMA og historiske sammenligninger er verifisert.

Fullstendige bevis: handover_snapshot/MAIN_ENCODER_FIXED256_REVIEW_20260919.json,
MAIN_ENCODER_FIXED256_VERDICT_20260919.json og MAIN_ENCODER_DECISION_GAP_20260919.json.
Neste: én bundet native representasjonsmåling, NATIVE_MAIN_ENCODER_REPRESENTATION_20260919. Inputkontroll og 16 kontrakttester bestod. Ny initialfunksjon er kildebundet. Recipe/campaign er ikke forberedt; se VEIEN_VIDERE.md. Modellkode er uendret.

Én agent/én tung jobb; gjenbruk ferdige analyser og beståtte tester. Bevar alle200
features/familier/tidsrammer, kausalitet, kostnader og originalfiler. Ingen
terskel-/tapsvektsøk, ny trening/full VAL/CONTROL/TEST, live/paper eller spending.
TRAIN-funn er ikke generalisering eller profitt. Stående offentlig push gjelder
ferdig kode/docs/stier/aggregater, uten rådata, vekter eller hemmeligheter.
