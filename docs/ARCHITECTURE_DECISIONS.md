# Architecture Decision Log (ADR)

Korte beslutningsnotater og lenker til audit/report for større endringer i GX1.

---

## ADR: Entry v10 ctx training dataset – monolith delt (done subsystem)

**Status:** Frosset. Dette området er ferdig; endringer skal være eksplisitte, kontrakt-endringer eller nye filer — ikke gjenbruk av gamle navn.

**Beslutning:** Base-script `build_entry_v10_ctx_training_dataset.py` er erstattet med stub (fail-fast). Kanoniske entrypoints er `build_entry_v10_ctx_training_dataset_signal_only.py`, `build_entry_v10_ctx_training_dataset_legacy.py` og stabil importflate `gx1.datasets.entry_v10_ctx_legacy`.

**Kjede:**
- Audit: [AUDIT_build_entry_v10_ctx_training_dataset_references.md](AUDIT_build_entry_v10_ctx_training_dataset_references.md)
- Final verification: [REPORT_build_entry_v10_ctx_routing_verification.md](REPORT_build_entry_v10_ctx_routing_verification.md)

**Oppskrift (mal for andre monolitter):**  
Audit → stub → stabil importflate → report → GO. Gjenbruk denne prosessen når dere deler opp andre monolitter i GX1.
