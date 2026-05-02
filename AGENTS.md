# GX1 Agent Guardrails

- Use `/home/andre2/src/GX1_ENGINE/.venv/bin/python` for Python commands.
- Treat `PROJECT_STATE.md` as the current local project state.
- Never use dummy, synthetic, or degraded fallback inputs for decisioning.
- Never use implicit latest/glob artifact selection for decisioning.
- Never use in-sample scores as decision-valid evidence.
- Never select old invalidated V3 artifacts for decisioning.
- Do not run R6, freeze, promo, live, or package build without an explicit green gate.
- Keep historical artifacts as history unless an explicit selection contract marks them active.
