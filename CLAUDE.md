# CLAUDE.md — Project Guidance (XAUUSD AI)

Denne filen styrer hvordan du (Claude Code) skal jobbe i dette repoet. Vær presis, konservativ med endringer, og arbeid i små, verifiserbare trinn.

## Stack & miljø
- Språk: Python 3.10+
- Pakker: Poetry (pyproject.toml)
- Test: pytest
- Lint/format: ruff, black
- Data: `data/` (ikke commite rådata)
- Kjøring skjer i venv laget av Poetry.

## Kjernekommandoer (bash)
```bash
# Setup
poetry install
poetry run python -V

# Test & kvalitetskontroll
poetry run pytest -q
poetry run ruff check .
poetry run black --check .

# Format
poetry run ruff check . --fix
poetry run black .

# Kjør moduler (eksempel)
poetry run python -m src.app
```