"""
Compat-wrapper for exec_smoke_test.

Historisk lokasjon: gx1/execution/exec_smoke_test.py
Ny lokasjon:        gx1/tools/exec_smoke_test.py

Beholdt for backwards compatibility. Bruk helst:
    python -m gx1.tools.exec_smoke_test
eller:
    python gx1/tools/exec_smoke_test.py
"""
from gx1.tools.exec_smoke_test import main, run_smoke_test

if __name__ == "__main__":
    exit(main())

