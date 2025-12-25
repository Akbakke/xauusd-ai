"""
Compat-wrapper for debug_oanda_ping.

Historisk lokasjon: gx1/execution/debug_oanda_ping.py
Ny lokasjon:        gx1/tools/debug_oanda_ping.py

Beholdt for backwards compatibility. Bruk helst:
    python -m gx1.tools.debug_oanda_ping
eller:
    python gx1/tools/debug_oanda_ping.py
"""
from gx1.tools.debug_oanda_ping import main

if __name__ == "__main__":
    main()

