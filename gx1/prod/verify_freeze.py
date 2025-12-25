"""
Compat-wrapper for verify_freeze.

Historisk lokasjon: gx1/prod/verify_freeze.py
Ny lokasjon:        gx1/tools/verify_freeze.py

Beholdt for backwards compatibility. Bruk helst:
    python -m gx1.tools.verify_freeze
eller:
    python gx1/tools/verify_freeze.py
"""
from gx1.tools.verify_freeze import verify_prod_freeze, compute_file_hash

# Re-export main function if it exists
try:
    from gx1.tools.verify_freeze import main
except ImportError:
    # If main doesn't exist, create a simple wrapper
    def main():
        import sys
        success = verify_prod_freeze()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

