"""PyInstaller entry point. Kept separate from `faceid/__main__.py` because
PyInstaller needs a real script path, not a `-m` module target."""

import sys

from faceid.cli import main

if __name__ == "__main__":
    sys.exit(main())
