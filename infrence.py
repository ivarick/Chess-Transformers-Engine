"""Backward-compatible shim for the old misspelled entry point.

Use ``python inference.py`` for new scripts.
"""

from inference import main

if __name__ == "__main__":
    raise SystemExit(main())
