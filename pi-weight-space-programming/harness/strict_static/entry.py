"""Sandbox entrypoint. Executed only after the mount namespace pivot."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strict_static.worker import main

if __name__ == "__main__":
    main()
