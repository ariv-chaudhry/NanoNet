#!/usr/bin/env python3
"""Fail if the GitHub release tag does not match NanoNet's package version.

Expects ``RELEASE_TAG`` (e.g. ``v0.2.0``) in the environment.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "nanonet_ml" / "_version.py"

version_data = runpy.run_path(str(VERSION_FILE))
__version__ = version_data["__version__"]


def main() -> int:
    tag = os.environ.get("RELEASE_TAG", "").strip()

    if not tag:
        print("RELEASE_TAG is not set", file=sys.stderr)
        return 1

    expected = f"v{__version__}"

    if tag != expected:
        print(
            f"Release tag {tag!r} does not match package version {expected!r}",
            file=sys.stderr,
        )
        return 1

    print(f"Release tag {tag} matches package version {__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())