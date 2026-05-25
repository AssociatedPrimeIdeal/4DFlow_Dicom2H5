"""Backward-compatible script entry point."""

import os
import sys


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(REPO_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from dicom2h5.converter import main


if __name__ == "__main__":
    raise SystemExit(main())
