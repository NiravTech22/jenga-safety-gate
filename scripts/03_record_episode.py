#!/usr/bin/env python3
"""Milestone 1 passive overhead/wrist recording; see docs/RECORDING.md."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.record_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
