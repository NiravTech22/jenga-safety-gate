#!/usr/bin/env python3
"""Step 0: confirm the hardware split before installing anything else.

Run this first. It doesn't do anything except sanity-check config.yaml and
tell you what's still undecided.
"""
import sys
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    hw = cfg["hardware"]
    problems = []

    if hw["host_pc"]["has_gpu"] is None or hw["host_pc"]["has_gpu"] is False:
        if hw["distillation_machine"] == "host_pc":
            problems.append(
                "host_pc.has_gpu is false but distillation_machine is 'host_pc'. "
                "Step 6 (V-JEPA distillation) needs a GPU somewhere — set "
                "distillation_machine to 'cloud' or 'borrowed' and sort that "
                "out now, per PLAN.md Step 0."
            )

    print(f"host PC has GPU: {hw['host_pc']['has_gpu']}")
    print(f"distillation machine: {hw['distillation_machine']}")
    print(f"edge device: {hw['edge_device']}")

    if problems:
        print("\nOPEN ISSUES:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    print("\nHardware split looks decided. Proceed to Step 1.")


if __name__ == "__main__":
    main()
