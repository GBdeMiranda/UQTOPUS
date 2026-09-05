"""
Builds 0.orig from a finished uncontrolled run.

Copies every field of a time directory into 0.orig and puts the uqtopus
controller placeholder back into the jet patch of U, so that every episode
starts from the same saturated state.

Usage:
    python3 tools/make_start_state.py <time directory> [destination]
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

CONTROLLER_BLOCK = """{
        type            uqtopusBoundaryCondition;
        value           uniform (0 0 0);
    }"""


def replace_patch_entry(text: str, patch: str, block: str) -> str:
    """
    Replace one boundaryField patch entry, braces included.
    """
    start = text.find("boundaryField")
    if start < 0:
        raise ValueError("no boundaryField entry in the field file")

    match = re.compile(rf"^\s*{re.escape(patch)}\s*$", re.M).search(text, start)
    if match is None:
        raise ValueError(f"no '{patch}' entry in boundaryField")

    opening = text.index("{", match.end())
    depth = 0
    for i in range(opening, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[:opening] + block + text[i + 1:]
    raise ValueError(f"unbalanced braces in the '{patch}' entry")


def set_location(text: str, location: str = "0") -> str:
    """Rewrite the location entry of the FoamFile header."""
    return re.sub(r'(location\s+)"[^"]*"', rf'\g<1>"{location}"', text, count=1)


def main(argv: list[str]) -> int:
    if not 2 <= len(argv) <= 3:
        print(__doc__)
        return 1

    source = Path(argv[1])
    destination = Path(argv[2]) if len(argv) == 3 else Path("0.orig")

    if not source.is_dir():
        print(f"not a directory: {source}")
        return 1

    fields = sorted(p for p in source.iterdir() if p.is_file())
    if not fields:
        print(f"no field files in {source}")
        return 1

    destination.mkdir(parents=True, exist_ok=True)
    for field in fields:
        text = set_location(field.read_text())
        if field.name == "U":
            text = replace_patch_entry(text, "jet", CONTROLLER_BLOCK)
        (destination / field.name).write_text(text)

    print(f"{destination}: {' '.join(f.name for f in fields)} from {source}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
