"""Name normalization rules for deterministic person-crosswalk workflows."""

from __future__ import annotations

import re


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_person_name(name: str) -> str:
    """Normalize a raw executive name for conservative exact-match workflows."""

    lowered = name.casefold().strip()
    collapsed = _NON_ALNUM.sub("", lowered)
    return collapsed
