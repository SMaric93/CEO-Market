"""Cross-fitting utilities for fold construction and out-of-fold predictions."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.model_selection import KFold


def assign_event_folds(
    event_panel: pl.DataFrame,
    *,
    n_folds: int,
    time_ordered: bool,
    random_seed: int,
) -> dict[str, int]:
    """Assign succession events to cross-fitting folds."""

    ordered = event_panel.select("event_id", "succession_year").sort(["succession_year", "event_id"])
    event_ids = ordered["event_id"].to_list()
    if time_ordered:
        chunks = np.array_split(np.arange(len(event_ids)), n_folds)
        return {
            str(event_ids[idx]): fold
            for fold, indices in enumerate(chunks)
            for idx in indices.tolist()
        }
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    folds: dict[str, int] = {}
    for fold, (_, test_idx) in enumerate(kf.split(event_ids)):
        for idx in test_idx:
            folds[str(event_ids[int(idx)])] = fold
    return folds
