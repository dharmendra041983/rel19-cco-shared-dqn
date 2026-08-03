#!/usr/bin/env python3
"""Aggregate same-structure per-seed JSON files into mean/std summaries.

This helper is suitable for ordinary numeric metrics. The experiment
scripts remain the authoritative way to reproduce specialized paired
effects and confidence intervals reported in the archived results.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def aggregate(values: list[Any]) -> Any:
    """Recursively aggregate values that share the same JSON structure."""
    first = values[0]

    if all(numeric(v) for v in values):
        arr = np.asarray(values, dtype=float)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "n_seeds": int(arr.size),
            "per_seed": [float(v) for v in arr],
        }

    if all(isinstance(v, dict) for v in values):
        common = set(values[0])
        for value in values[1:]:
            common &= set(value)
        return {
            key: aggregate([value[key] for value in values])
            for key in sorted(common)
        }

    if all(v == first for v in values):
        return first

    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Per-seed JSON files with the same nested structure.",
    )
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args()

    documents = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if not documents:
        raise SystemExit("No input files were supplied.")

    result = aggregate(documents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
