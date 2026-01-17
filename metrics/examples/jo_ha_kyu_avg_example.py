#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template example: multi-sequence Jo–Ha–Kyu averaging (Fisher z + 95% CI).

This script is intentionally a TEMPLATE:
- The repository runnable examples assume a single sequence (one BVH + one WAV).
- If you have multiple sequences locally, you can use this template to:
  1) compute one Jo–Ha–Kyu correlation per sequence, then
  2) compute an averaged score and confidence interval via Fisher z.

Usage (template)
---------------
1) Create a text file listing your BVH/WAV pairs, one pair per line:

    /abs/path/to/clip_001Re.bvh\t/abs/path/to/clip_001.wav
    /abs/path/to/clip_002Re.bvh\t/abs/path/to/clip_002.wav
    ...

2) Run:

    python metrics/examples/jo_ha_kyu_avg_example.py --pairs your_pairs.tsv

If you only have a single sequence, use `metrics/examples/run_metrics_example.py` instead.
"""

import argparse
import os
import sys

# Add repository root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, repo_root)

from metrics import jo_ha_kyu  # noqa: E402


def parse_pairs_tsv(path: str):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    "Each line must contain exactly two tab-separated paths: BVH<TAB>WAV"
                )
            pairs.append((parts[0], parts[1]))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Template: multi-sequence Jo–Ha–Kyu averaging (Fisher z + 95% CI)."
    )
    parser.add_argument(
        "--pairs",
        required=True,
        help="Path to a TSV file listing BVH/WAV pairs (one pair per line).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for CI (default: 0.95).",
    )
    args = parser.parse_args()

    pairs = parse_pairs_tsv(args.pairs)
    if len(pairs) < 2:
        print("Need at least 2 sequences to compute an averaged Jo–Ha–Kyu score.")
        return

    results = []
    for bvh_path, wav_path in pairs:
        if not (os.path.exists(bvh_path) and os.path.exists(wav_path)):
            print(f"Skipping missing files: {bvh_path} | {wav_path}")
            continue

        r = jo_ha_kyu.compute_jo_ha_kyu_from_bvh_and_audio(
            bvh_path=bvh_path,
            audio_path=wav_path,
        )
        results.append(r)
        print(f"{os.path.basename(bvh_path)}: r={r.r:.4f}, p={r.p_value:.3e}")

    if len(results) < 2:
        print("Not enough valid sequences after filtering missing files.")
        return

    avg = jo_ha_kyu.compute_jo_ha_kyu_avg(tuple(results), confidence_level=args.confidence)
    print("\n--- Jo–Ha–Kyu average (Fisher z) ---")
    print(f"r_avg   = {avg.r_avg:.4f}")
    print(f"SE_r    = {avg.se_r:.6f}")
    print(f"CI({args.confidence:.2f}) = [{avg.ci_lower:.4f}, {avg.ci_upper:.4f}]")
    print(f"r_std   = {avg.r_std:.6f}")


if __name__ == "__main__":
    main()

