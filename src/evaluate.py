#!/usr/bin/env python3
"""
CLI evaluation script for Business Entity Resolution.
Evaluates matching_results.tsv and candidate_pairs.tsv against ground truth.

Usage:
    python src/evaluate.py --ground-truth Dataset/train/train_ground_truth.tsv --matching output/matching_results.tsv [--candidate output/candidate_pairs.tsv]
"""

import argparse
import sys
import os
from typing import Dict, Set

# Allow importing local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import evaluate_resolution

DELIM = "\t"


def load_id_map(path: str, col_index: int = 1) -> Dict[str, Set[str]]:
    """Loads a TSV mapping source1_entity_id -> set of matched/candidate IDs."""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split(DELIM)
            s1_id = parts[0].strip()
            if len(parts) > col_index and parts[col_index].strip():
                target_ids = {x.strip() for x in parts[col_index].split(",") if x.strip()}
            else:
                target_ids = set()
            mapping[s1_id] = target_ids
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Evaluate Entity Resolution predictions.")
    parser.add_argument("--ground-truth", "-g", required=True, help="Path to ground truth TSV")
    parser.add_argument("--matching", "-m", required=True, help="Path to predicted matching_results.tsv")
    parser.add_argument("--candidate", "-c", default=None, help="Path to candidate_pairs.tsv (optional)")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit evaluation to first N ground truth rows (for quick checks)")
    args = parser.parse_args()

    if not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth file '{args.ground_truth}' not found.")
        sys.exit(1)
    if not os.path.exists(args.matching):
        print(f"Error: Matching file '{args.matching}' not found.")
        sys.exit(1)

    print(f"Loading ground truth from: {args.ground_truth}")
    gt = load_id_map(args.ground_truth)
    if args.limit and args.limit < len(gt):
        keys = list(gt.keys())[:args.limit]
        gt = {k: gt[k] for k in keys}
        print(f"Restricted evaluation to first {args.limit} entities.")
    print(f"Loaded {len(gt):,} ground truth entities.")

    print(f"Loading matching results from: {args.matching}")
    preds = load_id_map(args.matching)
    print(f"Loaded {len(preds):,} prediction rows.")

    cands = None
    if args.candidate:
        if os.path.exists(args.candidate):
            print(f"Loading candidate pairs from: {args.candidate}")
            cands = load_id_map(args.candidate)
            print(f"Loaded {len(cands):,} candidate rows.")
        else:
            print(f"Warning: Candidate file '{args.candidate}' not found. Skipping candidate diagnostics.")

    print("\n" + "=" * 60)
    print("           ENTITY RESOLUTION EVALUATION REPORT")
    print("=" * 60)

    results = evaluate_resolution(gt, preds, cands)

    print(f"Total Evaluated Entities : {results['total_entities']:,}")
    print(f"Singletons (No Match)    : {results['singleton_count']:,} ({results['singleton_count']/results['total_entities']*100:.1f}%)")
    print(f"Singleton Accuracy       : {results['singleton_accuracy']*100:.2f}%")
    print("-" * 60)
    print(f"Macro Precision          : {results['macro_precision']*100:.2f}%")
    print(f"Macro Recall             : {results['macro_recall']*100:.2f}%")
    print(f"Non-Singleton Macro F0.5 : {results['non_singleton_macro_f05']:.4f}")
    print("=" * 60)
    print(f"===> OFFICIAL MACRO F_0.5 SCORE : {results['macro_f05']:.5f} <===")
    print("=" * 60)

    if "candidate_recall_ceiling" in results:
        print("\nCandidate Blocking Diagnostics:")
        print(f"  * Candidate Recall Ceiling : {results['candidate_recall_ceiling']*100:.2f}% (max possible model recall)")
        print(f"  * Avg Candidates / Entity  : {results['avg_candidates_per_entity']:.1f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
