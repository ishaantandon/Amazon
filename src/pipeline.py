#!/usr/bin/env python3
"""
High-Performance Baseline Pipeline for Business Entity Resolution Challenge.
- Modular, memory-efficient, and optimized for Macro F_0.5.
- Handles blocking by country + dual (Name & Address) inverted indexing.
- Includes fuzzy similarity ranking with RapidFuzz.
- Generates candidate_pairs.tsv and matching_results.tsv.
- Built-in evaluation and validation mode.

Usage:
  # Quick validation on 5,000 samples (runs in ~15-30s on CPU)
  python src/pipeline.py --mode validate --sample-size 5000

  # Full test prediction
  python src/pipeline.py --mode test
"""

import os
import re
import sys
import time
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
from rapidfuzz import fuzz

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import evaluate_resolution

DELIM = "\t"

# Common business suffixes to strip for normalized matching
COMMON_LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "pvt", "private", "co", "company", "enterprises", "services", "group",
    "holdings", "sa", "sarl", "gmbh", "center", "centre"
}

# Common address abbreviations
ADDRESS_ABBRS = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "hwy": "highway", "pkwy": "parkway", "apt": "apartment", "ste": "suite"
}


def clean_text(text: str) -> str:
    """Normalize text: lowercase, remove special characters, clean extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def clean_name(name: str) -> str:
    """Clean business name, normalizing domain names and stripping legal suffixes."""
    if not isinstance(name, str):
        return ""
    name = name.lower()
    # Normalize domain-like names (e.g. maurewilliamscolombier.com -> maure williams colombier)
    name = re.sub(r"\.(com|org|net|in|us|fr|co|io)\b", " ", name)
    name = re.sub(r"[^\w\s]", " ", name)
    tokens = [t for t in name.split() if t not in COMMON_LEGAL_SUFFIXES]
    return " ".join(tokens)


def clean_address(addr: str) -> str:
    """Clean address, standardizing abbreviations."""
    if not isinstance(addr, str):
        return ""
    addr = clean_text(addr)
    tokens = [ADDRESS_ABBRS.get(t, t) for t in addr.split()]
    return " ".join(tokens)


def extract_blocking_keys(name: str, addr: str) -> Set[str]:
    """
    Extracts high-recall candidate blocking keys from name and address.
    Generates:
    - Name prefix & significant words
    - Address house/unit numbers + road/city tokens
    """
    keys = set()
    
    # 1. Name keys
    if name:
        name_tokens = name.split()
        if len(name_tokens) >= 1:
            # First token (if length >= 3)
            if len(name_tokens[0]) >= 3:
                keys.add(f"N1:{name_tokens[0]}")
            # First two tokens combined
            if len(name_tokens) >= 2:
                keys.add(f"N2:{name_tokens[0]}_{name_tokens[1]}")
        # Compact name (first 8 chars)
        compact = "".join(name_tokens)
        if len(compact) >= 5:
            keys.add(f"NC:{compact[:7]}")

    # 2. Address keys (street number + street name / city)
    if addr:
        addr_tokens = addr.split()
        nums = [t for t in addr_tokens if t.isdigit() or (any(c.isdigit() for c in t) and len(t) <= 6)]
        words = [t for t in addr_tokens if not any(c.isdigit() for c in t) and len(t) >= 4]
        if nums and words:
            # Street number + primary word
            keys.add(f"A:{nums[0]}_{words[0]}")
            if len(words) > 1:
                keys.add(f"A:{nums[0]}_{words[1]}")
        elif nums:
            keys.add(f"ANUM:{nums[0]}")

    return keys


class EntityIndex:
    """Inverted index for fast candidate retrieval partitioned by country."""

    def __init__(self):
        # country -> key -> list of entity_ids
        self.index = defaultdict(lambda: defaultdict(list))
        # entity_id -> (clean_name, clean_addr, country)
        self.records = {}

    def add_records(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            eid = row["entity_id"]
            c_name = clean_name(row["business_name"])
            c_addr = clean_address(row["business_address"])
            country = str(row["country"]).strip().upper()
            
            self.records[eid] = (c_name, c_addr, country)
            keys = extract_blocking_keys(c_name, c_addr)
            for k in keys:
                self.index[country][k].append(eid)

    def retrieve_candidates(self, c_name: str, c_addr: str, country: str, max_candidates: int = 30) -> Set[str]:
        keys = extract_blocking_keys(c_name, c_addr)
        country_index = self.index.get(country, {})
        counts = defaultdict(int)
        
        for k in keys:
            for cand_id in country_index.get(k, []):
                counts[cand_id] += 1
                
        if not counts:
            return set()
            
        # Prioritize candidates matching multiple blocking keys
        sorted_cands = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return {cand_id for cand_id, _ in sorted_cands[:max_candidates]}


def score_pair(name1: str, addr1: str, name2: str, addr2: str) -> float:
    """
    Computes a composite similarity score between two business entities.
    Precision-optimized for F_0.5.
    """
    # Name similarity
    if name1 and name2:
        name_ratio = fuzz.ratio(name1, name2)
        token_sort = fuzz.token_sort_ratio(name1, name2)
        name_score = max(name_ratio, token_sort)
    else:
        name_score = 0.0

    # Address similarity
    if addr1 and addr2:
        addr_ratio = fuzz.token_set_ratio(addr1, addr2)
        addr_partial = fuzz.partial_ratio(addr1, addr2)
        addr_score = max(addr_ratio, addr_partial)
    else:
        addr_score = None  # Missing address

    # Composite logic:
    # 1. Both name and address present
    if addr_score is not None:
        if name_score >= 88 and addr_score >= 70:
            return (0.6 * name_score + 0.4 * addr_score) / 100.0
        elif name_score >= 95:
            return (0.8 * name_score + 0.2 * addr_score) / 100.0
        elif addr_score >= 90 and name_score >= 60:
            return (0.4 * name_score + 0.6 * addr_score) / 100.0
        else:
            return (0.5 * name_score + 0.5 * addr_score) / 100.0
    else:
        # Address is missing in one or both records: require very high name confidence
        if name_score >= 90:
            return (name_score * 0.95) / 100.0
        return (name_score * 0.7) / 100.0


def run_pipeline(
    mode: str = "validate",
    data_dir: Optional[str] = None,
    sample_size: Optional[int] = None,
    match_threshold: float = 0.82,
    output_dir: str = "output"
):
    print("=" * 65)
    print(f"  RUNNING ENTITY RESOLUTION PIPELINE | Mode: {mode.upper()}")
    print("=" * 65)
    os.makedirs(output_dir, exist_ok=True)

    if data_dir is None:
        if mode == "validate":
            data_dir = "Dataset/val_sample" if os.path.exists("Dataset/val_sample/val_source1.tsv") else "Dataset/train"
        else:
            data_dir = "Dataset/test"

    is_val_sample = "val_sample" in data_dir
    prefix = "val" if is_val_sample else ("train" if mode == "validate" else "test")
    
    s1_file = os.path.join(data_dir, f"{prefix}_source1.tsv")
    s2_file = os.path.join(data_dir, f"{prefix}_source2.tsv")
    s3_file = os.path.join(data_dir, f"{prefix}_source3.tsv")

    print(f"Loading Source 1 records from {s1_file}...")
    s1_df = pd.read_csv(s1_file, sep="\t", nrows=sample_size)
    print(f"Loaded {len(s1_df):,} Source 1 records.")

    print(f"Building candidate index from Source 2 ({s2_file}) and Source 3 ({s3_file})...")
    t0 = time.time()
    index = EntityIndex()
    
    s2_df = pd.read_csv(s2_file, sep="\t")
    index.add_records(s2_df)
    del s2_df
    
    s3_df = pd.read_csv(s3_file, sep="\t")
    index.add_records(s3_df)
    del s3_df
    print(f"Indexed {len(index.records):,} S2/S3 target records in {time.time() - t0:.1f}s.")

    print("Running candidate generation (blocking) and matching...")
    t0 = time.time()
    
    candidate_dict = {}
    matched_dict = {}

    for _, row in s1_df.iterrows():
        s1_id = row["entity_id"]
        c_name = clean_name(row["business_name"])
        c_addr = clean_address(row["business_address"])
        country = str(row["country"]).strip().upper()

        # Step 1: Blocking candidates
        cands = index.retrieve_candidates(c_name, c_addr, country, max_candidates=35)
        candidate_dict[s1_id] = cands

        # Step 2: Scoring pairs
        matches = []
        for cand_id in cands:
            cand_name, cand_addr, _ = index.records[cand_id]
            score = score_pair(c_name, c_addr, cand_name, cand_addr)
            if score >= match_threshold:
                matches.append((cand_id, score))

        # Precision filter: keep matches meeting the high F_0.5 bar
        matched_dict[s1_id] = {m[0] for m in matches}

    print(f"Processed {len(s1_df):,} queries in {time.time() - t0:.1f}s.")

    # Write output files
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    print(f"Writing outputs to {output_dir}...")
    with open(matching_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_df["entity_id"]:
            m_list = ",".join(sorted(matched_dict.get(s1_id, set())))
            f.write(f"{s1_id}\t{m_list}\n")

    with open(candidate_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_df["entity_id"]:
            c_list = ",".join(sorted(candidate_dict.get(s1_id, set())))
            f.write(f"{s1_id}\t{c_list}\n")

    print(f"Successfully saved:")
    print(f"  - {matching_path}")
    print(f"  - {candidate_path}")

    # If in validate mode, evaluate against Ground Truth immediately
    if mode == "validate":
        gt_file = os.path.join(data_dir, f"{prefix}_ground_truth.tsv")
        print("\nEvaluating against ground truth...")
        gt_df = pd.read_csv(gt_file, sep="\t", nrows=len(s1_df))
        gt_dict = {}
        for _, r in gt_df.iterrows():
            m = str(r["matched_entity_ids"])
            if m and m != "nan":
                gt_dict[r["source1_entity_id"]] = {x.strip() for x in m.split(",") if x.strip()}
            else:
                gt_dict[r["source1_entity_id"]] = set()

        metrics = evaluate_resolution(gt_dict, matched_dict, candidate_dict)
        print("\n" + "=" * 60)
        print("          VALIDATION REPORT (F_0.5 SCORE)")
        print("=" * 60)
        print(f"Evaluated Entities       : {metrics['total_entities']:,}")
        print(f"Singletons (True Empty)  : {metrics['singleton_count']:,} ({metrics['singleton_count']/metrics['total_entities']*100:.1f}%)")
        print(f"Singleton Accuracy       : {metrics['singleton_accuracy']*100:.2f}%")
        print(f"Candidate Recall Ceiling : {metrics['candidate_recall_ceiling']*100:.2f}%")
        print(f"Avg Candidates / Entity  : {metrics['avg_candidates_per_entity']:.1f}")
        print("-" * 60)
        print(f"Macro Precision          : {metrics['macro_precision']*100:.2f}%")
        print(f"Macro Recall             : {metrics['macro_recall']*100:.2f}%")
        print(f"Non-Singleton Macro F0.5 : {metrics['non_singleton_macro_f05']:.4f}")
        print("=" * 60)
        print(f"===> OFFICIAL MACRO F_0.5 SCORE : {metrics['macro_f05']:.5f} <===")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Entity Resolution Pipeline.")
    parser.add_argument("--mode", choices=["validate", "test"], default="validate")
    parser.add_argument("--data-dir", default=None, help="Custom data directory")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of S1 samples (None for all)")
    parser.add_argument("--threshold", type=float, default=0.80, help="F_0.5 decision threshold")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    args = parser.parse_args()

    run_pipeline(
        mode=args.mode,
        data_dir=args.data_dir,
        sample_size=args.sample_size,
        match_threshold=args.threshold,
        output_dir=args.output_dir
    )
