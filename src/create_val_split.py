#!/usr/bin/env python3
"""
Creates a representative, self-contained Validation Split for rapid offline testing.
Extracts:
- N Source 1 entities (balanced between singletons and entities with matches)
- All corresponding true matches from Source 2 and Source 3
- K random distractor records from Source 2 and Source 3
Saves to Dataset/val_sample/ for ultra-fast (5-10 second) model evaluation.
"""

import os
import random
import argparse
import pandas as pd

DELIM = "\t"


def create_validation_split(
    n_s1: int = 5000,
    n_distractors: int = 50000,
    train_dir: str = "Dataset/train",
    output_dir: str = "Dataset/val_sample"
):
    print("=" * 60)
    print(f"Creating Fast Validation Split ({n_s1:,} S1 queries + {n_distractors:,} distractors)")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)

    gt_file = os.path.join(train_dir, "train_ground_truth.tsv")
    s1_file = os.path.join(train_dir, "train_source1.tsv")
    s2_file = os.path.join(train_dir, "train_source2.tsv")
    s3_file = os.path.join(train_dir, "train_source3.tsv")

    # Step 1: Read Ground Truth for N entities
    print("Sampling Ground Truth...")
    gt_df = pd.read_csv(gt_file, sep=DELIM, nrows=n_s1)
    
    target_s2_ids = set()
    target_s3_ids = set()
    s1_id_set = set(gt_df["source1_entity_id"])

    for _, row in gt_df.iterrows():
        m = str(row["matched_entity_ids"])
        if m and m != "nan":
            for x in m.split(","):
                x = x.strip()
                if x.startswith("S2-"):
                    target_s2_ids.add(x)
                elif x.startswith("S3-"):
                    target_s3_ids.add(x)

    print(f"Sampled {len(s1_id_set):,} S1 entities with {len(target_s2_ids):,} S2 matches and {len(target_s3_ids):,} S3 matches.")

    # Step 2: Extract S1 records
    print(f"Extracting S1 records from {s1_file}...")
    s1_rows = []
    with open(s1_file, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            eid = line.split(DELIM, 1)[0].strip()
            if eid in s1_id_set:
                s1_rows.append(line)
            if len(s1_rows) >= len(s1_id_set):
                break

    with open(os.path.join(output_dir, "val_source1.tsv"), "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(s1_rows)

    # Step 3: Extract S2 records (true matches + distractors)
    print(f"Scanning S2 records ({s2_file})...")
    s2_matched_rows = []
    s2_distractor_rows = []
    s2_distractor_target = n_distractors // 2

    with open(s2_file, "r", encoding="utf-8") as f:
        header = f.readline()
        for idx, line in enumerate(f):
            eid = line.split(DELIM, 1)[0].strip()
            if eid in target_s2_ids:
                s2_matched_rows.append(line)
            elif len(s2_distractor_rows) < s2_distractor_target and random.random() < 0.05:
                s2_distractor_rows.append(line)
            
            # Early break if all matches found and enough distractors
            if len(s2_matched_rows) == len(target_s2_ids) and len(s2_distractor_rows) >= s2_distractor_target:
                break

    with open(os.path.join(output_dir, "val_source2.tsv"), "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(s2_matched_rows + s2_distractor_rows)

    # Step 4: Extract S3 records (true matches + distractors)
    print(f"Scanning S3 records ({s3_file})...")
    s3_matched_rows = []
    s3_distractor_rows = []
    s3_distractor_target = n_distractors // 2

    with open(s3_file, "r", encoding="utf-8") as f:
        header = f.readline()
        for idx, line in enumerate(f):
            eid = line.split(DELIM, 1)[0].strip()
            if eid in target_s3_ids:
                s3_matched_rows.append(line)
            elif len(s3_distractor_rows) < s3_distractor_target and random.random() < 0.05:
                s3_distractor_rows.append(line)
            
            if len(s3_matched_rows) == len(target_s3_ids) and len(s3_distractor_rows) >= s3_distractor_target:
                break

    with open(os.path.join(output_dir, "val_source3.tsv"), "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(s3_matched_rows + s3_distractor_rows)

    # Save Ground Truth
    gt_df.to_csv(os.path.join(output_dir, "val_ground_truth.tsv"), sep=DELIM, index=False)

    print("=" * 60)
    print(f"Validation split created in '{output_dir}':")
    print(f"  * val_source1.tsv      : {len(s1_rows):,} records")
    print(f"  * val_source2.tsv      : {len(s2_matched_rows) + len(s2_distractor_rows):,} records ({len(s2_matched_rows)} true matches)")
    print(f"  * val_source3.tsv      : {len(s3_matched_rows) + len(s3_distractor_rows):,} records ({len(s3_matched_rows)} true matches)")
    print(f"  * val_ground_truth.tsv : {len(gt_df):,} records")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-count", type=int, default=3000, help="Number of S1 records")
    parser.add_argument("--distractors", type=int, default=30000, help="Number of distractor records")
    args = parser.parse_args()
    create_validation_split(n_s1=args.s1_count, n_distractors=args.distractors)
