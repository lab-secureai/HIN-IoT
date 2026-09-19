#!/usr/bin/env python
"""Write the SHA-256 manifests and the train/test partition of every run.

The partitions are regenerated with the same split functions and seeds used for
training, so they can be published without the binaries or the feature files.

Outputs (under --out-dir):
  manifests/primary_manifest.csv          sample id, label, architecture (and family if available)
  manifests/temporal_manifest.csv         same for the evaluation corpus (if --temporal is given)
  manifests/corpus_summary.md             class x architecture counts, duplicates, cross-corpus overlap
  partitions/holdout/seed_<s>.csv.gz      split of every sample (train / test)
  partitions/ablation/seed_<s>.csv.gz
  partitions/family/seed_<s>.csv.gz
  partitions/loao/<arch>/seed_<s>.csv.gz  train / id_test / ood_test

Example:
  python scripts/export_partitions.py --primary data/primary.csv --temporal data/temporal_2024.csv --out-dir data
"""
import argparse
import os

import pandas as pd

import _bootstrap  # noqa: F401

from hin_iot.cli import parse_list, parse_seeds
from hin_iot.config import ARCHITECTURES, DEFAULT_SEEDS
from hin_iot.data import (describe_corpus, family_disjoint_split, id_column, loao_split, read_corpus,
                          stratified_holdout_split, to_markdown)


def _manifest(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    cols = [id_col, "label", "arch"] + (["malware_family"] if "malware_family" in df.columns else [])
    return df[cols].rename(columns={id_col: "sha256"})


def _write_split(path: str, parts: dict, id_col: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = [pd.DataFrame({"sha256": df[id_col].values, "split": name}) for name, df in parts.items()]
    pd.concat(frames, ignore_index=True).to_csv(path, index=False, compression="gzip")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--temporal", default=None)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--id-col", default="sha256")
    parser.add_argument("--seeds", type=parse_seeds, default=list(DEFAULT_SEEDS))
    parser.add_argument("--architectures", type=parse_list, default=list(ARCHITECTURES))
    args = parser.parse_args()

    manifest_dir = os.path.join(args.out_dir, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    report = ["# Corpus summary", ""]

    primary = read_corpus(args.primary, standardize=True)
    primary_raw_arch = read_corpus(args.primary, standardize=False)
    id_col = id_column(primary, args.id_col)
    _manifest(primary, id_col).to_csv(os.path.join(manifest_dir, "primary_manifest.csv"), index=False)
    report += ["## Primary corpus", "", to_markdown(describe_corpus(primary)), "",
               f"Duplicate identifiers: {int(primary[id_col].duplicated().sum())}", ""]

    if args.temporal:
        temporal = read_corpus(args.temporal, standardize=True)
        t_id = id_column(temporal, args.id_col)
        _manifest(temporal, t_id).to_csv(os.path.join(manifest_dir, "temporal_manifest.csv"), index=False)
        overlap = set(primary[id_col].astype(str).str.lower()) & set(temporal[t_id].astype(str).str.lower())
        report += ["## Evaluation corpus", "", to_markdown(describe_corpus(temporal)), "",
                   f"Duplicate identifiers: {int(temporal[t_id].duplicated().sum())}",
                   f"Identifiers shared with the primary corpus: {len(overlap)}", ""]

    part_dir = os.path.join(args.out_dir, "partitions")
    for seed in args.seeds:
        tr, te = stratified_holdout_split(primary, seed)
        _write_split(os.path.join(part_dir, "holdout", f"seed_{seed}.csv.gz"), {"train": tr, "test": te}, id_col)
        tr, te = stratified_holdout_split(primary_raw_arch, seed)
        _write_split(os.path.join(part_dir, "ablation", f"seed_{seed}.csv.gz"), {"train": tr, "test": te}, id_col)
        tr, te = family_disjoint_split(primary, seed)
        _write_split(os.path.join(part_dir, "family", f"seed_{seed}.csv.gz"), {"train": tr, "test": te}, id_col)
        for arch in args.architectures:
            tr, id_te, ood = loao_split(primary, arch, seed)
            _write_split(os.path.join(part_dir, "loao", arch, f"seed_{seed}.csv.gz"),
                         {"train": tr, "id_test": id_te, "ood_test": ood}, id_col)

    with open(os.path.join(manifest_dir, "corpus_summary.md"), "w") as fh:
        fh.write("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"Manifests and partitions written under {args.out_dir}")


if __name__ == "__main__":
    main()
