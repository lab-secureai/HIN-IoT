#!/usr/bin/env python
"""Paired significance tests between two models.

wilcoxon : two-sided Wilcoxon signed-rank test on per-seed metrics from results_raw.csv
           (the test reported in the paper for RH-SAGE vs. end-to-end fusion).
mcnemar  : exact McNemar test per seed on per-sample predictions (predictions.csv.gz,
           written with --save-predictions). It accounts for the fact that all seeds
           share the same evaluation corpus.

Examples:
  python scripts/stats_tests.py wilcoxon --results runs/holdout_2024/results_raw.csv \
      --model-a RH-SAGE --model-b "End-to-End Fusion" --metric accuracy
  python scripts/stats_tests.py mcnemar --predictions runs/holdout_2024/predictions.csv.gz \
      --model-a rh_sage --model-b fusion
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats


def _filter(df: pd.DataFrame, where):
    for cond in where or []:
        key, value = cond.split("=", 1)
        df = df[df[key].astype(str) == value]
    return df


def wilcoxon(args):
    df = _filter(pd.read_csv(args.results), args.where)
    keys = [c for c in ("held_out_arch", "ablation_mode", "split", "seed") if c in df.columns]
    a = df[df["model"] == args.model_a].set_index(keys)[args.metric]
    b = df[df["model"] == args.model_b].set_index(keys)[args.metric]
    paired = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if paired.empty:
        raise SystemExit("No paired runs found; check --model-a/--model-b/--where.")
    diff = paired["a"] - paired["b"]
    exact_ok = len(diff) <= 50 and (diff != 0).all() and diff.abs().nunique() == len(diff)
    method = "exact" if exact_ok else "approx"
    res = stats.wilcoxon(paired["a"], paired["b"], alternative="two-sided", method=method)
    print(f"Wilcoxon signed-rank ({method}), metric={args.metric}, n={len(diff)}")
    print(f"  {args.model_a}: mean={paired['a'].mean():.4f}  {args.model_b}: mean={paired['b'].mean():.4f}")
    print(f"  mean diff={diff.mean():.4f}  median diff={diff.median():.4f}  "
          f"wins/ties/losses={int((diff > 0).sum())}/{int((diff == 0).sum())}/{int((diff < 0).sum())}")
    print(f"  statistic={res.statistic:.4f}  p={res.pvalue:.5f}")


def mcnemar(args):
    df = _filter(pd.read_csv(args.predictions), args.where)
    keys = [c for c in ("held_out_arch", "ablation_mode", "split", "seed") if c in df.columns]
    rows = []
    for group, grp in df.groupby(keys):
        a = grp[grp["model_key"] == args.model_a].sort_values("sample_id")
        b = grp[grp["model_key"] == args.model_b].sort_values("sample_id")
        if a.empty or b.empty:
            continue
        if not np.array_equal(a["sample_id"].values, b["sample_id"].values):
            raise SystemExit("Prediction files are not aligned by sample_id.")
        ca = a["pred"].values == a["label"].values
        cb = b["pred"].values == b["label"].values
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        rows.append({**dict(zip(keys, group if isinstance(group, tuple) else (group,))),
                     f"only_{args.model_a}_correct": n01, f"only_{args.model_b}_correct": n10, "p_exact": p})
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="test", required=True)
    w = sub.add_parser("wilcoxon")
    w.add_argument("--results", required=True)
    w.add_argument("--model-a", required=True, help="Paper model name, e.g. RH-SAGE")
    w.add_argument("--model-b", required=True, help='Paper model name, e.g. "End-to-End Fusion"')
    w.add_argument("--metric", default="accuracy")
    w.add_argument("--where", action="append", help="Filter, e.g. --where held_out_arch=MIPS --where split=OOD")
    m = sub.add_parser("mcnemar")
    m.add_argument("--predictions", required=True)
    m.add_argument("--model-a", required=True, help="Model key, e.g. rh_sage")
    m.add_argument("--model-b", required=True, help="Model key, e.g. fusion")
    m.add_argument("--where", action="append")
    args = parser.parse_args()
    wilcoxon(args) if args.test == "wilcoxon" else mcnemar(args)


if __name__ == "__main__":
    main()
