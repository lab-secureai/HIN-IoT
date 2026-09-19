#!/usr/bin/env python
"""Graph-component ablation for every heterogeneous encoder (paper Table 7 and computational cost).

Example:
  python scripts/ablation.py --data data/primary.csv --work-dir runs/ablation
"""
import argparse

import _bootstrap  # noqa: F401

from hin_iot.cli import add_common_args, add_modes_arg, add_training_args
from hin_iot.config import get_protocol
from hin_iot.experiments import run_ablation
from hin_iot.utils import get_device, write_run_info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Primary corpus CSV (see docs/DATA.md).")
    parser.add_argument("--work-dir", required=True)
    add_common_args(parser)
    add_training_args(parser)
    add_modes_arg(parser)
    args = parser.parse_args()

    write_run_info(args.work_dir, vars(args), get_protocol("ablation", smoke=args.smoke))
    summary = run_ablation(args.data, args.work_dir, args.seeds, get_device(args.device), modes=args.modes,
                           smoke=args.smoke, deterministic=args.deterministic,
                           save_predictions=args.save_predictions, id_col=args.id_col)
    cols = ["ablation_mode", "model", "accuracy", "macro_f1", "mcc", "pr_auc",
            "train_ms_per_epoch", "infer_ms_per_sample", "peak_gpu_mb"]
    print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
