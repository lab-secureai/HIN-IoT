#!/usr/bin/env python
"""Train and test all models under one protocol.

  holdout : ten stratified 80/20 splits of the primary corpus        (paper Table 3)
  loao    : leave-one-architecture-out                               (paper Table 4)
  family  : malware-family-disjoint grouped splits                   (paper Table 6, primary columns)

Example:
  python scripts/train.py --protocol holdout --data data/primary.csv --work-dir runs/holdout
"""
import argparse

import _bootstrap  # noqa: F401

from hin_iot.cli import add_arch_arg, add_common_args, add_training_args
from hin_iot.config import get_protocol
from hin_iot.experiments import run_training
from hin_iot.utils import get_device, write_run_info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", required=True, choices=["holdout", "loao", "family"])
    parser.add_argument("--data", required=True, help="Primary corpus CSV (see docs/DATA.md).")
    parser.add_argument("--work-dir", required=True, help="Directory for per-seed artifacts and result tables.")
    add_common_args(parser)
    add_training_args(parser)
    add_arch_arg(parser)
    args = parser.parse_args()

    write_run_info(args.work_dir, vars(args), get_protocol(args.protocol, smoke=args.smoke))
    summary = run_training(args.protocol, args.data, args.work_dir, args.seeds, get_device(args.device),
                           architectures=args.architectures, smoke=args.smoke,
                           deterministic=args.deterministic, save_predictions=args.save_predictions,
                           id_col=args.id_col)
    cols = [c for c in ("held_out_arch", "split", "model", "accuracy", "binary_f1", "mcc", "pr_auc")
            if c in summary.columns]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
