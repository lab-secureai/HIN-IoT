#!/usr/bin/env python
"""Apply trained models to an independently collected corpus without fine-tuning.

Uses the artifacts written by scripts/train.py or scripts/ablation.py. Nothing is
refitted on the new corpus: scalers, TF-IDF vocabulary, tokenizer, API vocabulary
and architecture encoder all come from the training partition of each seed, and
HIN models see one isolated ego graph per file.

  --protocol holdout  -> paper Table 5
  --protocol family   -> paper Table 6, 2024 columns
  --protocol ablation -> paper Table 7, 2024 columns
  --protocol loao     -> each held-out architecture of the new corpus, using the model trained without it

Example:
  python scripts/evaluate.py --protocol holdout --data data/temporal_2024.csv \
      --work-dir runs/holdout --out-dir runs/holdout_2024
"""
import argparse

import _bootstrap  # noqa: F401

from hin_iot.cli import add_arch_arg, add_common_args, add_modes_arg
from hin_iot.config import get_protocol
from hin_iot.experiments import evaluate_new_corpus
from hin_iot.utils import get_device, write_run_info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", required=True, choices=["holdout", "loao", "family", "ablation"])
    parser.add_argument("--data", required=True, help="Labelled evaluation corpus CSV (e.g. the 2024 corpus).")
    parser.add_argument("--work-dir", required=True, help="Directory produced by train.py / ablation.py.")
    parser.add_argument("--out-dir", required=True, help="Where to write the evaluation tables.")
    add_common_args(parser)
    add_arch_arg(parser)
    add_modes_arg(parser)
    args = parser.parse_args()

    write_run_info(args.out_dir, vars(args), get_protocol(args.protocol))
    summary = evaluate_new_corpus(args.protocol, args.data, args.work_dir, args.out_dir, args.seeds,
                                  get_device(args.device), architectures=args.architectures, modes=args.modes,
                                  save_predictions=args.save_predictions, id_col=args.id_col)
    cols = [c for c in ("held_out_arch", "ablation_mode", "model", "accuracy", "binary_f1", "macro_f1",
                        "mcc", "pr_auc") if c in summary.columns]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
