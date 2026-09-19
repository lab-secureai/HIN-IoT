"""Shared command-line helpers for the scripts in ``scripts/``."""

import argparse
from typing import List

from .config import ABLATION_MODES, ARCHITECTURES, DEFAULT_SEEDS


def parse_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_seeds(value: str) -> List[int]:
    return [int(v) for v in parse_list(value)]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seeds", type=parse_seeds, default=list(DEFAULT_SEEDS),
                        help="Comma-separated random seeds (default: the ten seeds used in the paper).")
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda or cpu (default: auto).")
    parser.add_argument("--id-col", default=None,
                        help="Sample identifier column written to prediction files (default: sha256 if present).")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Also write per-sample predictions (predictions.csv.gz).")


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--smoke", action="store_true",
                        help="Very short training for a pipeline check (results are not meaningful).")
    det = parser.add_mutually_exclusive_group()
    det.add_argument("--deterministic", dest="deterministic", action="store_true", default=None,
                     help="Force cudnn.deterministic=True (overrides the protocol default).")
    det.add_argument("--nondeterministic", dest="deterministic", action="store_false",
                     help="Force cudnn.deterministic=False (overrides the protocol default).")


def add_arch_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--architectures", type=parse_list, default=list(ARCHITECTURES),
                        help="Architectures for the loao protocol (default: MIPS,ARM,PowerPC,x64,x86).")


def add_modes_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--modes", type=parse_list, default=list(ABLATION_MODES),
                        help="Ablation modes (default: Full_Graph,No_API_Node,No_Arch_Node,Static_Opcode_Only).")
