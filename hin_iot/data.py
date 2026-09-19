"""Corpus loading, field parsing and train/test partitioning.

The expected CSV schema is documented in ``docs/DATA.md``. This module depends
only on NumPy, pandas and scikit-learn.
"""

import ast
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

BENIGN_FAMILY = "Benign"


def standardize_arch(value) -> str:
    """Map a raw ELF machine string to ARM, MIPS, PowerPC, x64, x86 or Other."""
    arch = str(value).strip().lower()
    if "arm" in arch or "aarch64" in arch:
        return "ARM"
    if "mips" in arch:
        return "MIPS"
    if "powerpc" in arch or "ppc" in arch:
        return "PowerPC"
    if "64" in arch or "amd64" in arch or "x86_64" in arch:
        return "x64"
    if "86" in arch or "i386" in arch or "i686" in arch:
        return "x86"
    return "Other"


def parse_text_sequence(value) -> str:
    """Turn an opcode/API field into a whitespace-separated token string.

    A dictionary string such as ``"{'mov': 3, 'bl': 1}"`` is expanded into
    repeated tokens, with each token repeated at most 20 times.
    """
    if isinstance(value, str):
        if value.startswith("{"):
            try:
                counts = ast.literal_eval(value)
                return " ".join([f"{k} " * min(int(v), 20) for k, v in counts.items()])
            except Exception:
                pass
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(map(str, value))
    return ""


def parse_api_list(value) -> list:
    """Parse the API trace field into a list of API names (order is irrelevant for the HIN)."""
    try:
        if isinstance(value, str) and value.startswith("["):
            return ast.literal_eval(value)
        return [] if pd.isna(value) else str(value).split()
    except Exception:
        return []


def api_column(df: pd.DataFrame) -> str:
    return "api_sequence" if "api_sequence" in df.columns else "api_list"


def opcode_column(df: pd.DataFrame) -> str:
    return "opcode_counts" if "opcode_counts" in df.columns else "opcode_sequence"


def api_lists(df: pd.DataFrame) -> pd.Series:
    return df.get(api_column(df), pd.Series([[]] * len(df))).apply(parse_api_list)


def read_corpus(path: str, standardize: bool = True) -> pd.DataFrame:
    """Read a corpus CSV. ``standardize`` maps the ``arch`` column to five canonical names."""
    df = pd.read_csv(path)
    missing = [c for c in ("label", "arch") if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    if standardize:
        df["arch"] = df["arch"].apply(standardize_arch)
    return df


def _stratify_key(df: pd.DataFrame) -> pd.Series:
    return df["label"].astype(str) + "_" + df["arch"].astype(str)


def stratified_holdout_split(df: pd.DataFrame, seed: int, test_size: float = 0.2
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 split stratified by the joint (label, architecture) key.

    Strata with a single sample cannot be stratified and are dropped, as in the
    experiments reported in the paper.
    """
    key = _stratify_key(df)
    counts = key.value_counts()
    keep = key.isin(counts[counts > 1].index)
    df_valid = df[keep].copy()
    df_train, df_test = train_test_split(
        df_valid, test_size=test_size, stratify=key[keep], random_state=seed
    )
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def loao_split(df: pd.DataFrame, held_out_arch: str, seed: int, test_size: float = 0.2
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leave-one-architecture-out split.

    Returns (train, in-distribution test, out-of-distribution test). The held-out
    architecture never appears in training; the remaining architectures are
    split 80/20 with :func:`stratified_holdout_split`.
    """
    df_ood = df[df["arch"] == held_out_arch].reset_index(drop=True)
    df_id = df[df["arch"] != held_out_arch].reset_index(drop=True)
    df_train, df_id_test = stratified_holdout_split(df_id, seed, test_size)
    return df_train, df_id_test, df_ood


def family_groups(df: pd.DataFrame) -> np.ndarray:
    """Group labels for family-disjoint splitting.

    Malware samples are grouped by ``malware_family``; every benign sample forms
    its own group.
    """
    families = df["malware_family"] if "malware_family" in df.columns else pd.Series(BENIGN_FAMILY, index=df.index)
    families = families.fillna(BENIGN_FAMILY)
    return np.array([
        f"benign_{idx}" if label == 0 else str(family)
        for idx, label, family in zip(df.index, df["label"], families)
    ])


def family_disjoint_split(df: pd.DataFrame, seed: int, n_splits: int = 5
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Family-disjoint split: the first fold of ``StratifiedGroupKFold(n_splits, shuffle=True)``."""
    df_clean = df.copy()
    df_clean["arch"] = df_clean["arch"].apply(standardize_arch)
    if "malware_family" not in df_clean.columns:
        df_clean["malware_family"] = BENIGN_FAMILY
    df_clean["malware_family"] = df_clean["malware_family"].fillna(BENIGN_FAMILY)
    groups = family_groups(df_clean)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df_clean, df_clean["label"], groups=groups))
    return (df_clean.iloc[train_idx].reset_index(drop=True),
            df_clean.iloc[test_idx].reset_index(drop=True))


def describe_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Class x architecture counts (Table 1 / Table 2 of the paper)."""
    return pd.crosstab(df["label"].map({0: "Benign", 1: "Malware"}), df["arch"],
                       margins=True, margins_name="Total")


def to_markdown(table: pd.DataFrame) -> str:
    try:
        return table.to_markdown()
    except ImportError:  # tabulate not installed
        return "```\n" + table.to_string() + "\n```"


def ensure_binary_labels(y: np.ndarray, where: str) -> None:
    values = set(np.unique(y).tolist())
    if not values <= {0, 1}:
        raise ValueError(f"Labels in {where} must be 0 (benign) or 1 (malware); found {sorted(values)}")


def id_column(df: pd.DataFrame, preferred: str = "sha256") -> str:
    candidates: List[str] = [preferred, "sha256", "SHA256", "hash", "file_hash", "md5", "file_name", "filename"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "No sample identifier column found. Add a 'sha256' column to the corpus CSV "
        "(see docs/DATA.md) or pass --id-col."
    )
