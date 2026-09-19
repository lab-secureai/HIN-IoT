"""File-node features: static numerical, opcode TF-IDF, numerical runtime, API token sequence.

Every data-dependent transformation (z-score scalers, opcode TF-IDF vocabulary,
API tokenizer) is fitted on the training partition only and then applied
unchanged to test partitions and to the independent 2024 corpus.

This module does not import PyTorch.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .data import api_column, opcode_column, parse_text_sequence

# s_i in the paper: mean/max/min entropy, file size, total number of opcodes.
CORE_STATIC_COLUMNS: List[str] = ["mean_entropy", "max_entropy", "min_entropy", "file_size_mb", "total_opcodes"]
N_ENTROPY_COLUMNS = 3

# d_i in the paper: numerical runtime counters (the ones present in the training data are used).
DYNAMIC_COUNTER_CANDIDATES: List[str] = [
    "socket_calls", "connect_calls", "send_calls", "fork_calls",
    "exec_calls", "execve_calls", "sys_calls", "file_ops",
    "net_ops", "proc_ops", "total_calls", "unique_calls",
]


class PyTorchApiTokenizer:
    """Word-level API tokenizer for the Bi-GRU and fusion baselines.

    Index 0 is padding and index 1 is the out-of-vocabulary token. The class name
    is kept identical to the original experiment code so that saved tokenizers
    can be unpickled.
    """

    def __init__(self, max_vocab: int = 5000):
        self.max_vocab = max_vocab
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    @property
    def word_index(self):
        return self.word2idx

    def fit_on_texts(self, texts: Iterable) -> None:
        counts = Counter()
        for text in texts:
            tokens = text.split() if isinstance(text, str) else text
            counts.update(tokens)
        for token, _ in counts.most_common(self.max_vocab - 2):
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def texts_to_sequences(self, texts: Iterable, max_len: int = 200) -> np.ndarray:
        seqs = []
        for text in texts:
            tokens = text.split() if isinstance(text, str) else text
            ids = [self.word2idx.get(t, 1) for t in tokens[:max_len]]
            if len(ids) < max_len:
                ids = ids + [0] * (max_len - len(ids))
            seqs.append(ids)
        return np.array(seqs, dtype=np.int64)


def _with_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if not missing:
        return df
    df = df.copy()
    for c in missing:
        df[c] = 0.0
    return df


def _numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    # Non-numeric or missing values become 0 before scaling.
    return df[columns].apply(pd.to_numeric, errors="coerce").fillna(0)


def _text(df: pd.DataFrame, column: str) -> pd.Series:
    return df.get(column, pd.Series([""] * len(df))).fillna("").apply(parse_text_sequence)


@dataclass
class FeatureSet:
    core: np.ndarray                 # 5 standardized static features
    opcode: np.ndarray               # opcode TF-IDF
    dynamic: np.ndarray              # standardized runtime counters
    api_seq: Optional[np.ndarray]    # padded API token ids (baselines only)

    @property
    def static_full(self) -> np.ndarray:
        """Static branch of the fusion baseline: [s_i || o_i]."""
        return np.hstack([self.core, self.opcode])

    @property
    def hin_file(self) -> np.ndarray:
        """File-node vector of the HIN, Eq. (1): [s_i || o_i || d_i]."""
        return np.hstack([self.core, self.opcode, self.dynamic])


class FeaturePipeline:
    """Fit on training data, then transform any partition or corpus."""

    def __init__(self, opcode_max_features: int = 100, api_vocab_size: int = 5000, api_max_len: int = 200):
        self.opcode_max_features = opcode_max_features
        self.api_vocab_size = api_vocab_size
        self.api_max_len = api_max_len
        self.core_cols: List[str] = list(CORE_STATIC_COLUMNS)
        self.dyn_cols: List[str] = []
        self.scaler_core: Optional[StandardScaler] = None
        self.tfidf_op: Optional[TfidfVectorizer] = None
        self.scaler_dyn: Optional[StandardScaler] = None
        self.tokenizer: Optional[PyTorchApiTokenizer] = None

    # ------------------------------------------------------------------ fit
    def fit(self, df_train: pd.DataFrame) -> "FeaturePipeline":
        df = _with_columns(df_train, self.core_cols)
        self.scaler_core = StandardScaler().fit(_numeric(df, self.core_cols))

        self.tfidf_op = TfidfVectorizer(max_features=self.opcode_max_features)
        self.tfidf_op.fit(_text(df, opcode_column(df)))

        self.dyn_cols = [c for c in DYNAMIC_COUNTER_CANDIDATES if c in df.columns] or ["total_calls"]
        df = _with_columns(df, self.dyn_cols)
        self.scaler_dyn = StandardScaler().fit(_numeric(df, self.dyn_cols))

        self.tokenizer = PyTorchApiTokenizer(max_vocab=self.api_vocab_size)
        self.tokenizer.fit_on_texts(_text(df, api_column(df)).tolist())
        return self

    def fit_transform(self, df_train: pd.DataFrame) -> FeatureSet:
        """Fit on the training partition and return its features.

        The opcode TF-IDF of the training rows comes from ``fit_transform`` (as in
        the experiments), which can differ from ``fit`` + ``transform`` in the last
        floating-point bit.
        """
        self.fit(df_train)
        feats = self.transform(df_train)
        feats.opcode = self.tfidf_op.fit_transform(_text(df_train, opcode_column(df_train))).toarray()
        return feats

    # ------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame) -> FeatureSet:
        if self.scaler_core is None:
            raise RuntimeError("FeaturePipeline must be fitted before transform().")
        df = _with_columns(df, self.core_cols + self.dyn_cols)
        core = self.scaler_core.transform(_numeric(df, self.core_cols))
        if self.tfidf_op is not None:
            opcode = self.tfidf_op.transform(_text(df, opcode_column(df))).toarray()
        else:
            opcode = np.zeros((len(df), 0))
        dynamic = self.scaler_dyn.transform(_numeric(df, self.dyn_cols))
        api_seq = None
        if self.tokenizer is not None:
            api_seq = self.tokenizer.texts_to_sequences(_text(df, api_column(df)).tolist(), max_len=self.api_max_len)
        return FeatureSet(core=core, opcode=opcode, dynamic=dynamic, api_seq=api_seq)

    # ----------------------------------------------------------- properties
    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer.word2idx) if self.tokenizer is not None else 0

    @property
    def core_dim(self) -> int:
        return len(self.core_cols)

    @property
    def opcode_dim(self) -> int:
        return len(self.tfidf_op.vocabulary_) if self.tfidf_op is not None else 0

    # ------------------------------------------------------- serialization
    def to_state(self) -> dict:
        """Dictionary saved with joblib.

        Keys used by all earlier versions of the experiment code are written, so
        the file can be read by either code base.
        """
        return {
            "core_static_cols": self.core_cols, "core_cols": self.core_cols,
            "top_dyn_cols": self.dyn_cols,
            "scaler_core_static": self.scaler_core, "scaler_core_stat": self.scaler_core,
            "scaler_core": self.scaler_core,
            "scaler_top_dyn": self.scaler_dyn, "scaler_dyn": self.scaler_dyn,
            "tfidf_op": self.tfidf_op,
            "tokenizer": self.tokenizer,
            "vocab_size": self.vocab_size,
            "api_max_len": self.api_max_len,
            "core_dim": self.core_dim, "op_dim": self.opcode_dim, "dyn_dim": len(self.dyn_cols),
        }

    @classmethod
    def from_state(cls, state: dict) -> "FeaturePipeline":
        def first(*keys):
            for k in keys:
                if state.get(k) is not None:
                    return state[k]
            return None

        pipe = cls(api_max_len=state.get("api_max_len", 200))
        pipe.core_cols = list(first("core_static_cols", "core_cols") or CORE_STATIC_COLUMNS)
        pipe.dyn_cols = list(first("top_dyn_cols") or [])
        pipe.scaler_core = first("scaler_core_static", "scaler_core_stat", "scaler_core")
        pipe.scaler_dyn = first("scaler_top_dyn", "scaler_dyn")
        pipe.tfidf_op = first("tfidf_op")
        pipe.tokenizer = first("tokenizer")
        if pipe.scaler_core is None or pipe.scaler_dyn is None:
            raise ValueError("Saved preprocessor state is missing a fitted scaler.")
        if pipe.tokenizer is not None:
            pipe.api_vocab_size = pipe.tokenizer.max_vocab
        return pipe


def static_opcode_only(x_file: np.ndarray, core_dim: int, opcode_dim: int, keep_opcode: bool = True) -> np.ndarray:
    """File features for the "Static + opcode only" ablation.

    Removes the three entropy features and all numerical runtime features, and
    keeps file size, total opcode count and (optionally) the opcode TF-IDF.
    Graph relations are not affected by this function.
    """
    x = x_file.copy()
    x[:, 0:N_ENTROPY_COLUMNS] = 0.0
    if not keep_opcode:
        x[:, core_dim:core_dim + opcode_dim] = 0.0
    x[:, core_dim + opcode_dim:] = 0.0
    return x
