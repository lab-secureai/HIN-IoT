"""HIN construction (Section 3.3) and isolated per-file evaluation graphs (Section 4.3).

Node types: ``file``, ``api``, ``arch``.
Relations: ``(file, calls, api)``, ``(file, runs_on, arch)`` and their reverses
``rev_calls`` / ``rev_runs_on`` added by ``ToUndirected``.

API and architecture nodes are categorical entities with one-hot (identity)
features. Their vocabularies are defined by the training partition only.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData

from .data import api_lists

FULL_GRAPH = "Full_Graph"
NO_API = "No_API_Node"
NO_ARCH = "No_Arch_Node"

CALLS = ("file", "calls", "api")
RUNS_ON = ("file", "runs_on", "arch")


def _empty_edges() -> torch.Tensor:
    return torch.empty((2, 0), dtype=torch.long)


def build_train_hin(df_train: pd.DataFrame, x_file: np.ndarray, mode: str = FULL_GRAPH
                    ) -> Tuple[HeteroData, Dict[str, int], LabelEncoder]:
    """Training HIN over all training files (transductive, full batch).

    Returns the graph, the API vocabulary and the architecture encoder. Both
    mappings are reused unchanged at evaluation time.
    """
    api_seqs = api_lists(df_train)
    all_apis = sorted({api for seq in api_seqs for api in seq})
    api_to_id = {name: i for i, name in enumerate(all_apis)}

    le_arch = LabelEncoder()
    arch_ids = le_arch.fit_transform(df_train["arch"].astype(str))

    data = HeteroData()
    data["file"].x = torch.tensor(x_file, dtype=torch.float)
    data["file"].y = torch.tensor(df_train["label"].values, dtype=torch.long)
    data["api"].x = torch.eye(max(len(api_to_id), 1), dtype=torch.float)
    data["arch"].x = torch.eye(len(le_arch.classes_), dtype=torch.float)

    if mode != NO_API and len(api_to_id) > 0:
        src, dst = [], []
        for f_idx, seq in enumerate(api_seqs):
            for api in seq:
                if api in api_to_id:
                    src.append(f_idx)
                    dst.append(api_to_id[api])
        data[CALLS].edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long) if src else _empty_edges()
    else:
        data[CALLS].edge_index = _empty_edges()

    if mode != NO_ARCH:
        valid = arch_ids != -1
        edges = np.array([np.arange(len(df_train))[valid], arch_ids[valid]])
        data[RUNS_ON].edge_index = torch.tensor(edges, dtype=torch.long) if valid.any() else _empty_edges()
    else:
        data[RUNS_ON].edge_index = _empty_edges()

    return T.ToUndirected()(data), api_to_id, le_arch


def encode_arch(values: pd.Series, le_arch: LabelEncoder) -> np.ndarray:
    """Architecture index from the training encoder, or -1 if the architecture was not seen in training."""
    index = {str(c): i for i, c in enumerate(le_arch.classes_)}
    return np.array([index.get(str(v), -1) for v in values], dtype=np.int64)


def build_ego_graphs(df: pd.DataFrame, x_file: np.ndarray, api_to_id: Dict[str, int],
                     le_arch: LabelEncoder, mode: str = FULL_GRAPH) -> List[HeteroData]:
    """One isolated graph per evaluation file.

    Each graph holds the file node, the API nodes it invokes that exist in the
    training vocabulary, and its architecture node if that architecture was seen
    in training. There are no edges between evaluation files, so batching these
    graphs does not let test files exchange messages.
    """
    api_seqs = api_lists(df)
    arch_ids = encode_arch(df["arch"], le_arch)
    labels = df["label"].values if "label" in df.columns else np.zeros(len(df), dtype=np.int64)

    num_apis = max(len(api_to_id), 1)
    num_archs = len(le_arch.classes_)
    eye_api = torch.eye(num_apis, dtype=torch.float)
    eye_arch = torch.eye(num_archs, dtype=torch.float)

    graphs = []
    for i in range(len(df)):
        data = HeteroData()
        data["file"].x = torch.tensor(x_file[i:i + 1], dtype=torch.float)
        data["file"].y = torch.tensor([labels[i]], dtype=torch.long)

        api_ids = []
        if mode != NO_API and len(api_to_id) > 0:
            api_ids = sorted({api_to_id[api] for api in api_seqs.iloc[i] if api in api_to_id})
        if api_ids:
            data["api"].x = eye_api[api_ids]
            data[CALLS].edge_index = torch.tensor(np.array([[0] * len(api_ids), list(range(len(api_ids)))]),
                                                  dtype=torch.long)
        else:
            data["api"].x = torch.empty((0, num_apis), dtype=torch.float)
            data[CALLS].edge_index = _empty_edges()

        if mode != NO_ARCH and arch_ids[i] != -1:
            data["arch"].x = eye_arch[arch_ids[i:i + 1]]
            data[RUNS_ON].edge_index = torch.tensor(np.array([[0], [0]]), dtype=torch.long)
        else:
            data["arch"].x = torch.empty((0, num_archs), dtype=torch.float)
            data[RUNS_ON].edge_index = _empty_edges()

        graphs.append(T.ToUndirected()(data))
    return graphs


# ----------------------------------------------------------------- homogeneous kNN graph (baseline)

def build_knn_file_graph(x: np.ndarray, k: int = 5) -> torch.Tensor:
    """kNN edges between training files for the homogeneous GCN baseline."""
    k_actual = min(k, x.shape[0] - 1) if x.shape[0] > 1 else 1
    if k_actual <= 0:
        return _empty_edges()
    adj = kneighbors_graph(x, n_neighbors=k_actual, mode="connectivity", include_self=False, n_jobs=-1).tocoo()
    return torch.tensor(np.array([adj.row, adj.col]), dtype=torch.long)


def inductive_knn_edges(x_train: np.ndarray, x_test: np.ndarray, k: int = 5, symmetric: bool = False
                        ) -> torch.Tensor:
    """Edges from each test file to its k nearest training files.

    Node indices: training files first, then test files. With ``symmetric=False``
    messages flow only from training to test files. ``symmetric=True`` also adds
    test-to-training edges (used by the original family-disjoint evaluation on the
    2024 corpus).
    """
    k_actual = min(k, len(x_train))
    nn_support = NearestNeighbors(n_neighbors=k_actual, n_jobs=-1).fit(x_train)
    adj = nn_support.kneighbors_graph(x_test, mode="connectivity").tocoo()
    n_train = len(x_train)
    src = torch.tensor(adj.col, dtype=torch.long)
    dst = torch.tensor(adj.row + n_train, dtype=torch.long)
    edges = torch.stack([src, dst], dim=0)
    if symmetric:
        edges = torch.cat([edges, torch.stack([dst, src], dim=0)], dim=1)
    return edges
