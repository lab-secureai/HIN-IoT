"""Training and inference routines for the HIN encoders and the baselines."""

import time
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from .config import TrainSettings
from .graph import build_knn_file_graph, inductive_knn_edges
from .models import BiGRUClassifier, FusionClassifier, HomogeneousGCN
from .utils import peak_memory_mb, reset_peak_memory, synchronize


class SequenceDenseDataset(Dataset):
    def __init__(self, x_seq, x_dense=None, y=None):
        self.x_seq = torch.tensor(x_seq, dtype=torch.long)
        self.x_dense = torch.tensor(x_dense, dtype=torch.float) if x_dense is not None else None
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.x_seq)

    def __getitem__(self, idx):
        item = {"x_seq": self.x_seq[idx]}
        if self.x_dense is not None:
            item["x_dense"] = self.x_dense[idx]
        if self.y is not None:
            item["y"] = self.y[idx]
        return item


def balanced_class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights n / (2 n_c) used in the HIN cross-entropy loss."""
    y = np.asarray(y)
    num_neg, num_pos = (y == 0).sum(), (y == 1).sum()
    w_neg = len(y) / (2.0 * max(num_neg, 1))
    w_pos = len(y) / (2.0 * max(num_pos, 1))
    return torch.tensor([w_neg, w_pos], dtype=torch.float).to(device)


# ============================================================ HIN encoders

def train_hin_model(model, graph, class_weights, s: TrainSettings, device) -> Tuple[float, float]:
    """Full-batch training on the training HIN.

    Adam (lr, weight decay) with cosine annealing over ``hin_epochs`` epochs and
    class-weighted cross-entropy on file nodes. Models that define
    ``compute_orthogonality_loss`` (DR-HGT) add ``ortho_weight`` times that term.
    Returns (training time per epoch in ms, peak GPU memory in MB).
    """
    with torch.no_grad():  # materializes lazy layers
        _ = model(graph.x_dict, graph.edge_index_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=s.hin_lr, weight_decay=s.hin_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=s.hin_epochs, eta_min=s.hin_eta_min)

    reset_peak_memory(device)
    t0 = time.perf_counter()
    model.train()
    for _ in range(s.hin_epochs):
        optimizer.zero_grad()
        out = model(graph.x_dict, graph.edge_index_dict)
        loss = F.cross_entropy(out, graph["file"].y, weight=class_weights)
        if hasattr(model, "compute_orthogonality_loss"):
            loss = loss + s.ortho_weight * model.compute_orthogonality_loss()
        loss.backward()
        optimizer.step()
        scheduler.step()
    synchronize(device)
    train_ms = (time.perf_counter() - t0) * 1000 / s.hin_epochs
    return train_ms, peak_memory_mb(device)


def predict_ego_graphs(model, graphs, device, batch_size: int = 128) -> Tuple[np.ndarray, np.ndarray, float]:
    """Predict isolated per-file graphs. Returns (labels, malware probabilities, ms per sample)."""
    model.eval()
    loader = PyGDataLoader(graphs, batch_size=batch_size, shuffle=False)
    preds, probs = [], []
    synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            preds.extend(out.argmax(dim=1).cpu().numpy())
    synchronize(device)
    infer_ms = (time.perf_counter() - t0) * 1000 / len(graphs)
    return np.array(preds), np.array(probs), infer_ms


def load_hin_model(model, state_dict, sample_graph, device):
    """Load saved weights into a freshly built model (lazy layers are materialized first)."""
    model = model.to(device)
    sample = next(iter(PyGDataLoader([sample_graph] if not isinstance(sample_graph, list) else sample_graph,
                                     batch_size=128, shuffle=False))).to(device)
    with torch.no_grad():
        _ = model(sample.x_dict, sample.edge_index_dict)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ============================================================ sequence baselines

def _train_sequence_model(model, loader, epochs, lr, device, dense: bool):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    reset_peak_memory(device)
    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            if dense:
                out = model(batch["x_seq"].to(device), batch["x_dense"].to(device))
            else:
                out = model(batch["x_seq"].to(device))
            loss = F.cross_entropy(out, batch["y"].to(device))
            loss.backward()
            optimizer.step()
    synchronize(device)
    return (time.perf_counter() - t0) * 1000 / epochs, peak_memory_mb(device)


def train_bigru(x_seq, y, vocab_size, s: TrainSettings, device):
    loader = DataLoader(SequenceDenseDataset(x_seq, y=y), batch_size=s.seq_batch_size, shuffle=True)
    model = BiGRUClassifier(vocab_size=vocab_size, embed_dim=s.embed_dim, hidden_dim=s.gru_hidden).to(device)
    train_ms, peak = _train_sequence_model(model, loader, s.bigru_epochs, s.seq_lr, device, dense=False)
    return model, train_ms, peak


def train_fusion(x_seq, x_dense, y, vocab_size, s: TrainSettings, device):
    loader = DataLoader(SequenceDenseDataset(x_seq, x_dense, y), batch_size=s.seq_batch_size, shuffle=True)
    model = FusionClassifier(vocab_size=vocab_size, dense_dim=x_dense.shape[1],
                             embed_dim=s.embed_dim, gru_hidden=s.gru_hidden).to(device)
    train_ms, peak = _train_sequence_model(model, loader, s.fusion_epochs, s.seq_lr, device, dense=True)
    return model, train_ms, peak


def predict_sequence_model(model, x_seq, device, x_dense=None, batch_size: int = 256):
    model.eval()
    loader = DataLoader(SequenceDenseDataset(x_seq, x_dense), batch_size=batch_size, shuffle=False)
    preds, probs = [], []
    synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            if x_dense is not None:
                out = model(batch["x_seq"].to(device), batch["x_dense"].to(device))
            else:
                out = model(batch["x_seq"].to(device))
            probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            preds.extend(out.argmax(dim=1).cpu().numpy())
    synchronize(device)
    infer_ms = (time.perf_counter() - t0) * 1000 / len(x_seq)
    return np.array(preds), np.array(probs), infer_ms


# ============================================================ homogeneous GCN

def train_gcn(x_train, y, s: TrainSettings, device):
    edge_index = build_knn_file_graph(x_train, k=s.knn_k)
    x = torch.tensor(x_train, dtype=torch.float).to(device)
    edge_index = edge_index.to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)

    model = HomogeneousGCN(in_channels=x_train.shape[1], hidden_dim=s.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=s.gcn_lr)
    reset_peak_memory(device)
    t0 = time.perf_counter()
    model.train()
    for _ in range(s.gcn_epochs):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x, edge_index), y_t)
        loss.backward()
        optimizer.step()
    synchronize(device)
    return model, (time.perf_counter() - t0) * 1000 / s.gcn_epochs, peak_memory_mb(device)


def predict_gcn_inductive(model, x_train, x_test, device, k: int = 5, symmetric: bool = False):
    """Attach each test file to its k nearest training files and run the trained GCN."""
    model.eval()
    edge_index = inductive_knn_edges(x_train, x_test, k=k, symmetric=symmetric).to(device)
    x_all = torch.tensor(np.vstack([x_train, x_test]), dtype=torch.float).to(device)
    synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x_all, edge_index)[len(x_train):]
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
    synchronize(device)
    return preds, probs, (time.perf_counter() - t0) * 1000 / len(x_test)


# ============================================================ LightGBM

def train_lgbm(x_core, y, seed: int, s: TrainSettings):
    """Static LightGBM on the five numerical ELF features, trained on a randomly undersampled training set."""
    from imblearn.under_sampling import RandomUnderSampler
    from lightgbm import LGBMClassifier

    x_res, y_res = RandomUnderSampler(random_state=seed).fit_resample(x_core, y)
    t0 = time.perf_counter()
    model = LGBMClassifier(n_estimators=s.lgbm_estimators, learning_rate=s.lgbm_lr,
                           random_state=seed, n_jobs=-1, verbose=-1)
    model.fit(np.asarray(x_res), y_res)
    train_ms = (time.perf_counter() - t0) * 1000 / s.lgbm_estimators
    return model, train_ms


def predict_lgbm(model, x_core) -> Tuple[np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()
    preds = model.predict(x_core)
    probs = model.predict_proba(x_core)[:, 1]
    return preds, probs, (time.perf_counter() - t0) * 1000 / len(x_core)

