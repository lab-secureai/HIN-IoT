# -*- coding: utf-8 -*-
"""Leave-one-architecture-out evaluation across the supported CPU architectures."""

import os
import ast
import time
import random
import gc
import joblib
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, HGTConv, GATConv, HANConv, GCNConv

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DEFAULT_SEEDS = [10, 42, 100, 314, 555, 777, 999, 1234, 2024, 2026]
DEFAULT_DATASET_PATH = '/content/drive/MyDrive/dataset/dataset.csv'
DEFAULT_BASE_SAVE_DIR = '/content/drive/MyDrive/dataset/HIN/results_scenario_2_cross_isa'
TARGET_ARCHITECTURES = ['MIPS', 'ARM', 'PowerPC', 'x64', 'x86']

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def check_colab_environment():
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive'):
            drive.mount('/content/drive')
    except ImportError:
        pass

class PyTorchApiTokenizer:
    def __init__(self, max_vocab=5000):
        self.max_vocab = max_vocab
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    @property
    def word_index(self):
        return self.word2idx

    def fit_on_texts(self, texts):
        counts = Counter()
        for text in texts:
            tokens = text.split() if isinstance(text, str) else text
            counts.update(tokens)
        most_common = counts.most_common(self.max_vocab - 2)
        for token, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def texts_to_sequences(self, texts, max_len=200):
        seqs = []
        for text in texts:
            tokens = text.split() if isinstance(text, str) else text
            ids = [self.word2idx.get(t, 1) for t in tokens[:max_len]]
            if len(ids) < max_len:
                ids = ids + [0] * (max_len - len(ids))
            seqs.append(ids)
        return np.array(seqs, dtype=np.int64)

class SequenceDenseDataset(torch.utils.data.Dataset):
    def __init__(self, x_seq, x_dense=None, y=None):
        self.x_seq = torch.tensor(x_seq, dtype=torch.long)
        self.x_dense = torch.tensor(x_dense, dtype=torch.float) if x_dense is not None else None
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.x_seq)

    def __getitem__(self, idx):
        item = {'x_seq': self.x_seq[idx]}
        if self.x_dense is not None:
            item['x_dense'] = self.x_dense[idx]
        if self.y is not None:
            item['y'] = self.y[idx]
        return item

def standardize_elf_architecture(arch_str):
    arch_clean = str(arch_str).strip().lower()
    if 'arm' in arch_clean or 'aarch64' in arch_clean:
        return 'ARM'
    elif 'mips' in arch_clean:
        return 'MIPS'
    elif 'powerpc' in arch_clean or 'ppc' in arch_clean:
        return 'PowerPC'
    elif '64' in arch_clean or 'amd64' in arch_clean or 'x86_64' in arch_clean:
        return 'x64'
    elif '86' in arch_clean or 'i386' in arch_clean or 'i686' in arch_clean:
        return 'x86'
    return 'Other'

def parse_text_sequence(val):
    if isinstance(val, str):
        if val.startswith('{'):
            try:
                d = ast.literal_eval(val)
                return " ".join([f"{k} "*min(int(v), 20) for k, v in d.items()])
            except:
                pass
        return val
    elif isinstance(val, (list, tuple)):
        return " ".join(map(str, val))
    return ""

def split_loao_dataset(df_raw, held_out_arch, seed=42):
    df_ood = df_raw[df_raw['arch'] == held_out_arch].reset_index(drop=True)
    df_id_all = df_raw[df_raw['arch'] != held_out_arch].reset_index(drop=True)

    df_id_all['stratify_key'] = df_id_all['label'].astype(str) + "_" + df_id_all['arch'].astype(str)
    valid_counts = df_id_all['stratify_key'].value_counts()
    valid_keys = valid_counts[valid_counts > 1].index
    df_id_valid = df_id_all[df_id_all['stratify_key'].isin(valid_keys)].copy()

    df_id_train, df_id_test = train_test_split(
        df_id_valid, test_size=0.2, stratify=df_id_valid['stratify_key'], random_state=seed
    )
    df_id_train = df_id_train.drop(columns=['stratify_key']).reset_index(drop=True)
    df_id_test = df_id_test.drop(columns=['stratify_key']).reset_index(drop=True)

    return df_id_train, df_id_test, df_ood

def extract_loao_streamlined_features(df_train, df_id_test, df_ood_test):
    core_static_cols = ['mean_entropy', 'max_entropy', 'min_entropy', 'file_size_mb', 'total_opcodes']
    for c in core_static_cols:
        if c not in df_train.columns: df_train[c] = 0.0
        if c not in df_id_test.columns: df_id_test[c] = 0.0
        if c not in df_ood_test.columns: df_ood_test[c] = 0.0

    scaler_core_stat = StandardScaler()
    x_tr_core_stat = scaler_core_stat.fit_transform(df_train[core_static_cols].apply(pd.to_numeric, errors='coerce').fillna(0))
    x_id_core_stat = scaler_core_stat.transform(df_id_test[core_static_cols].apply(pd.to_numeric, errors='coerce').fillna(0))
    x_ood_core_stat = scaler_core_stat.transform(df_ood_test[core_static_cols].apply(pd.to_numeric, errors='coerce').fillna(0))

    op_col = 'opcode_counts' if 'opcode_counts' in df_train.columns else 'opcode_sequence'
    tfidf_op = TfidfVectorizer(max_features=100)
    x_tr_op = tfidf_op.fit_transform(df_train.get(op_col, pd.Series([""]*len(df_train))).fillna("").apply(parse_text_sequence)).toarray()
    x_id_op = tfidf_op.transform(df_id_test.get(op_col, pd.Series([""]*len(df_id_test))).fillna("").apply(parse_text_sequence)).toarray()
    x_ood_op = tfidf_op.transform(df_ood_test.get(op_col, pd.Series([""]*len(df_ood_test))).fillna("").apply(parse_text_sequence)).toarray()

    X_tr_stat_full = np.hstack([x_tr_core_stat, x_tr_op])
    X_id_stat_full = np.hstack([x_id_core_stat, x_id_op])
    X_ood_stat_full = np.hstack([x_ood_core_stat, x_ood_op])

    top_dyn_candidates = [
        'socket_calls', 'connect_calls', 'send_calls', 'fork_calls',
        'exec_calls', 'execve_calls', 'sys_calls', 'file_ops',
        'net_ops', 'proc_ops', 'total_calls', 'unique_calls'
    ]
    top_dyn_cols = [c for c in top_dyn_candidates if c in df_train.columns]
    if not top_dyn_cols:
        fallback_candidates = ['total_calls', 'unique_calls', 'file_ops', 'net_ops', 'proc_ops', 'sys_calls', 'socket_calls']
        top_dyn_cols = [c for c in fallback_candidates if c in df_train.columns]
        if not top_dyn_cols:
            top_dyn_cols = ['total_calls']

    for c in top_dyn_cols:
        if c not in df_train.columns: df_train[c] = 0.0
        if c not in df_id_test.columns: df_id_test[c] = 0.0
        if c not in df_ood_test.columns: df_ood_test[c] = 0.0

    scaler_top_dyn = StandardScaler()
    x_tr_top_dyn = scaler_top_dyn.fit_transform(df_train[top_dyn_cols].apply(pd.to_numeric, errors='coerce').fillna(0))
    x_id_top_dyn = scaler_top_dyn.transform(df_id_test[top_dyn_cols].apply(pd.to_numeric, errors='coerce').fillna(0))
    x_ood_top_dyn = scaler_top_dyn.transform(df_ood_test[top_dyn_cols].apply(pd.to_numeric, errors='coerce').fillna(0))

    api_col = 'api_sequence' if 'api_sequence' in df_train.columns else 'api_list'
    tfidf_api = TfidfVectorizer(max_features=100)
    tfidf_api.fit(df_train.get(api_col, pd.Series([""]*len(df_train))).fillna("").apply(parse_text_sequence))

    X_tr_dyn_stream = x_tr_top_dyn
    X_id_dyn_stream = x_id_top_dyn
    X_ood_dyn_stream = x_ood_top_dyn

    X_tr_hin_file = np.hstack([X_tr_stat_full, X_tr_dyn_stream])
    X_id_hin_file = np.hstack([X_id_stat_full, X_id_dyn_stream])
    X_ood_hin_file = np.hstack([X_ood_stat_full, X_ood_dyn_stream])

    tokenizer = PyTorchApiTokenizer(max_vocab=5000)
    seq_tr = df_train.get(api_col, pd.Series([""]*len(df_train))).fillna("").apply(parse_text_sequence).tolist()
    seq_id = df_id_test.get(api_col, pd.Series([""]*len(df_id_test))).fillna("").apply(parse_text_sequence).tolist()
    seq_ood = df_ood_test.get(api_col, pd.Series([""]*len(df_ood_test))).fillna("").apply(parse_text_sequence).tolist()
    tokenizer.fit_on_texts(seq_tr)
    X_tr_seq = tokenizer.texts_to_sequences(seq_tr, max_len=200)
    X_id_seq = tokenizer.texts_to_sequences(seq_id, max_len=200)
    X_ood_seq = tokenizer.texts_to_sequences(seq_ood, max_len=200)

    preprocessors = {
        'scaler_core_stat': scaler_core_stat,
        'tfidf_op': tfidf_op,
        'scaler_top_dyn': scaler_top_dyn,
        'tfidf_api': tfidf_api,
        'tokenizer': tokenizer,
        'core_static_cols': core_static_cols,
        'top_dyn_cols': top_dyn_cols,
        'vocab_size': len(tokenizer.word2idx),
        'baseline1_dim': x_tr_core_stat.shape[1],
        'static_full_dim': X_tr_stat_full.shape[1],
        'dynamic_streamlined_dim': X_tr_dyn_stream.shape[1],
        'hin_file_dim': X_tr_hin_file.shape[1]
    }
    return (x_tr_core_stat, x_id_core_stat, x_ood_core_stat), (X_tr_stat_full, X_id_stat_full, X_ood_stat_full), (X_tr_dyn_stream, X_id_dyn_stream, X_ood_dyn_stream), (X_tr_hin_file, X_id_hin_file, X_ood_hin_file), (X_tr_seq, X_id_seq, X_ood_seq), preprocessors

def build_loao_hetero_graph_train(df_train, X_tr_hin_file):
    def parse_api(s):
        try:
            return ast.literal_eval(s) if isinstance(s, str) and s.startswith('[') else ([] if pd.isna(s) else str(s).split())
        except:
            return []

    api_col = 'api_sequence' if 'api_sequence' in df_train.columns else 'api_list'
    api_seqs = df_train.get(api_col, pd.Series([[]]*len(df_train))).apply(parse_api)

    all_apis = sorted(list({api for seq in api_seqs for api in seq}))
    api_to_id = {name: i for i, name in enumerate(all_apis)}
    le_arch = LabelEncoder()
    arch_ids = le_arch.fit_transform(df_train['arch'].astype(str))

    src_f, dst_a = [], []
    for f_idx, seq in enumerate(api_seqs):
        for api in seq:
            if api in api_to_id:
                src_f.append(f_idx)
                dst_a.append(api_to_id[api])

    data = HeteroData()
    data['file'].x = torch.tensor(X_tr_hin_file, dtype=torch.float)
    data['file'].y = torch.tensor(df_train['label'].values, dtype=torch.long)
    data['api'].x = torch.eye(len(api_to_id) if len(api_to_id) > 0 else 1, dtype=torch.float)
    data['arch'].x = torch.eye(len(le_arch.classes_), dtype=torch.float)

    data['file', 'calls', 'api'].edge_index = torch.tensor(np.array([src_f, dst_a]), dtype=torch.long) if src_f else torch.empty((2, 0), dtype=torch.long)
    mask = arch_ids != -1
    edge_runs = np.array([np.arange(len(df_train))[mask], arch_ids[mask]])
    data['file', 'runs_on', 'arch'].edge_index = torch.tensor(edge_runs, dtype=torch.long) if mask.any() else torch.empty((2, 0), dtype=torch.long)

    return T.ToUndirected()(data), api_to_id, le_arch

def build_loao_independent_ego_graphs(df_test, X_te_hin_file, api_to_id, le_arch):
    def parse_api(s):
        try:
            return ast.literal_eval(s) if isinstance(s, str) and s.startswith('[') else ([] if pd.isna(s) else str(s).split())
        except:
            return []

    api_col = 'api_sequence' if 'api_sequence' in df_test.columns else 'api_list'
    api_seqs = df_test.get(api_col, pd.Series([[]]*len(df_test))).apply(parse_api)
    known_archs = set(le_arch.classes_)
    arch_ids = df_test['arch'].apply(lambda x: le_arch.transform([str(x)])[0] if str(x) in known_archs else -1).values

    num_apis = len(api_to_id) if len(api_to_id) > 0 else 1
    num_archs = len(le_arch.classes_)

    eye_api = torch.eye(num_apis, dtype=torch.float)
    eye_arch = torch.eye(num_archs, dtype=torch.float)

    graphs = []
    for i in range(len(df_test)):
        data = HeteroData()
        data['file'].x = torch.tensor(X_te_hin_file[i:i+1], dtype=torch.float)
        data['file'].y = torch.tensor([df_test['label'].iloc[i]], dtype=torch.long)

        seq = api_seqs.iloc[i]
        valid_api_ids = sorted(list({api_to_id[api] for api in seq if api in api_to_id}))

        if len(valid_api_ids) > 0:
            data['api'].x = eye_api[valid_api_ids]
            local_api_indices = list(range(len(valid_api_ids)))
            data['file', 'calls', 'api'].edge_index = torch.tensor(np.array([[0]*len(valid_api_ids), local_api_indices]), dtype=torch.long)
        else:
            data['api'].x = torch.empty((0, num_apis), dtype=torch.float)
            data['file', 'calls', 'api'].edge_index = torch.empty((2, 0), dtype=torch.long)

        if arch_ids[i] != -1:
            data['arch'].x = eye_arch[arch_ids[i:i+1]]
            data['file', 'runs_on', 'arch'].edge_index = torch.tensor(np.array([[0], [0]]), dtype=torch.long)
        else:
            data['arch'].x = torch.empty((0, num_archs), dtype=torch.float)
            data['file', 'runs_on', 'arch'].edge_index = torch.empty((2, 0), dtype=torch.long)

        graphs.append(T.ToUndirected()(data))
    return graphs

class Baseline2_BiGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, out_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))

    def forward(self, x_seq):
        out, _ = self.gru(self.embedding(x_seq))
        return self.fc(torch.mean(out, dim=1))

class Baseline3_EndToEndHybrid(nn.Module):
    def __init__(self, vocab_size, dense_dim, embed_dim=128, gru_hidden=128, out_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, gru_hidden, batch_first=True, bidirectional=True)
        self.static_mlp = nn.Sequential(nn.Linear(dense_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear((gru_hidden * 2) + 64, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, out_dim))

    def forward(self, x_seq, x_dense):
        seq_feat = torch.mean(self.gru(self.embedding(x_seq))[0], dim=1)
        static_feat = self.static_mlp(x_dense)
        return self.classifier(torch.cat([seq_feat, static_feat], dim=1))

def build_homogeneous_file_graph(X_features, y_labels=None, k=5):
    k_actual = min(k, X_features.shape[0] - 1) if X_features.shape[0] > 1 else 1
    if k_actual > 0:
        adj = kneighbors_graph(X_features, n_neighbors=k_actual, mode='connectivity', include_self=False, n_jobs=-1)
        adj = adj.tocoo()
        edge_index = torch.tensor(np.array([adj.row, adj.col]), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    x = torch.tensor(X_features, dtype=torch.float)
    if y_labels is not None:
        y = torch.tensor(y_labels, dtype=torch.long)
        return x, edge_index, y
    return x, edge_index

def predict_bigru_batched(model, X_seq, device, batch_size=256):
    model.eval()
    dataset = SequenceDenseDataset(X_seq)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_probs = [], []

    if device.type == 'cuda': torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            out = model(batch['x_seq'].to(device))
            all_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            all_preds.extend(out.argmax(dim=1).cpu().numpy())

    if device.type == 'cuda': torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t0) * 1000 / len(X_seq)
    return np.array(all_preds), np.array(all_probs), infer_ms

def predict_fusion_batched(model, X_seq, X_dense, device, batch_size=256):
    model.eval()
    dataset = SequenceDenseDataset(X_seq, X_dense)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_probs = [], []

    if device.type == 'cuda': torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            out = model(batch['x_seq'].to(device), batch['x_dense'].to(device))
            all_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            all_preds.extend(out.argmax(dim=1).cpu().numpy())

    if device.type == 'cuda': torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t0) * 1000 / len(X_seq)
    return np.array(all_preds), np.array(all_probs), infer_ms

class Baseline4_HomogeneousGCN(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, out_dim=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)

def evaluate_inductive_homogeneous_gcn(model, X_tr, X_te, device, k=5):
    model.eval()
    nn_support = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X_tr)
    adj_bipartite = nn_support.kneighbors_graph(X_te, mode='connectivity').tocoo()

    N_tr = len(X_tr)
    src = torch.tensor(adj_bipartite.col, dtype=torch.long)
    dst = torch.tensor(adj_bipartite.row + N_tr, dtype=torch.long)
    edge_index_te = torch.stack([src, dst], dim=0).to(device)

    X_all = torch.tensor(np.vstack([X_tr, X_te]), dtype=torch.float).to(device)

    if device.type == 'cuda': torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        out_all = model(X_all, edge_index_te)
        out_te = out_all[N_tr:]
        probs = F.softmax(out_te, dim=1)[:, 1].cpu().numpy()
        preds = out_te.argmax(dim=1).cpu().numpy()

    if device.type == 'cuda': torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t_start) * 1000 / len(X_te)

    return preds, probs, infer_ms

class BaseHeteroSAGE(nn.Module):
    def __init__(self, hidden, out, metadata):
        super().__init__()
        self.conv1 = HeteroConv({et: SAGEConv((-1, -1), hidden) for et in metadata[1]}, aggr='sum')
        self.conv2 = HeteroConv({et: SAGEConv((-1, -1), hidden) for et in metadata[1]}, aggr='sum')
        self.lin = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        out1 = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(out1.get(k, x_dict[k])) for k in x_dict.keys()}
        out2 = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(out2.get(k, x_dict[k])) for k in x_dict.keys()}
        return self.lin(x_dict['file'])

class HGT(nn.Module):
    def __init__(self, hidden, out, heads, layers, metadata):
        super().__init__()
        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in metadata[0]})
        self.convs = nn.ModuleList([HGTConv(hidden, hidden, metadata, heads) for _ in range(layers)])
        self.lin = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {node: F.relu(self.lin_dict[node](x)) for node, x in x_dict.items()}
        for conv in self.convs:
            out = conv(x_dict, edge_index_dict)
            x_dict = {k: out.get(k, x_dict[k]) for k in x_dict.keys()}
        return self.lin(x_dict['file'])

class GATModel(nn.Module):
    def __init__(self, hidden, out, heads, metadata):
        super().__init__()
        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in metadata[0]})
        self.conv1 = HeteroConv({et: GATConv((-1, -1), hidden // heads, heads=heads, add_self_loops=False) for et in metadata[1]}, aggr='sum')
        self.conv2 = HeteroConv({et: GATConv((-1, -1), hidden // heads, heads=heads, add_self_loops=False) for et in metadata[1]}, aggr='sum')
        self.lin = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items()}
        out1 = self.conv1(x, edge_index_dict)
        x = {k: F.relu(out1.get(k, x[k]) + x[k]) for k in x.keys()}
        out2 = self.conv2(x, edge_index_dict)
        x = {k: F.relu(out2.get(k, x[k]) + x[k]) for k in x.keys()}
        return self.lin(x['file'])

class HANModel(nn.Module):
    def __init__(self, hidden, out, heads, metadata):
        super().__init__()
        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in metadata[0]})
        self.conv1 = HANConv(hidden, hidden, metadata, heads=heads)
        self.lin = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items()}
        out_dict = self.conv1(x, edge_index_dict)
        return self.lin(F.relu(out_dict.get('file', x['file']) + x['file']))

class MetaPathSAGE(nn.Module):
    def __init__(self, hidden, out, metadata):
        super().__init__()
        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in metadata[0]})
        self.mp1_f2a = SAGEConv((-1, -1), hidden)
        self.mp1_a2f = SAGEConv((-1, -1), hidden)
        self.mp2_f2ar = SAGEConv((-1, -1), hidden)
        self.mp2_ar2f = SAGEConv((-1, -1), hidden)
        self.att_vector = nn.Parameter(torch.randn(1, hidden))
        self.lin = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](x_dict[node])) for node in x_dict.keys()}

        edge_f_a = edge_index_dict.get(('file', 'calls', 'api'), torch.empty((2, 0), dtype=torch.long))
        if edge_f_a.numel() > 0:
            edge_a_f = edge_f_a[[1, 0]]
            h_api = F.relu(self.mp1_f2a((x['file'], x['api']), edge_f_a))
            h_file_mp1 = F.relu(self.mp1_a2f((h_api, x['file']), edge_a_f))
        else:
            h_file_mp1 = x['file']

        edge_f_ar = edge_index_dict.get(('file', 'runs_on', 'arch'), torch.empty((2, 0), dtype=torch.long))
        if edge_f_ar.numel() > 0:
            edge_ar_f = edge_f_ar[[1, 0]]
            h_arch = F.relu(self.mp2_f2ar((x['file'], x['arch']), edge_f_ar))
            h_file_mp2 = F.relu(self.mp2_ar2f((h_arch, x['file']), edge_ar_f))
        else:
            h_file_mp2 = x['file']

        stacked = torch.stack([h_file_mp1, h_file_mp2], dim=1)
        beta = F.softmax((stacked * self.att_vector).sum(dim=-1), dim=1).unsqueeze(-1)
        return self.lin((stacked * beta).sum(dim=1))

class RHSAGE(nn.Module):
    def __init__(self, hidden, out, metadata):
        super().__init__()
        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in metadata[0]})
        self.conv1 = HeteroConv({et: SAGEConv(hidden, hidden) for et in metadata[1]}, aggr='sum')
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = HeteroConv({et: SAGEConv(hidden, hidden) for et in metadata[1]}, aggr='sum')
        self.norm2 = nn.LayerNorm(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out)
        )

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items()}
        out1 = self.conv1(x, edge_index_dict)
        x = {k: self.norm1(F.relu(out1.get(k, x[k])) + x[k]) for k in x.keys()}
        out2 = self.conv2(x, edge_index_dict)
        x = {k: self.norm2(F.relu(out2.get(k, x[k])) + x[k]) for k in x.keys()}
        return self.classifier(x['file'])

class DRHGT(nn.Module):
    def __init__(self, hidden, out, heads, layers, metadata):
        super().__init__()
        self.hidden = hidden
        self.half_hidden = hidden // 2
        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in metadata[0]})
        self.hgt_convs = nn.ModuleList([HGTConv(hidden, hidden, metadata, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.fc_malware = nn.Linear(hidden, self.half_hidden)
        self.fc_arch = nn.Linear(hidden, self.half_hidden)
        self.skip_proj = nn.Linear(hidden, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, out)
        )

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items()}
        h_init = x['file']
        for conv in self.hgt_convs:
            out = conv(x, edge_index_dict)
            x = {k: out.get(k, x[k]) for k in x.keys()}

        h_file = self.norm(x['file']) + F.relu(self.skip_proj(h_init))
        z_mal = F.relu(self.fc_malware(h_file))
        z_ar = F.relu(self.fc_arch(h_file))
        return self.classifier(torch.cat([z_mal, z_ar], dim=-1))

    def compute_orthogonality_loss(self):
        w_mal = F.normalize(self.fc_malware.weight, p=2, dim=1)
        w_ar = F.normalize(self.fc_arch.weight, p=2, dim=1)
        prod = torch.matmul(w_mal, w_ar.T)
        return torch.norm(prod, p='fro') ** 2 / (self.half_hidden * self.half_hidden)

def predict_loao_hin(model, df_test, X_te_hin_file, apis, archs, device, batch_size=128):
    model.eval()
    graphs = build_loao_independent_ego_graphs(df_test, X_te_hin_file, apis, archs)
    loader = PyGDataLoader(graphs, batch_size=batch_size, shuffle=False)
    all_probs, all_preds = [], []

    if device.type == 'cuda': torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            all_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            all_preds.extend(out.argmax(dim=1).cpu().numpy())

    if device.type == 'cuda': torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t_start) * 1000 / len(df_test)

    return np.array(all_preds), np.array(all_probs), infer_ms

def run_scenario_2(
    dataset_path=DEFAULT_DATASET_PATH,
    base_save_dir=DEFAULT_BASE_SAVE_DIR,
    seeds=DEFAULT_SEEDS,
    architectures=TARGET_ARCHITECTURES
):
    os.makedirs(base_save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Starting Scenario 2 LOAO Benchmark on {len(seeds)} seeds across architectures: {architectures}")

    df_raw = pd.read_csv(dataset_path)
    df_raw['arch'] = df_raw['arch'].apply(standardize_elf_architecture)

    all_loao_results = []

    for held_out_arch in architectures:
        print(f"\nEvaluating LOAO Target: [Held-out Architecture: {held_out_arch}]")

        for seed in seeds:
            print(f"Seed {seed} | LOAO Architecture: {held_out_arch}...")
            seed_dir = os.path.join(base_save_dir, f"arch_{held_out_arch}", f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            seed_everything(seed)

            df_tr, df_id_te, df_ood_te = split_loao_dataset(df_raw, held_out_arch, seed=seed)
            y_tr, y_id_te, y_ood_te = df_tr['label'].values, df_id_te['label'].values, df_ood_te['label'].values

            (x_tr_core, x_id_core, x_ood_core), (X_tr_stat_full, X_id_stat_full, X_ood_stat_full), _, (X_tr_hin_file, X_id_hin_file, X_ood_hin_file), (X_tr_seq, X_id_seq, X_ood_seq), prep = extract_loao_streamlined_features(df_tr, df_id_te, df_ood_te)

            train_g, apis, archs = build_loao_hetero_graph_train(df_tr, X_tr_hin_file)
            train_g = train_g.to(device)

            artifacts_to_save = {
                'preprocessors': prep,
                'api_to_id': apis,
                'le_arch': archs,
                'X_tr_hin_file': X_tr_hin_file
            }
            joblib.dump(artifacts_to_save, os.path.join(seed_dir, "preprocessors.joblib"))
            np.save(os.path.join(seed_dir, "X_tr_hin_file.npy"), X_tr_hin_file)
            x_res, y_res = RandomUnderSampler(random_state=seed).fit_resample(x_tr_core, y_tr)
            lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=seed, n_jobs=-1, verbose=-1)
            lgbm.fit(np.asarray(x_res), y_res)

            joblib.dump(lgbm, os.path.join(seed_dir, "baseline1_lgbm.joblib"))

            pred_id_lgbm = lgbm.predict(x_id_core)
            pred_ood_lgbm = lgbm.predict(x_ood_core)
            pb_id_lgbm = lgbm.predict_proba(x_id_core)[:, 1]
            pb_ood_lgbm = lgbm.predict_proba(x_ood_core)[:, 1]

            id_acc_lgbm = accuracy_score(y_id_te, pred_id_lgbm)
            ood_acc_lgbm = accuracy_score(y_ood_te, pred_ood_lgbm)
            id_f1_lgbm = f1_score(y_id_te, pred_id_lgbm, average='binary', zero_division=0)
            ood_f1_lgbm = f1_score(y_ood_te, pred_ood_lgbm, average='binary', zero_division=0)

            try: id_pr_lgbm = average_precision_score(y_id_te, pb_id_lgbm)
            except: id_pr_lgbm = 0.5
            try: ood_pr_lgbm = average_precision_score(y_ood_te, pb_ood_lgbm)
            except: ood_pr_lgbm = 0.5

            all_loao_results.append({
                'Held-out Arch': held_out_arch, 'Model': 'Baseline 1: LightGBM', 'Seed': seed,
                'ID Accuracy': id_acc_lgbm, 'OOD Accuracy': ood_acc_lgbm,
                'ID Binary F1': id_f1_lgbm, 'OOD Binary F1': ood_f1_lgbm,
                'ID PR-AUC': id_pr_lgbm, 'OOD PR-AUC': ood_pr_lgbm,
                'Accuracy Drop': id_acc_lgbm - ood_acc_lgbm
            })
            dataset_gru = SequenceDenseDataset(X_tr_seq, y=y_tr)
            loader_gru = torch.utils.data.DataLoader(dataset_gru, batch_size=128, shuffle=True)
            bigru_net = Baseline2_BiGRU(vocab_size=prep['vocab_size'], embed_dim=128, hidden_dim=128).to(device)
            opt_gru = torch.optim.Adam(bigru_net.parameters(), lr=0.001)

            bigru_net.train()
            for epoch in range(15):
                for batch in loader_gru:
                    opt_gru.zero_grad()
                    out = bigru_net(batch['x_seq'].to(device))
                    loss = F.cross_entropy(out, batch['y'].to(device))
                    loss.backward()
                    opt_gru.step()

            torch.save(bigru_net.state_dict(), os.path.join(seed_dir, "baseline2_bigru.pt"))

            pred_id_b2, pb_id_b2, _ = predict_bigru_batched(bigru_net, X_id_seq, device, batch_size=256)
            pred_ood_b2, pb_ood_b2, _ = predict_bigru_batched(bigru_net, X_ood_seq, device, batch_size=256)

            id_acc_b2 = accuracy_score(y_id_te, pred_id_b2)
            ood_acc_b2 = accuracy_score(y_ood_te, pred_ood_b2)
            id_f1_b2 = f1_score(y_id_te, pred_id_b2, average='binary', zero_division=0)
            ood_f1_b2 = f1_score(y_ood_te, pred_ood_b2, average='binary', zero_division=0)

            try: id_pr_b2 = average_precision_score(y_id_te, pb_id_b2)
            except: id_pr_b2 = 0.5
            try: ood_pr_b2 = average_precision_score(y_ood_te, pb_ood_b2)
            except: ood_pr_b2 = 0.5

            all_loao_results.append({
                'Held-out Arch': held_out_arch, 'Model': 'Baseline 2: Dynamic Bi-GRU', 'Seed': seed,
                'ID Accuracy': id_acc_b2, 'OOD Accuracy': ood_acc_b2,
                'ID Binary F1': id_f1_b2, 'OOD Binary F1': ood_f1_b2,
                'ID PR-AUC': id_pr_b2, 'OOD PR-AUC': ood_pr_b2,
                'Accuracy Drop': id_acc_b2 - ood_acc_b2
            })
            clear_gpu_memory()
            dataset_fusion = SequenceDenseDataset(X_tr_seq, X_tr_stat_full, y_tr)
            loader_fusion = torch.utils.data.DataLoader(dataset_fusion, batch_size=128, shuffle=True)
            fusion_net = Baseline3_EndToEndHybrid(vocab_size=prep['vocab_size'], dense_dim=X_tr_stat_full.shape[1]).to(device)
            opt_fusion = torch.optim.Adam(fusion_net.parameters(), lr=0.001)

            fusion_net.train()
            for epoch in range(15):
                for batch in loader_fusion:
                    opt_fusion.zero_grad()
                    out = fusion_net(batch['x_seq'].to(device), batch['x_dense'].to(device))
                    loss = F.cross_entropy(out, batch['y'].to(device))
                    loss.backward()
                    opt_fusion.step()

            torch.save(fusion_net.state_dict(), os.path.join(seed_dir, "baseline3_fusion.pt"))

            pred_id, pb_id_f, _ = predict_fusion_batched(fusion_net, X_id_seq, X_id_stat_full, device, batch_size=256)
            pred_ood, pb_ood_f, _ = predict_fusion_batched(fusion_net, X_ood_seq, X_ood_stat_full, device, batch_size=256)

            try: id_pr_f = average_precision_score(y_id_te, pb_id_f)
            except: id_pr_f = 0.5
            try: ood_pr_f = average_precision_score(y_ood_te, pb_ood_f)
            except: ood_pr_f = 0.5

            all_loao_results.append({
                'Held-out Arch': held_out_arch, 'Model': 'Baseline 3: End-to-End Fusion', 'Seed': seed,
                'ID Accuracy': accuracy_score(y_id_te, pred_id), 'OOD Accuracy': accuracy_score(y_ood_te, pred_ood),
                'ID Binary F1': f1_score(y_id_te, pred_id, average='binary', zero_division=0),
                'OOD Binary F1': f1_score(y_ood_te, pred_ood, average='binary', zero_division=0),
                'ID PR-AUC': id_pr_f, 'OOD PR-AUC': ood_pr_f,
                'Accuracy Drop': accuracy_score(y_id_te, pred_id) - accuracy_score(y_ood_te, pred_ood)
            })
            clear_gpu_memory()
            x_tr_g, edge_tr_g, y_tr_g = build_homogeneous_file_graph(X_tr_hin_file, y_tr, k=5)
            x_tr_g, edge_tr_g, y_tr_g = x_tr_g.to(device), edge_tr_g.to(device), y_tr_g.to(device)

            b4_model = Baseline4_HomogeneousGCN(in_channels=X_tr_hin_file.shape[1], hidden_dim=64).to(device)
            opt_b4 = torch.optim.Adam(b4_model.parameters(), lr=0.01)

            b4_model.train()
            for epoch in range(100):
                opt_b4.zero_grad()
                out = b4_model(x_tr_g, edge_tr_g)
                loss = F.cross_entropy(out, y_tr_g)
                loss.backward()
                opt_b4.step()

            torch.save(b4_model.state_dict(), os.path.join(seed_dir, "baseline4_gcn.pt"))

            p4_id, pb4_id, _ = evaluate_inductive_homogeneous_gcn(b4_model, X_tr_hin_file, X_id_hin_file, device, k=5)
            p4_ood, pb4_ood, _ = evaluate_inductive_homogeneous_gcn(b4_model, X_tr_hin_file, X_ood_hin_file, device, k=5)

            id_acc_gcn = accuracy_score(y_id_te, p4_id)
            ood_acc_gcn = accuracy_score(y_ood_te, p4_ood)
            id_f1_gcn = f1_score(y_id_te, p4_id, average='binary', zero_division=0)
            ood_f1_gcn = f1_score(y_ood_te, p4_ood, average='binary', zero_division=0)

            try: id_pr_gcn = average_precision_score(y_id_te, pb4_id)
            except: id_pr_gcn = 0.5
            try: ood_pr_gcn = average_precision_score(y_ood_te, pb4_ood)
            except: ood_pr_gcn = 0.5

            all_loao_results.append({
                'Held-out Arch': held_out_arch, 'Model': 'Baseline 4: Homogeneous GCN', 'Seed': seed,
                'ID Accuracy': id_acc_gcn, 'OOD Accuracy': ood_acc_gcn,
                'ID Binary F1': id_f1_gcn, 'OOD Binary F1': ood_f1_gcn,
                'ID PR-AUC': id_pr_gcn, 'OOD PR-AUC': ood_pr_gcn,
                'Accuracy Drop': id_acc_gcn - ood_acc_gcn
            })
            clear_gpu_memory()
            num_neg = (y_tr == 0).sum()
            num_pos = (y_tr == 1).sum()
            w_neg = len(y_tr) / (2.0 * max(num_neg, 1))
            w_pos = len(y_tr) / (2.0 * max(num_pos, 1))
            class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float).to(device)
            def eval_loao_hin(model_fn, model_name, weight_filename):
                m = model_fn(train_g.metadata()).to(device)
                with torch.no_grad():
                    _ = m(train_g.x_dict, train_g.edge_index_dict)
                opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-4)

                m.train()
                for epoch in range(100):
                    opt.zero_grad()
                    out = m(train_g.x_dict, train_g.edge_index_dict)
                    loss = F.cross_entropy(out, train_g['file'].y, weight=class_weights)

                    if hasattr(m, 'compute_orthogonality_loss'):
                        loss = loss + 0.01 * m.compute_orthogonality_loss()

                    loss.backward()
                    opt.step()
                    sched.step()

                torch.save(m.state_dict(), os.path.join(seed_dir, weight_filename))

                pred_id, pb_id, _ = predict_loao_hin(m, df_id_te, X_id_hin_file, apis, archs, device)
                pred_ood, pb_ood, _ = predict_loao_hin(m, df_ood_te, X_ood_hin_file, apis, archs, device)

                id_acc = accuracy_score(y_id_te, pred_id)
                ood_acc = accuracy_score(y_ood_te, pred_ood)
                id_f1 = f1_score(y_id_te, pred_id, average='binary', zero_division=0)
                ood_f1 = f1_score(y_ood_te, pred_ood, average='binary', zero_division=0)

                try: id_pr = average_precision_score(y_id_te, pb_id)
                except: id_pr = 0.5
                try: ood_pr = average_precision_score(y_ood_te, pb_ood)
                except: ood_pr = 0.5

                all_loao_results.append({
                    'Held-out Arch': held_out_arch, 'Model': model_name, 'Seed': seed,
                    'ID Accuracy': id_acc, 'OOD Accuracy': ood_acc,
                    'ID Binary F1': id_f1, 'OOD Binary F1': ood_f1,
                    'ID PR-AUC': id_pr, 'OOD PR-AUC': ood_pr,
                    'Accuracy Drop': id_acc - ood_acc
                })
                clear_gpu_memory()

            eval_loao_hin(lambda *args, **kwargs: BaseHeteroSAGE(hidden=64, out=2, metadata=args[0] if len(args) > 0 else kwargs.get("metadata")), 'Base HeteroSAGE', 'hin_rgcn.pt')
            eval_loao_hin(lambda *args, **kwargs: HGT(hidden=64, out=2, heads=4, layers=2, metadata=args[0] if len(args) > 0 else kwargs.get("metadata")), 'HIN: HGT', 'hin_hgt.pt')
            eval_loao_hin(lambda *args, **kwargs: GATModel(hidden=64, out=2, heads=4, metadata=args[0] if len(args) > 0 else kwargs.get("metadata")), 'HIN: GAT', 'hin_gat.pt')
            eval_loao_hin(lambda *args, **kwargs: HANModel(hidden=64, out=2, heads=4, metadata=args[0] if len(args) > 0 else kwargs.get("metadata")), 'HIN: HAN', 'hin_han.pt')
            eval_loao_hin(lambda *args, **kwargs: MetaPathSAGE(hidden=64, out=2, metadata=args[0] if len(args) > 0 else kwargs.get("metadata")), 'Meta-path SAGE', 'hin_magnn.pt')
            eval_loao_hin(lambda *args, **kwargs: RHSAGE(hidden=64, out=2, metadata=args[0] if len(args) > 0 else kwargs.get("metadata")), 'RH-SAGE', 'hin_improved_rgcn.pt')
            eval_loao_hin(lambda *args, **kwargs: DRHGT(hidden=64, out=2, heads=4, layers=2, metadata=args[0] if len(args) > 0 else kwargs.get("metadata")), 'DR-HGT', 'hin_improved_hgt.pt')

    df_loao_raw = pd.DataFrame(all_loao_results)
    raw_loao_path = os.path.join(base_save_dir, "scenario_2_loao_raw_results_10seeds.csv")
    df_loao_raw.to_csv(raw_loao_path, index=False)

    summary_loao = []
    for (arch, model_name), group in df_loao_raw.groupby(['Held-out Arch', 'Model'], sort=False):
        summary_loao.append({
            'Held-out Architecture': arch,
            'Model Architecture': model_name,
            'ID Accuracy': f"{group['ID Accuracy'].mean():.4f} ± {group['ID Accuracy'].std():.4f}",
            'OOD Accuracy': f"{group['OOD Accuracy'].mean():.4f} ± {group['OOD Accuracy'].std():.4f}",
            'ID Binary F1': f"{group['ID Binary F1'].mean():.4f} ± {group['ID Binary F1'].std():.4f}",
            'OOD Binary F1': f"{group['OOD Binary F1'].mean():.4f} ± {group['OOD Binary F1'].std():.4f}",
            'ID PR-AUC': f"{group['ID PR-AUC'].mean():.4f} ± {group['ID PR-AUC'].std():.4f}",
            'OOD PR-AUC': f"{group['OOD PR-AUC'].mean():.4f} ± {group['OOD PR-AUC'].std():.4f}",
            'Accuracy Drop': f"{group['Accuracy Drop'].mean():.4f} ± {group['Accuracy Drop'].std():.4f}"
        })

    df_loao_sum = pd.DataFrame(summary_loao)
    sum_loao_path = os.path.join(base_save_dir, "scenario_2_loao_summary_results_10seeds.csv")
    df_loao_sum.to_csv(sum_loao_path, index=False)

    print("\nLOAO Summary Results:")
    print(df_loao_sum.to_string(index=False))
    print(f"\nSaved summary results to: {sum_loao_path}")

    return df_loao_sum, df_loao_raw

if __name__ == "__main__":
    run_scenario_2()