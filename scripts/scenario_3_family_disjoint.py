"""Family-disjoint malware detection experiments."""

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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DEFAULT_SEEDS = [10, 42, 100, 314, 555, 777, 999, 1234, 2024, 2026]
DEFAULT_DATASET_PATH = '/content/drive/MyDrive/dataset/dataset.csv'
DEFAULT_BASE_SAVE_DIR = '/content/drive/MyDrive/dataset/HIN/results_scenario_3_lofo'

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

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import DataLoader as PyGDataLoader
    import torch_geometric.transforms as T
    from torch_geometric.nn import HeteroConv, SAGEConv, Linear, HGTConv, GATConv, HANConv, GCNConv
except ImportError:
    cuda_ver = ('cu' + torch.version.cuda.replace('.', '')) if torch.cuda.is_available() else 'cpu'
    torch_ver = torch.__version__.split('+')[0]
    os.system(f"pip install -q pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-{torch_ver}+{cuda_ver}.html")
    os.system("pip install -q torch_geometric lightgbm scikit-learn pandas imbalanced-learn")
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import DataLoader as PyGDataLoader
    import torch_geometric.transforms as T
    from torch_geometric.nn import HeteroConv, SAGEConv, Linear, HGTConv, GATConv, HANConv, GCNConv

class PyTorchApiTokenizer:
    def __init__(self, max_vocab=5000):
        self.max_vocab = max_vocab
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    @property
    def word_index(self): return self.word2idx

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
            if len(ids) < max_len: ids = ids + [0] * (max_len - len(ids))
            seqs.append(ids)
        return np.array(seqs, dtype=np.int64)

class SequenceDenseDataset(torch.utils.data.Dataset):
    def __init__(self, x_seq, x_dense=None, y=None):
        self.x_seq = torch.tensor(x_seq, dtype=torch.long)
        self.x_dense = torch.tensor(x_dense, dtype=torch.float) if x_dense is not None else None
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self): return len(self.x_seq)
    def __getitem__(self, idx):
        item = {'x_seq': self.x_seq[idx]}
        if self.x_dense is not None: item['x_dense'] = self.x_dense[idx]
        if self.y is not None: item['y'] = self.y[idx]
        return item

def standardize_elf_architecture(arch_str):
    arch_clean = str(arch_str).strip().lower()
    if 'arm' in arch_clean or 'aarch64' in arch_clean: return 'ARM'
    elif 'mips' in arch_clean: return 'MIPS'
    elif 'powerpc' in arch_clean or 'ppc' in arch_clean: return 'PowerPC'
    elif '64' in arch_clean or 'amd64' in arch_clean or 'x86_64' in arch_clean: return 'x64'
    elif '86' in arch_clean or 'i386' in arch_clean or 'i686' in arch_clean: return 'x86'
    return 'Other'

def parse_text_sequence(val):
    if isinstance(val, str):
        if val.startswith('{'):
            try:
                d = ast.literal_eval(val)
                return " ".join([f"{k} "*min(int(v), 20) for k, v in d.items()])
            except: pass
        return val
    elif isinstance(val, (list, tuple)):
        return " ".join(map(str, val))
    return ""

def split_family_disjoint_dataset(df_raw, seed=42):
    df_clean = df_raw.copy()
    df_clean['arch'] = df_clean['arch'].apply(standardize_elf_architecture)

    if 'malware_family' not in df_clean.columns:
        df_clean['malware_family'] = 'Benign'
    df_clean['malware_family'] = df_clean['malware_family'].fillna('Benign')

    groups = []
    for idx, row in df_clean.iterrows():
        if row['label'] == 0:
            groups.append(f"benign_{idx}")
        else:
            groups.append(str(row['malware_family']))
    groups = np.array(groups)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df_clean, df_clean['label'], groups=groups))

    df_train = df_clean.iloc[train_idx].reset_index(drop=True)
    df_test = df_clean.iloc[test_idx].reset_index(drop=True)

    return df_train, df_test

def extract_lofo_streamlined_features(df_train, df_test):
    core_static_cols = ['mean_entropy', 'max_entropy', 'min_entropy', 'file_size_mb', 'total_opcodes']
    for c in core_static_cols:
        if c not in df_train.columns: df_train[c] = 0.0
        if c not in df_test.columns: df_test[c] = 0.0

    scaler_core_stat = StandardScaler()
    x_tr_core_stat = scaler_core_stat.fit_transform(df_train[core_static_cols].apply(pd.to_numeric, errors='coerce').fillna(0))
    x_te_core_stat = scaler_core_stat.transform(df_test[core_static_cols].apply(pd.to_numeric, errors='coerce').fillna(0))

    op_col = 'opcode_counts' if 'opcode_counts' in df_train.columns else 'opcode_sequence'
    tfidf_op = TfidfVectorizer(max_features=100)
    x_tr_op = tfidf_op.fit_transform(df_train.get(op_col, pd.Series([""]*len(df_train))).fillna("").apply(parse_text_sequence)).toarray()
    x_te_op = tfidf_op.transform(df_test.get(op_col, pd.Series([""]*len(df_test))).fillna("").apply(parse_text_sequence)).toarray()

    X_tr_stat_full = np.hstack([x_tr_core_stat, x_tr_op])
    X_te_stat_full = np.hstack([x_te_core_stat, x_te_op])

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
        if c not in df_test.columns: df_test[c] = 0.0

    scaler_top_dyn = StandardScaler()
    x_tr_top_dyn = scaler_top_dyn.fit_transform(df_train[top_dyn_cols].apply(pd.to_numeric, errors='coerce').fillna(0))
    x_te_top_dyn = scaler_top_dyn.transform(df_test[top_dyn_cols].apply(pd.to_numeric, errors='coerce').fillna(0))

    api_col = 'api_sequence' if 'api_sequence' in df_train.columns else 'api_list'
    tfidf_api = TfidfVectorizer(max_features=100)
    x_tr_api = tfidf_api.fit_transform(df_train.get(api_col, pd.Series([""]*len(df_train))).fillna("").apply(parse_text_sequence)).toarray()
    x_te_api = tfidf_api.transform(df_test.get(api_col, pd.Series([""]*len(df_test))).fillna("").apply(parse_text_sequence)).toarray()

    X_tr_dyn_stream = x_tr_top_dyn
    X_te_dyn_stream = x_te_top_dyn

    X_tr_hin_file = np.hstack([X_tr_stat_full, X_tr_dyn_stream])
    X_te_hin_file = np.hstack([X_te_stat_full, X_te_dyn_stream])

    tokenizer = PyTorchApiTokenizer(max_vocab=5000)
    seq_tr = df_train.get(api_col, pd.Series([""]*len(df_train))).fillna("").apply(parse_text_sequence).tolist()
    seq_te = df_test.get(api_col, pd.Series([""]*len(df_test))).fillna("").apply(parse_text_sequence).tolist()
    tokenizer.fit_on_texts(seq_tr)
    X_tr_seq = tokenizer.texts_to_sequences(seq_tr, max_len=200)
    X_te_seq = tokenizer.texts_to_sequences(seq_te, max_len=200)

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
    return (x_tr_core_stat, x_te_core_stat), (X_tr_stat_full, X_te_stat_full), (X_tr_dyn_stream, X_te_dyn_stream), (X_tr_hin_file, X_te_hin_file), (X_tr_seq, X_te_seq), preprocessors

def build_hetero_graph_train(df_train, X_tr_hin_file):
    def parse_api(s):
        try: return ast.literal_eval(s) if isinstance(s, str) and s.startswith('[') else ([] if pd.isna(s) else str(s).split())
        except: return []

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

def build_independent_ego_graphs(df_test, X_te_hin_file, api_to_id, le_arch):
    def parse_api(s):
        try: return ast.literal_eval(s) if isinstance(s, str) and s.startswith('[') else ([] if pd.isna(s) else str(s).split())
        except: return []

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
        data['file'].y = torch.tensor([df_test['label'].iloc[i]], dtype=torch.long) if 'label' in df_test.columns else torch.tensor([0], dtype=torch.long)

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
        self.mp1_f2a = SAGEConv((-1, -1), hidden); self.mp1_a2f = SAGEConv((-1, -1), hidden)
        self.mp2_f2ar = SAGEConv((-1, -1), hidden); self.mp2_ar2f = SAGEConv((-1, -1), hidden)
        self.att_vector = nn.Parameter(torch.randn(1, hidden))
        self.lin = Linear(hidden, out)
    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](x_dict[node])) for node in x_dict.keys()}

        edge_f_a = edge_index_dict.get(('file', 'calls', 'api'), torch.empty((2, 0), dtype=torch.long))
        if edge_f_a.numel() > 0:
            edge_a_f = edge_f_a[[1, 0]]
            h_api = F.relu(self.mp1_f2a((x['file'], x['api']), edge_f_a))
            h_file_mp1 = F.relu(self.mp1_a2f((h_api, x['file']), edge_a_f))
        else: h_file_mp1 = x['file']

        edge_f_ar = edge_index_dict.get(('file', 'runs_on', 'arch'), torch.empty((2, 0), dtype=torch.long))
        if edge_f_ar.numel() > 0:
            edge_ar_f = edge_f_ar[[1, 0]]
            h_arch = F.relu(self.mp2_f2ar((x['file'], x['arch']), edge_f_ar))
            h_file_mp2 = F.relu(self.mp2_ar2f((h_arch, x['file']), edge_ar_f))
        else: h_file_mp2 = x['file']

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
        loss_orth = torch.norm(prod, p='fro') ** 2 / (self.half_hidden * self.half_hidden)
        return loss_orth

def predict_lofo_hin(model, df_test, X_te_hin_file, apis, archs, device, batch_size=128):
    model.eval()
    graphs = build_independent_ego_graphs(df_test, X_te_hin_file, apis, archs)
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

def calculate_metrics(y_true, preds, probs, name, seed, train_ms=0.0, infer_ms=0.0, mem_mb=0.0):
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1_bin = f1_score(y_true, preds, average='binary', zero_division=0)
    f1_macro = f1_score(y_true, preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, preds)

    try: pr_auc = average_precision_score(y_true, probs)
    except: pr_auc = 0.5

    try: roc_auc = roc_auc_score(y_true, probs)
    except: roc_auc = 0.5

    return {
        'Model Name': name, 'Seed': seed, 'Accuracy': acc, 'Precision': prec,
        'Recall': rec, 'Binary F1': f1_bin, 'Macro F1': f1_macro, 'MCC': mcc,
        'PR-AUC': pr_auc, 'ROC-AUC': roc_auc, 'Train Time (ms/epoch)': train_ms,
        'Infer Time (ms/sample)': infer_ms, 'Peak GPU VRAM (MB)': mem_mb
    }

def run_scenario_3(
    dataset_path=DEFAULT_DATASET_PATH,
    base_save_dir=DEFAULT_BASE_SAVE_DIR,
    seeds=DEFAULT_SEEDS
):
    os.makedirs(base_save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Running Scenario 3 ({len(seeds)} seeds) | Device: {device}")

    df_raw = pd.read_csv(dataset_path)
    all_lofo_results = []

    for seed in seeds:
        print(f"\n--- Execution Seed = {seed} ---")

        seed_dir = os.path.join(base_save_dir, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        seed_everything(seed)

        df_tr, df_te = split_family_disjoint_dataset(df_raw, seed=seed)
        y_tr, y_te = df_tr['label'].values, df_te['label'].values
        print(f"Train samples: {len(df_tr)} | Test samples: {len(df_te)}")

        (x_tr_core, x_te_core), (X_tr_stat_full, X_te_stat_full), (X_tr_dyn_stream, X_te_dyn_stream), (X_tr_hin_file, X_te_hin_file), (X_tr_seq, X_te_seq), prep = extract_lofo_streamlined_features(df_tr, df_te)

        train_g, apis, archs = build_hetero_graph_train(df_tr, X_tr_hin_file)
        train_g = train_g.to(device)

        joblib.dump(prep, os.path.join(seed_dir, 'preprocessors.joblib'))
        joblib.dump({'api_to_id': apis, 'le_arch': archs}, os.path.join(seed_dir, 'graph_mappings.joblib'))
        np.save(os.path.join(seed_dir, 'X_tr_hin_file.npy'), X_tr_hin_file)
        print("Training Baseline 1: LightGBM...")
        x_res, y_res = RandomUnderSampler(random_state=seed).fit_resample(x_tr_core, y_tr)
        t0 = time.perf_counter()
        lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=seed, n_jobs=-1, verbose=-1)
        lgbm.fit(np.asarray(x_res), y_res)
        train_ms = (time.perf_counter() - t0) * 1000 / 300

        joblib.dump(lgbm, os.path.join(seed_dir, 'baseline1_lgbm.joblib'))

        t_inf = time.perf_counter()
        p1 = lgbm.predict(x_te_core)
        pr1 = lgbm.predict_proba(x_te_core)[:, 1]
        infer_ms = (time.perf_counter() - t_inf) * 1000 / len(y_te)
        all_lofo_results.append(calculate_metrics(y_te, p1, pr1, 'Baseline 1: LightGBM', seed, train_ms, infer_ms))
        print("Training Baseline 2: Bi-GRU...")
        dataset_seq_tr = SequenceDenseDataset(X_tr_seq, y=y_tr)
        loader_seq_tr = torch.utils.data.DataLoader(dataset_seq_tr, batch_size=128, shuffle=True)
        bigru = Baseline2_BiGRU(vocab_size=prep['vocab_size'], embed_dim=128, hidden_dim=128).to(device)
        opt_gru = torch.optim.Adam(bigru.parameters(), lr=0.001)

        if device.type == 'cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        bigru.train()
        for epoch in range(15):
            for batch in loader_seq_tr:
                opt_gru.zero_grad()
                out = bigru(batch['x_seq'].to(device))
                loss = F.cross_entropy(out, batch['y'].to(device))
                loss.backward()
                opt_gru.step()
        if device.type == 'cuda': torch.cuda.synchronize()
        train_ms = (time.perf_counter() - t0) * 1000 / 15
        peak_mem = torch.cuda.max_memory_allocated(device)/(1024**2) if device.type == 'cuda' else 0.0

        torch.save(bigru.state_dict(), os.path.join(seed_dir, 'baseline2_bigru.pth'))

        p2, pr2, infer_ms_gru = predict_bigru_batched(bigru, X_te_seq, device, batch_size=256)
        all_lofo_results.append(calculate_metrics(y_te, p2, pr2, 'Baseline 2: Bi-GRU', seed, train_ms, infer_ms_gru, peak_mem))
        clear_gpu_memory()
        print("Training Baseline 3: End-to-End Fusion...")
        dataset_fusion = SequenceDenseDataset(X_tr_seq, X_tr_stat_full, y_tr)
        loader_fusion = torch.utils.data.DataLoader(dataset_fusion, batch_size=128, shuffle=True)
        fusion_net = Baseline3_EndToEndHybrid(vocab_size=prep['vocab_size'], dense_dim=X_tr_stat_full.shape[1]).to(device)
        opt_fusion = torch.optim.Adam(fusion_net.parameters(), lr=0.001)

        if device.type == 'cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        fusion_net.train()
        for epoch in range(20):
            for batch in loader_fusion:
                opt_fusion.zero_grad()
                out = fusion_net(batch['x_seq'].to(device), batch['x_dense'].to(device))
                loss = F.cross_entropy(out, batch['y'].to(device))
                loss.backward()
                opt_fusion.step()
        if device.type == 'cuda': torch.cuda.synchronize()
        train_ms = (time.perf_counter() - t0) * 1000 / 20
        peak_mem = torch.cuda.max_memory_allocated(device)/(1024**2) if device.type == 'cuda' else 0.0

        torch.save(fusion_net.state_dict(), os.path.join(seed_dir, 'baseline3_fusion.pth'))

        p3, pr3, infer_ms_fusion = predict_fusion_batched(fusion_net, X_te_seq, X_te_stat_full, device, batch_size=256)
        all_lofo_results.append(calculate_metrics(y_te, p3, pr3, 'Baseline 3: End-to-End Fusion', seed, train_ms, infer_ms_fusion, peak_mem))
        clear_gpu_memory()
        print("Training Baseline 4: Homogeneous GCN...")
        x_tr_g, edge_tr_g, y_tr_g = build_homogeneous_file_graph(X_tr_hin_file, y_tr, k=5)
        x_tr_g, edge_tr_g, y_tr_g = x_tr_g.to(device), edge_tr_g.to(device), y_tr_g.to(device)

        b4_model = Baseline4_HomogeneousGCN(in_channels=X_tr_hin_file.shape[1], hidden_dim=64).to(device)
        opt_b4 = torch.optim.Adam(b4_model.parameters(), lr=0.01)

        if device.type == 'cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        b4_model.train()
        for epoch in range(100):
            opt_b4.zero_grad()
            out = b4_model(x_tr_g, edge_tr_g)
            loss = F.cross_entropy(out, y_tr_g)
            loss.backward()
            opt_b4.step()
        if device.type == 'cuda': torch.cuda.synchronize()
        train_ms = (time.perf_counter() - t0) * 1000 / 100
        peak_mem = torch.cuda.max_memory_allocated(device)/(1024**2) if device.type == 'cuda' else 0.0

        torch.save(b4_model.state_dict(), os.path.join(seed_dir, 'baseline4_gcn.pth'))

        p4, pr4, infer_ms_gcn = evaluate_inductive_homogeneous_gcn(b4_model, X_tr_hin_file, X_te_hin_file, device, k=5)
        all_lofo_results.append(calculate_metrics(y_te, p4, pr4, 'Baseline 4: Homogeneous GCN', seed, train_ms, infer_ms_gcn, peak_mem))
        clear_gpu_memory()

        num_neg = (y_tr == 0).sum()
        num_pos = (y_tr == 1).sum()
        w_neg = len(y_tr) / (2.0 * max(num_neg, 1))
        w_pos = len(y_tr) / (2.0 * max(num_pos, 1))
        class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float).to(device)

        def train_and_eval_family_hin(model_fn, model_filename, display_name):
            print(f"Training {display_name}...")
            model = model_fn(train_g.metadata()).to(device)
            with torch.no_grad():
                _ = model(train_g.x_dict, train_g.edge_index_dict)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)

            if device.type == 'cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                out = model(train_g.x_dict, train_g.edge_index_dict)
                loss = F.cross_entropy(out, train_g['file'].y, weight=class_weights)

                if hasattr(model, 'compute_orthogonality_loss'):
                    loss = loss + 0.01 * model.compute_orthogonality_loss()

                loss.backward()
                optimizer.step()
                scheduler.step()
            if device.type == 'cuda': torch.cuda.synchronize()
            train_ms = (time.perf_counter() - t0) * 1000 / 100
            peak_mem = torch.cuda.max_memory_allocated(device)/(1024**2) if device.type == 'cuda' else 0.0

            torch.save(model.state_dict(), os.path.join(seed_dir, f"{model_filename}.pth"))

            preds, probs, infer_ms_hin = predict_lofo_hin(model, df_te, X_te_hin_file, apis, archs, device)

            res = calculate_metrics(y_te, preds, probs, display_name, seed, train_ms, infer_ms_hin, peak_mem)
            all_lofo_results.append(res)
            print(f"Result -> {display_name} | Acc: {res['Accuracy']:.4f} | F1: {res['Binary F1']:.4f} | MCC: {res['MCC']:.4f}")
            clear_gpu_memory()

        train_and_eval_family_hin(lambda *args, **kwargs: BaseHeteroSAGE(hidden=64, out=2, metadata=args[0] if len(args) > 0 else kwargs.get("metadata", kwargs.get("meta"))), 'HIN_RGCN', 'Base HeteroSAGE')
        train_and_eval_family_hin(lambda *args, **kwargs: HGT(hidden=64, out=2, heads=4, layers=2, metadata=args[0] if len(args) > 0 else kwargs.get("metadata", kwargs.get("meta"))), 'HIN_HGT', 'HIN: HGT')
        train_and_eval_family_hin(lambda *args, **kwargs: GATModel(hidden=64, out=2, heads=4, metadata=args[0] if len(args) > 0 else kwargs.get("metadata", kwargs.get("meta"))), 'HIN_GAT', 'HIN: GAT')
        train_and_eval_family_hin(lambda *args, **kwargs: HANModel(hidden=64, out=2, heads=4, metadata=args[0] if len(args) > 0 else kwargs.get("metadata", kwargs.get("meta"))), 'HIN_HAN', 'HIN: HAN')
        train_and_eval_family_hin(lambda *args, **kwargs: MetaPathSAGE(hidden=64, out=2, metadata=args[0] if len(args) > 0 else kwargs.get("metadata", kwargs.get("meta"))), 'HIN_MAGNN', 'Meta-path SAGE')

        train_and_eval_family_hin(lambda *args, **kwargs: RHSAGE(hidden=64, out=2, metadata=args[0] if len(args) > 0 else kwargs.get("metadata", kwargs.get("meta"))), 'IMPROVED_RGCN', 'RH-SAGE')
        train_and_eval_family_hin(lambda *args, **kwargs: DRHGT(hidden=64, out=2, heads=4, layers=2, metadata=args[0] if len(args) > 0 else kwargs.get("metadata", kwargs.get("meta"))), 'IMPROVED_HGT', 'DR-HGT')

    df_lofo_raw = pd.DataFrame(all_lofo_results)
    raw_lofo_path = os.path.join(base_save_dir, "scenario_3_lofo_raw_results_10seeds.csv")
    df_lofo_raw.to_csv(raw_lofo_path, index=False)

    summary_lofo = []
    metric_cols = ['Accuracy', 'Precision', 'Recall', 'Binary F1', 'Macro F1', 'MCC', 'PR-AUC', 'ROC-AUC', 'Train Time (ms/epoch)', 'Infer Time (ms/sample)', 'Peak GPU VRAM (MB)']

    for model_name, group in df_lofo_raw.groupby('Model Name', sort=False):
        row = {'Model Architecture': model_name}
        for m in metric_cols:
            mean_val = group[m].mean()
            std_val = group[m].std()
            row[f'{m} (Mean ± Std)'] = f"{mean_val:.4f} ± {std_val:.4f}"
        summary_lofo.append(row)

    df_lofo_sum = pd.DataFrame(summary_lofo)
    sum_lofo_path = os.path.join(base_save_dir, "scenario_3_lofo_summary_results_10seeds.csv")
    df_lofo_sum.to_csv(sum_lofo_path, index=False)

    print("\n" + "="*80)
    print("Family-disjoint summary")
    print("="*80)
    print(df_lofo_sum.to_string(index=False))

    return df_lofo_sum, df_lofo_raw

if __name__ == "__main__":
    run_scenario_3()