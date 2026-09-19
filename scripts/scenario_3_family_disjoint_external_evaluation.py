"""Evaluation of saved family-disjoint models on the later-period corpus."""

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

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DEFAULT_SEEDS = [10, 42, 100, 314, 555, 777, 999, 1234, 2024, 2026]
DEFAULT_NEW_DATASET_PATH = '/content/drive/MyDrive/dataset/HIN/final_dataset_arch_13286.csv'
DEFAULT_SAVED_MODELS_DIR = '/content/drive/MyDrive/dataset/HIN/results_scenario_3_lofo'
DEFAULT_OUTPUT_DIR = '/content/drive/MyDrive/dataset/HIN/results_scenario_3_lofo/results_test_new_dataset'

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

def extract_test_features_with_preprocessors(df_test, prep):
    df_te = df_test.copy()
    if 'arch' in df_te.columns:
        df_te['arch'] = df_te['arch'].apply(standardize_elf_architecture)
    else:
        df_te['arch'] = 'Other'

    core_static_cols = prep.get('core_static_cols', prep.get('base1_static_cols', ['mean_entropy', 'max_entropy', 'min_entropy']))
    for c in core_static_cols:
        if c not in df_te.columns:
            df_te[c] = 0.0

    scaler_core = prep.get('scaler_core_stat', prep.get('scaler_base1', None))
    if scaler_core is not None:
        x_te_core_stat = scaler_core.transform(df_te[core_static_cols].apply(pd.to_numeric, errors='coerce').fillna(0))
    else:
        x_te_core_stat = df_te[core_static_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values

    op_col = 'opcode_counts' if 'opcode_counts' in df_te.columns else 'opcode_sequence'
    if 'tfidf_op' in prep:
        x_te_op = prep['tfidf_op'].transform(df_te.get(op_col, pd.Series([""]*len(df_te))).fillna("").apply(parse_text_sequence)).toarray()
        X_te_stat_full = np.hstack([x_te_core_stat, x_te_op])
    else:
        X_te_stat_full = x_te_core_stat

    top_dyn_cols = prep.get('top_dyn_cols', prep.get('core_5_static_cols', []))
    if len(top_dyn_cols) > 0 and 'scaler_top_dyn' in prep:
        for c in top_dyn_cols:
            if c not in df_te.columns:
                df_te[c] = 0.0
        x_te_top_dyn = prep['scaler_top_dyn'].transform(df_te[top_dyn_cols].apply(pd.to_numeric, errors='coerce').fillna(0))
    elif len(top_dyn_cols) > 0 and 'scaler_core_5' in prep:
        for c in top_dyn_cols:
            if c not in df_te.columns:
                df_te[c] = 0.0
        x_te_top_dyn = prep['scaler_core_5'].transform(df_te[top_dyn_cols].apply(pd.to_numeric, errors='coerce').fillna(0))
    else:
        x_te_top_dyn = np.zeros((len(df_te), 0))

    X_te_dyn_stream = x_te_top_dyn

    if X_te_dyn_stream.shape[1] > 0:
        X_te_hin_file = np.hstack([X_te_stat_full, X_te_dyn_stream])
    else:
        X_te_hin_file = X_te_stat_full

    api_col = 'api_sequence' if 'api_sequence' in df_te.columns else 'api_list'
    seq_te = df_te.get(api_col, pd.Series([""]*len(df_te))).fillna("").apply(parse_text_sequence).tolist()
    if 'tokenizer' in prep:
        X_te_seq = prep['tokenizer'].texts_to_sequences(seq_te, max_len=200)
    else:
        X_te_seq = np.zeros((len(df_te), 200), dtype=np.int64)

    return x_te_core_stat, X_te_stat_full, X_te_dyn_stream, X_te_hin_file, X_te_seq

def build_independent_ego_graphs(df_test, X_te_hin_file, api_to_id, le_arch):
    def parse_api(s):
        try: return ast.literal_eval(s) if isinstance(s, str) and s.startswith('[') else ([] if pd.isna(s) else str(s).split())
        except: return []

    api_col = 'api_sequence' if 'api_sequence' in df_test.columns else 'api_list'
    api_seqs = df_test.get(api_col, pd.Series([[]]*len(df_test))).apply(parse_api)
    known_archs = set(le_arch.classes_) if hasattr(le_arch, 'classes_') else set()
    arch_ids = df_test['arch'].apply(lambda x: le_arch.transform([str(x)])[0] if str(x) in known_archs else -1).values

    num_apis = len(api_to_id) if len(api_to_id) > 0 else 1
    num_archs = len(le_arch.classes_) if hasattr(le_arch, 'classes_') else 1

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

def get_dummy_hin_metadata():
    data = HeteroData()
    data['file'].x = torch.empty((1, 1))
    data['api'].x = torch.empty((1, 1))
    data['arch'].x = torch.empty((1, 1))
    data['file', 'calls', 'api'].edge_index = torch.empty((2, 0), dtype=torch.long)
    data['file', 'runs_on', 'arch'].edge_index = torch.empty((2, 0), dtype=torch.long)
    return T.ToUndirected()(data).metadata()

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

def predict_inductive_homogeneous_gcn(model, X_tr, X_te, device, k=5):
    model.eval()
    k_actual = min(k, len(X_tr))
    nn_support = NearestNeighbors(n_neighbors=k_actual, n_jobs=-1).fit(X_tr)
    adj_bipartite = nn_support.kneighbors_graph(X_te, mode='connectivity').tocoo()

    N_tr = len(X_tr)
    src = torch.tensor(adj_bipartite.col, dtype=torch.long)
    dst = torch.tensor(adj_bipartite.row + N_tr, dtype=torch.long)

    edge_index_sym = torch.cat([
        torch.stack([src, dst], dim=0),
        torch.stack([dst, src], dim=0)
    ], dim=1).to(device)

    X_all = torch.tensor(np.vstack([X_tr, X_te]), dtype=torch.float).to(device)

    if device.type == 'cuda': torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        out_all = model(X_all, edge_index_sym)
        out_te = out_all[N_tr:]
        probs = F.softmax(out_te, dim=1)[:, 1].cpu().numpy()
        preds = out_te.argmax(dim=1).cpu().numpy()

    if device.type == 'cuda': torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t_start) * 1000 / len(X_te)

    return preds, probs, infer_ms

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

def calculate_metrics(y_true, preds, probs, name, seed, infer_ms=0.0):
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
        'PR-AUC': pr_auc, 'ROC-AUC': roc_auc, 'Infer Time (ms/sample)': infer_ms
    }

def run_test_pipeline(
    new_dataset_path=DEFAULT_NEW_DATASET_PATH,
    saved_models_dir=DEFAULT_SAVED_MODELS_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    seeds=DEFAULT_SEEDS
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Running Inference Pipeline ({len(seeds)} seeds) | Device: {device}")

    if not os.path.exists(new_dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {new_dataset_path}")

    df_test_raw = pd.read_csv(new_dataset_path)
    if 'label' not in df_test_raw.columns:
        df_test_raw['label'] = 0

    if 'arch' in df_test_raw.columns:
        df_test_raw['arch'] = df_test_raw['arch'].apply(standardize_elf_architecture)

    y_te = df_test_raw['label'].values
    all_test_results = []
    metadata = get_dummy_hin_metadata()

    for seed in seeds:
        print(f"\n--- Inference Seed = {seed} ---")

        seed_dir = os.path.join(saved_models_dir, f'seed_{seed}')
        if not os.path.exists(seed_dir):
            print(f"Saved weights not found for seed {seed}. Skipping.")
            continue

        seed_everything(seed)

        prep = joblib.load(os.path.join(seed_dir, 'preprocessors.joblib'))
        mappings = joblib.load(os.path.join(seed_dir, 'graph_mappings.joblib'))

        tr_hin_path = os.path.join(seed_dir, 'X_tr_hin_file.npy')
        X_tr_hin_file = np.load(tr_hin_path) if os.path.exists(tr_hin_path) else None

        apis = mappings['api_to_id']
        archs = mappings['le_arch']

        (x_te_core, X_te_stat_full, X_te_dyn_stream, X_te_hin_file, X_te_seq) = extract_test_features_with_preprocessors(df_test_raw, prep)
        sample_batch = build_independent_ego_graphs(df_test_raw.iloc[:1], X_te_hin_file[:1], apis, archs)[0].to(device)
        lgbm_path = os.path.join(seed_dir, 'baseline1_lgbm.joblib')
        if os.path.exists(lgbm_path):
            lgbm = joblib.load(lgbm_path)
            t_inf = time.perf_counter()
            p1 = lgbm.predict(x_te_core)
            pr1 = lgbm.predict_proba(x_te_core)[:, 1]
            infer_ms = (time.perf_counter() - t_inf) * 1000 / len(y_te)
            all_test_results.append(calculate_metrics(y_te, p1, pr1, 'Baseline 1: LightGBM', seed, infer_ms))
        gru_path = os.path.join(seed_dir, 'baseline2_bigru.pth')
        if os.path.exists(gru_path):
            vocab_sz = prep.get('vocab_size', 5000)
            bigru = Baseline2_BiGRU(vocab_size=vocab_sz, embed_dim=128, hidden_dim=128).to(device)
            bigru.load_state_dict(torch.load(gru_path, map_location=device))
            p2, pr2, infer_ms_gru = predict_bigru_batched(bigru, X_te_seq, device, batch_size=256)
            all_test_results.append(calculate_metrics(y_te, p2, pr2, 'Baseline 2: Bi-GRU', seed, infer_ms_gru))
            clear_gpu_memory()
        fusion_path = os.path.join(seed_dir, 'baseline3_fusion.pth')
        if os.path.exists(fusion_path):
            vocab_sz = prep.get('vocab_size', 5000)
            fusion_net = Baseline3_EndToEndHybrid(vocab_size=vocab_sz, dense_dim=X_te_stat_full.shape[1]).to(device)
            fusion_net.load_state_dict(torch.load(fusion_path, map_location=device))
            p3, pr3, infer_ms_fusion = predict_fusion_batched(fusion_net, X_te_seq, X_te_stat_full, device, batch_size=256)
            all_test_results.append(calculate_metrics(y_te, p3, pr3, 'Baseline 3: End-to-End Fusion', seed, infer_ms_fusion))
            clear_gpu_memory()
        gcn_path = os.path.join(seed_dir, 'baseline4_gcn.pth')
        if os.path.exists(gcn_path) and X_tr_hin_file is not None:
            b4_model = Baseline4_HomogeneousGCN(in_channels=X_te_hin_file.shape[1], hidden_dim=64).to(device)
            b4_model.load_state_dict(torch.load(gcn_path, map_location=device))
            p4, pr4, infer_ms_gcn = predict_inductive_homogeneous_gcn(b4_model, X_tr_hin_file, X_te_hin_file, device, k=5)
            all_test_results.append(calculate_metrics(y_te, p4, pr4, 'Baseline 4: Homogeneous GCN', seed, infer_ms_gcn))
            clear_gpu_memory()

        def test_family_hin_model(model_fn, model_filename, display_name):
            weight_path = os.path.join(seed_dir, f"{model_filename}.pth")
            if not os.path.exists(weight_path):
                return
            model = model_fn(metadata).to(device)
            with torch.no_grad():
                _ = model(sample_batch.x_dict, sample_batch.edge_index_dict)
            model.load_state_dict(torch.load(weight_path, map_location=device))
            preds, probs, infer_ms_hin = predict_lofo_hin(model, df_test_raw, X_te_hin_file, apis, archs, device)
            res = calculate_metrics(y_te, preds, probs, display_name, seed, infer_ms_hin)
            all_test_results.append(res)
            print(f"Result -> {display_name} | Acc: {res['Accuracy']:.4f} | F1: {res['Binary F1']:.4f} | MCC: {res['MCC']:.4f}")
            clear_gpu_memory()

        test_family_hin_model(lambda meta: BaseHeteroSAGE(hidden=64, out=2, metadata=meta), 'HIN_RGCN', 'Base HeteroSAGE')
        test_family_hin_model(lambda meta: HGT(hidden=64, out=2, heads=4, layers=2, metadata=meta), 'HIN_HGT', 'HIN: HGT')
        test_family_hin_model(lambda meta: GATModel(hidden=64, out=2, heads=4, metadata=meta), 'HIN_GAT', 'HIN: GAT')
        test_family_hin_model(lambda meta: HANModel(hidden=64, out=2, heads=4, metadata=meta), 'HIN_HAN', 'HIN: HAN')
        test_family_hin_model(lambda meta: MetaPathSAGE(hidden=64, out=2, metadata=meta), 'HIN_MAGNN', 'Meta-path SAGE')

        test_family_hin_model(lambda meta: RHSAGE(hidden=64, out=2, metadata=meta), 'IMPROVED_RGCN', 'RH-SAGE')
        test_family_hin_model(lambda meta: DRHGT(hidden=64, out=2, heads=4, layers=2, metadata=meta), 'IMPROVED_HGT', 'DR-HGT')

    if not all_test_results:
        print("No valid results computed.")
        return None, None

    df_test_raw_res = pd.DataFrame(all_test_results)
    raw_save_path = os.path.join(output_dir, "f1dataset_scenario_3_raw_results_10seeds.csv")
    df_test_raw_res.to_csv(raw_save_path, index=False)

    summary_test = []
    metric_cols = ['Accuracy', 'Precision', 'Recall', 'Binary F1', 'Macro F1', 'MCC', 'PR-AUC', 'ROC-AUC', 'Infer Time (ms/sample)']

    for model_name, group in df_test_raw_res.groupby('Model Name', sort=False):
        row = {'Model Architecture': model_name}
        for m in metric_cols:
            mean_val = group[m].mean()
            std_val = group[m].std()
            std_val = 0.0 if np.isnan(std_val) else std_val
            row[f'{m} (Mean ± Std)'] = f"{mean_val:.4f} ± {std_val:.4f}"
        summary_test.append(row)

    df_test_summary = pd.DataFrame(summary_test)
    sum_save_path = os.path.join(output_dir, "f1dataset_scenario_3_summary_results_10seeds.csv")
    df_test_summary.to_csv(sum_save_path, index=False)

    print("\n" + "="*80)
    print("Inference Summary Results")
    print("="*80)
    print(df_test_summary.to_string(index=False))

    return df_test_summary, df_test_raw_res

if __name__ == "__main__":
    run_test_pipeline()