# -*- coding: utf-8 -*-
"""Ablation experiments for the heterogeneous graph models."""

import os
import ast
import time
import random
import gc
import joblib
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, average_precision_score
from google.colab import drive, files

warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DEFAULT_DATASET_PATH = '/content/drive/MyDrive/dataset/dataset.csv'
DEFAULT_BASE_SAVE_DIR = '/content/drive/MyDrive/dataset/HIN/ablation_saved_models'
DEFAULT_SEEDS = [10, 42, 100, 314, 555, 777, 999, 1234, 2024, 2026]

ABLATION_MODES = ['Full_Graph', 'No_API_Node', 'No_Arch_Node', 'Static_Opcode_Only']

MODEL_DISPLAY_NAMES = {
    'RGCN': 'Base HeteroSAGE',
    'HGT': 'HGT',
    'GAT': 'GAT',
    'HAN': 'HAN',
    'MAGNN': 'Meta-path SAGE',
    'R2-RGCN': 'RH-SAGE',
    'DR-HGT': 'DR-HGT',
}
KEEP_OPCODE_TFIDF = True


def install_dependencies():
    print("Installing PyTorch Geometric dependencies...")
    cuda_ver = 'cu' + torch.version.cuda.replace('.', '') if torch.cuda.is_available() else 'cpu'
    torch_ver = torch.__version__.split('+')[0]
    os.system(f"pip install -q pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-{torch_ver}+{cuda_ver}.html")
    os.system("pip install -q torch_geometric scikit-learn pandas openpyxl imbalanced-learn")


try:
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import DataLoader as PyGDataLoader
    import torch_geometric.transforms as T
    from torch_geometric.nn import HeteroConv, SAGEConv, Linear, HGTConv, GATConv, HANConv
except ImportError:
    install_dependencies()
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import DataLoader as PyGDataLoader
    import torch_geometric.transforms as T
    from torch_geometric.nn import HeteroConv, SAGEConv, Linear, HGTConv, GATConv, HANConv


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def parse_text_sequence(val):
    if isinstance(val, str):
        if val.startswith('{'):
            try:
                d = ast.literal_eval(val)
                return " ".join([f"{k} " * min(int(v), 20) for k, v in d.items()])
            except Exception:
                pass
        return val
    elif isinstance(val, (list, tuple)):
        return " ".join(map(str, val))
    return ""


def extract_4component_features(df_train, df_test):
    core_cols = ['mean_entropy', 'max_entropy', 'min_entropy', 'file_size_mb', 'total_opcodes']
    for c in core_cols:
        if c not in df_train.columns:
            df_train[c] = 0.0
        if c not in df_test.columns:
            df_test[c] = 0.0

    scaler_core = StandardScaler()
    x_tr_core = scaler_core.fit_transform(df_train[core_cols].apply(pd.to_numeric, errors='coerce').fillna(0))
    x_te_core = scaler_core.transform(df_test[core_cols].apply(pd.to_numeric, errors='coerce').fillna(0))

    op_col = 'opcode_counts' if 'opcode_counts' in df_train.columns else 'opcode_sequence'
    tfidf_op = TfidfVectorizer(max_features=100)
    x_tr_op = tfidf_op.fit_transform(df_train.get(op_col, pd.Series([""] * len(df_train))).fillna("").apply(parse_text_sequence)).toarray()
    x_te_op = tfidf_op.transform(df_test.get(op_col, pd.Series([""] * len(df_test))).fillna("").apply(parse_text_sequence)).toarray()

    dyn_candidates = [
        'socket_calls', 'connect_calls', 'send_calls', 'fork_calls',
        'exec_calls', 'execve_calls', 'sys_calls', 'file_ops',
        'net_ops', 'proc_ops', 'total_calls', 'unique_calls'
    ]
    top_dyn_cols = [c for c in dyn_candidates if c in df_train.columns] or ['total_calls']

    for c in top_dyn_cols:
        if c not in df_train.columns:
            df_train[c] = 0.0
        if c not in df_test.columns:
            df_test[c] = 0.0

    scaler_dyn = StandardScaler()
    x_tr_dyn = scaler_dyn.fit_transform(df_train[top_dyn_cols].apply(pd.to_numeric, errors='coerce').fillna(0))
    x_te_dyn = scaler_dyn.transform(df_test[top_dyn_cols].apply(pd.to_numeric, errors='coerce').fillna(0))

    api_col = 'api_sequence' if 'api_sequence' in df_train.columns else 'api_list'
    tfidf_api = TfidfVectorizer(max_features=100)
    x_tr_api = tfidf_api.fit_transform(df_train.get(api_col, pd.Series([""] * len(df_train))).fillna("").apply(parse_text_sequence)).toarray()
    x_te_api = tfidf_api.transform(df_test.get(api_col, pd.Series([""] * len(df_test))).fillna("").apply(parse_text_sequence)).toarray()

    X_tr_file = np.hstack([x_tr_core, x_tr_op, x_tr_dyn])
    X_te_file = np.hstack([x_te_core, x_te_op, x_te_dyn])

    preprocessors = {
        'scaler_core': scaler_core,
        'tfidf_op': tfidf_op,
        'scaler_dyn': scaler_dyn,
        'tfidf_api': tfidf_api,
        'core_cols': core_cols,
        'top_dyn_cols': top_dyn_cols,
        'feature_dim': X_tr_file.shape[1],
        'core_dim': x_tr_core.shape[1],
        'op_dim': x_tr_op.shape[1],
        'dyn_dim': x_tr_dyn.shape[1],
        'api_dim': x_tr_api.shape[1]
    }
    return X_tr_file, X_te_file, preprocessors


def apply_static_opcode_only_ablation(x_matrix, keep_opcode_tfidf=True):
    x_ablated = x_matrix.copy()
    x_ablated[:, 0:3] = 0.0
    if not keep_opcode_tfidf:
        x_ablated[:, 5:105] = 0.0
    dyn_start_idx = 5 + 100
    x_ablated[:, dyn_start_idx:] = 0.0
    return x_ablated


def is_static_opcode_mode(mode_str):
    return mode_str in ['Static_Opcode_Only', 'No_Dynamic_No_Entropy', 'No_File_Features']


def build_hetero_graph_train(df_train, X_tr_file, ablation_mode='Full_Graph'):
    def parse_api(s):
        try:
            return ast.literal_eval(s) if isinstance(s, str) and s.startswith('[') else ([] if pd.isna(s) else str(s).split())
        except Exception:
            return []

    api_col = 'api_sequence' if 'api_sequence' in df_train.columns else 'api_list'
    api_seqs = df_train.get(api_col, pd.Series([[]] * len(df_train))).apply(parse_api)

    all_apis = sorted(list({api for seq in api_seqs for api in seq}))
    api_to_id = {name: i for i, name in enumerate(all_apis)}

    le_arch = LabelEncoder()
    arch_ids = le_arch.fit_transform(df_train['arch'].astype(str))

    data = HeteroData()

    if is_static_opcode_mode(ablation_mode):
        x_tr_processed = apply_static_opcode_only_ablation(X_tr_file, keep_opcode_tfidf=KEEP_OPCODE_TFIDF)
        data['file'].x = torch.tensor(x_tr_processed, dtype=torch.float)
    else:
        data['file'].x = torch.tensor(X_tr_file, dtype=torch.float)

    data['file'].y = torch.tensor(df_train['label'].values, dtype=torch.long)
    data['api'].x = torch.eye(max(len(api_to_id), 1), dtype=torch.float)
    data['arch'].x = torch.eye(len(le_arch.classes_), dtype=torch.float)

    if ablation_mode != 'No_API_Node' and len(api_to_id) > 0:
        src_f, dst_a = [], []
        for f_idx, seq in enumerate(api_seqs):
            for api in seq:
                if api in api_to_id:
                    src_f.append(f_idx)
                    dst_a.append(api_to_id[api])
        data['file', 'calls', 'api'].edge_index = torch.tensor(np.array([src_f, dst_a]), dtype=torch.long) if src_f else torch.empty((2, 0), dtype=torch.long)
    else:
        data['file', 'calls', 'api'].edge_index = torch.empty((2, 0), dtype=torch.long)

    if ablation_mode != 'No_Arch_Node':
        valid_arch = arch_ids != -1
        src_ar = np.arange(len(df_train))[valid_arch]
        dst_ar = arch_ids[valid_arch]
        data['file', 'runs_on', 'arch'].edge_index = torch.tensor(np.array([src_ar, dst_ar]), dtype=torch.long) if valid_arch.any() else torch.empty((2, 0), dtype=torch.long)
    else:
        data['file', 'runs_on', 'arch'].edge_index = torch.empty((2, 0), dtype=torch.long)

    return T.ToUndirected()(data), api_to_id, le_arch


def build_independent_ego_graphs(df_test, X_te_file, api_to_id, le_arch, ablation_mode='Full_Graph'):
    def parse_api(s):
        try:
            return ast.literal_eval(s) if isinstance(s, str) and s.startswith('[') else ([] if pd.isna(s) else str(s).split())
        except Exception:
            return []

    api_col = 'api_sequence' if 'api_sequence' in df_test.columns else 'api_list'
    api_seqs = df_test.get(api_col, pd.Series([[]] * len(df_test))).apply(parse_api)

    known_archs = set(le_arch.classes_)
    arch_ids = df_test['arch'].apply(lambda x: le_arch.transform([str(x)])[0] if str(x) in known_archs else -1).values

    num_apis = max(len(api_to_id), 1)
    num_archs = len(le_arch.classes_)

    eye_api = torch.eye(num_apis, dtype=torch.float)
    eye_arch = torch.eye(num_archs, dtype=torch.float)

    graphs = []
    for i in range(len(df_test)):
        data = HeteroData()

        if is_static_opcode_mode(ablation_mode):
            x_te_sample = apply_static_opcode_only_ablation(X_te_file[i:i + 1], keep_opcode_tfidf=KEEP_OPCODE_TFIDF)
            data['file'].x = torch.tensor(x_te_sample, dtype=torch.float)
        else:
            data['file'].x = torch.tensor(X_te_file[i:i + 1], dtype=torch.float)

        data['file'].y = torch.tensor([df_test['label'].iloc[i]], dtype=torch.long) if 'label' in df_test.columns else torch.tensor([0], dtype=torch.long)

        if ablation_mode != 'No_API_Node' and len(api_to_id) > 0:
            seq = api_seqs.iloc[i]
            valid_api_ids = sorted(list({api_to_id[api] for api in seq if api in api_to_id}))
            if len(valid_api_ids) > 0:
                data['api'].x = eye_api[valid_api_ids]
                local_indices = list(range(len(valid_api_ids)))
                data['file', 'calls', 'api'].edge_index = torch.tensor(np.array([[0] * len(valid_api_ids), local_indices]), dtype=torch.long)
            else:
                data['api'].x = torch.empty((0, num_apis), dtype=torch.float)
                data['file', 'calls', 'api'].edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            data['api'].x = torch.empty((0, num_apis), dtype=torch.float)
            data['file', 'calls', 'api'].edge_index = torch.empty((2, 0), dtype=torch.long)

        if ablation_mode != 'No_Arch_Node' and arch_ids[i] != -1:
            data['arch'].x = eye_arch[arch_ids[i:i + 1]]
            data['file', 'runs_on', 'arch'].edge_index = torch.tensor(np.array([[0], [0]]), dtype=torch.long)
        else:
            data['arch'].x = torch.empty((0, num_archs), dtype=torch.float)
            data['file', 'runs_on', 'arch'].edge_index = torch.empty((2, 0), dtype=torch.long)

        graphs.append(T.ToUndirected()(data))

    return graphs


class BaseHeteroSAGE(nn.Module):
    def __init__(self, hidden, out, metadata):
        super().__init__()
        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in metadata[0]})
        self.conv1 = HeteroConv({et: SAGEConv(hidden, hidden) for et in metadata[1]}, aggr='sum')
        self.conv2 = HeteroConv({et: SAGEConv(hidden, hidden) for et in metadata[1]}, aggr='sum')
        self.lin = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items()}
        out1 = self.conv1(x, edge_index_dict)
        x = {k: F.relu(out1.get(k, x[k])) for k in x.keys()}
        out2 = self.conv2(x, edge_index_dict)
        x = {k: F.relu(out2.get(k, x[k])) for k in x.keys()}
        return self.lin(x['file'])


class HGT(nn.Module):
    def __init__(self, hidden, out, heads, layers, metadata):
        super().__init__()
        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in metadata[0]})
        self.convs = nn.ModuleList([HGTConv(hidden, hidden, metadata, heads) for _ in range(layers)])
        self.lin = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items()}
        for conv in self.convs:
            out = conv(x, edge_index_dict)
            x = {k: out.get(k, x[k]) for k in x.keys()}
        return self.lin(x['file'])


class GATModel(nn.Module):
    def __init__(self, hidden, out, heads, metadata):
        super().__init__()
        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in metadata[0]})
        self.conv1 = HeteroConv({et: GATConv(hidden, hidden // heads, heads=heads, add_self_loops=False) for et in metadata[1]}, aggr='sum')
        self.conv2 = HeteroConv({et: GATConv(hidden, hidden // heads, heads=heads, add_self_loops=False) for et in metadata[1]}, aggr='sum')
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
        self.mp1_f2a = SAGEConv((hidden, hidden), hidden)
        self.mp1_a2f = SAGEConv((hidden, hidden), hidden)
        self.mp2_f2ar = SAGEConv((hidden, hidden), hidden)
        self.mp2_ar2f = SAGEConv((hidden, hidden), hidden)
        self.att_vector = nn.Parameter(torch.randn(1, hidden))
        self.lin = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](x_dict[node])) for node in x_dict.keys()}

        edge_f_a = edge_index_dict.get(('file', 'calls', 'api'), torch.empty((2, 0), dtype=torch.long))
        if edge_f_a.numel() > 0:
            edge_a_f = edge_index_dict.get(('api', 'rev_calls', 'file'), edge_f_a[[1, 0]])
            h_api = F.relu(self.mp1_f2a((x['file'], x['api']), edge_f_a))
            h_file_mp1 = F.relu(self.mp1_a2f((h_api, x['file']), edge_a_f))
        else:
            h_file_mp1 = x['file']

        edge_f_ar = edge_index_dict.get(('file', 'runs_on', 'arch'), torch.empty((2, 0), dtype=torch.long))
        if edge_f_ar.numel() > 0:
            edge_ar_f = edge_index_dict.get(('arch', 'rev_runs_on', 'file'), edge_f_ar[[1, 0]])
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


def predict_independent_ego_graphs(model, test_graphs, device, batch_size=128):
    model.eval()
    loader = PyGDataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    all_preds, all_probs = [], []

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            all_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            all_preds.extend(out.argmax(dim=1).cpu().numpy())

    if device.type == 'cuda':
        torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t_start) * 1000 / len(test_graphs)

    return np.array(all_preds), np.array(all_probs), infer_ms


def run_hin_ablation_benchmark(
    dataset_path=DEFAULT_DATASET_PATH,
    base_save_dir=DEFAULT_BASE_SAVE_DIR,
    seeds=DEFAULT_SEEDS
):
    os.makedirs(base_save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(dataset_path):
        uploaded = files.upload()
        dataset_path = list(uploaded.keys())[0]

    df_raw = pd.read_csv(dataset_path)

    model_factories = [
        ("BaseHeteroSAGE", lambda meta: BaseHeteroSAGE(hidden=64, out=2, metadata=meta)),
        ("HGT", lambda meta: HGT(hidden=64, out=2, heads=4, layers=2, metadata=meta)),
        ("GAT", lambda meta: GATModel(hidden=64, out=2, heads=4, metadata=meta)),
        ("HAN", lambda meta: HANModel(hidden=64, out=2, heads=4, metadata=meta)),
        ("MAGNN", lambda meta: MetaPathSAGE(hidden=64, out=2, metadata=meta)),
        ("R2-RGCN", lambda meta: RHSAGE(hidden=64, out=2, metadata=meta)),
        ("DR-HGT", lambda meta: DRHGT(hidden=64, out=2, heads=4, layers=2, metadata=meta))
    ]

    all_benchmark_rows = []
    print(f"Starting HIN ablation benchmark on {len(seeds)} seeds with 7 models (Device: {device})")

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        seed_dir = os.path.join(base_save_dir, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        seed_everything(seed)

        df_raw['stratify_col'] = df_raw['label'].astype(str) + "_" + df_raw['arch'].astype(str)
        counts = df_raw['stratify_col'].value_counts()
        valid_idx = df_raw['stratify_col'].isin(counts[counts > 1].index)
        df_valid = df_raw[valid_idx].copy()

        df_tr, df_te = train_test_split(df_valid, test_size=0.2, stratify=df_valid['stratify_col'], random_state=seed)
        df_tr = df_tr.drop(columns=['stratify_col']).reset_index(drop=True)
        df_te = df_te.drop(columns=['stratify_col']).reset_index(drop=True)
        y_te = df_te['label'].values

        X_tr_file, X_te_file, preprocessors = extract_4component_features(df_tr, df_te)
        joblib.dump(preprocessors, os.path.join(seed_dir, 'preprocessors.joblib'))

        num_neg, num_pos = (df_tr['label'] == 0).sum(), (df_tr['label'] == 1).sum()
        w_neg = len(df_tr) / (2.0 * max(num_neg, 1))
        w_pos = len(df_tr) / (2.0 * max(num_pos, 1))
        class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float).to(device)

        for mode in ABLATION_MODES:
            print(f"  Mode: {mode}")
            mode_dir = os.path.join(seed_dir, mode)
            os.makedirs(mode_dir, exist_ok=True)

            train_g, api_to_id, le_arch = build_hetero_graph_train(df_tr, X_tr_file, ablation_mode=mode)
            train_g = train_g.to(device)

            joblib.dump({'api_to_id': api_to_id, 'le_arch': le_arch}, os.path.join(mode_dir, 'mappings.joblib'))
            test_graphs = build_independent_ego_graphs(df_te, X_te_file, api_to_id, le_arch, ablation_mode=mode)

            for model_name, model_fn in model_factories:
                clear_gpu_memory()

                model = model_fn(train_g.metadata()).to(device)

                with torch.no_grad():
                    _ = model(train_g.x_dict, train_g.edge_index_dict)

                optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
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

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                train_epoch_ms = ((time.perf_counter() - t0) / 100) * 1000
                peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0.0

                model_save_path = os.path.join(mode_dir, f"{model_name}.pth")
                torch.save(model.state_dict(), model_save_path)

                preds, probs, infer_ms = predict_independent_ego_graphs(model, test_graphs, device, batch_size=128)

                acc = accuracy_score(y_te, preds)
                f1_macro = f1_score(y_te, preds, average='macro', zero_division=0)
                mcc = matthews_corrcoef(y_te, preds)
                try:
                    pr_auc = average_precision_score(y_te, probs)
                except Exception:
                    pr_auc = 0.5

                all_benchmark_rows.append({
                    'Seed': seed,
                    'Ablation Mode': mode,
                    'Model': MODEL_DISPLAY_NAMES.get(model_name, model_name),
                    'Accuracy': acc,
                    'Macro F1-Score': f1_macro,
                    'MCC': mcc,
                    'PR-AUC (AP)': pr_auc,
                    'Train Time/Epoch (ms)': train_epoch_ms,
                    'Inference Time/Sample (ms)': infer_ms,
                    'Peak GPU Memory (MB)': peak_mem
                })

                print(f"    [{mode}] {MODEL_DISPLAY_NAMES.get(model_name, model_name):16s} | Acc: {acc:.4f} | F1: {f1_macro:.4f} | MCC: {mcc:.4f} | PR-AUC: {pr_auc:.4f} | Infer: {infer_ms:.4f} ms")

    df_all = pd.DataFrame(all_benchmark_rows)
    raw_csv = os.path.join(base_save_dir, "hin_ablation_10seeds_raw.csv")
    df_all.to_csv(raw_csv, index=False)

    summary_list = []
    group_cols = ['Ablation Mode', 'Model']
    metric_cols = ['Accuracy', 'Macro F1-Score', 'MCC', 'PR-AUC (AP)', 'Train Time/Epoch (ms)', 'Inference Time/Sample (ms)', 'Peak GPU Memory (MB)']

    for (ablation_mode, model_name), grp in df_all.groupby(group_cols, sort=False):
        row = {'Ablation Mode': ablation_mode, 'Model': model_name}
        for col in metric_cols:
            row[col] = grp[col].mean()
        summary_list.append(row)

    df_summary = pd.DataFrame(summary_list)
    sum_csv = os.path.join(base_save_dir, "hin_ablation_10seeds_summary.csv")
    df_summary.to_csv(sum_csv, index=False)

    excel_path = os.path.join(base_save_dir, "HIN_Ablation_Report_10Seeds.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary_10Seeds_Mean', index=False)
        df_all.to_excel(writer, sheet_name='Raw_10Seeds_Data', index=False)

    print("\nBenchmark completed.")
    print(df_summary.to_string(index=False))

    return df_summary, df_all


def predict_unseen_dataset(
    new_csv_path,
    model_name="R2-RGCN",
    ablation_mode="Full_Graph",
    seed=42,
    base_save_dir=DEFAULT_BASE_SAVE_DIR
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_dir = os.path.join(base_save_dir, f'seed_{seed}')
    mode_dir = os.path.join(seed_dir, ablation_mode)

    if not os.path.exists(mode_dir) and is_static_opcode_mode(ablation_mode):
        for alt in ['Static_Opcode_Only', 'No_File_Features', 'No_Dynamic_No_Entropy']:
            alt_dir = os.path.join(seed_dir, alt)
            if os.path.exists(alt_dir):
                mode_dir = alt_dir
                break

    if not os.path.exists(os.path.join(seed_dir, 'preprocessors.joblib')):
        raise FileNotFoundError(f"Preprocessor parameters not found at {seed_dir}.")

    prep = joblib.load(os.path.join(seed_dir, 'preprocessors.joblib'))
    mappings = joblib.load(os.path.join(mode_dir, 'mappings.joblib'))
    api_to_id = mappings['api_to_id']
    le_arch = mappings['le_arch']

    df_new = pd.read_csv(new_csv_path)

    for c in prep['core_cols']:
        if c not in df_new.columns:
            df_new[c] = 0.0
    for c in prep['top_dyn_cols']:
        if c not in df_new.columns:
            df_new[c] = 0.0

    x_core = prep['scaler_core'].transform(df_new[prep['core_cols']].apply(pd.to_numeric, errors='coerce').fillna(0))
    op_col = 'opcode_counts' if 'opcode_counts' in df_new.columns else 'opcode_sequence'
    x_op = prep['tfidf_op'].transform(df_new.get(op_col, pd.Series([""] * len(df_new))).fillna("").apply(parse_text_sequence)).toarray()
    x_dyn = prep['scaler_dyn'].transform(df_new[prep['top_dyn_cols']].apply(pd.to_numeric, errors='coerce').fillna(0))

    X_new_file = np.hstack([x_core, x_op, x_dyn])

    test_graphs = build_independent_ego_graphs(df_new, X_new_file, api_to_id, le_arch, ablation_mode=ablation_mode)

    meta = test_graphs[0].metadata()
    model_dict = {
        "BaseHeteroSAGE": BaseHeteroSAGE(hidden=64, out=2, metadata=meta),
        "HGT": HGT(hidden=64, out=2, heads=4, layers=2, metadata=meta),
        "GAT": GATModel(hidden=64, out=2, heads=4, metadata=meta),
        "HAN": HANModel(hidden=64, out=2, heads=4, metadata=meta),
        "MAGNN": MetaPathSAGE(hidden=64, out=2, metadata=meta),
        "R2-RGCN": RHSAGE(hidden=64, out=2, metadata=meta),
        "DR-HGT": DRHGT(hidden=64, out=2, heads=4, layers=2, metadata=meta)
    }

    if model_name not in model_dict:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model_dict[model_name]

    sample_loader = PyGDataLoader(test_graphs[:1], batch_size=1)
    sample_batch = next(iter(sample_loader))
    with torch.no_grad():
        _ = model(sample_batch.x_dict, sample_batch.edge_index_dict)

    weights_path = os.path.join(mode_dir, f"{model_name}.pth")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)

    preds, probs, infer_ms = predict_independent_ego_graphs(model, test_graphs, device, batch_size=128)
    df_new['Predicted_Label'] = preds
    df_new['Malware_Probability'] = probs

    out_csv = os.path.join(base_save_dir, f"inference_results_{model_name}_{ablation_mode}_seed{seed}.csv")
    df_new.to_csv(out_csv, index=False)
    print(f"Inference complete for {len(df_new)} samples | Latency: {infer_ms:.4f} ms/sample")
    return df_new


if __name__ == "__main__":
    drive.mount('/content/drive')
    df_res, _ = run_hin_ablation_benchmark()