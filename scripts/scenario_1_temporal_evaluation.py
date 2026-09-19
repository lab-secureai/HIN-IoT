# -*- coding: utf-8 -*-
"""Evaluation of saved binary-detection models on the later-period corpus."""

import gc
import os
import random
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DEFAULT_SEEDS = [10, 42, 100, 314, 555, 777, 999, 1234, 2024, 2026]
NEW_DATASET_PATH = '/content/drive/MyDrive/dataset/HIN/final_dataset_arch_13286.csv'
BASE_SAVE_DIR = '/content/drive/MyDrive/dataset/HIN/results_scenario_1_binary'

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import DataLoader as PyGDataLoader
    import torch_geometric.transforms as T
    from torch_geometric.nn import HeteroConv, SAGEConv, Linear, HGTConv, GATConv, HANConv, GCNConv
except ImportError:
    if torch.cuda.is_available():
        cuda_ver = 'cu' + torch.version.cuda.replace('.', '')
    else:
        cuda_ver = 'cpu'
    torch_ver = torch.__version__.split('+')[0]
    os.system(f"pip install -q pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-{torch_ver}+{cuda_ver}.html")
    os.system("pip install -q torch_geometric lightgbm scikit-learn pandas imbalanced-learn")
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import DataLoader as PyGDataLoader
    import torch_geometric.transforms as T
    from torch_geometric.nn import HeteroConv, SAGEConv, Linear, HGTConv, GATConv, HANConv, GCNConv


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


class PyTorchApiTokenizer:
    def __init__(self, max_vocab=5000):
        self.max_vocab = max_vocab
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}

    def texts_to_sequences(self, texts, max_len=200):
        seqs = []
        for text in texts:
            tokens = text.split() if isinstance(text, str) else text
            ids = [self.word2idx.get(t, 1) for t in tokens[:max_len]]
            if len(ids) < max_len:
                ids = ids + [0] * (max_len - len(ids))
            seqs.append(ids)
        return np.array(seqs, dtype=np.int64)


class Baseline2_BiGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, out_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim)
        )

    def forward(self, x_seq):
        x = self.embedding(x_seq)
        out, _ = self.gru(x)
        h = torch.mean(out, dim=1)
        return self.fc(h)


class Baseline3_EndToEndHybrid(nn.Module):
    def __init__(self, vocab_size, dense_dim, embed_dim=128, gru_hidden=128, out_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, gru_hidden, batch_first=True, bidirectional=True)
        self.static_mlp = nn.Sequential(
            nn.Linear(dense_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear((gru_hidden * 2) + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_dim)
        )

    def forward(self, x_seq, x_dense):
        x_emb = self.embedding(x_seq)
        gru_out, _ = self.gru(x_emb)
        seq_feat = torch.mean(gru_out, dim=1)
        static_feat = self.static_mlp(x_dense)
        fused = torch.cat([seq_feat, static_feat], dim=1)
        return self.classifier(fused)


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

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        out_all = model(X_all, edge_index_te)
        out_te = out_all[N_tr:]
        probs = F.softmax(out_te, dim=1)[:, 1].cpu().numpy()
        preds = out_te.argmax(dim=1).cpu().numpy()

    if device.type == 'cuda':
        torch.cuda.synchronize()
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
        h_file = F.relu(out_dict.get('file', x['file']) + x['file'])
        return self.lin(h_file)


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

        stacked_h = torch.stack([h_file_mp1, h_file_mp2], dim=1)
        w = (stacked_h * self.att_vector).sum(dim=-1)
        beta = F.softmax(w, dim=1).unsqueeze(-1)
        return self.lin((stacked_h * beta).sum(dim=1))


class RHSAGE(nn.Module):
    def __init__(self, hidden, out, metadata):
        super().__init__()
        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in metadata[0]})
        self.conv1 = HeteroConv({et: SAGEConv(hidden, hidden) for et in metadata[1]}, aggr='sum')
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = HeteroConv({et: SAGEConv(hidden, hidden) for et in metadata[1]}, aggr='sum')
        self.norm2 = nn.LayerNorm(hidden)
        self.classifier = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, out))

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
        self.classifier = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, out))

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
                return " ".join([f"{k} " * min(int(v), 20) for k, v in d.items()])
            except Exception:
                pass
        return val
    elif isinstance(val, (list, tuple)):
        return " ".join(map(str, val))
    return ""


def transform_new_dataset_features(df_new, prep):
    for c in prep['core_static_cols']:
        if c not in df_new.columns:
            df_new[c] = 0.0
    x_new_core_static = prep['scaler_core_static'].transform(df_new[prep['core_static_cols']].apply(pd.to_numeric, errors='coerce').fillna(0))

    op_col = 'opcode_counts' if 'opcode_counts' in df_new.columns else 'opcode_sequence'
    x_new_op = prep['tfidf_op'].transform(df_new.get(op_col, pd.Series([""] * len(df_new))).fillna("").apply(parse_text_sequence)).toarray()
    X_new_static_full = np.hstack([x_new_core_static, x_new_op])

    for c in prep['top_dyn_cols']:
        if c not in df_new.columns:
            df_new[c] = 0.0
    x_new_top_dyn = prep['scaler_top_dyn'].transform(df_new[prep['top_dyn_cols']].apply(pd.to_numeric, errors='coerce').fillna(0))

    X_new_dynamic_streamlined = x_new_top_dyn
    X_new_hin_file = np.hstack([X_new_static_full, X_new_dynamic_streamlined])

    api_col = 'api_sequence' if 'api_sequence' in df_new.columns else 'api_list'
    seq_new = df_new.get(api_col, pd.Series([""] * len(df_new))).fillna("").apply(parse_text_sequence).tolist()
    X_new_seq = prep['tokenizer'].texts_to_sequences(seq_new, max_len=200)

    return x_new_core_static, X_new_static_full, X_new_dynamic_streamlined, X_new_hin_file, X_new_seq


def build_independent_ego_graphs_test(df_new, X_hin_file, api_to_id, le_arch):
    def parse_api(s):
        try:
            return ast.literal_eval(s) if isinstance(s, str) and s.startswith('[') else ([] if pd.isna(s) else str(s).split())
        except Exception:
            return []

    api_col = 'api_sequence' if 'api_sequence' in df_new.columns else 'api_list'
    api_seqs = df_new.get(api_col, pd.Series([[]] * len(df_new))).apply(parse_api)
    known_archs = set(le_arch.classes_)
    arch_ids = df_new['arch'].apply(lambda x: le_arch.transform([str(x)])[0] if str(x) in known_archs else -1).values

    graphs = []
    num_apis = len(api_to_id) if len(api_to_id) > 0 else 1
    num_archs = len(le_arch.classes_)

    eye_api = torch.eye(num_apis, dtype=torch.float)
    eye_arch = torch.eye(num_archs, dtype=torch.float)

    for i in range(len(df_new)):
        data = HeteroData()
        data['file'].x = torch.tensor(X_hin_file[i:i + 1], dtype=torch.float)
        data['file'].y = torch.tensor([df_new['label'].iloc[i]], dtype=torch.long)

        seq = api_seqs.iloc[i]
        valid_api_ids = sorted(list({api_to_id[api] for api in seq if api in api_to_id}))
        if len(valid_api_ids) > 0:
            data['api'].x = eye_api[valid_api_ids]
            local_api_indices = list(range(len(valid_api_ids)))
            data['file', 'calls', 'api'].edge_index = torch.tensor(np.array([[0] * len(valid_api_ids), local_api_indices]), dtype=torch.long)
        else:
            data['api'].x = torch.empty((0, num_apis), dtype=torch.float)
            data['file', 'calls', 'api'].edge_index = torch.empty((2, 0), dtype=torch.long)

        if arch_ids[i] != -1:
            data['arch'].x = eye_arch[arch_ids[i:i + 1]]
            data['file', 'runs_on', 'arch'].edge_index = torch.tensor(np.array([[0], [0]]), dtype=torch.long)
        else:
            data['arch'].x = torch.empty((0, num_archs), dtype=torch.float)
            data['file', 'runs_on', 'arch'].edge_index = torch.empty((2, 0), dtype=torch.long)

        graphs.append(T.ToUndirected()(data))
    return graphs


def calculate_metrics(y_true, preds, probs, name, seed, infer_ms=0.0, mem_mb=0.0):
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1_bin = f1_score(y_true, preds, average='binary', zero_division=0)
    f1_macro = f1_score(y_true, preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, preds)

    try:
        pr_auc = average_precision_score(y_true, probs)
    except Exception:
        pr_auc = 0.5

    try:
        roc_auc = roc_auc_score(y_true, probs)
    except Exception:
        roc_auc = 0.5

    return {
        'name': name,
        'seed': seed,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'Binary F1': f1_bin,
        'Macro F1': f1_macro,
        'MCC': mcc,
        'PR-AUC': pr_auc,
        'ROC-AUC': roc_auc,
        'Infer Time (ms/sample)': infer_ms,
        'Peak GPU VRAM (MB)': mem_mb
    }


def evaluate_new_dataset(
    new_dataset_path=NEW_DATASET_PATH,
    base_save_dir=BASE_SAVE_DIR,
    seeds=DEFAULT_SEEDS
):
    if not os.path.exists(new_dataset_path):
        raise FileNotFoundError(f"New dataset file not found at: {new_dataset_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating model performance on new dataset | Device: {device}")

    df_new = pd.read_csv(new_dataset_path)
    df_new['arch'] = df_new['arch'].apply(standardize_elf_architecture)
    y_true = df_new['label'].values
    print(f"Test samples: {len(df_new)} | Malware: {np.sum(y_true == 1)} | Benign: {np.sum(y_true == 0)}")

    all_raw_results = []

    for seed in seeds:
        print(f"\n--- Evaluating Seed {seed} ---")

        seed_dir = os.path.join(base_save_dir, f'seed_{seed}')
        prep_path = os.path.join(seed_dir, 'preprocessors.pkl')

        if not os.path.exists(prep_path):
            print(f"Skipping Seed {seed}: Preprocessor file not found.")
            continue

        seed_everything(seed)

        all_prep = joblib.load(prep_path)
        prep = all_prep['preprocessors']
        apis = all_prep['api_to_id']
        archs = all_prep['le_arch']

        x_te_core_stat, X_te_stat_full, X_te_dyn_stream, X_te_hin_file, X_te_seq = transform_new_dataset_features(df_new, prep)
        lgbm_path = os.path.join(seed_dir, 'baseline_1_lgbm.joblib')
        if not os.path.exists(lgbm_path):
            lgbm_path = os.path.join(seed_dir, 'lgbm_model.pkl')

        if os.path.exists(lgbm_path):
            print("Inference Baseline 1: Static LightGBM...")
            lgbm = joblib.load(lgbm_path)
            t_inf = time.perf_counter()
            p1 = lgbm.predict(x_te_core_stat)
            pr1 = lgbm.predict_proba(x_te_core_stat)[:, 1]
            infer_ms = (time.perf_counter() - t_inf) * 1000 / len(y_true)
            all_raw_results.append(calculate_metrics(y_true, p1, pr1, "Baseline 1: Static (LightGBM)", seed, infer_ms))
        gru_path = os.path.join(seed_dir, 'baseline_2_bigru.pt')
        if os.path.exists(gru_path):
            print("Inference Baseline 2: Dynamic Bi-GRU...")
            bigru = Baseline2_BiGRU(vocab_size=prep['vocab_size']).to(device)
            bigru.load_state_dict(torch.load(gru_path, map_location=device))
            bigru.eval()

            dataset = torch.utils.data.TensorDataset(torch.tensor(X_te_seq, dtype=torch.long))
            loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
            all_preds, all_probs = [], []

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_inf = time.perf_counter()

            with torch.no_grad():
                for (x_b,) in loader:
                    out = bigru(x_b.to(device))
                    all_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
                    all_preds.extend(out.argmax(dim=1).cpu().numpy())

            if device.type == 'cuda':
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t_inf) * 1000 / len(y_true)

            all_raw_results.append(calculate_metrics(y_true, np.array(all_preds), np.array(all_probs), "Baseline 2: Dynamic (Bi-GRU)", seed, infer_ms))
            clear_gpu_memory()
        fusion_path = os.path.join(seed_dir, 'baseline_3_fusion.pt')
        if os.path.exists(fusion_path):
            print("Inference Baseline 3: End-to-End Fusion...")
            fusion_net = Baseline3_EndToEndHybrid(vocab_size=prep['vocab_size'], dense_dim=X_te_stat_full.shape[1]).to(device)
            fusion_net.load_state_dict(torch.load(fusion_path, map_location=device))
            fusion_net.eval()

            dataset = torch.utils.data.TensorDataset(torch.tensor(X_te_seq, dtype=torch.long), torch.tensor(X_te_stat_full, dtype=torch.float))
            loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
            all_preds, all_probs = [], []

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_inf = time.perf_counter()

            with torch.no_grad():
                for x_s, x_d in loader:
                    out = fusion_net(x_s.to(device), x_d.to(device))
                    all_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
                    all_preds.extend(out.argmax(dim=1).cpu().numpy())

            if device.type == 'cuda':
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t_inf) * 1000 / len(y_true)

            all_raw_results.append(calculate_metrics(y_true, np.array(all_preds), np.array(all_probs), "Baseline 3: End-to-End Fusion", seed, infer_ms))
            clear_gpu_memory()
        gcn_path = os.path.join(seed_dir, 'baseline_4_gcn.pt')
        tr_hin_path = os.path.join(seed_dir, 'X_tr_hin_file.npy')
        if os.path.exists(gcn_path) and os.path.exists(tr_hin_path):
            print("Inference Baseline 4: Homogeneous Graph GCN...")
            b4_model = Baseline4_HomogeneousGCN(in_channels=X_te_hin_file.shape[1]).to(device)
            b4_model.load_state_dict(torch.load(gcn_path, map_location=device))
            b4_model.eval()

            X_tr_hin_file = np.load(tr_hin_path)
            p4, pr4, infer_ms_gcn = evaluate_inductive_homogeneous_gcn(b4_model, X_tr_hin_file, X_te_hin_file, device, k=5)
            all_raw_results.append(calculate_metrics(y_true, p4, pr4, "Baseline 4: Homogeneous GCN", seed, infer_ms_gcn))
            clear_gpu_memory()
        graphs_test = build_independent_ego_graphs_test(df_new, X_te_hin_file, apis, archs)
        test_loader = PyGDataLoader(graphs_test, batch_size=128, shuffle=False)
        sample_meta = graphs_test[0].metadata()
        sample_batch = next(iter(test_loader)).to(device)

        hin_models_info = [
            ("HIN_RGCN", "Base HeteroSAGE", lambda meta: BaseHeteroSAGE(hidden=64, out=2, metadata=meta)),
            ("HIN_HGT", "HIN: HGT", lambda meta: HGT(hidden=64, out=2, heads=4, layers=2, metadata=meta)),
            ("HIN_HAN", "HIN: HAN", lambda meta: HANModel(hidden=64, out=2, heads=4, metadata=meta)),
            ("HIN_MAGNN", "Meta-path SAGE", lambda meta: MetaPathSAGE(hidden=64, out=2, metadata=meta)),
            ("HIN_GAT", "HIN: GAT", lambda meta: GATModel(hidden=64, out=2, heads=4, metadata=meta)),
            ("IMPROVED_RGCN", "RH-SAGE", lambda meta: RHSAGE(hidden=64, out=2, metadata=meta)),
            ("IMPROVED_HGT", "DR-HGT", lambda meta: DRHGT(hidden=64, out=2, heads=4, layers=2, metadata=meta)),
        ]

        for model_file, display_name, model_builder in hin_models_info:
            weight_file = os.path.join(seed_dir, f"{model_file}.pt")
            if not os.path.exists(weight_file):
                continue

            print(f"Inference {display_name}...")
            model = model_builder(sample_meta).to(device)
            with torch.no_grad():
                _ = model(sample_batch.x_dict, sample_batch.edge_index_dict)
            model.load_state_dict(torch.load(weight_file, map_location=device))
            model.eval()

            all_probs, all_preds = [], []
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t_inf = time.perf_counter()

            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch.x_dict, batch.edge_index_dict)
                    all_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
                    all_preds.extend(out.argmax(dim=1).cpu().numpy())

            if device.type == 'cuda':
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t_inf) * 1000 / len(y_true)
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0.0

            res = calculate_metrics(y_true, np.array(all_preds), np.array(all_probs), display_name, seed, infer_ms, peak_mem)
            all_raw_results.append(res)
            print(f"Result -> {display_name} | Acc: {res['Accuracy']:.4f} | F1: {res['Binary F1']:.4f} | MCC: {res['MCC']:.4f}")
            clear_gpu_memory()

    if not all_raw_results:
        print("No valid results found.")
        return None, None

    df_raw = pd.DataFrame(all_raw_results)
    raw_csv_path = os.path.join(base_save_dir, "F1_dataset_raw_results_10seeds.csv")
    df_raw.to_csv(raw_csv_path, index=False)

    summary_list = []
    metric_cols = ['Accuracy', 'Precision', 'Recall', 'Binary F1', 'Macro F1', 'MCC', 'PR-AUC', 'ROC-AUC', 'Infer Time (ms/sample)', 'Peak GPU VRAM (MB)']

    for model_name, group in df_raw.groupby('name', sort=False):
        row = {'Model Architecture': model_name}
        for m in metric_cols:
            mean_val = group[m].mean()
            std_val = group[m].std()
            row[f'{m} (Mean ± Std)'] = f"{mean_val:.4f} ± {std_val:.4f}"
        summary_list.append(row)

    df_summary = pd.DataFrame(summary_list)
    summary_csv_path = os.path.join(base_save_dir, "F1_dataset_summary_results_10seeds.csv")
    df_summary.to_csv(summary_csv_path, index=False)

    print("\nSummary Results on New Dataset:")
    print(df_summary.to_string(index=False))
    return df_summary, df_raw


if __name__ == "__main__":
    evaluate_new_dataset()