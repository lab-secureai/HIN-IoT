# -*- coding: utf-8 -*-
"""Evaluation of saved ablation models on the later-period corpus."""

import os
import sys
import ast
import time
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
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, average_precision_score
from google.colab import drive, files

warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DEFAULT_BASE_SAVE_DIR = '/content/drive/MyDrive/dataset/HIN/ablation_saved_models'
DEFAULT_NEW_DATASET_PATH = '/content/drive/MyDrive/dataset/final_dataset_arch_13286.csv'
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
    os.system("pip install -q torch_geometric scikit-learn pandas openpyxl")


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


def is_static_opcode_mode(mode_str):
    return mode_str in ['Static_Opcode_Only', 'No_Dynamic_No_Entropy', 'No_File_Features']


def apply_static_opcode_only_ablation(x_matrix, keep_opcode_tfidf=True):
    x_ablated = x_matrix.copy()
    x_ablated[:, 0:3] = 0.0
    if not keep_opcode_tfidf:
        x_ablated[:, 5:105] = 0.0
    dyn_start_idx = 5 + 100
    x_ablated[:, dyn_start_idx:] = 0.0
    return x_ablated


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
    has_labels = 'label' in df_test.columns

    for i in range(len(df_test)):
        data = HeteroData()

        if is_static_opcode_mode(ablation_mode):
            x_te_sample = apply_static_opcode_only_ablation(X_te_file[i:i + 1], keep_opcode_tfidf=KEEP_OPCODE_TFIDF)
            data['file'].x = torch.tensor(x_te_sample, dtype=torch.float)
        else:
            data['file'].x = torch.tensor(X_te_file[i:i + 1], dtype=torch.float)

        data['file'].y = torch.tensor([df_test['label'].iloc[i]], dtype=torch.long) if has_labels else torch.tensor([0], dtype=torch.long)

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


def evaluate_new_dataset_10seeds(
    new_dataset_path=DEFAULT_NEW_DATASET_PATH,
    base_save_dir=DEFAULT_BASE_SAVE_DIR,
    seeds=DEFAULT_SEEDS,
    ablation_modes=ABLATION_MODES
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(new_dataset_path):
        uploaded = files.upload()
        if uploaded:
            new_dataset_path = list(uploaded.keys())[0]
        else:
            sys.exit("Dataset file not provided.")

    df_new = pd.read_csv(new_dataset_path)
    has_labels = 'label' in df_new.columns
    y_true = df_new['label'].values if has_labels else None

    print(f"Evaluating {len(df_new)} samples across {len(seeds)} seeds (Device: {device})")

    model_factories = [
        ("BaseHeteroSAGE", lambda meta: BaseHeteroSAGE(hidden=64, out=2, metadata=meta)),
        ("HGT", lambda meta: HGT(hidden=64, out=2, heads=4, layers=2, metadata=meta)),
        ("GAT", lambda meta: GATModel(hidden=64, out=2, heads=4, metadata=meta)),
        ("HAN", lambda meta: HANModel(hidden=64, out=2, heads=4, metadata=meta)),
        ("MAGNN", lambda meta: MetaPathSAGE(hidden=64, out=2, metadata=meta)),
        ("R2-RGCN", lambda meta: RHSAGE(hidden=64, out=2, metadata=meta)),
        ("DR-HGT", lambda meta: DRHGT(hidden=64, out=2, heads=4, layers=2, metadata=meta))
    ]

    all_results = []
    sample_predictions = df_new.copy()

    for seed in seeds:
        seed_dir = os.path.join(base_save_dir, f'seed_{seed}')
        if not os.path.exists(seed_dir):
            continue

        prep_path = os.path.join(seed_dir, 'preprocessors.joblib')
        if not os.path.exists(prep_path):
            continue

        print(f"\nEvaluating Seed {seed}")
        prep = joblib.load(prep_path)

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

        for mode in ablation_modes:
            mode_dir = os.path.join(seed_dir, mode)
            if not os.path.exists(mode_dir) and is_static_opcode_mode(mode):
                for alt in ['Static_Opcode_Only', 'No_File_Features', 'No_Dynamic_No_Entropy']:
                    alt_dir = os.path.join(seed_dir, alt)
                    if os.path.exists(alt_dir):
                        mode_dir = alt_dir
                        break

            if not os.path.exists(mode_dir):
                continue

            mappings_path = os.path.join(mode_dir, 'mappings.joblib')
            if not os.path.exists(mappings_path):
                continue

            mappings = joblib.load(mappings_path)
            api_to_id = mappings['api_to_id']
            le_arch = mappings['le_arch']

            test_graphs = build_independent_ego_graphs(df_new, X_new_file, api_to_id, le_arch, ablation_mode=mode)
            meta = test_graphs[0].metadata()

            sample_loader = PyGDataLoader(test_graphs[:1], batch_size=1)
            sample_batch = next(iter(sample_loader)).to(device)

            for model_name, model_fn in model_factories:
                weights_path = os.path.join(mode_dir, f"{model_name}.pth")
                if not os.path.exists(weights_path):
                    continue

                clear_gpu_memory()
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()

                model = model_fn(meta).to(device)

                with torch.no_grad():
                    _ = model(sample_batch.x_dict, sample_batch.edge_index_dict)

                model.load_state_dict(torch.load(weights_path, map_location=device))
                model.eval()

                preds, probs, infer_ms = predict_independent_ego_graphs(model, test_graphs, device, batch_size=128)
                peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0.0

                record = {
                    'Seed': seed,
                    'Ablation Mode': mode,
                    'Model': MODEL_DISPLAY_NAMES.get(model_name, model_name),
                    'Inference Time/Sample (ms)': infer_ms,
                    'Peak GPU Memory (MB)': peak_mem
                }

                if has_labels:
                    record['Accuracy'] = accuracy_score(y_true, preds)
                    record['Macro F1-Score'] = f1_score(y_true, preds, average='macro', zero_division=0)
                    record['MCC'] = matthews_corrcoef(y_true, preds)
                    try:
                        record['PR-AUC (AP)'] = average_precision_score(y_true, probs)
                    except Exception:
                        record['PR-AUC (AP)'] = 0.5
                    print(f"   [{mode:18s}] {MODEL_DISPLAY_NAMES.get(model_name, model_name):16s} | Acc: {record['Accuracy']:.4f} | F1: {record['Macro F1-Score']:.4f} | MCC: {record['MCC']:.4f} | PR-AUC: {record['PR-AUC (AP)']:.4f} | Infer: {infer_ms:.4f} ms")
                else:
                    print(f"   [{mode:18s}] {MODEL_DISPLAY_NAMES.get(model_name, model_name):16s} | Evaluated {len(preds)} samples | Infer: {infer_ms:.4f} ms")

                all_results.append(record)

                if mode == 'Full_Graph' and model_name == 'R2-RGCN':
                    sample_predictions[f'Pred_R2_RGCN_Seed_{seed}'] = preds
                    sample_predictions[f'Prob_R2_RGCN_Seed_{seed}'] = probs

    df_raw_eval = pd.DataFrame(all_results)
    if len(df_raw_eval) == 0:
        print("No evaluation results found.")
        return None, None

    raw_eval_csv = os.path.join(base_save_dir, "new_dataset_evaluation_10seeds_raw.csv")
    df_raw_eval.to_csv(raw_eval_csv, index=False)

    summary_eval = []
    group_cols = ['Ablation Mode', 'Model']
    metric_cols = ['Accuracy', 'Macro F1-Score', 'MCC', 'PR-AUC (AP)', 'Inference Time/Sample (ms)', 'Peak GPU Memory (MB)'] if has_labels else ['Inference Time/Sample (ms)', 'Peak GPU Memory (MB)']

    for (ablation_mode, model_name), grp in df_raw_eval.groupby(group_cols, sort=False):
        row = {'Ablation Mode': ablation_mode, 'Model': model_name}
        for col in metric_cols:
            row[col] = grp[col].mean()
        summary_eval.append(row)

    df_summary_eval = pd.DataFrame(summary_eval)
    sum_eval_csv = os.path.join(base_save_dir, "new_dataset_evaluation_10seeds_summary.csv")
    df_summary_eval.to_csv(sum_eval_csv, index=False)

    excel_eval_path = os.path.join(base_save_dir, "New_Dataset_Evaluation_Report_10Seeds.xlsx")
    with pd.ExcelWriter(excel_eval_path, engine='openpyxl') as writer:
        df_summary_eval.to_excel(writer, sheet_name='Evaluation_10Seeds_Mean', index=False)
        df_raw_eval.to_excel(writer, sheet_name='Raw_Seeds_Data', index=False)

    preds_csv_path = os.path.join(base_save_dir, "new_dataset_detailed_predictions.csv")
    sample_predictions.to_csv(preds_csv_path, index=False)

    print("\nEvaluation Summary:")
    print(df_summary_eval.to_string(index=False))

    return df_summary_eval, sample_predictions


if __name__ == "__main__":
    df_sum_res, df_preds = evaluate_new_dataset_10seeds()