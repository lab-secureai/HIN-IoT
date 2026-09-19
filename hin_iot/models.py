"""Graph encoders and baselines evaluated in the paper.

Paper name        | Class               | Name in the original experiment code
------------------|---------------------|-------------------------------------
Base HeteroSAGE   | BaseHeteroSAGE      | RGCN
HGT               | HGT                 | HGT
GAT               | RelationGAT         | GAT_Model
HAN               | HAN                 | HAN_Model
Meta-path SAGE    | MetaPathSAGE        | MAGNN_Model
RH-SAGE           | RHSAGE              | Improved_RGCN_Model / R2-RGCN
DR-HGT            | DRHGT               | Improved_HGT_Model
Dynamic Bi-GRU    | BiGRUClassifier     | Baseline2_BiGRU
End-to-End Fusion | FusionClassifier    | Baseline3_EndToEndHybrid
Homogeneous GCN   | HomogeneousGCN      | Baseline4_HomogeneousGCN

Parameter names are unchanged, so state dicts saved by the original code load
directly into these classes.

Two construction variants exist. The ``main`` variant is used by the holdout,
LOAO and family-disjoint experiments; the ``ablation`` variant is used by the
ablation study. They differ only for Base HeteroSAGE (the ablation variant adds
a per-type input projection), and for GAT and Meta-path SAGE (lazy vs. fixed
input sizes). See docs/REPRODUCIBILITY.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, HANConv, HeteroConv, HGTConv, Linear, SAGEConv

HIN_MODEL_KEYS = ("base_heterosage", "hgt", "gat", "han", "metapath_sage", "rh_sage", "dr_hgt")


# ============================================================ heterogeneous encoders

class BaseHeteroSAGE(nn.Module):
    """Relation-specific GraphSAGE (one SAGEConv per relation, summed by HeteroConv)."""

    def __init__(self, hidden, out, metadata, input_projection: bool = False):
        super().__init__()
        self.input_projection = input_projection
        if input_projection:
            self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in metadata[0]})
            in_channels = hidden
        else:
            in_channels = (-1, -1)
        self.conv1 = HeteroConv({et: SAGEConv(in_channels, hidden) for et in metadata[1]}, aggr="sum")
        self.conv2 = HeteroConv({et: SAGEConv(in_channels, hidden) for et in metadata[1]}, aggr="sum")
        self.lin = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        if self.input_projection:
            x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items()}
        else:
            x = dict(x_dict)
        out1 = self.conv1(x, edge_index_dict)
        x = {k: F.relu(out1.get(k, x[k])) for k in x.keys()}
        out2 = self.conv2(x, edge_index_dict)
        x = {k: F.relu(out2.get(k, x[k])) for k in x.keys()}
        return self.lin(x["file"])


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
        return self.lin(x["file"])


class RelationGAT(nn.Module):
    """Relation-specific GAT with residual connections."""

    def __init__(self, hidden, out, heads, metadata, lazy_input: bool = True):
        super().__init__()
        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in metadata[0]})
        in_channels = (-1, -1) if lazy_input else hidden
        self.conv1 = HeteroConv({et: GATConv(in_channels, hidden // heads, heads=heads, add_self_loops=False)
                                 for et in metadata[1]}, aggr="sum")
        self.conv2 = HeteroConv({et: GATConv(in_channels, hidden // heads, heads=heads, add_self_loops=False)
                                 for et in metadata[1]}, aggr="sum")
        self.lin = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items()}
        out1 = self.conv1(x, edge_index_dict)
        x = {k: F.relu(out1.get(k, x[k]) + x[k]) for k in x.keys()}
        out2 = self.conv2(x, edge_index_dict)
        x = {k: F.relu(out2.get(k, x[k]) + x[k]) for k in x.keys()}
        return self.lin(x["file"])


class HAN(nn.Module):
    def __init__(self, hidden, out, heads, metadata):
        super().__init__()
        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in metadata[0]})
        self.conv1 = HANConv(hidden, hidden, metadata, heads=heads)
        self.lin = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items()}
        out_dict = self.conv1(x, edge_index_dict)
        return self.lin(F.relu(out_dict.get("file", x["file"]) + x["file"]))


class MetaPathSAGE(nn.Module):
    """GraphSAGE over the meta-paths file-API-file and file-arch-file, fused by semantic attention."""

    def __init__(self, hidden, out, metadata, lazy_input: bool = True):
        super().__init__()
        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in metadata[0]})
        in_channels = (-1, -1) if lazy_input else (hidden, hidden)
        self.mp1_f2a = SAGEConv(in_channels, hidden)
        self.mp1_a2f = SAGEConv(in_channels, hidden)
        self.mp2_f2ar = SAGEConv(in_channels, hidden)
        self.mp2_ar2f = SAGEConv(in_channels, hidden)
        self.att_vector = nn.Parameter(torch.randn(1, hidden))
        self.lin = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](x_dict[node])) for node in x_dict.keys()}

        edge_f_a = edge_index_dict.get(("file", "calls", "api"), torch.empty((2, 0), dtype=torch.long))
        if edge_f_a.numel() > 0:
            edge_a_f = edge_f_a[[1, 0]]
            h_api = F.relu(self.mp1_f2a((x["file"], x["api"]), edge_f_a))
            h_file_mp1 = F.relu(self.mp1_a2f((h_api, x["file"]), edge_a_f))
        else:
            h_file_mp1 = x["file"]

        edge_f_ar = edge_index_dict.get(("file", "runs_on", "arch"), torch.empty((2, 0), dtype=torch.long))
        if edge_f_ar.numel() > 0:
            edge_ar_f = edge_f_ar[[1, 0]]
            h_arch = F.relu(self.mp2_f2ar((x["file"], x["arch"]), edge_f_ar))
            h_file_mp2 = F.relu(self.mp2_ar2f((h_arch, x["file"]), edge_ar_f))
        else:
            h_file_mp2 = x["file"]

        stacked = torch.stack([h_file_mp1, h_file_mp2], dim=1)
        beta = F.softmax((stacked * self.att_vector).sum(dim=-1), dim=1).unsqueeze(-1)
        return self.lin((stacked * beta).sum(dim=1))


class RHSAGE(nn.Module):
    """Residual Heterogeneous GraphSAGE (RH-SAGE), Eq. (2).

    Each node type is projected to the hidden space; every layer applies
    relation-specific SAGEConv operators, sums the messages, and updates
    ``h <- LayerNorm(h + ReLU(sum_r SAGE_r(h, N_r)))``. Only file-node
    representations are passed to the MLP classifier.
    """

    def __init__(self, hidden, out, metadata):
        super().__init__()
        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in metadata[0]})
        self.conv1 = HeteroConv({et: SAGEConv(hidden, hidden) for et in metadata[1]}, aggr="sum")
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = HeteroConv({et: SAGEConv(hidden, hidden) for et in metadata[1]}, aggr="sum")
        self.norm2 = nn.LayerNorm(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out),
        )

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items()}
        out1 = self.conv1(x, edge_index_dict)
        x = {k: self.norm1(F.relu(out1.get(k, x[k])) + x[k]) for k in x.keys()}
        out2 = self.conv2(x, edge_index_dict)
        x = {k: self.norm2(F.relu(out2.get(k, x[k])) + x[k]) for k in x.keys()}
        return self.classifier(x["file"])


class DRHGT(nn.Module):
    """HGT with a skip projection and two parallel projections regularized toward orthogonality."""

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
            nn.Linear(hidden, out),
        )

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items()}
        h_init = x["file"]
        for conv in self.hgt_convs:
            out = conv(x, edge_index_dict)
            x = {k: out.get(k, x[k]) for k in x.keys()}
        h_file = self.norm(x["file"]) + F.relu(self.skip_proj(h_init))
        z_mal = F.relu(self.fc_malware(h_file))
        z_ar = F.relu(self.fc_arch(h_file))
        return self.classifier(torch.cat([z_mal, z_ar], dim=-1))

    def compute_orthogonality_loss(self):
        w_mal = F.normalize(self.fc_malware.weight, p=2, dim=1)
        w_ar = F.normalize(self.fc_arch.weight, p=2, dim=1)
        prod = torch.matmul(w_mal, w_ar.T)
        return torch.norm(prod, p="fro") ** 2 / (self.half_hidden * self.half_hidden)


def build_hin_model(key: str, metadata, variant: str = "main", hidden: int = 64, heads: int = 4,
                    layers: int = 2, out: int = 2) -> nn.Module:
    if variant not in ("main", "ablation"):
        raise ValueError(f"Unknown model variant '{variant}'")
    ablation = variant == "ablation"
    if key == "base_heterosage":
        return BaseHeteroSAGE(hidden, out, metadata, input_projection=ablation)
    if key == "hgt":
        return HGT(hidden, out, heads, layers, metadata)
    if key == "gat":
        return RelationGAT(hidden, out, heads, metadata, lazy_input=not ablation)
    if key == "han":
        return HAN(hidden, out, heads, metadata)
    if key == "metapath_sage":
        return MetaPathSAGE(hidden, out, metadata, lazy_input=not ablation)
    if key == "rh_sage":
        return RHSAGE(hidden, out, metadata)
    if key == "dr_hgt":
        return DRHGT(hidden, out, heads, layers, metadata)
    raise KeyError(f"Unknown HIN model '{key}'. Choose from: {', '.join(HIN_MODEL_KEYS)}")


# ============================================================ baselines

class BiGRUClassifier(nn.Module):
    """Dynamic baseline: bidirectional GRU over the ordered API trace (mean pooled)."""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, out_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, out_dim))

    def forward(self, x_seq):
        out, _ = self.gru(self.embedding(x_seq))
        return self.fc(torch.mean(out, dim=1))


class FusionClassifier(nn.Module):
    """End-to-end static-dynamic fusion: Bi-GRU(API sequence) || MLP(static numerical + opcode TF-IDF)."""

    def __init__(self, vocab_size, dense_dim, embed_dim=128, gru_hidden=128, out_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, gru_hidden, batch_first=True, bidirectional=True)
        self.static_mlp = nn.Sequential(
            nn.Linear(dense_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear((gru_hidden * 2) + 64, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, out_dim),
        )

    def forward(self, x_seq, x_dense):
        seq_feat = torch.mean(self.gru(self.embedding(x_seq))[0], dim=1)
        static_feat = self.static_mlp(x_dense)
        return self.classifier(torch.cat([seq_feat, static_feat], dim=1))


class HomogeneousGCN(nn.Module):
    """Two-layer GCN over a kNN graph of file nodes (no API or architecture nodes)."""

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
