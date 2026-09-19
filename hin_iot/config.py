"""Experiment settings for every evaluation protocol reported in the paper.

Each :class:`Protocol` fixes the order in which models are trained, the random
number generator settings and the file names of saved artifacts. Training order
matters: all models of one seed share the global PyTorch RNG, so reordering them
changes the initial weights of every later model.

Artifact file names match the ones produced by the original Colab scripts, so
evaluation scripts can load previously trained weights.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

DEFAULT_SEEDS: Tuple[int, ...] = (10, 42, 100, 314, 555, 777, 999, 1234, 2024, 2026)

# Order used for leave-one-architecture-out (LOAO) runs.
ARCHITECTURES: Tuple[str, ...] = ("MIPS", "ARM", "PowerPC", "x64", "x86")

ABLATION_MODES: Tuple[str, ...] = ("Full_Graph", "No_API_Node", "No_Arch_Node", "Static_Opcode_Only")

# Internal model keys -> names used in the paper.
PAPER_NAMES: Dict[str, str] = {
    "lgbm": "Static LightGBM",
    "bigru": "Dynamic Bi-GRU",
    "fusion": "End-to-End Fusion",
    "gcn": "Homogeneous GCN",
    "base_heterosage": "Base HeteroSAGE",
    "hgt": "HGT",
    "gat": "GAT",
    "han": "HAN",
    "metapath_sage": "Meta-path SAGE",
    "rh_sage": "RH-SAGE",
    "dr_hgt": "DR-HGT",
}

BASELINE_KEYS: Tuple[str, ...] = ("lgbm", "bigru", "fusion", "gcn")


@dataclass(frozen=True)
class TrainSettings:
    """Hyperparameters shared by all protocols unless overridden."""

    # Heterogeneous graph models
    hidden: int = 64
    heads: int = 4
    hgt_layers: int = 2
    hin_epochs: int = 100
    hin_lr: float = 0.01
    hin_weight_decay: float = 5e-4
    hin_eta_min: float = 1e-4          # CosineAnnealingLR lower bound (T_max = hin_epochs)
    ortho_weight: float = 0.01         # DR-HGT orthogonality penalty weight
    # Sequence baselines (Bi-GRU and end-to-end fusion)
    bigru_epochs: int = 15
    fusion_epochs: int = 20
    seq_lr: float = 1e-3
    seq_batch_size: int = 128
    embed_dim: int = 128
    gru_hidden: int = 128
    # Homogeneous GCN baseline
    gcn_epochs: int = 100
    gcn_lr: float = 0.01
    knn_k: int = 5
    # Static LightGBM baseline
    lgbm_estimators: int = 300
    lgbm_lr: float = 0.05
    # Feature extraction
    opcode_max_features: int = 100
    api_vocab_size: int = 5000
    api_max_len: int = 200
    # Evaluation
    hin_eval_batch_size: int = 128
    seq_eval_batch_size: int = 256
    test_size: float = 0.2


SMOKE_SETTINGS = dict(
    hin_epochs=2, bigru_epochs=1, fusion_epochs=1, gcn_epochs=2, lgbm_estimators=10,
)


@dataclass(frozen=True)
class Protocol:
    name: str
    description: str
    hin_order: Tuple[str, ...]
    hin_files: Dict[str, str]
    baseline_files: Dict[str, str] = field(default_factory=dict)
    model_variant: str = "main"            # "main" or "ablation" (see hin_iot.models)
    standardize_arch: bool = True
    train_deterministic: bool = False      # cudnn.deterministic during training
    eval_deterministic: Optional[bool] = None  # None: do not reseed during evaluation
    gcn_symmetric_test_edges: bool = False # evaluation of the homogeneous GCN on a new corpus
    settings: TrainSettings = TrainSettings()

    def with_settings(self, **overrides) -> "Protocol":
        return replace(self, settings=replace(self.settings, **overrides))


_MAIN_ORDER_A = ("base_heterosage", "hgt", "gat", "han", "metapath_sage", "rh_sage", "dr_hgt")
_MAIN_ORDER_HOLDOUT = ("base_heterosage", "hgt", "han", "metapath_sage", "gat", "rh_sage", "dr_hgt")

PROTOCOLS: Dict[str, Protocol] = {
    # Paper Section 5.1 (Table 3) and Section 5.3 (Table 5).
    "holdout": Protocol(
        name="holdout",
        description="Ten stratified 80/20 train-test splits of the primary corpus",
        hin_order=_MAIN_ORDER_HOLDOUT,
        hin_files={
            "base_heterosage": "HIN_RGCN.pt", "hgt": "HIN_HGT.pt", "han": "HIN_HAN.pt",
            "metapath_sage": "HIN_MAGNN.pt", "gat": "HIN_GAT.pt",
            "rh_sage": "IMPROVED_RGCN.pt", "dr_hgt": "IMPROVED_HGT.pt",
        },
        baseline_files={
            "lgbm": "baseline_1_lgbm.joblib", "bigru": "baseline_2_bigru.pt",
            "fusion": "baseline_3_fusion.pt", "gcn": "baseline_4_gcn.pt",
            "preprocessors": "preprocessors.pkl",
        },
        train_deterministic=False,
        eval_deterministic=True,
    ),
    # Paper Section 5.2 (Table 4).
    "loao": Protocol(
        name="loao",
        description="Leave-one-architecture-out",
        hin_order=_MAIN_ORDER_A,
        hin_files={
            "base_heterosage": "hin_rgcn.pt", "hgt": "hin_hgt.pt", "gat": "hin_gat.pt",
            "han": "hin_han.pt", "metapath_sage": "hin_magnn.pt",
            "rh_sage": "hin_improved_rgcn.pt", "dr_hgt": "hin_improved_hgt.pt",
        },
        baseline_files={
            "lgbm": "baseline1_lgbm.joblib", "bigru": "baseline2_bigru.pt",
            "fusion": "baseline3_fusion.pt", "gcn": "baseline4_gcn.pt",
            "preprocessors": "preprocessors.joblib",
        },
        train_deterministic=False,
        eval_deterministic=None,
        settings=TrainSettings(fusion_epochs=15),
    ),
    # Paper Section 5.4 (Table 6).
    "family": Protocol(
        name="family",
        description="Malware-family-disjoint grouped split (first fold of StratifiedGroupKFold, k=5)",
        hin_order=_MAIN_ORDER_A,
        hin_files={
            "base_heterosage": "HIN_RGCN.pth", "hgt": "HIN_HGT.pth", "gat": "HIN_GAT.pth",
            "han": "HIN_HAN.pth", "metapath_sage": "HIN_MAGNN.pth",
            "rh_sage": "IMPROVED_RGCN.pth", "dr_hgt": "IMPROVED_HGT.pth",
        },
        baseline_files={
            "lgbm": "baseline1_lgbm.joblib", "bigru": "baseline2_bigru.pth",
            "fusion": "baseline3_fusion.pth", "gcn": "baseline4_gcn.pth",
            "preprocessors": "preprocessors.joblib", "mappings": "graph_mappings.joblib",
        },
        train_deterministic=False,
        eval_deterministic=False,
        gcn_symmetric_test_edges=True,
    ),
    # Paper Section 5.5 (Table 7) and computational cost.
    "ablation": Protocol(
        name="ablation",
        description="Graph-component ablation over the stratified holdout splits",
        hin_order=_MAIN_ORDER_A,
        hin_files={
            "base_heterosage": "RGCN.pth", "hgt": "HGT.pth", "gat": "GAT.pth", "han": "HAN.pth",
            "metapath_sage": "MAGNN.pth", "rh_sage": "R2-RGCN.pth", "dr_hgt": "DR-HGT.pth",
        },
        baseline_files={"preprocessors": "preprocessors.joblib", "mappings": "mappings.joblib"},
        model_variant="ablation",
        standardize_arch=False,
        train_deterministic=True,
        eval_deterministic=None,
    ),
}


def get_protocol(name: str, smoke: bool = False) -> Protocol:
    if name not in PROTOCOLS:
        raise KeyError(f"Unknown protocol '{name}'. Choose from: {', '.join(PROTOCOLS)}")
    protocol = PROTOCOLS[name]
    if smoke:
        protocol = protocol.with_settings(**SMOKE_SETTINGS)
    return protocol
