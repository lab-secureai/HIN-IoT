# Reproducibility notes

## Protocols

| Protocol | Split | Model training order (one seed) | cudnn.deterministic |
|---|---|---|---|
| `holdout` | 80/20, stratified by label × architecture; strata with one file are dropped | LightGBM, Bi-GRU, fusion, GCN, Base HeteroSAGE, HGT, HAN, Meta-path SAGE, GAT, RH-SAGE, DR-HGT | False |
| `loao` | held-out architecture = test (OOD); the remaining architectures are split 80/20 as in `holdout` (ID) | LightGBM, Bi-GRU, fusion, GCN, Base HeteroSAGE, HGT, GAT, HAN, Meta-path SAGE, RH-SAGE, DR-HGT | False |
| `family` | first fold of `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)`; malware grouped by family, each benign file its own group | same as `loao` | False |
| `ablation` | same split as `holdout` (architecture strings are not normalized before stratification) | for each mode: Base HeteroSAGE, HGT, GAT, HAN, Meta-path SAGE, RH-SAGE, DR-HGT | True |

All models of one seed share the global PyTorch random number generator, so the training order is part of the protocol. The orders above are those used for the reported results.

## Model construction variants

`hin_iot.models.build_hin_model(..., variant=...)` builds the encoders exactly as in the reported experiments:

| Encoder | `main` variant (holdout, LOAO, family) | `ablation` variant |
|---|---|---|
| Base HeteroSAGE | SAGEConv with lazy input sizes applied directly to the raw node features | per-type linear projection to 64 dims, then SAGEConv(64, 64) |
| GAT | GATConv with lazy input sizes | GATConv(64, 16 × 4 heads) |
| Meta-path SAGE | SAGEConv with lazy input sizes | SAGEConv((64, 64), 64) |
| RH-SAGE, HGT, HAN, DR-HGT | identical in both variants | identical in both variants |

## Evaluation on a new corpus

* HIN encoders: each file is an isolated ego graph (file node, its APIs present in the training vocabulary, its architecture if seen in training), evaluated in batches of 128 disconnected graphs.
* Homogeneous GCN: each evaluation file is connected to its 5 nearest training files. Messages flow from training files to the evaluation file only, except for the `family` protocol, whose original 2024 evaluation also added the reverse edges (`gcn_symmetric_test_edges=True` in `hin_iot/config.py`).
* Bi-GRU and fusion: API tokens outside the training vocabulary map to `<UNK>`.
* Saved HIN models are rebuilt with the node and relation types of the training graph, in the same order, before their weights are loaded (HGT indexes relation-specific weights by this order).

## Saved artifacts

Artifact file names are the ones written by the original experiment scripts, so earlier runs can be evaluated with `scripts/evaluate.py`:

| Protocol | Preprocessing | HIN weights | Baselines |
|---|---|---|---|
| `holdout` | `seed_*/preprocessors.pkl`, `X_tr_hin_file.npy` | `HIN_RGCN.pt`, `HIN_HGT.pt`, `HIN_HAN.pt`, `HIN_MAGNN.pt`, `HIN_GAT.pt`, `IMPROVED_RGCN.pt` (RH-SAGE), `IMPROVED_HGT.pt` (DR-HGT) | `baseline_1_lgbm.joblib`, `baseline_2_bigru.pt`, `baseline_3_fusion.pt`, `baseline_4_gcn.pt` |
| `loao` | `arch_*/seed_*/preprocessors.joblib` | `hin_rgcn.pt`, …, `hin_improved_rgcn.pt`, `hin_improved_hgt.pt` | `baseline1_lgbm.joblib`, `baseline2_bigru.pt`, `baseline3_fusion.pt`, `baseline4_gcn.pt` |
| `family` | `seed_*/preprocessors.joblib`, `graph_mappings.joblib` | `HIN_RGCN.pth`, …, `IMPROVED_RGCN.pth`, `IMPROVED_HGT.pth` | `baseline1_lgbm.joblib`, `baseline2_bigru.pth`, `baseline3_fusion.pth`, `baseline4_gcn.pth` |
| `ablation` | `seed_*/preprocessors.joblib`, `seed_*/<mode>/mappings.joblib` | `seed_*/<mode>/RGCN.pth`, …, `R2-RGCN.pth` (RH-SAGE), `DR-HGT.pth` | – |

Model names in the original code: `RGCN` = Base HeteroSAGE, `MAGNN` = Meta-path SAGE, `Improved RGCN` / `R2-RGCN` = RH-SAGE, `Improved HGT` = DR-HGT.

## Sources of run-to-run variation

* CUDA scatter/gather kernels used by PyTorch Geometric are non-deterministic, even with `cudnn.deterministic=True`.
* `holdout`, `loao` and `family` enable cudnn benchmarking (`cudnn.deterministic=False`) for speed; use `--deterministic` to override.
* Library versions change kernels and default initializations. `run_info.json` records the versions of every run.
