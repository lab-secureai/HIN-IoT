# HIN-IoT

Code for the paper **"Cross-Architecture IoT Malware Detection under Temporal Shift via Heterogeneous Information Networks"** by Dac-Tot Tran, Anh-Tu Tran, Khuong Nguyen-An and The-Dung Luong.

ELF executables, the runtime APIs they invoke, and their CPU architectures are modelled as a heterogeneous information network (HIN). File nodes carry static numerical features, opcode TF-IDF and numerical runtime counters. APIs and architectures enter only through typed relations (`file –calls→ api`, `file –runs_on→ arch`). The main encoder is **Residual Heterogeneous GraphSAGE (RH-SAGE)**. It is compared with static, dynamic, static–dynamic fusion, homogeneous-graph and six other heterogeneous encoders under four protocols:

* repeated stratified holdout splits;
* leave-one-architecture-out (LOAO);
* malware-family-disjoint splits;
* evaluation without fine-tuning on an independently collected 2024 corpus.

## Contents

```
hin_iot/                 library code
  data.py                CSV loading, field parsing, holdout / LOAO / family-disjoint splits
  features.py            static, opcode TF-IDF and runtime features (fitted on training data only)
  graph.py               training HIN and isolated per-file evaluation graphs
  models.py              RH-SAGE, Base HeteroSAGE, HGT, GAT, HAN, Meta-path SAGE, DR-HGT and baselines
  training.py            training and inference loops
  experiments.py         protocol runners used by the scripts
  config.py              seeds, hyperparameters and per-protocol settings
  metrics.py             metrics and mean ± std aggregation
scripts/
  train.py               holdout, LOAO and family-disjoint experiments
  ablation.py            graph-component ablation and computational cost
  evaluate.py            evaluation of saved models on a new corpus (no fine-tuning)
  stats_tests.py         paired Wilcoxon and McNemar tests
  export_partitions.py   SHA-256 manifests and per-seed train/test partitions
  make_synthetic_corpus.py  random corpus with the expected schema (for testing)
data/manifests/          SHA-256 manifests of both corpora
data/partitions/         train/test membership of every split and seed
docs/                    data format and reproducibility notes
tests/                   unit tests and an end-to-end smoke test
```

## Installation

Python ≥ 3.9. A CUDA GPU is recommended; the paper's experiments used an NVIDIA Tesla T4 on Google Colab.

```bash
git clone https://github.com/lab-secureai/HIN-IoT.git
cd HIN-IoT
pip install -r requirements.txt        # install PyTorch first if it is not already present
```

On Google Colab, PyTorch is preinstalled:

```python
!git clone https://github.com/lab-secureai/HIN-IoT.git
%cd HIN-IoT
!pip install -q torch_geometric lightgbm imbalanced-learn
```

Check the installation on a synthetic corpus (a few minutes on CPU):

```bash
pip install pytest
pytest -q tests
```

## Data

The experiments use two feature tables, one row per ELF file. Their schema is given in [docs/DATA.md](docs/DATA.md).

| Corpus | Files | Benign | Malware | Source |
|---|---:|---:|---:|---|
| Primary | 16,364 | 6,565 (OpenWrt firmware) | 9,799 (IoTPOT Dataset C, Sep 2018 – May 2020) | ARM, MIPS, PowerPC, x64, x86 |
| 2024 (temporal) | 13,286 | 6,645 (OpenWrt firmware, 2024) | 6,641 (IoTPOT Dataset F-1, 2024) | SHA-256-disjoint from the primary corpus |

**Malware binaries and derived feature tables are not redistributed.** The IoTPOT collections are available to researchers on request from the Yokohama National University IoT security group (<https://sec.ynu.codes/iot/>). This repository provides:

* `data/manifests/` — the SHA-256, label, architecture (and malware family, for the primary corpus) of every file in both corpora, together with the class × architecture counts and the cross-corpus overlap audit;
* `data/partitions/` — for every protocol and seed, which files were used for training and for testing.

With these files and the original binaries, the exact corpora and splits can be rebuilt. Feature extraction (radare2, Capstone and binwalk for static analysis; Qiling/Unicorn emulation for dynamic analysis) is described in [`appendix_feature_extraction_reproducibility.md`](appendix_feature_extraction_reproducibility.md).

## Reproducing the results

The commands below assume the feature tables are at `data/primary.csv` and `data/temporal_2024.csv`. By default every command runs the ten seeds used in the paper (`10, 42, 100, 314, 555, 777, 999, 1234, 2024, 2026`). Use `--seeds` to run a subset.

| Paper | Command | Result file |
|---|---|---|
| Tables 1–2 | `python scripts/export_partitions.py --primary data/primary.csv --temporal data/temporal_2024.csv --out-dir data` | `data/manifests/corpus_summary.md` |
| Table 3 (primary corpus) | `python scripts/train.py --protocol holdout --data data/primary.csv --work-dir runs/holdout` | `runs/holdout/results_summary.csv` |
| Table 4 (LOAO) | `python scripts/train.py --protocol loao --data data/primary.csv --work-dir runs/loao` | `runs/loao/results_summary.csv`, rows `split = OOD` |
| Table 5 (2024 corpus) | `python scripts/evaluate.py --protocol holdout --data data/temporal_2024.csv --work-dir runs/holdout --out-dir runs/holdout_2024 --save-predictions` | `runs/holdout_2024/results_summary.csv` |
| Table 6 (family-disjoint), primary | `python scripts/train.py --protocol family --data data/primary.csv --work-dir runs/family` | `runs/family/results_summary.csv` |
| Table 6, 2024 corpus | `python scripts/evaluate.py --protocol family --data data/temporal_2024.csv --work-dir runs/family --out-dir runs/family_2024` | `runs/family_2024/results_summary.csv` |
| Table 7 (ablation), primary + cost | `python scripts/ablation.py --data data/primary.csv --work-dir runs/ablation` | `runs/ablation/results_summary.csv` |
| Table 7, 2024 corpus | `python scripts/evaluate.py --protocol ablation --data data/temporal_2024.csv --work-dir runs/ablation --out-dir runs/ablation_2024` | `runs/ablation_2024/results_summary.csv` |
| Wilcoxon tests (Sections 5.1, 5.3) | `python scripts/stats_tests.py wilcoxon --results runs/holdout_2024/results_raw.csv --model-a RH-SAGE --model-b "End-to-End Fusion" --metric accuracy` | printed |

Every run directory contains:

* `results_raw.csv` — one row per seed × model (× held-out architecture or ablation mode);
* `results_summary.csv` — mean, standard deviation and `mean ± std` per metric;
* `run_info.json` — arguments, protocol settings and library/GPU versions;
* `seed_*/` — fitted preprocessing objects, API and architecture vocabularies, and model weights.

The reported metrics are accuracy, balanced accuracy, precision, recall, specificity, binary F1 (malware = positive class), macro F1, MCC, PR-AUC (average precision) and ROC-AUC. LOAO tables also report `all_malware_f1`: the binary F1 of a detector that labels every file as malware, which is the reference level for binary F1 on an imbalanced held-out architecture.

`scripts/evaluate.py` never refits anything on the evaluation corpus. The scalers, opcode TF-IDF vocabulary, API tokenizer, API vocabulary and architecture encoder all come from each seed's training partition. Each evaluation file is classified through its own ego graph (the file, its known APIs and its known architecture), so evaluation files never exchange messages.

## Model and training settings

| Component | Setting |
|---|---|
| File-node features | 5 standardized static features, opcode TF-IDF (≤ 100 terms), standardized runtime counters |
| API / architecture nodes | one-hot identity features; vocabularies from the training partition |
| RH-SAGE | per-type linear projection to 64 dims; 2 layers of relation-specific SAGEConv (sum over relations) with residual connection and LayerNorm; MLP classifier (64-64-2, dropout 0.2) |
| HIN training | full batch, 100 epochs, Adam (lr 0.01, weight decay 5e-4), cosine annealing to 1e-4, class-weighted cross-entropy |
| DR-HGT | 2 HGT layers (4 heads), skip projection, two 32-dim projections with orthogonality penalty (weight 0.01) |
| HGT / GAT / HAN | 4 heads, hidden size 64 |
| Static LightGBM | 5 static features, random undersampling of the training set, 300 trees, learning rate 0.05 |
| Dynamic Bi-GRU | API vocabulary 5,000, sequence length 200, embedding 128, hidden 128, Adam lr 1e-3, batch 128, 15 epochs |
| End-to-end fusion | Bi-GRU branch + MLP over static features and opcode TF-IDF; 20 epochs (15 in LOAO) |
| Homogeneous GCN | 2-layer GCN over a k = 5 nearest-neighbour graph of file features, 100 epochs; test files attached to their 5 nearest training files |

Per-protocol details are in [`hin_iot/config.py`](hin_iot/config.py) and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

**Determinism.** Seeds fix the data splits, weight initialization and data order. PyTorch Geometric scatter operations are not bit-deterministic on CUDA, so GPU reruns can differ slightly from the published numbers; CPU runs are deterministic.

## Citation

```bibtex
@unpublished{tran2026hiniot,
  title  = {Cross-Architecture IoT Malware Detection under Temporal Shift via Heterogeneous Information Networks},
  author = {Tran, Dac-Tot and Tran, Anh-Tu and Nguyen-An, Khuong and Luong, The-Dung},
  note   = {Submitted to the Journal of Information Security and Applications},
  year   = {2026}
}
```

## License

Apache License 2.0 (see `LICENSE`). The IoTPOT data are subject to the terms of their providers.

## Contact

The-Dung Luong (thedungluong1@gmail.com), Khuong Nguyen-An (nakhuong@hcmut.edu.vn).
