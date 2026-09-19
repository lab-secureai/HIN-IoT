"""Experiment runners behind the command-line scripts in ``scripts/``.

* :func:`run_training` - train and test every model for the ``holdout``,
  ``loao`` and ``family`` protocols (Tables 3, 4 and 6 of the paper).
* :func:`run_ablation` - graph-component ablation (Table 7, computational cost).
* :func:`evaluate_new_corpus` - apply saved models, without any parameter
  update, to an independently collected corpus (Table 5 and the 2024 columns of
  Tables 6 and 7).

Every seed directory keeps the fitted preprocessing objects, the API and
architecture vocabularies and all model weights, so evaluations can be rerun
without retraining.
"""

import os
from typing import Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import torch

from .config import ABLATION_MODES, ARCHITECTURES, PAPER_NAMES, Protocol, get_protocol
from .data import (ensure_binary_labels, family_disjoint_split, id_column, loao_split, read_corpus,
                   stratified_holdout_split)
from .features import FeaturePipeline, FeatureSet, static_opcode_only
from .graph import FULL_GRAPH, build_ego_graphs, build_train_hin
from .metrics import classification_metrics, summarize
from .models import BiGRUClassifier, FusionClassifier, HomogeneousGCN, build_hin_model
from .training import (balanced_class_weights, load_hin_model, predict_ego_graphs, predict_gcn_inductive,
                       predict_lgbm, predict_sequence_model, train_bigru, train_fusion, train_gcn,
                       train_hin_model, train_lgbm)
from .utils import clear_gpu_memory, load_joblib, load_state_dict, seed_everything

STATIC_OPCODE_ONLY = "Static_Opcode_Only"


# ============================================================ result bookkeeping

class ResultLog:
    """Collects one metrics row per (run, model, test set) and, optionally, per-sample predictions."""

    def __init__(self, save_predictions: bool = False):
        self.rows: List[dict] = []
        self.predictions: List[pd.DataFrame] = []
        self.save_predictions = save_predictions

    def add(self, context: dict, model_key: str, y_true, y_pred, y_prob, sample_ids=None, **costs) -> dict:
        row = {**context, "model_key": model_key, "model": PAPER_NAMES[model_key]}
        row.update(classification_metrics(y_true, y_pred, y_prob))
        row.update({k: v for k, v in costs.items() if v is not None})
        self.rows.append(row)
        if self.save_predictions:
            frame = pd.DataFrame({
                "sample_id": sample_ids if sample_ids is not None else np.arange(len(y_true)),
                "label": np.asarray(y_true), "pred": np.asarray(y_pred), "prob_malware": np.asarray(y_prob),
            })
            for k, v in {**context, "model_key": model_key}.items():
                frame.insert(0, k, v)
            self.predictions.append(frame)
        tag = " ".join(f"{k}={v}" for k, v in context.items() if k != "protocol")
        print(f"  [{tag}] {PAPER_NAMES[model_key]:<18s} acc={row['accuracy']:.4f} "
              f"f1={row['binary_f1']:.4f} mcc={row['mcc']:.4f}")
        return row

    def write(self, out_dir: str, group_cols: Sequence[str]) -> pd.DataFrame:
        os.makedirs(out_dir, exist_ok=True)
        raw = pd.DataFrame(self.rows)
        raw.to_csv(os.path.join(out_dir, "results_raw.csv"), index=False)
        summary = summarize(raw, list(group_cols) + ["model"]) if len(raw) else raw
        summary.to_csv(os.path.join(out_dir, "results_summary.csv"), index=False)
        if self.predictions:
            pd.concat(self.predictions, ignore_index=True).to_csv(
                os.path.join(out_dir, "predictions.csv.gz"), index=False, compression="gzip")
        print(f"\nResults written to {out_dir}")
        return summary


def _sample_ids(df: pd.DataFrame, id_col: Optional[str]):
    if id_col and id_col in df.columns:
        return df[id_col].values
    return None


def _resolve_id_col(df: pd.DataFrame, id_col: Optional[str]) -> Optional[str]:
    try:
        return id_column(df, id_col or "sha256")
    except ValueError:
        return None


# ============================================================ seed artifacts

def _save_seed_artifacts(protocol: Protocol, seed_dir: str, pipe: FeaturePipeline, api_to_id, le_arch,
                         x_train_hin: np.ndarray) -> None:
    files = protocol.baseline_files
    state = pipe.to_state()
    path = os.path.join(seed_dir, files["preprocessors"])
    if protocol.name == "family":
        joblib.dump(state, path)
        joblib.dump({"api_to_id": api_to_id, "le_arch": le_arch}, os.path.join(seed_dir, files["mappings"]))
    else:
        bundle = {"preprocessors": state, "api_to_id": api_to_id, "le_arch": le_arch}
        if protocol.name == "loao":
            bundle["X_tr_hin_file"] = x_train_hin
        joblib.dump(bundle, path)
    np.save(os.path.join(seed_dir, "X_tr_hin_file.npy"), x_train_hin)


def load_seed_artifacts(protocol: Protocol, seed_dir: str):
    """Return (feature pipeline, API vocabulary, architecture encoder, training file features or None)."""
    files = protocol.baseline_files
    obj = load_joblib(os.path.join(seed_dir, files["preprocessors"]))
    x_train = None
    if isinstance(obj, dict) and "preprocessors" in obj:
        state, api_to_id, le_arch = obj["preprocessors"], obj["api_to_id"], obj["le_arch"]
        x_train = obj.get("X_tr_hin_file")
    else:
        state = obj
        mappings = load_joblib(os.path.join(seed_dir, files["mappings"]))
        api_to_id, le_arch = mappings["api_to_id"], mappings["le_arch"]
    npy = os.path.join(seed_dir, "X_tr_hin_file.npy")
    if x_train is None and os.path.exists(npy):
        x_train = np.load(npy)
    return FeaturePipeline.from_state(state), api_to_id, le_arch, x_train


# ============================================================ training protocols

def _train_and_test_seed(protocol: Protocol, seed: int, df_train: pd.DataFrame, test_sets: Dict[str, pd.DataFrame],
                         seed_dir: str, device, log: ResultLog, context: dict, id_col: Optional[str]) -> None:
    s = protocol.settings
    files = protocol.baseline_files
    os.makedirs(seed_dir, exist_ok=True)

    pipe = FeaturePipeline(s.opcode_max_features, s.api_vocab_size, s.api_max_len)
    f_tr = pipe.fit_transform(df_train)
    f_te: Dict[str, FeatureSet] = {name: pipe.transform(df) for name, df in test_sets.items()}
    y_tr = df_train["label"].values
    ensure_binary_labels(y_tr, "training data")

    train_graph, api_to_id, le_arch = build_train_hin(df_train, f_tr.hin_file)
    train_graph = train_graph.to(device)
    _save_seed_artifacts(protocol, seed_dir, pipe, api_to_id, le_arch, f_tr.hin_file)
    class_weights = balanced_class_weights(y_tr, device)
    print(f"  train={len(df_train)} " + " ".join(f"{k}={len(v)}" for k, v in test_sets.items())
          + f" | APIs={len(api_to_id)} archs={list(le_arch.classes_)}")

    def record(model_key, split, pred, prob, infer_ms, train_ms=None, peak=None):
        df_te = test_sets[split]
        log.add({**context, "split": split}, model_key, df_te["label"].values, pred, prob,
                sample_ids=_sample_ids(df_te, id_col),
                train_ms_per_epoch=train_ms, infer_ms_per_sample=infer_ms, peak_gpu_mb=peak)

    # Static LightGBM (five numerical ELF features)
    lgbm, train_ms = train_lgbm(f_tr.core, y_tr, seed, s)
    joblib.dump(lgbm, os.path.join(seed_dir, files["lgbm"]))
    for split in test_sets:
        pred, prob, infer_ms = predict_lgbm(lgbm, f_te[split].core)
        record("lgbm", split, pred, prob, infer_ms, train_ms)

    # Dynamic Bi-GRU (ordered API sequence)
    bigru, train_ms, peak = train_bigru(f_tr.api_seq, y_tr, pipe.vocab_size, s, device)
    torch.save(bigru.state_dict(), os.path.join(seed_dir, files["bigru"]))
    for split in test_sets:
        pred, prob, infer_ms = predict_sequence_model(bigru, f_te[split].api_seq, device,
                                                      batch_size=s.seq_eval_batch_size)
        record("bigru", split, pred, prob, infer_ms, train_ms, peak)
    del bigru
    clear_gpu_memory()

    # End-to-end static-dynamic fusion
    fusion, train_ms, peak = train_fusion(f_tr.api_seq, f_tr.static_full, y_tr, pipe.vocab_size, s, device)
    torch.save(fusion.state_dict(), os.path.join(seed_dir, files["fusion"]))
    for split in test_sets:
        pred, prob, infer_ms = predict_sequence_model(fusion, f_te[split].api_seq, device,
                                                      x_dense=f_te[split].static_full,
                                                      batch_size=s.seq_eval_batch_size)
        record("fusion", split, pred, prob, infer_ms, train_ms, peak)
    del fusion
    clear_gpu_memory()

    # Homogeneous GCN over a kNN file graph
    gcn, train_ms, peak = train_gcn(f_tr.hin_file, y_tr, s, device)
    torch.save(gcn.state_dict(), os.path.join(seed_dir, files["gcn"]))
    for split in test_sets:
        pred, prob, infer_ms = predict_gcn_inductive(gcn, f_tr.hin_file, f_te[split].hin_file, device, k=s.knn_k)
        record("gcn", split, pred, prob, infer_ms, train_ms, peak)
    del gcn
    clear_gpu_memory()

    # Heterogeneous graph encoders
    test_graphs = {split: build_ego_graphs(test_sets[split], f_te[split].hin_file, api_to_id, le_arch)
                   for split in test_sets}
    for key in protocol.hin_order:
        model = build_hin_model(key, train_graph.metadata(), protocol.model_variant,
                                s.hidden, s.heads, s.hgt_layers).to(device)
        train_ms, peak = train_hin_model(model, train_graph, class_weights, s, device)
        torch.save(model.state_dict(), os.path.join(seed_dir, protocol.hin_files[key]))
        for split, graphs in test_graphs.items():
            pred, prob, infer_ms = predict_ego_graphs(model, graphs, device, s.hin_eval_batch_size)
            record(key, split, pred, prob, infer_ms, train_ms, peak)
        del model
        clear_gpu_memory()


def run_training(protocol_name: str, data_path: str, work_dir: str, seeds: Iterable[int], device,
                 architectures: Optional[Sequence[str]] = None, smoke: bool = False,
                 deterministic: Optional[bool] = None, save_predictions: bool = False,
                 id_col: Optional[str] = None) -> pd.DataFrame:
    """Train and test all models for the ``holdout``, ``loao`` or ``family`` protocol."""
    seeds = list(seeds)
    protocol = get_protocol(protocol_name, smoke=smoke)
    if protocol.name not in ("holdout", "loao", "family"):
        raise ValueError("run_training supports the holdout, loao and family protocols; use run_ablation for ablation.")
    det = protocol.train_deterministic if deterministic is None else deterministic
    s = protocol.settings
    df_raw = read_corpus(data_path, standardize=protocol.standardize_arch)
    ensure_binary_labels(df_raw["label"].values, data_path)
    id_col = _resolve_id_col(df_raw, id_col)
    log = ResultLog(save_predictions)
    os.makedirs(work_dir, exist_ok=True)
    print(f"Protocol: {protocol.description} | device={device} | seeds={list(seeds)}")

    if protocol.name == "loao":
        for arch in (architectures or ARCHITECTURES):
            for seed in seeds:
                print(f"\n=== held-out architecture {arch} | seed {seed} ===")
                seed_everything(seed, det)
                df_tr, df_id, df_ood = loao_split(df_raw, arch, seed, s.test_size)
                _train_and_test_seed(protocol, seed, df_tr, {"ID": df_id, "OOD": df_ood},
                                     os.path.join(work_dir, f"arch_{arch}", f"seed_{seed}"), device, log,
                                     {"protocol": protocol.name, "held_out_arch": arch, "seed": seed}, id_col)
        return log.write(work_dir, ["held_out_arch", "split"])

    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        seed_everything(seed, det)
        if protocol.name == "holdout":
            df_tr, df_te = stratified_holdout_split(df_raw, seed, s.test_size)
        else:
            df_tr, df_te = family_disjoint_split(df_raw, seed)
        _train_and_test_seed(protocol, seed, df_tr, {"test": df_te}, os.path.join(work_dir, f"seed_{seed}"),
                             device, log, {"protocol": protocol.name, "seed": seed}, id_col)
    return log.write(work_dir, ["split"])


# ============================================================ ablation

def _ablation_inputs(mode: str, x: np.ndarray, pipe: FeaturePipeline) -> np.ndarray:
    if mode == STATIC_OPCODE_ONLY:
        return static_opcode_only(x, pipe.core_dim, pipe.opcode_dim, keep_opcode=True)
    return x


def run_ablation(data_path: str, work_dir: str, seeds: Iterable[int], device,
                 modes: Sequence[str] = ABLATION_MODES, smoke: bool = False,
                 deterministic: Optional[bool] = None, save_predictions: bool = False,
                 id_col: Optional[str] = None) -> pd.DataFrame:
    """Train every heterogeneous encoder under each graph ablation mode.

    Modes: Full_Graph, No_API_Node (no file-API edges), No_Arch_Node (no
    file-architecture edges), Static_Opcode_Only (entropy and runtime counters
    removed from file nodes; graph relations kept).
    """
    seeds = list(seeds)
    protocol = get_protocol("ablation", smoke=smoke)
    s = protocol.settings
    det = protocol.train_deterministic if deterministic is None else deterministic
    df_raw = read_corpus(data_path, standardize=protocol.standardize_arch)
    ensure_binary_labels(df_raw["label"].values, data_path)
    id_col = _resolve_id_col(df_raw, id_col)
    log = ResultLog(save_predictions)
    os.makedirs(work_dir, exist_ok=True)

    for seed in seeds:
        print(f"\n=== ablation | seed {seed} ===")
        seed_dir = os.path.join(work_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        seed_everything(seed, det)

        df_tr, df_te = stratified_holdout_split(df_raw, seed, s.test_size)
        pipe = FeaturePipeline(s.opcode_max_features, s.api_vocab_size, s.api_max_len)
        f_tr, f_te = pipe.fit_transform(df_tr), pipe.transform(df_te)
        joblib.dump(pipe.to_state(), os.path.join(seed_dir, protocol.baseline_files["preprocessors"]))
        class_weights = balanced_class_weights(df_tr["label"].values, device)

        for mode in modes:
            mode_dir = os.path.join(seed_dir, mode)
            os.makedirs(mode_dir, exist_ok=True)
            x_tr = _ablation_inputs(mode, f_tr.hin_file, pipe)
            x_te = _ablation_inputs(mode, f_te.hin_file, pipe)
            train_graph, api_to_id, le_arch = build_train_hin(df_tr, x_tr, mode=mode)
            train_graph = train_graph.to(device)
            joblib.dump({"api_to_id": api_to_id, "le_arch": le_arch},
                        os.path.join(mode_dir, protocol.baseline_files["mappings"]))
            test_graphs = build_ego_graphs(df_te, x_te, api_to_id, le_arch, mode=mode)

            for key in protocol.hin_order:
                clear_gpu_memory()
                model = build_hin_model(key, train_graph.metadata(), protocol.model_variant,
                                        s.hidden, s.heads, s.hgt_layers).to(device)
                train_ms, peak = train_hin_model(model, train_graph, class_weights, s, device)
                torch.save(model.state_dict(), os.path.join(mode_dir, protocol.hin_files[key]))
                pred, prob, infer_ms = predict_ego_graphs(model, test_graphs, device, s.hin_eval_batch_size)
                log.add({"protocol": "ablation", "seed": seed, "ablation_mode": mode, "split": "test"}, key,
                        df_te["label"].values, pred, prob, sample_ids=_sample_ids(df_te, id_col),
                        train_ms_per_epoch=train_ms, infer_ms_per_sample=infer_ms, peak_gpu_mb=peak)
                del model
    return log.write(work_dir, ["ablation_mode", "split"])


# ============================================================ evaluation on a new corpus

def _evaluate_saved_baselines(protocol: Protocol, seed_dir: str, feats: FeatureSet, pipe: FeaturePipeline,
                              x_train_hin: Optional[np.ndarray], device, emit) -> None:
    s = protocol.settings
    files = protocol.baseline_files

    lgbm_path = os.path.join(seed_dir, files["lgbm"])
    if not os.path.exists(lgbm_path) and protocol.name == "holdout":
        lgbm_path = os.path.join(seed_dir, "lgbm_model.pkl")
    if os.path.exists(lgbm_path):
        pred, prob, infer_ms = predict_lgbm(load_joblib(lgbm_path), feats.core)
        emit("lgbm", pred, prob, infer_ms)

    vocab = pipe.vocab_size or s.api_vocab_size
    bigru_path = os.path.join(seed_dir, files["bigru"])
    if os.path.exists(bigru_path) and feats.api_seq is not None:
        model = BiGRUClassifier(vocab_size=vocab, embed_dim=s.embed_dim, hidden_dim=s.gru_hidden).to(device)
        model.load_state_dict(load_state_dict(bigru_path, device))
        pred, prob, infer_ms = predict_sequence_model(model, feats.api_seq, device, batch_size=s.seq_eval_batch_size)
        emit("bigru", pred, prob, infer_ms)
        clear_gpu_memory()

    fusion_path = os.path.join(seed_dir, files["fusion"])
    if os.path.exists(fusion_path) and feats.api_seq is not None:
        model = FusionClassifier(vocab_size=vocab, dense_dim=feats.static_full.shape[1],
                                 embed_dim=s.embed_dim, gru_hidden=s.gru_hidden).to(device)
        model.load_state_dict(load_state_dict(fusion_path, device))
        pred, prob, infer_ms = predict_sequence_model(model, feats.api_seq, device, x_dense=feats.static_full,
                                                      batch_size=s.seq_eval_batch_size)
        emit("fusion", pred, prob, infer_ms)
        clear_gpu_memory()

    gcn_path = os.path.join(seed_dir, files["gcn"])
    if os.path.exists(gcn_path) and x_train_hin is not None:
        model = HomogeneousGCN(in_channels=feats.hin_file.shape[1], hidden_dim=s.hidden).to(device)
        model.load_state_dict(load_state_dict(gcn_path, device))
        pred, prob, infer_ms = predict_gcn_inductive(model, x_train_hin, feats.hin_file, device, k=s.knn_k,
                                                     symmetric=protocol.gcn_symmetric_test_edges)
        emit("gcn", pred, prob, infer_ms)
        clear_gpu_memory()


def _evaluate_saved_hin(protocol: Protocol, model_dir: str, graphs, device, emit) -> None:
    s = protocol.settings
    metadata = graphs[0].metadata()
    for key in protocol.hin_order:
        path = os.path.join(model_dir, protocol.hin_files[key])
        if not os.path.exists(path):
            continue
        model = build_hin_model(key, metadata, protocol.model_variant, s.hidden, s.heads, s.hgt_layers)
        model = load_hin_model(model, load_state_dict(path, device), graphs[:s.hin_eval_batch_size], device)
        pred, prob, infer_ms = predict_ego_graphs(model, graphs, device, s.hin_eval_batch_size)
        emit(key, pred, prob, infer_ms)
        del model
        clear_gpu_memory()


def evaluate_new_corpus(protocol_name: str, data_path: str, work_dir: str, out_dir: str,
                        seeds: Iterable[int], device, architectures: Optional[Sequence[str]] = None,
                        modes: Sequence[str] = ABLATION_MODES, save_predictions: bool = False,
                        id_col: Optional[str] = None) -> pd.DataFrame:
    """Apply models saved under ``work_dir`` to a new labelled corpus without any parameter update.

    The preprocessing objects, API vocabulary and architecture encoder of each
    seed are the ones fitted on that seed's training partition; nothing is refitted
    on the new corpus. HIN models see one isolated ego graph per file.
    """
    seeds = list(seeds)
    protocol = get_protocol(protocol_name)
    df_new = read_corpus(data_path, standardize=protocol.standardize_arch)
    ensure_binary_labels(df_new["label"].values, data_path)
    id_col = _resolve_id_col(df_new, id_col)
    log = ResultLog(save_predictions)
    print(f"Evaluating '{protocol.name}' models on {data_path} ({len(df_new)} files) | device={device}")

    def emitter(df, context):
        y = df["label"].values
        ids = _sample_ids(df, id_col)

        def emit(model_key, pred, prob, infer_ms):
            log.add(context, model_key, y, pred, prob, sample_ids=ids, infer_ms_per_sample=infer_ms)
        return emit

    if protocol.name == "loao":
        for arch in (architectures or ARCHITECTURES):
            df_arch = df_new[df_new["arch"] == arch].reset_index(drop=True)
            if df_arch.empty:
                print(f"No files with architecture {arch}; skipped.")
                continue
            for seed in seeds:
                seed_dir = os.path.join(work_dir, f"arch_{arch}", f"seed_{seed}")
                if not os.path.exists(os.path.join(seed_dir, protocol.baseline_files["preprocessors"])):
                    print(f"Missing artifacts in {seed_dir}; skipped.")
                    continue
                pipe, api_to_id, le_arch, x_train = load_seed_artifacts(protocol, seed_dir)
                feats = pipe.transform(df_arch)
                emit = emitter(df_arch, {"protocol": "loao", "held_out_arch": arch, "seed": seed, "split": "new"})
                _evaluate_saved_baselines(protocol, seed_dir, feats, pipe, x_train, device, emit)
                graphs = build_ego_graphs(df_arch, feats.hin_file, api_to_id, le_arch)
                _evaluate_saved_hin(protocol, seed_dir, graphs, device, emit)
        return log.write(out_dir, ["held_out_arch", "split"])

    if protocol.name == "ablation":
        for seed in seeds:
            seed_dir = os.path.join(work_dir, f"seed_{seed}")
            prep_path = os.path.join(seed_dir, protocol.baseline_files["preprocessors"])
            if not os.path.exists(prep_path):
                print(f"Missing artifacts in {seed_dir}; skipped.")
                continue
            pipe = FeaturePipeline.from_state(load_joblib(prep_path))
            x_new = pipe.transform(df_new).hin_file
            for mode in modes:
                mode_dir = os.path.join(seed_dir, mode)
                mappings_path = os.path.join(mode_dir, protocol.baseline_files["mappings"])
                if not os.path.exists(mappings_path):
                    continue
                mappings = load_joblib(mappings_path)
                graphs = build_ego_graphs(df_new, _ablation_inputs(mode, x_new, pipe),
                                          mappings["api_to_id"], mappings["le_arch"], mode=mode)
                emit = emitter(df_new, {"protocol": "ablation", "seed": seed, "ablation_mode": mode, "split": "new"})
                _evaluate_saved_hin(protocol, mode_dir, graphs, device, emit)
        return log.write(out_dir, ["ablation_mode", "split"])

    for seed in seeds:
        seed_dir = os.path.join(work_dir, f"seed_{seed}")
        if not os.path.exists(os.path.join(seed_dir, protocol.baseline_files["preprocessors"])):
            print(f"Missing artifacts in {seed_dir}; skipped.")
            continue
        print(f"\n=== seed {seed} ===")
        if protocol.eval_deterministic is not None:
            seed_everything(seed, protocol.eval_deterministic)
        pipe, api_to_id, le_arch, x_train = load_seed_artifacts(protocol, seed_dir)
        feats = pipe.transform(df_new)
        emit = emitter(df_new, {"protocol": protocol.name, "seed": seed, "split": "new"})
        _evaluate_saved_baselines(protocol, seed_dir, feats, pipe, x_train, device, emit)
        graphs = build_ego_graphs(df_new, feats.hin_file, api_to_id, le_arch)
        _evaluate_saved_hin(protocol, seed_dir, graphs, device, emit)
    return log.write(out_dir, ["split"])


__all__ = ["run_training", "run_ablation", "evaluate_new_corpus", "load_seed_artifacts", "FULL_GRAPH"]
