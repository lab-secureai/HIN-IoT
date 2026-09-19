"""Checks that need only NumPy, pandas and scikit-learn."""
import numpy as np
import pandas as pd

from hin_iot.data import (family_disjoint_split, loao_split, read_corpus, standardize_arch,
                          stratified_holdout_split)
from hin_iot.features import FeaturePipeline, static_opcode_only
from hin_iot.metrics import classification_metrics, summarize


def test_standardize_arch():
    cases = {"ARM, EABI5": "ARM", "aarch64": "ARM", "MIPS R3000": "MIPS", "PowerPC or cisco 4500": "PowerPC",
             "x86-64": "x64", "Intel 80386": "x86", "SuperH": "Other"}
    for raw, expected in cases.items():
        assert standardize_arch(raw) == expected
        assert standardize_arch(standardize_arch(raw)) == expected  # idempotent


def test_holdout_split_is_reproducible_and_disjoint(primary_csv):
    df = read_corpus(primary_csv)
    a_tr, a_te = stratified_holdout_split(df, 42)
    b_tr, b_te = stratified_holdout_split(df, 42)
    assert a_tr["sha256"].tolist() == b_tr["sha256"].tolist()
    assert not set(a_tr["sha256"]) & set(a_te["sha256"])
    assert abs(len(a_te) / (len(a_tr) + len(a_te)) - 0.2) < 0.01


def test_loao_split_holds_out_architecture(primary_csv):
    df = read_corpus(primary_csv)
    tr, id_te, ood = loao_split(df, "MIPS", 10)
    assert (ood["arch"] == "MIPS").all()
    assert "MIPS" not in set(tr["arch"]) | set(id_te["arch"])
    assert len(tr) + len(id_te) + len(ood) <= len(df)


def test_family_split_is_family_disjoint(primary_csv):
    df = read_corpus(primary_csv)
    for seed in (10, 42, 100):
        tr, te = family_disjoint_split(df, seed)
        fam_tr = set(tr.loc[tr["label"] == 1, "malware_family"])
        fam_te = set(te.loc[te["label"] == 1, "malware_family"])
        assert not fam_tr & fam_te


def test_pipeline_fit_on_train_only_and_state_roundtrip(primary_csv, temporal_csv):
    df = read_corpus(primary_csv)
    tr, te = stratified_holdout_split(df, 10)
    pipe = FeaturePipeline()
    f_tr = pipe.fit_transform(tr)
    assert np.allclose(f_tr.core.mean(axis=0), 0, atol=1e-8)          # scaler fitted on train
    assert f_tr.hin_file.shape[1] == pipe.core_dim + pipe.opcode_dim + len(pipe.dyn_cols)

    new = read_corpus(temporal_csv)
    restored = FeaturePipeline.from_state(pipe.to_state())
    a, b = pipe.transform(new), restored.transform(new)
    assert np.array_equal(a.hin_file, b.hin_file) and np.array_equal(a.api_seq, b.api_seq)

    legacy = {"scaler_core_stat": pipe.scaler_core, "scaler_top_dyn": pipe.scaler_dyn, "tfidf_op": pipe.tfidf_op,
              "tokenizer": pipe.tokenizer, "core_static_cols": pipe.core_cols, "top_dyn_cols": pipe.dyn_cols}
    assert np.array_equal(FeaturePipeline.from_state(legacy).transform(new).hin_file, a.hin_file)


def test_static_opcode_only_mask():
    x = np.arange(2 * 12, dtype=float).reshape(2, 12) + 1
    out = static_opcode_only(x, core_dim=5, opcode_dim=4)
    assert (out[:, :3] == 0).all() and (out[:, 3:9] == x[:, 3:9]).all() and (out[:, 9:] == 0).all()


def test_metrics_and_summary():
    y = np.array([1, 1, 1, 0])
    m = classification_metrics(y, np.ones(4, dtype=int), np.full(4, 0.9))
    assert abs(m["binary_f1"] - m["all_malware_f1"]) < 1e-12
    assert m["specificity"] == 0 and m["balanced_accuracy"] == 0.5
    raw = pd.DataFrame({"model": ["A", "A"], "split": ["t", "t"], "accuracy": [0.9, 0.8]})
    s = summarize(raw, ["split", "model"], ["accuracy"])
    assert abs(s.loc[0, "accuracy_mean"] - 0.85) < 1e-12 and abs(s.loc[0, "accuracy_std"] - 0.0707107) < 1e-6


if __name__ == "__main__":  # allows running without pytest
    import tempfile
    from pathlib import Path

    from conftest import make_corpus
    tmp = Path(tempfile.mkdtemp())
    p, t = make_corpus(tmp / "p.csv", 400, 0), make_corpus(tmp / "t.csv", 200, 1)
    test_standardize_arch()
    test_holdout_split_is_reproducible_and_disjoint(p)
    test_loao_split_holds_out_architecture(p)
    test_family_split_is_family_disjoint(p)
    test_pipeline_fit_on_train_only_and_state_roundtrip(p, t)
    test_static_opcode_only_mask()
    test_metrics_and_summary()
    print("all data/feature tests passed")
