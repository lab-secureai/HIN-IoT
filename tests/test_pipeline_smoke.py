"""End-to-end smoke test on a small synthetic corpus (needs PyTorch, PyTorch Geometric, LightGBM).

Run with:  pytest -q tests/test_pipeline_smoke.py
It takes a few minutes on CPU. The numbers produced are meaningless; the test
checks that every protocol trains, saves, reloads and evaluates without error,
and that reloaded models reproduce the predictions made right after training.
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("lightgbm")

from hin_iot.data import read_corpus, stratified_holdout_split  # noqa: E402
from hin_iot.experiments import evaluate_new_corpus, run_ablation, run_training  # noqa: E402

DEVICE = torch.device("cpu")
SEEDS = [10]
N_MODELS = 11


def _raw(path):
    return pd.read_csv(path / "results_raw.csv")


def test_holdout_train_and_reload(primary_csv, tmp_path):
    work = tmp_path / "holdout"
    run_training("holdout", str(primary_csv), str(work), SEEDS, DEVICE, smoke=True, save_predictions=True)
    raw = _raw(work)
    assert len(raw) == N_MODELS and raw["accuracy"].between(0, 1).all()
    for f in ["preprocessors.pkl", "X_tr_hin_file.npy", "baseline_1_lgbm.joblib", "IMPROVED_RGCN.pt"]:
        assert (work / "seed_10" / f).exists()

    # Re-evaluate the saved models on the same test partition: results must match training-time predictions.
    _, df_te = stratified_holdout_split(read_corpus(str(primary_csv)), 10)
    test_csv = tmp_path / "test_partition.csv"
    df_te.to_csv(test_csv, index=False)
    evaluate_new_corpus("holdout", str(test_csv), str(work), str(tmp_path / "re_eval"), SEEDS, DEVICE)
    again = _raw(tmp_path / "re_eval").set_index("model_key")
    first = raw.set_index("model_key")
    for key in first.index:
        assert np.isclose(first.loc[key, "accuracy"], again.loc[key, "accuracy"]), key


def test_holdout_temporal_evaluation(primary_csv, temporal_csv, tmp_path):
    work = tmp_path / "holdout"
    run_training("holdout", str(primary_csv), str(work), SEEDS, DEVICE, smoke=True)
    evaluate_new_corpus("holdout", str(temporal_csv), str(work), str(tmp_path / "t2024"), SEEDS, DEVICE,
                        save_predictions=True)
    assert len(_raw(tmp_path / "t2024")) == N_MODELS
    assert (tmp_path / "t2024" / "predictions.csv.gz").exists()


def test_loao(primary_csv, temporal_csv, tmp_path):
    work = tmp_path / "loao"
    run_training("loao", str(primary_csv), str(work), SEEDS, DEVICE, architectures=["MIPS"], smoke=True)
    raw = _raw(work)
    assert set(raw["split"]) == {"ID", "OOD"} and len(raw) == 2 * N_MODELS
    assert "all_malware_f1" in raw.columns
    evaluate_new_corpus("loao", str(temporal_csv), str(work), str(tmp_path / "loao_new"), SEEDS, DEVICE,
                        architectures=["MIPS"])
    assert len(_raw(tmp_path / "loao_new")) == N_MODELS


def test_family(primary_csv, temporal_csv, tmp_path):
    work = tmp_path / "family"
    run_training("family", str(primary_csv), str(work), SEEDS, DEVICE, smoke=True)
    assert len(_raw(work)) == N_MODELS
    evaluate_new_corpus("family", str(temporal_csv), str(work), str(tmp_path / "family_new"), SEEDS, DEVICE)
    assert len(_raw(tmp_path / "family_new")) == N_MODELS


def test_ablation(primary_csv, temporal_csv, tmp_path):
    work = tmp_path / "ablation"
    run_ablation(str(primary_csv), str(work), SEEDS, DEVICE, smoke=True)
    raw = _raw(work)
    assert len(raw) == 4 * 7 and set(raw["ablation_mode"]) == {
        "Full_Graph", "No_API_Node", "No_Arch_Node", "Static_Opcode_Only"}
    evaluate_new_corpus("ablation", str(temporal_csv), str(work), str(tmp_path / "abl_new"), SEEDS, DEVICE)
    assert len(_raw(tmp_path / "abl_new")) == 4 * 7
