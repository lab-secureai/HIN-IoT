import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_corpus(path: Path, n: int, seed: int) -> Path:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_corpus.py"),
                    "--n", str(n), "--out", str(path), "--seed", str(seed)], check=True)
    return path


@pytest.fixture(scope="session")
def primary_csv(tmp_path_factory):
    return make_corpus(tmp_path_factory.mktemp("data") / "primary.csv", 400, 0)


@pytest.fixture(scope="session")
def temporal_csv(tmp_path_factory):
    return make_corpus(tmp_path_factory.mktemp("data") / "temporal.csv", 200, 1)
