"""HIN-IoT: heterogeneous information networks for cross-architecture IoT malware detection.

Reference implementation for:
    D.-T. Tran, A.-T. Tran, K. Nguyen-An, T.-D. Luong,
    "Cross-Architecture IoT Malware Detection under Temporal Shift via
    Heterogeneous Information Networks".

Modules that only need NumPy/pandas/scikit-learn (``config``, ``data``,
``features``, ``metrics``) can be imported without PyTorch. Graph construction,
models and training live in ``graph``, ``models`` and ``training`` and require
PyTorch and PyTorch Geometric.
"""

__version__ = "1.0.0"
