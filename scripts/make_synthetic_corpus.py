#!/usr/bin/env python
"""Generate a small random corpus with the expected schema, for pipeline checks only.

Example:
  python scripts/make_synthetic_corpus.py --n 600 --out data/synthetic_primary.csv --seed 0
  python scripts/make_synthetic_corpus.py --n 300 --out data/synthetic_2024.csv --seed 1
"""
import argparse
import hashlib
import os

import numpy as np
import pandas as pd

ARCHS = ["ARM", "MIPS", "PowerPC", "x64", "x86"]
OPCODES = {
    "ARM": ["ldr", "str", "mov", "bl", "cmp", "b", "add", "sub", "push", "pop"],
    "MIPS": ["lw", "sw", "addiu", "jal", "beq", "bne", "move", "lui", "jr", "nop"],
    "PowerPC": ["lwz", "stw", "addi", "bl", "cmpwi", "beq", "mr", "li", "blr", "mflr"],
    "x64": ["mov", "call", "push", "pop", "lea", "cmp", "jmp", "je", "xor", "ret"],
    "x86": ["mov", "call", "push", "pop", "lea", "cmp", "jmp", "jne", "add", "ret"],
}
BENIGN_APIS = ["open", "read", "write", "close", "mmap", "brk", "fstat", "getuid", "ioctl", "exit"]
MALWARE_APIS = ["socket", "connect", "send", "recv", "fork", "execve", "kill", "setsid", "bind", "prctl"]
FAMILIES = ["mirai", "gafgyt", "tsunami", "mozi", "hajime", "dofloo"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = []
    for i in range(args.n):
        label = int(rng.random() < 0.6)
        arch = ARCHS[rng.integers(len(ARCHS))]
        vocab = OPCODES[arch] + [f"{arch.lower()}_op{j}" for j in range(40)]
        ops = rng.choice(vocab, size=12, replace=False)
        pool = MALWARE_APIS + BENIGN_APIS[:4] if label else BENIGN_APIS + MALWARE_APIS[:2]
        apis = list(rng.choice(pool, size=int(rng.integers(3, 12))))
        rows.append({
            "sha256": hashlib.sha256(f"{args.seed}-{i}".encode()).hexdigest(),
            "label": label,
            "arch": arch,
            "malware_family": FAMILIES[rng.integers(len(FAMILIES))] if label else None,
            "mean_entropy": rng.normal(6.5 if label else 5.5, 0.5),
            "max_entropy": rng.normal(7.5, 0.3),
            "min_entropy": rng.normal(1.0, 0.5),
            "file_size_mb": abs(rng.normal(0.1 if label else 0.3, 0.05)),
            "total_opcodes": int(rng.integers(500, 20000)),
            "opcode_counts": str({op: int(rng.integers(1, 40)) for op in ops}),
            "api_sequence": str(apis),
            "socket_calls": apis.count("socket"), "connect_calls": apis.count("connect"),
            "send_calls": apis.count("send"), "fork_calls": apis.count("fork"),
            "exec_calls": apis.count("execve"), "execve_calls": apis.count("execve"),
            "sys_calls": len(apis), "file_ops": sum(a in ("open", "read", "write") for a in apis),
            "net_ops": sum(a in ("socket", "connect", "send", "recv", "bind") for a in apis),
            "proc_ops": sum(a in ("fork", "execve", "kill", "setsid") for a in apis),
            "total_calls": len(apis), "unique_calls": len(set(apis)),
        })
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {args.n} rows to {args.out}")


if __name__ == "__main__":
    main()
