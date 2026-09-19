# Feature table format

Each corpus is a CSV file with one row per ELF executable. After static and dynamic analysis, the two outputs are aligned per executable, and only files present in both are kept.

## Required columns

| Column | Type | Description |
|---|---|---|
| `sha256` | string | SHA-256 of the ELF file. Used as the sample identifier in manifests, partitions and prediction files. |
| `label` | int | `0` = benign, `1` = malware. |
| `arch` | string | Target architecture from the ELF header. Raw strings such as `ARM, EABI5` or `Intel 80386` are mapped to `ARM`, `MIPS`, `PowerPC`, `x64`, `x86` (anything else becomes `Other`). |

## Static features (file node, s_i and o_i)

| Column | Description |
|---|---|
| `mean_entropy`, `max_entropy`, `min_entropy` | Mean, maximum and minimum Shannon entropy (see `appendix_feature_extraction_reproducibility.md` for the unit over which entropy is computed). |
| `file_size_mb` | File size in MB. |
| `total_opcodes` | Number of disassembled instructions. |
| `opcode_counts` | Opcode frequency dictionary, e.g. `"{'mov': 120, 'bl': 33}"`. Each opcode is repeated `min(count, 20)` times before TF-IDF. Alternatively `opcode_sequence`: a whitespace-separated opcode string. |

## Dynamic features

| Column | Description |
|---|---|
| `api_sequence` | Ordered API trace as a Python list literal, e.g. `"['socket', 'connect', 'send']"`. Alternatively `api_list`: a whitespace-separated string. The HIN uses the set of APIs; the Bi-GRU and fusion baselines use the first 200 calls in order. |
| `socket_calls`, `connect_calls`, `send_calls`, `fork_calls`, `exec_calls`, `execve_calls`, `sys_calls`, `file_ops`, `net_ops`, `proc_ops`, `total_calls`, `unique_calls` | Numerical runtime counters (d_i). The counters present in the training data are used. Missing or non-numeric values are set to 0 before standardization. |

## Optional columns

| Column | Used by | Description |
|---|---|---|
| `malware_family` | `--protocol family` | Family label of malware samples (empty for benign files). Only used to form groups; never used as a feature. |

Columns not listed above are ignored.

## Preprocessing

All data-dependent transformations are fitted on the training partition of each split and applied unchanged to its test partition and to the 2024 corpus:

| Object | Fitted on | Use at evaluation time |
|---|---|---|
| z-score scalers (static, runtime) | training partition | applied unchanged |
| opcode TF-IDF vocabulary and IDF (≤ 100 terms) | training partition | applied unchanged |
| API tokenizer (baselines, ≤ 5,000 tokens) | training partition | unknown APIs → `<UNK>` |
| API node vocabulary (HIN) | training HIN | only known APIs create edges |
| architecture encoder (HIN) | training HIN | an unseen architecture creates no edge |

## Example

`python scripts/make_synthetic_corpus.py --n 600 --out data/synthetic.csv` writes a random table with this schema. It can be used to test the pipeline; its results mean nothing.
