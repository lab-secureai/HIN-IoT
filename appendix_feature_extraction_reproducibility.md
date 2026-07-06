# Appendix: Feature Extraction and Reproducibility Notes

This document provides implementation-level details for the feature extraction and reproducibility pipeline used in the HIN-based cross-architecture IoT malware detection framework. It is intended as a companion appendix for the paper. The main paper reports the methodology and experimental results, while this document provides additional details about static analysis, dynamic analysis, feature definitions, and reproducibility controls.

> **Note:** Replace repository paths, script names, and directory names with the exact structure of your released GitHub repository.

---

## Table of Contents

- [A. Static Feature Extraction](#a-static-feature-extraction)
- [B. Dynamic Feature Extraction](#b-dynamic-feature-extraction)
- [C. Reproducibility Notes](#c-reproducibility-notes)
- [D. Suggested Repository Structure](#d-suggested-repository-structure)
- [E. Citation from the Paper](#e-citation-from-the-paper)

---

# A. Static Feature Extraction

## A.1 Objective

Static feature extraction is the first step of the proposed data processing pipeline. This step extracts features from ELF binaries without executing them. The objective is to capture file identity, architecture metadata, entropy and packing characteristics, code-level properties, and ELF structural organization.

Static analysis is useful because it can be applied safely and efficiently to large binary collections. It also provides information about the structure and layout of the executable before runtime behavior is observed.

## A.2 Input Directories

The extraction script scans two input directories:

```text
/hin_iot/data/malicious
/hin_iot/data/benign
```

Files under `/hin_iot/data/malicious` are assigned label `1`, corresponding to malware. Files under `/hin_iot/data/benign` are assigned label `0`, corresponding to benign executables.

## A.3 Extracted Static Features

The static extraction stage produces the following fields:

| Feature | Description |
|---|---|
| `file_path` | Absolute path of the binary file, used for traceability and data management. |
| `file_hash` | SHA-256 hash of the binary file. |
| `label` | Ground-truth label: `1` for malware and `0` for benign. |
| `arch` | CPU architecture of the ELF file. |
| `bits` | Word size of the binary, such as 32-bit or 64-bit. |
| `endian` | Byte order of the binary. |
| `is_packed` | Boolean indicator of whether the binary appears packed. |
| `mean_entropy` | Mean Shannon entropy of the binary. |
| `max_entropy` | Maximum entropy across file blocks. |
| `min_entropy` | Minimum entropy across file blocks. |
| `entropy_sequence` | Sequence of entropy values computed over smaller file regions. |
| `num_functions` | Number of detected functions. |
| `avg_func_size` | Average detected function size. |
| `num_strings` | Number of printable strings. |
| `syscall_count` | Number of detected system calls. |
| `opcode_counts` | Frequency distribution of opcode types. |
| `total_opcodes` | Total number of counted opcodes. |
| `num_sections` | Number of ELF sections. |
| `total_section_size` | Total size of all ELF sections. |
| `text_size` | Size of the `.text` section. |
| `data_size` | Size of the `.data` section. |
| `bss_size` | Size of the `.bss` section. |
| `num_segments` | Number of ELF program segments. |
| `file_size_mb` | File size in megabytes. |

## A.4 Feature Groups

### A.4.1 Identity and Label Features

This group contains attributes used for sample identification, dataset verification, and duplicate control. These fields are not intended to directly represent malware behavior.

- **`file_hash`**: SHA-256 fingerprint of the binary. This value is used for duplicate detection and can support external lookup using services such as VirusTotal.
- **`label`**: Ground-truth class label. A value of `1` denotes malware, while a value of `0` denotes benign software.

### A.4.2 Architecture and Metadata Features

IoT malware is commonly compiled for multiple hardware platforms. Therefore, architecture metadata is important for cross-platform analysis.

- **`arch`**: CPU architecture of the ELF binary, such as ARM, MIPS, PowerPC, Hitachi SH, x86, or x64.
- **`bits`**: Binary word size, indicating whether the executable is 32-bit or 64-bit.
- **`endian`**: Byte order of the binary. This is important for accurate disassembly, especially for architectures such as MIPS, which may appear in both little-endian and big-endian variants.
- **`file_size_mb`**: Physical file size measured in megabytes.

### A.4.3 Entropy and Packing Features

Entropy-related features capture obfuscation, compression, encryption, and packing characteristics.

- **`is_packed`**: Boolean indicator showing whether packing is detected, for example using UPX-like signatures.
- **`mean_entropy`**: Average Shannon entropy of the binary. Values close to 8 may indicate compressed or encrypted content.
- **`max_entropy`** and **`min_entropy`**: Maximum and minimum entropy values across file blocks.
- **`entropy_sequence`**: Entropy values computed over smaller file regions. A low-entropy header followed by high-entropy regions may indicate packed or encrypted executable content.

### A.4.4 Code and Logic Features

This group reflects the internal code structure of the executable.

- **`num_functions`**: Total number of detected functions.
- **`avg_func_size`**: Average size of detected functions. Obfuscated malware may contain many small meaningless functions or unusually large functions.
- **`num_strings`**: Number of readable strings. Packed binaries often contain fewer readable strings, while unpacked binaries may reveal IP addresses, URLs, shell commands, or configuration values.
- **`syscall_count`**: Number of detected system calls.
- **`total_opcodes`**: Total number of machine instructions counted in the binary.
- **`opcode_counts`**: Frequency distribution of opcode types, such as `mov`, `xor`, and `add`.

### A.4.5 ELF Structure Features

This group describes how the ELF file is organized on disk and mapped into memory.

- **`num_sections`**: Number of ELF sections, such as `.text`, `.data`, and `.bss`.
- **`total_section_size`**: Total size of all sections.
- **`text_size`**: Size of the `.text` section, which contains executable code.
- **`data_size`**: Size of the `.data` section, which stores initialized global variables.
- **`bss_size`**: Size of the `.bss` section, which stores uninitialized global variables and reserved memory.
- **`num_segments`**: Number of program segments used when the binary is loaded into memory.

The `bss_size` feature is relevant in IoT malware analysis. Some IoT malware families, including Mirai-like variants, may allocate large `.bss` regions to store scanning buffers, IP address lists, or data structures used for DDoS activity. Therefore, an unusually large `.bss` region relative to file size can be a suspicious indicator.

## A.5 Output File

The output of the static analysis step is:

```text
/hin_iot/data/static_features.csv.gz
```

This compressed CSV file contains approximately 19,485 rows, where each row corresponds to one ELF binary sample. The file includes all extracted static fields listed above.

## A.6 Logging

During extraction, the script records processing logs for monitoring and debugging. The logs include the number of successfully processed files, extraction errors, skipped files, failures during CFG or binary analysis, unsupported or corrupted ELF files, and missing file paths.

---

# B. Dynamic Feature Extraction

## B.1 Objective

Dynamic feature extraction is the second step of the data processing pipeline. In this step, ELF binaries are emulated using the Qiling framework in a controlled environment. Instead of executing binaries directly on physical IoT devices, Qiling simulates program execution and records runtime behavior.

Dynamic analysis captures behavior that may not be visible from static structure alone, such as API invocations, networking activity, process creation, file access, memory operations, and API-call ordering.

## B.2 Emulation Protocol

The dynamic analysis script scans ELF binaries and excludes unsupported files, including shared libraries with the `.so` extension, kernel modules with the `.ko` extension, non-ELF files, and samples unsupported by the emulation environment.

The architecture of each binary is detected using the `file` command. Based on the detected architecture, the corresponding root filesystem is selected. Examples include:

```text
/usr/x86_64-linux-gnu
/usr/arm-linux-gnueabi
```

Each sample is emulated with a fixed timeout of:

```text
500,000 microseconds
```

This timeout prevents infinite loops, hanging execution, and excessive runtime. The same timeout is applied across samples to maintain a consistent execution budget.

## B.3 Extracted Dynamic Features

The dynamic extraction stage produces the following fields:

| Feature | Description |
|---|---|
| `file_path` | Absolute path of the analyzed file. |
| `file_hash` | SHA-256 hash of the binary. |
| `label` | Ground-truth label: `1` for malware and `0` for benign. |
| `socket_calls` | Number of socket creation calls. |
| `connect_calls` | Number of connection attempts. |
| `bind_calls` | Number of socket binding calls. |
| `listen_calls` | Number of listening calls. |
| `accept_calls` | Number of accepted incoming connections. |
| `send_calls` | Number of send operations. |
| `recv_calls` | Number of receive operations. |
| `fork_calls` | Number of process duplication calls. |
| `execve_calls` | Number of external program execution calls. |
| `open_calls` | Number of file open operations. |
| `write_calls` | Number of file write operations. |
| `read_calls` | Number of file read operations. |
| `stat_calls` | Number of file metadata queries. |
| `gettimeofday_calls` | Number of system time queries. |
| `mmap_calls` | Number of memory mapping calls. |
| `close_calls` | Number of file close operations. |
| `setsockopt_calls` | Number of socket option configuration calls. |
| `ioctl_calls` | Number of input/output control calls. |
| `nanosleep_calls` | Number of sleep calls. |
| `poll_calls` | Number of polling calls. |
| `mem_hook_count` | Number of memory hook events. |
| `loop_16590` | Loop-related execution label. |
| `loop_165f8` | Loop-related execution label. |
| `loop_165fc` | Loop-related execution label. |
| `instr_count` | Total number of executed instructions. |
| `api_sequence` | Ordered sequence of API calls observed during emulation. |
| `unique_apis` | Number of distinct API types invoked. |

## B.4 Feature Groups

### B.4.1 Networking Behavior

Networking behavior is one of the most important dynamic indicators for IoT malware. Many IoT malware families communicate with command-and-control servers, perform scanning, propagate across networks, or launch DDoS attacks.

- **`socket_calls`**: Number of socket creation calls.
- **`connect_calls`**: Number of attempts to connect to remote addresses.
- **`bind_calls`**: Number of calls used to bind a socket to a local port.
- **`listen_calls`** and **`accept_calls`**: Calls used to listen for and accept incoming connections.
- **`send_calls`** and **`recv_calls`**: Data transmission and reception calls.
- **`setsockopt_calls`**: Socket option configuration calls. Malware may use this function to optimize scanning or DDoS behavior.
- **`poll_calls`**: Calls used to monitor multiple file descriptors or sockets.

### B.4.2 Process and System Behavior

- **`fork_calls`**: Process duplication calls. Malware may use `fork()` to run in the background, daemonize itself, or create worker processes.
- **`execve_calls`**: Calls used to execute external programs, such as `/bin/sh`.
- **`gettimeofday_calls`**: Calls used to retrieve system time.
- **`nanosleep_calls`**: Calls used to suspend execution. Malware may use sleep behavior to delay execution or evade time-limited sandboxes.

### B.4.3 File and I/O Activity

- **`open_calls`** and **`close_calls`**: File open and close operations.
- **`read_calls`** and **`write_calls`**: File read and write operations.
- **`stat_calls`**: File metadata queries.

These calls may indicate access to configuration files, credential files, device files, or payload-writing behavior.

### B.4.4 Memory and Device Interaction

- **`mmap_calls`**: Memory mapping calls.
- **`mem_hook_count`**: Memory hook events captured during emulation.
- **`ioctl_calls`**: Input/output control calls. IoT malware may use `ioctl()` to interact with network interfaces, device drivers, or watchdog timers.

### B.4.5 Execution Summary and API Sequence

- **`instr_count`**: Total number of executed instructions during emulation.
- **`unique_apis`**: Number of distinct API types invoked.
- **`api_sequence`**: Ordered sequence of API calls observed during emulation.

The `api_sequence` feature is particularly important because API order can reveal behavioral patterns. For example:

```text
socket -> connect -> execve
```

may indicate reverse-shell behavior, whereas:

```text
open -> read
```

may represent ordinary file access.

### B.4.6 Loop Label Features

The features `loop_16590`, `loop_165f8`, and `loop_165fc` represent loop-related execution labels. The hexadecimal suffixes correspond to code offsets or addresses identified during binary analysis.

These loop labels may capture repeated execution patterns associated with decryption loops, scanner loops, and attack loops. Loop behavior is relevant in IoT malware because botnets often rely on continuous scanning, brute-force attempts, and repeated attack traffic generation.

## B.5 Output File

The output of the dynamic analysis step is:

```text
/hin_iot/data/dynamic_features.csv.gz
```

This compressed CSV file contains approximately 20,000 rows, where each row corresponds to one analyzed sample. The file includes all extracted dynamic fields listed above.

## B.6 Logging

The dynamic analysis script records logs to monitor the completeness and reliability of the emulation process. The logs include successfully processed samples, execution errors, skipped files, unsupported architectures, Qiling emulation failures, missing root filesystem configurations, corrupted binaries, and unsupported file types.

---

# C. Reproducibility Notes

To ensure reliability and reproducibility, the pipeline is implemented as an automated process covering static feature extraction, dynamic feature extraction, graph construction, model training, and evaluation.

## C.1 Dataset Deduplication

All executable files are identified using SHA-256 hashes. Duplicate binaries are removed or controlled to prevent the same executable from appearing multiple times in the dataset.

## C.2 Controlled Dynamic Execution

Dynamic analysis is performed in a controlled Qiling-based emulation environment rather than on physical IoT devices. Unsupported files are excluded before execution. A fixed timeout of 500,000 microseconds is applied to each sample to avoid hanging executions and ensure consistent runtime constraints.

## C.3 Architecture-Aware Emulation

The architecture of each ELF binary is detected before emulation, and the corresponding root filesystem is selected accordingly. This reduces execution errors caused by incompatible runtime environments and improves consistency across architectures.

## C.4 Deterministic Setup

Random seeds are fixed for data splitting and model initialization. This ensures that repeated experiments under the same configuration produce consistent training and evaluation results.

## C.5 Leakage Prevention

Graph construction is performed independently for each training fold. This prevents structural information from the test set from leaking into the training graph and ensures a fair evaluation setting.

## C.6 Executable-Level Evaluation

All evaluation metrics are computed at the executable level. This avoids inflated performance caused by duplicated fragments, extracted subcomponents, or graph nodes that do not correspond to independent executable samples.

## C.7 Logging and Error Tracking

Both static and dynamic extraction scripts record processing logs, including successfully processed files, skipped samples, unsupported binaries, extraction failures, and emulation errors. These logs make it possible to audit the dataset construction process and reproduce the same filtering decisions.

---

# D. Suggested Repository Structure

A clean GitHub repository can use the following structure:

```text
hin-iot-malware/
├── README.md
├── requirements.txt
├── configs/
│   ├── static_extraction.yaml
│   ├── dynamic_extraction.yaml
│   └── train_config.yaml
├── scripts/
│   ├── extract_static_features.py
│   ├── extract_dynamic_features.py
│   ├── build_hetero_graph.py
│   ├── train_models.py
│   └── evaluate_models.py
├── data/
│   ├── README.md
│   └── sample_schema/
│       ├── static_features_schema.csv
│       └── dynamic_features_schema.csv
├── docs/
│   └── appendix_feature_extraction.md
├── results/
│   ├── binary_detection/
│   ├── per_architecture/
│   ├── leave_one_architecture_out/
│   ├── temporal_2024/
│   └── ablation/
└── LICENSE
```

If the raw malware binaries cannot be shared, the repository should provide feature schemas, preprocessing scripts, graph construction scripts, model definitions, evaluation scripts, instructions for reconstructing the dataset from the original sources, and checksums or metadata when legally permissible.

---

# E. Citation from the Paper

In the main paper, the appendix can be referenced as:

```latex
\appendix
\section{Feature Extraction and Reproducibility Details}
\label{app:github_appendix}

Detailed descriptions of the static feature extraction pipeline, dynamic emulation protocol, extracted feature definitions, logging procedure, and reproducibility controls are provided in the public repository:

\begin{center}
\url{https://github.com/YOUR_USERNAME/YOUR_REPOSITORY}
\end{center}

The repository also contains the graph construction scripts, model training configuration, and evaluation code used in the experiments.
```

If the repository is not public yet, use:

```latex
The source code and supplementary appendix will be released upon publication.
```
