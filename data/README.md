# Data

Feature tables (`primary.csv`, `temporal_2024.csv`) are not distributed with this repository; see the main README for access to the IoTPOT collections. Their format is described in `docs/DATA.md`.

* `manifests/` — SHA-256, label and architecture of every file in both corpora, and the corpus summary (class × architecture counts, duplicates, cross-corpus overlap).
* `partitions/<protocol>/seed_<s>.csv.gz` — the split (`train`, `test`, `id_test`, `ood_test`) of every file for each protocol and seed.

Both are generated with:

```bash
python scripts/export_partitions.py --primary data/primary.csv --temporal data/temporal_2024.csv --out-dir data
```
