# Reproduction guide

## What runs now

The `er_lab` package uses only the Python standard library and supports Python 3.12+. The demo starts from synthetic scored pairs and exercises the final decision/evaluation stages. It does not train or run a multilingual model.

```bash
PYTHONPATH=src python3 -m er_lab demo
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 scripts/check_repository.py
```

These commands do not install anything or download data. To use `er-lab` as a command on any OS, create an optional environment and run `python -m pip install -e .`.

For your own labelled pair lists, call:

```bash
er-lab score --truth truth.tsv --predictions matching_results.tsv --candidates candidate_pairs.tsv
```

Truth and predictions must use the header `source1_entity_id<TAB>matched_entity_ids`. Candidates must use `source1_entity_id<TAB>candidate_entity_ids`. Each S1 appears exactly once in all files. Empty sets are written as an empty second field. Target lists are comma-separated S2/S3 IDs with no duplicate IDs. The evaluator checks coverage and candidate membership; it does not verify target IDs against original source tables.

## Historical source layout

`experiments/hackathon/` preserves all available top-level `.py` scripts and the small learned alias map. `upgrade_trial_02/src/` retains its original relative nesting inside that folder so the source's `parents[2]` project-root logic remains meaningful. Python sources were copied byte-for-byte; their original checksums and destinations are in [source_archive.json](../reports/manifests/source_archive.json).

Historical scripts are archived experiments, not a tested current API. Some execute work when imported, expect files in the current working directory, overwrite their own experiment outputs, or use library APIs tied to the original environment. Inspect a script before running it.

## Optional reconstruction with your own authorized dataset

The original dataset is not bundled. If you have access to it, restore it outside the repository or into the ignored historical workspace. No download URL or redistribution permission is assumed.

The following is the *available early-stage dependency order*, not a claim that a complete V11 pipeline can be reproduced from this archive:

```bash
# Run from the repository root, in a separate environment you create yourself.
python -m pip install -r requirements/legacy.txt
cd experiments/hackathon
python 00_to_parquet.py /path/to/student_resource
python 01_eda.py
python 02_eda2.py
python 03_norm_eval.py
# Inspect and fix the split policy before using this for a fresh evaluation.
python 09_build_dev_candidates.py
python 10_build_dev_features.py
python 11_build_train_candidates.py
python 12_build_train_features.py
python 13_train_lgbm_baseline.py
python 18_build_features_v2.py
python 19_train_lgbm_v2.py
python 20_tune_thresholds_v2.py
python 21_eval_one_owner_v2.py
```

The candidate builders import earlier blocking modules that are included in the same directory. `04`–`08`, `14`–`17`, `22`, and `23` retain the individual blocking, threshold, error-analysis, and model experiments described in [the workflow](workflow.md).

The dependency files are reconstructed from imports. They are **not pinned historical lockfiles** and installation or ML execution was not tested during this cleanup. The deleted environment cannot be recovered by pretending its package versions are known. A retained model configuration separately records PyTorch 2.14.0, Sentence Transformers 6.1.0, and Transformers 5.17.0; this is partial historical metadata, not a complete or newly validated environment specification.

## Later components and missing inputs

| Component | Preserved entry point | What must be supplied or reconstructed |
| --- | --- | --- |
| Character address/name development retrieval | Reports only | Original V8/V9 terminal code, vectorizer settings, and source data |
| Original multilingual development retrieval | Reports and later inference worker | Exact V10 script and input filters |
| Bi-encoder fine-tuning | Reports and metadata | Training script, pair generation details, data, environment, and retrained weights |
| Fine-tuned inference | `upgrade_trial_02/src/neural_mac.py` | Handoff inputs and locally trained encoder |
| Targeted Indic inference | `upgrade_trial_02/src/indic_retrieval.py` | Source Parquet files, encoder, Apple MPS |
| Rule candidates | `upgrade_trial_02/src/rule_candidates.py` | Source Parquet tables and alias map |
| Hard-negative training/evaluation | `upgrade_trial_02/src/experiment.py` | V5/V11 features and retrieval candidates removed during cleanup |
| Batched scoring and assembly | `upgrade_trial_02/src/mac_finish_fast.py` | Candidate shards, source tables, 35-feature model(s), completed manifests |
| Candidate export | `upgrade_trial_02/src/export_candidate_pairs.py` | The exact candidate inputs actually scored and matching output |

Configuration snapshots under `reports/manifests/` are evidence rather than plug-and-play runtime configs. Local home-directory prefixes were removed; model paths now refer to the historical workspace and the corresponding weights are absent.

## Tests and CI

CI runs the public library's tests on Python 3.12 and 3.13, checks source syntax without importing heavy ML packages, verifies archived checksums, and rejects accidental large binary artifacts or local absolute paths. It does not train a model or recompute historical scores.

After changes, run the same commands locally. The tests cover singleton treatment, entity-level versus pooled metrics, oracle versus measured scores, malformed TSVs, deterministic ownership, prediction coverage, and round-trip synthetic output.

