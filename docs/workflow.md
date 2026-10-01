# Experiment history

This timeline was reconstructed from the retained source and reports. Numbered scripts reflect the early experiment order; later retrieval work ran partly through terminal blocks. An experiment number is not a guarantee that its output was deployed in the submitted trial.

| Stage | Motive and implementation | Outcome / retained evidence |
| --- | --- | --- |
| 1. Parse and store | `00_to_parquet.py` converted explicit-tab, all-string TSV input into compressed Parquet. | EDA recorded 2,206,821 training S1 and 1,732,544 test S1 entities. |
| 2. EDA | `01_eda.py`, `02_eda2.py` examined missing addresses, duplication, countries, singleton frequency, decoys, and cross-source ownership. | 123,247 training singletons; no training target record had multiple S1 owners. France appeared only in test. |
| 3. Normalize | `norm.py`, `03_norm_eval.py` normalized legal suffixes, punctuation, Unicode text, address abbreviations, and learned final-address-component aliases. | 87 learned aliases. Normalization errors motivated conservative rewriting and later soft agreement features. |
| 4. Exact blocking V1 | `04_blocking_eval.py` established a development retrieval baseline. | S2 recall 0.714055; S3 recall 0.620689. |
| 5. Rule experiments V2–V4 | `05`–`07` explored rare name/address tokens, compound address words, numeric keys, and capped loose fallback. | Candidate budgets and recall measured for each variation; see original aggregate reports. |
| 6. V5 rule blocker | `08_blocking_eval_v5.py` fuzzy-ranked loose fallback candidates before union with tighter rules. | 2,530,505 development candidates, 85.286647% true-link recall. |
| 7. Label and featurize | `09`–`12` built separate development/training candidates and the first similarity features. | Development: 22,184 S1 entities and 2,530,505 pairs. Training: 44,139 S1 entities and 5,017,353 pairs. |
| 8. First LightGBM | `13` trained the 20-input baseline; `14` tuned thresholds; `15` added one-owner resolution. | Baseline best global-threshold macro F0.5: 0.871792. |
| 9. Error analysis | `16` separated false positives, classifier misses, and links absent from the blocker. | 1,676 false positives, 7,565 candidate-level false negatives, 11,295 links missed by blocking in that analysis. |
| 10. V6 rules + richer features | `17` tried address prefixes. `18`–`21` added 15 features, retrained LightGBM, tuned thresholds, and resolved ownership. | V6 maximum recall 0.853401, only a small gain over V5. Improved V2 model: 0.894354 at global 0.65; 0.894738 after country/source tuning and ownership. |
| 11. Slower training and transliteration | `22` reduced the learning rate; `23` tested transliterated India name-token rules. | V3 model global-threshold score 0.894431. V7 rule recall 0.852906: limited additional coverage. |
| 12. V8 character-address retrieval | Retrieved top 25 address neighbors per S1/source and unioned with V5. | 95.705186% recall; 3,468,496 candidates; 7,998 newly recovered true links. |
| 13. V9 character-name retrieval | Added top 15 name neighbors per S1/source. | 96.546693% recall; 3,801,006 candidates; 646 newly recovered true links. |
| 14. Remaining-miss analysis | Checked script mismatch and text sparsity among V9 misses. | 2,651 missed true links; 1,956 in India. 45.7566% of India misses had a name-script mismatch. |
| 15. V10 multilingual retrieval | Used multilingual embeddings and ANN search for India names, top 25 per source. | 96.996105% recall; 4,168,110 candidates; 345 newly recovered true links. |
| 16. Fine-tune a bi-encoder | Built 152,891 positive pairs from S1 buckets 0 and 1, then trained multilingual E5 for one epoch on Apple MPS. | Batch 32; length 192; learning rate 2e-5; 477 warmup steps; 103.50 minutes reported. |
| 17. V11 fine-tuned retrieval | Queried India sources using the fine-tuned model over business name + address, then unioned with V10. | 98.944859% recall; 4,533,116 candidates; 1,496 additional true links; 810 still missing. |
| 18. Hard-negative retrieval | Applied fine-tuned retrieval to training buckets 0/1. | 768,318 pairs new to V5: 10,071 positives and 758,247 hard negatives. |
| 19. Full-test deadline trial | Scored six address-character slices using the earlier V2 LightGBM and assembled complete S1 match lists. | 86,627,200 candidate pairs; 4,922,502 selected links; full 1,732,544-row export. Owner reported approximately 0.88 on the public leaderboard. |
| 20. Later upgrade | `upgrade_trial_02/src` added batch feature computation, hard-negative training, model blending, rule candidates, and targeted Indic retrieval. | Training manifest: 5,785,671 pairs, 140,847 positives, 647 boosting iterations. Selection config records development macro F0.5 0.9312823859417719. No scored upgraded submission is established by retained evidence. |
| 21. Public archive | Preserve available source, aggregate reports, configurations, and hashes; add a small tested metric demonstration. | Original data, environments, intermediate tables, model weights, and submission TSVs removed during owner-authorized cleanup. |

Source: [aggregate reports](../reports/historical/), [configuration manifests](../reports/manifests/), and [archived scripts](../experiments/hackathon/). Stages 18–20 overlapped near the deadline; the table describes their dependency order rather than asserting exact execution timestamps.

