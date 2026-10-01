# Results and evidence

## Three different quantities

For one S1 entity, with true set T and prediction set P:

`F0.5 = 1.25 × |T ∩ P| / (0.25 × |T| + |P|)`

If both sets are empty, the entity scores 1. If only one is empty, it scores 0. Macro F0.5 is the unweighted mean over every S1 entity, including singletons. It is not ordinary classification accuracy and is not the harmonic mean of dataset-wide precision and recall.

Candidate recall is the fraction of all true links present in the candidate set, pooled across S1 entities. Oracle macro F0.5 assumes an impossible perfect matcher that selects exactly the true links present in each candidate set and rejects every false candidate. The same pooled candidate recall can yield different macro oracle scores depending on which entities lost links.

The approximately 0.99649 value discussed during development was an oracle ceiling. It was never evidence of a 99.6% measured model score. No standalone oracle report remains in this archive, so it is not included as a verified result below.

## Retrieval progression

All rows below describe the same reported development cohort: 22,184 S1 entities and 76,767 true links. V8–V11 are cumulative unions. Added true links are relative to the preceding row.

| Candidate set | True-link recall | Candidates | Added candidates | Added true links |
| --- | ---: | ---: | ---: | ---: |
| V5 rules | 85.286647% | 2,530,505 | — | — |
| V8 + character addresses | 95.705186% | 3,468,496 | 937,991 | 7,998 |
| V9 + character names | 96.546693% | 3,801,006 | 332,510 | 646 |
| V10 + multilingual names | 96.996105% | 4,168,110 | 367,104 | 345 |
| V11 + fine-tuned India retrieval | 98.944859% | 4,533,116 | 365,006 | 1,496 |

V11 added **13.658212 percentage points** of candidate recall over V5. Candidate volume increased by approximately **79.14%**. It still missed **810** true links. These recall improvements created an opportunity for a better matcher; they did not measure the classifier's final precision.

Sources: [`blocking_v8_address_char_report.txt`](../reports/historical/blocking_v8_address_char_report.txt), [`blocking_v9_address_name_char_report.txt`](../reports/historical/blocking_v9_address_name_char_report.txt), [`blocking_v10_multilingual_report.txt`](../reports/historical/blocking_v10_multilingual_report.txt), [`blocking_v11_finetuned_india_report.txt`](../reports/historical/blocking_v11_finetuned_india_report.txt).

## Matching and submission results

| Configuration | Macro F0.5 | Status |
| --- | ---: | --- |
| 20-input baseline, global threshold 0.60 | 0.871792 | Historical development report |
| 35-input V2, global threshold 0.65 | 0.894354 | Historical development report |
| V2, tuned country/source thresholds, ownership | 0.894738 | Historical development report |
| Slower-learning V3, global threshold 0.60 | 0.894431 | Historical development report; not directly the same selection recipe as previous row |
| Later `address_rules` blend configuration | 0.9312823859417719 | Recorded in selection metadata; detailed evaluation table was removed before archival |
| Uploaded deadline trial | approximately 0.88 | Owner-reported public leaderboard score; no portal export or private score retained |

Sources: [baseline report](../reports/historical/lgbm_baseline_report.txt), [V2 report](../reports/historical/lgbm_v2_report.txt), [V2 ownership report](../reports/historical/one_owner_v2_report.txt), [V3 report](../reports/historical/lgbm_v3_report.txt), [later selection configuration](../reports/manifests/selection_rules.json). These development measurements were not independently rerun during archival.

The one-owner report contains a mislabeled heading: `global_0.600` actually lists 0.65 for every group. The explicit thresholds and numerical results take precedence over that heading.

## What was actually submitted

The [trial manifest](../reports/manifests/submission_manifest.json) lists six address-character candidate files, V2-based matching, and a France fallback. The [export report](../reports/manifests/candidate_export_report.json) records:

- 1,732,544 S1 rows.
- 86,627,200 unique candidate pairs.
- 4,922,502 selected matching links checked as candidate subsets.
- `PASS` for the export checks.

The final matching and candidate files were hashed again during repository preparation and matched their retained export checksums. Those hashes are in the [artifact inventory](../reports/manifests/artifact_inventory.json). The actual TSVs are not included and were deleted during cleanup.

The directory's historical `v11` name is not sufficient evidence that the full V11 recipe reached test inference. The later 0.931282 configuration also has no retained proof of an uploaded, scored result. No top-50 finish, private leaderboard result, or model score above 0.99 is claimed.

## Validation caveat

The seed-41 development bucket was used repeatedly for choosing blockers, features, models, and thresholds. Additionally, alias learning used a different split. These measurements are valuable experiment records but should not be interpreted as a final, untouched estimate of generalization. See [limitations](limitations.md).

