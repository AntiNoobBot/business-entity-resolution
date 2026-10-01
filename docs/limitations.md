# Limitations and lessons

## Validation independence needs a stronger audit

Most later scripts split S1 IDs by `Polars hash(seed=41) % 100`: training buckets 0 and 1, development bucket 7, and a proposed lockbox bucket 8. The saved split audit checks the main training/development query sets and declares the lockbox unused in that experiment.

However, `03_norm_eval.py` sampled labelled links and learned aliases using `hash(seed=1) % 10 < 7`. This can include S1 entities later assigned to seed-41 development or lockbox buckets. The map was used downstream. A downstream split check does not prove that preprocessing excluded those entities. The archive preserves this issue instead of retroactively claiming a fully isolated lockbox.

Future work should freeze and persist entity IDs first, then fit every learned transform and encoder only on the allowed training partition. It should also audit duplicate/same-entity collisions in contrastive batches and target reuse across splits. Polars hash assignments are version-sensitive; a seed alone is not a durable cross-version split manifest.

## Repeated development tuning

The same development cohort guided many experiment choices. Its best score may be optimistic even without direct label leakage. No preserved lockbox evaluation resolves this uncertainty, and no private leaderboard result is available. Keep threshold tuning separate from final reporting.

## Country shift

France had no labelled training examples. Historical encodings map all non-India records to one value, and the trial used an unvalidated France threshold of 0.65. Better country-independent features and held-out-country stress tests would be more informative than assuming the US/India development score transfers.

## Retrieval and classification are different bottlenecks

V11 recovered most development true links, but enlarged the candidate pool substantially. Many added pairs are hard negatives. The classifier trained on V5 candidates faced a changed inference distribution. The later hard-negative model addressed this in part; the trial's manifest still describes the earlier matcher and only address-character test retrieval.

A high oracle ceiling does not establish high model precision. The next useful experiment would compare retrieval channels with identical reranking, ownership, and validation conditions, reporting per-country precision/recall and singleton performance alongside macro F0.5.

## Compute and deployment

The Mac could run useful retrieval and LightGBM experiments, but encoding millions of records and generating full-test candidates consumed substantial time. Some jobs finished without all of their outputs reaching final inference. Parallel hardware would only help after defining transferable inputs, resumable shards, and a frozen recipe; faster hardware alone would not guarantee a higher score.

## Archive completeness

The owner requested space recovery after the challenge. Data, virtual environments, intermediate tables, trained weights, full prediction files, and example-bearing raw reports were removed. Several terminal-only retrieval/fine-tuning implementations were not available as standalone source files. Some upgrades are preserved as code and configuration without a final benchmark.

Checksums establish which retained files or deleted artifacts were observed. They cannot restore their contents or reproduce a score. The public demo and its tests validate metric behavior, not the historical models.

