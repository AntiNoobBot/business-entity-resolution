# Methodology

## Task and data contract

S1 is a deduplicated reference source. Each S1 entity can match zero, one, or multiple S2/S3 records. Input fields are `entity_id`, `business_name`, `business_address`, and `country`. The target is a set of record IDs, not a single multiclass label.

Training countries were India and US; test also included France. The data contained noisy spelling, legal suffixes, transliterated names, mixed scripts, abbreviations, reordered address components, and missing addresses. The preserved EDA reports count 5,034,616 training S2 and 5,285,603 training S3 records. Exhaustively comparing these to 2,206,821 S1 entities would require roughly 22.8 trillion pairs.

All development was intended to use the supplied business text and labelled training pairs. The architecture uses a pretrained text encoder; it does not require entity lookups, geocoding, or business registry APIs.

## Normalization

The historical normalizer lowercases text, folds selected Latin accents, preserves Unicode letters and combining marks, and folds punctuation into spaces. It creates ordered, unique-sorted, and compact business-name forms. Legal forms can be removed from the core name while remaining available to the classifier as separate raw-text features.

Address normalization expands selected abbreviations, with a France branch where `st` means `saint` rather than `street`. Single-letter compass tokens are deliberately preserved because they can also represent building identifiers. Numeric rewrites and learned final-address-component aliases have limitations; neither is proof of identity.

`03_norm_eval.py` learns a small alias map from labelled pairs using its own seed-1 split. Later model experiments use a seed-41 split. These splits were not aligned, so the repository does not claim that all preprocessing was independent of the later evaluation entities. See [limitations](limitations.md).

## Candidate generation

Rule blocking combines exact normalized fields and selected shared token/number keys. Loose address fallbacks are capped and ranked with string similarity before union and deduplication. V5 was the practical starting point for later retrieval: approximately 114 candidates per development S1, but 14.7% of true links were still absent.

Character address retrieval (V8) added top-25 neighbors per source. The report records a 90,000-character-feature budget per country/source slice. Character name retrieval (V9) added top-15 neighbors. These methods address spelling and token-boundary variation that exact blocks miss. The original terminal implementations for these two stages were not retained as standalone scripts, so exact vectorizer settings and filtering cannot be certified from the current archive alone.

V10 added multilingual name retrieval for India. V11 added a fine-tuned multilingual E5 encoder over name and address. The retained model metadata records 384-dimensional vectors, a sequence-length limit of 192, and MultipleNegativesRankingLoss. The fine-tuning report records 152,891 positive pairs, batch size 32, one epoch, learning rate 2e-5, and 477 warmup steps. The encoder was trained on positive pairs from both India and US; the reported new V11 development retrieval was applied to India.

The later handoff configuration uses cosine HNSW search with `M=16`, `ef_construction=200`, `ef_search=100`, and `top_k=25`. These are recorded settings for that later worker, not proof of the exact ANN settings in every earlier development run. Retrieval text uses `query:` / `passage:` prefixes.

## Pair features and LightGBM

The V2 classifier consumed 35 inputs:

| Group | Features |
| --- | --- |
| Name equality | `name_core_exact`, `name_sorted_exact`, `name_compact_exact`, `name_prefix4_exact` |
| Address equality and missingness | `address_token_exact`, `tail_exact`, `s_addr_missing`, `r_addr_missing` |
| Address numbers | `shared_number_count`, `number_union_count`, `number_jaccard`, `number_count_exact` |
| Lengths | `name_length_ratio`, `address_length_ratio` |
| Fuzzy similarities | `name_ratio`, `name_token_set`, `address_ratio`, `address_token_set` |
| Token overlap | `shared_name_tokens`, `name_token_union`, `name_token_jaccard`, `shared_address_tokens`, `address_token_union`, `address_token_jaccard` |
| Legal form | `s_legal_code`, `r_legal_code`, `legal_form_agrees`, `legal_form_conflicts`, `legal_form_one_missing` |
| Partial matches | `name_partial_ratio`, `address_partial_ratio`, `name_contains`, `address_contains` |
| Encodings | `country_code`, `source_code` |

The initial 20-input model used the first 18 features plus the two encodings. V2 added the remaining 15 features. Its saved report records 700 iterations. A slower-learning V3 reached 1,396 iterations but did not materially improve the global-threshold development score.

Training candidates are labelled by joining candidate IDs to known ground-truth links. Fine-tuned retrieval exposed difficult nonmatches absent from the V5 candidate distribution. The later hard-negative run added 768,318 pairs, producing a total of 5,785,671 rows and 140,847 positives. A candidate is not a positive merely because a retrieval method ranks it highly.

The archived V2 encoding maps India to 0 and every other country to 1. This makes France look like the non-India training group to the classifier; it is a known weakness rather than an open-world country representation.

## Thresholds and ownership

For the earlier V2 development run, the selected country/source thresholds were India/S2 0.60, India/S3 0.65, US/S2 0.675, and US/S3 0.65. A target S2/S3 record can be assigned to only one S1 owner, selected by model score. This permits multiple target records per S1; it is not one-to-one matching.

The EDA found zero training targets owned by multiple S1 entities, supporting this constraint for the observed data. Later assembly resolves score ties deterministically using S1 ID. Every S1 is emitted, including entities with empty predictions.

Later upgrade metadata records a 0.70 threshold for a model blend on India/US, and a 0.65 old-model fallback for France. A separate scoring configuration retains probabilities above 0.25 to allow later threshold selection. That 0.25 floor is not the final acceptance threshold.

## Scaling and safeguards

The project used a 16 GB Mac with M2/MPS. The deadline pipeline worked in country/source slices, read candidate Parquet batches, prepared entity-side features once per slice, and wrote selected-pair shards. Its default scoring batch size is 25,000; thread counts were bounded to reduce contention.

Content hashes identify candidate inputs, model/feature configurations, and completed shards. Assembly checks completion and fingerprints before writing results. Candidate export checks S1 coverage, deduplication, and that final matches are a subset of candidates. These integrity checks establish file consistency, not predictive quality.

The archived neural worker supports resumable embedding work and transfer manifests. A Windows/CUDA split was planned; retained evidence does not establish that it produced the scored trial.

