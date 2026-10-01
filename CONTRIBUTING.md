# Contributing

Keep changes focused and attach the evidence needed to evaluate them. Run the standard-library tests and repository audit from the README before submitting a change.

For a new ML result, record the exact data/split identity, candidate recipe, feature schema, dependency versions, random seeds, thresholds, singleton convention, runtime, and measured score. Clearly label development, untouched validation, and leaderboard measurements. Candidate recall and oracle scores must remain separate from model metrics.

Use synthetic fixtures in tests. Do not add challenge data, trained weights, real entity examples, credentials, or generated submission outputs to Git. Treat the historical scripts as an archive: put new experiments in a new folder and explain their relationship to the original work.

